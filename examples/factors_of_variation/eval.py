"""Factor-of-variation stress test: zero-shot planning sweep on a FIXED checkpoint.

Loads a trained world model once (using your ``main.build_jepa``) and evaluates
goal-conditioned planning across a grid of controllable Two Rooms perturbations
(``cfgs/eval.yaml:grid``), writing per-point success to JSON. Perturbations are applied
through the standard ``data:`` config path, so they reach the planning env (not the encoder
geometry). Aggregate several seeds into a figure with ``make_figure.py``.

    python -m examples.factors_of_variation.eval --model_folder <ckpt_dir>
"""
import json
import os
from pathlib import Path

import fire
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.jepa import JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint, setup_device, setup_seed
from examples.ac_video_jepa.eval import launch_plan_eval
from examples.factors_of_variation.main import build_jepa  # your recipe assembly

logger = get_logger(__name__)


def run(
    model_folder: str,
    eval_cfg: str = "examples/factors_of_variation/cfgs/eval.yaml",
    out: str = None,
    checkpoint: str = "latest.pth.tar",
    seed: int = 0,
):
    """Sweep planning success over the perturbation grid in ``eval_cfg``."""
    model_folder = Path(model_folder)
    cfg = OmegaConf.load(model_folder / "config.yaml")
    ev = yaml.safe_load(open(eval_cfg))
    setup_device("auto")
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_data = OmegaConf.to_container(cfg.data, resolve=True)
    base_data.pop("pipeline", None)  # eval env needs geometry only; never streams data
    loader, val_loader, data_config, _ = init_data(
        env_name=cfg.data.env_name,
        cfg_data={**base_data, "pipeline": {"mode": "online"}}, device=device)

    jepa = build_jepa(cfg, data_config, device)
    ckpt = load_checkpoint(model_folder / checkpoint, jepa, device=device)
    if not ckpt.get("resumed"):
        raise FileNotFoundError(f"No checkpoint at {model_folder / checkpoint}")
    feat_dim = jepa.encode(torch.zeros(
        1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size, device=device)).shape[1]
    xy_head = MLPXYHead(input_shape=feat_dim, normalizer=loader.dataset.normalizer).to(device)
    if "xy_head_state_dict" in ckpt:
        xy_head.load_state_dict(ckpt["xy_head_state_dict"])
    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())
    jepa.eval()

    plan_cfg = yaml.load(open(ev["plan_cfg_path"]), Loader=yaml.FullLoader)
    plan_cfg.setdefault("logging", {}).update(
        {"tqdm_silent": True, "optional_plots": False, "save_gif": False})

    results = []
    for point in ev["grid"]:
        label, overrides = point["label"], point.get("data", {})
        _, _, env_config, _ = init_data(
            env_name=cfg.data.env_name, cfg_data={**base_data, **overrides})

        def env_creator(_cfg=env_config):
            return create_env(cfg.data.env_name, config=_cfg,
                              n_allowed_steps=ev["n_allowed_steps"], level=ev["level"])

        res = launch_plan_eval(
            jepa, env_creator, model_folder / "fov_eval", 0, 0, suffix=f"_{label}",
            num_eval_episodes=ev["num_episodes"], n_parallel=ev["n_parallel"],
            loader=val_loader, prober=xy_prober, plan_cfg=plan_cfg, value_head=None)
        row = {"label": label, "overrides": overrides,
               "success_rate": float(res["success_rate"]),
               "mean_state_dist": float(res["mean_state_dist"]),
               "num_episodes": ev["num_episodes"]}
        results.append(row)
        logger.info(f"[{label}] success={row['success_rate']:.3f} dist={row['mean_state_dist']:.2f}")

    out = out or str(model_folder / "fov_sweep.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"model_folder": str(model_folder), "checkpoint": checkpoint,
               "results": results}, open(out, "w"), indent=2)
    logger.info(f"Wrote {out}")
    return results


if __name__ == "__main__":
    fire.Fire(run)
