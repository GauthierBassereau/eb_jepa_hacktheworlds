"""Factors of variation (Track 13) — AC-Video-JEPA training entrypoint (SKELETON).

The track stress-tests the *minimal* Two Rooms recipe: train an action-conditioned
Video-JEPA, then sweep planning success as you dial controllable factors of variation
out of distribution (``eval.py``), and find which factor / regularizer term breaks first.

The DATA + TRAINING LOOP are provided. The one modelling piece you implement is the
recipe assembly, marked ``# TODO`` below — ``build_jepa``. Everything you need is already
in ``eb_jepa`` (do NOT reimplement encoders/losses); ``examples/ac_video_jepa/main.py`` is
the full reference for the same recipe.

Run (via the launcher, HTW SLURM autoconfig):
    python -m examples.launch_sbatch --example factors_of_variation --full-sweep   # 3 seeds
    python -m examples.launch_sbatch --example factors_of_variation --model.dstc 128      # capacity
    python -m examples.launch_sbatch --example factors_of_variation --data.door_space 2   # train perturbed
or directly:
    python -m examples.factors_of_variation.main --fname examples/factors_of_variation/cfgs/train.yaml
"""
import os
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# RECIPE ASSEMBLY  — # TODO  (the one piece you implement)
# --------------------------------------------------------------------------- #
def build_jepa(cfg, data_config, device=None):
    """TODO: build and return the action-conditioned Video-JEPA ``eb_jepa.jepa.JEPA``
    for the minimal Two Rooms recipe. Reuse eb_jepa (do not reimplement):

      * encoder:    ``eb_jepa.architectures.ImpalaEncoder`` with
                    ``stack_sizes=(16, cfg.model.henc, cfg.model.dstc)``,
                    ``input_channels=cfg.model.dobs``,
                    ``input_shape=(cfg.model.dobs, img, img)``, ``final_ln=True``,
                    ``mlp_output_dim=512`` (``img = data_config.img_size``).
      * predictor:  ``eb_jepa.architectures.RNNPredictor(hidden_size=encoder.mlp_output_dim,
                    final_ln=encoder.final_ln)``.
      * action enc: ``nn.Identity()`` (actions are 2-D dot displacements).
      * regularizer (the part this track stresses): ``eb_jepa.losses.VC_IDM_Sim_Regularizer``
                    with cov/std/sim_t/idm coeffs from ``cfg.model.regularizer`` and an
                    ``eb_jepa.architectures.InverseDynamicsModel(state_dim=h*w*f, hidden_dim=256,
                    action_dim=2)`` (f, h, w = encoder output channels/height/width).
      * pred loss:  ``eb_jepa.losses.SquareLossSeq()``.

    Assemble via ``JEPA(encoder, action_encoder, predictor, regularizer, pred_loss)`` and
    ``.to(device)``. The training loop below drives it with ``jepa.unroll(x, a, ...)``.
    See ``examples/ac_video_jepa/main.py`` for the full reference implementation.
    """
    raise NotImplementedError("TODO: assemble the AC-Video-JEPA recipe (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(
    fname="examples/factors_of_variation/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            exp_name = folder.name.rsplit("_seed", 1)[0]
        else:
            exp_name = get_exp_name("ac_video_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="factors_of_variation", sweep_name=get_default_dev_name(),
                exp_name=exp_name, seed=cfg.meta.seed)
    else:
        folder = Path(folder)
        exp_name = folder.name.rsplit("_seed", 1)[0]
    os.makedirs(folder, exist_ok=True)

    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, val_loader, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    if data_pipeline is not None:
        data_pipeline.warm_up()
    setup_wandb(project="eb_jepa",
                config={"example": "factors_of_variation", **OmegaConf.to_container(cfg, resolve=True)},
                run_dir=folder, run_name=exp_name, tags=[f"seed_{cfg.meta.seed}", "factors_of_variation"],
                group=cfg.logging.get("wandb_group"), enabled=cfg.logging.get("log_wandb", False))
    log_data_info(cfg.data.env_name, len(loader), data_config.batch_size,
                  train_samples=data_config.size, val_samples=data_config.val_size)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)

    # -- MODEL (your build_jepa) + a position prober (used by some planners + as a diagnostic)
    jepa = build_jepa(cfg, data_config, device)
    with torch.no_grad():
        feat_dim = jepa.encode(
            torch.zeros(1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size,
                        device=device)).shape[1]
    xy_head = MLPXYHead(input_shape=feat_dim, normalizer=loader.dataset.normalizer).to(device)
    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())
    log_config(cfg)

    total_steps = cfg.optim.epochs * max(1, data_config.size // data_config.batch_size)
    jepa_opt = AdamW(jepa.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.get("weight_decay", 1e-6))
    jepa_sched = CosineWithWarmup(jepa_opt, total_steps, warmup_ratio=0.1)
    probe_opt = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_sched = CosineWithWarmup(probe_opt, total_steps, warmup_ratio=0.1)

    start_epoch = 0
    latest_ckpt = folder / "latest.pth.tar"
    if cfg.meta.get("load_model"):
        info = load_checkpoint(latest_ckpt, jepa, jepa_opt, jepa_sched, device=device)
        start_epoch = info.get("epoch", 0)
        if "xy_head_state_dict" in info:
            xy_head.load_state_dict(info["xy_head_state_dict"])
    OmegaConf.save(cfg, folder / "config.yaml")
    if torch.cuda.is_available() and cfg.model.get("compile"):
        jepa = torch.compile(jepa)

    for epoch in range(start_epoch, cfg.optim.epochs):
        t0 = time()
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}",
                    disable=cfg.logging.get("tqdm_silent", False))
        jepa_loss = xy_loss = regl = pl = torch.tensor(0.0, device=device)
        for idx, (x, a, loc, _, _) in pbar:
            x = x.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True)

            jepa_opt.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                    x, a, nsteps=cfg.model.nsteps, unroll_mode="autoregressive",
                    ctxt_window_time=1, compute_loss=True, return_all_steps=False)
            scaler.scale(jepa_loss).backward()
            if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                scaler.unscale_(jepa_opt)
                torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), cfg.optim.grad_clip_enc)
                torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), cfg.optim.grad_clip_pred)
            scaler.step(jepa_opt)
            scaler.update()
            jepa_sched.step()

            probe_opt.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                xy_loss = loader.dataset.normalizer.unnormalize_mse(
                    xy_prober(observations=x[:, :, :1], targets=loc[:, :, :1]))
            scaler.scale(xy_loss).backward()
            scaler.step(probe_opt)
            scaler.update()
            probe_sched.step()
            pbar.set_postfix({"loss": f"{jepa_loss.item():.3f}", "probe": f"{xy_loss.item():.3f}"})

        log_epoch(epoch, {"loss": jepa_loss.item(), "reg": regl.item(), "pred": pl.item(),
                          "probe": xy_loss.item()},
                  total_epochs=cfg.optim.epochs, elapsed_time=time() - t0)
        ck = dict(model=jepa, optimizer=jepa_opt, scheduler=jepa_sched, epoch=epoch,
                  step=epoch * len(loader), xy_head_state_dict=xy_head.state_dict())
        save_checkpoint(latest_ckpt, **ck)
        if epoch % cfg.logging.get("save_every_n_epochs", 1) == 0:
            save_checkpoint(folder / f"e-{epoch}.pth.tar", **ck)

    if data_pipeline is not None:
        data_pipeline.shutdown()


if __name__ == "__main__":
    fire.Fire(run)
