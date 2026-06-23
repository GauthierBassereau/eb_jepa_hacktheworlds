"""Benchmark surgical JEPA encoding + autoregressive latent rollout."""

from pathlib import Path
from time import perf_counter

import fire
import torch
from omegaconf import OmegaConf

from examples.surgical_jepa.decoder import clean_state_dict
from examples.surgical_jepa.main import build_surgical_jepa


def benchmark(path: str, batch_size: int, steps: int, warmup: int, repeats: int):
    device = torch.device("cuda")
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["jepa_config"])
    model = build_surgical_jepa(
        cfg.model, int(cfg.data.image_size), load_pretrained_encoder=False
    ).to(device)
    model.load_state_dict(clean_state_dict(state["jepa_state_dict"]))
    model.eval()

    frames = torch.rand(
        batch_size, 3, 1, cfg.data.image_size, cfg.data.image_size, device=device
    )
    actions = torch.randn(batch_size, cfg.model.action_dim, steps, device=device)
    context = max(1, int(getattr(model.predictor, "context_length", 1)))

    def run():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.unroll(
                frames,
                actions,
                nsteps=steps,
                unroll_mode="autoregressive",
                ctxt_window_time=context,
                compute_loss=False,
            )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        run()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start

    ms = 1000 * elapsed / repeats
    print(
        f"{Path(path).name:24s} {ms:8.2f} ms/rollout | "
        f"{batch_size * steps / (elapsed / repeats):8.1f} predicted steps/s"
    )


def main(
    batch_size: int = 1,
    steps: int = 8,
    warmup: int = 20,
    repeats: int = 100,
):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    print(f"device={torch.cuda.get_device_name(0)}")
    for path in ("checkpoints/baseline_eb_jepa.tar", "checkpoints/transformer.tar"):
        benchmark(path, batch_size, steps, warmup, repeats)


if __name__ == "__main__":
    fire.Fire(main)
