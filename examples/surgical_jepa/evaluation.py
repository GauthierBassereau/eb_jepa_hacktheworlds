"""Pixel-space LPIPS evaluation for the surgical action-conditioned JEPA."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any

import fire
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.open_h import OpenHDatasetConfig, OpenHWindowDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_config,
    setup_device,
    setup_seed,
    setup_wandb,
)

try:
    from examples.surgical_jepa.decoder import (
        autoregressive_latent_rollout,
        load_system_checkpoint,
        normalizer_from_state,
    )
except ModuleNotFoundError:  # Direct execution from examples/surgical_jepa.
    from decoder import (
        autoregressive_latent_rollout,
        load_system_checkpoint,
        normalizer_from_state,
    )

logger = get_logger(__name__)


def _merged_data_values(checkpoint: dict[str, Any], config) -> dict[str, Any]:
    values = OmegaConf.to_container(
        OmegaConf.create(checkpoint["jepa_config"]).data,
        resolve=True,
    )
    for key, value in OmegaConf.to_container(config.data, resolve=True).items():
        if value is not None:
            values[key] = value
    return values


def _build_evaluation_loader(
    checkpoint: dict[str, Any],
    config,
) -> tuple[DataLoader, OpenHDatasetConfig]:
    values = _merged_data_values(checkpoint, config)
    values["num_frames"] = int(config.evaluation.rollout_steps) + 1
    data_config = OpenHDatasetConfig.from_dict(values)
    dataset = OpenHWindowDataset(
        data_config,
        split="train_holdout",
        normalizer=normalizer_from_state(checkpoint["normalizer_state_dict"]),
        episode_indices=checkpoint["holdout_episode_indices"],
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": data_config.num_workers,
        "pin_memory": data_config.pin_mem,
        "persistent_workers": (
            data_config.persistent_workers and data_config.num_workers > 0
        ),
    }
    if data_config.num_workers > 0 and data_config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = data_config.prefetch_factor
    return (
        DataLoader(
            dataset,
            batch_size=data_config.val_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        ),
        data_config,
    )


def _build_visualization_dataset(
    checkpoint: dict[str, Any],
    config,
) -> OpenHWindowDataset:
    values = _merged_data_values(checkpoint, config)
    values.update(
        {
            "num_frames": int(config.evaluation.visualization.rollout_steps) + 1,
            "batch_size": 1,
            "val_batch_size": 1,
            "num_workers": 0,
            "pin_mem": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "drop_last": False,
        }
    )
    data_config = OpenHDatasetConfig.from_dict(values)
    return OpenHWindowDataset(
        data_config,
        split="train_holdout",
        normalizer=normalizer_from_state(checkpoint["normalizer_state_dict"]),
        episode_indices=checkpoint["holdout_episode_indices"],
    )


def _motion_score(sample) -> float:
    proprioception = sample.raw_proprioception
    left = (proprioception[0:3, -1] - proprioception[0:3, 0]).norm()
    right = (proprioception[8:11, -1] - proprioception[8:11, 0]).norm()
    grippers = (proprioception[7, -1] - proprioception[7, 0]).abs() + (
        proprioception[15, -1] - proprioception[15, 0]
    ).abs()
    return float(left + right + 0.1 * grippers)


def select_rollout_visualizations(
    dataset: OpenHWindowDataset,
    *,
    num_examples: int,
    selection: str,
    candidate_count: int,
    dataset_indices: list[int] | None = None,
) -> list[tuple[int, Any, float]]:
    """Select deterministic windows, preferring distinct held-out episodes."""
    if num_examples <= 0:
        return []
    if dataset_indices:
        invalid = [index for index in dataset_indices if not 0 <= index < len(dataset)]
        if invalid:
            raise ValueError(f"Visualization dataset indices out of range: {invalid}")
        return [
            (index, dataset[index], _motion_score(dataset[index]))
            for index in dataset_indices[:num_examples]
        ]
    if len(dataset) == 0:
        raise ValueError("No held-out windows support the visualization horizon")

    candidate_count = min(max(candidate_count, num_examples), len(dataset))
    indices = (
        torch.linspace(0, len(dataset) - 1, steps=candidate_count)
        .round()
        .to(torch.int64)
        .unique()
        .tolist()
    )
    candidates = [
        (index, dataset[index], _motion_score(dataset[index])) for index in indices
    ]
    if selection == "motion":
        candidates.sort(key=lambda item: item[2], reverse=True)
    elif selection != "even":
        raise ValueError("Visualization selection must be 'motion' or 'even'")

    selected = []
    selected_episodes = set()
    for item in candidates:
        episode = int(item[1].episode_index)
        if episode not in selected_episodes:
            selected.append(item)
            selected_episodes.add(episode)
        if len(selected) == num_examples:
            return selected
    selected_indices = {item[0] for item in selected}
    selected.extend(item for item in candidates if item[0] not in selected_indices)
    return selected[:num_examples]


def clean_context_latent_rollout(
    jepa,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Predict each future latent using only clean encoded latent context."""
    clean_latents = jepa.encoder(states[:, :, : steps + 1])
    encoded_actions = jepa.action_encoder(actions[:, :, :steps])
    context_length = max(1, int(getattr(jepa.predictor, "context_length", 1)))
    predictions = []
    for horizon in range(steps):
        start = max(0, horizon + 1 - context_length)
        context = clean_latents[:, :, start : horizon + 1]
        context_actions = encoded_actions[:, :, start : horizon + 1]
        predictions.append(jepa.predictor(context, context_actions)[:, :, -1:])
    return torch.cat([clean_latents[:, :, :1], *predictions], dim=2)


def _lpips_by_horizon(
    metric,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return LPIPS as [batch, horizon] for RGB clips in [0, 1]."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "LPIPS expects matching [B,3,T,H,W] clips, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    batch_size, channels, timesteps, height, width = prediction.shape
    if channels != 3:
        raise ValueError("LPIPS evaluation requires RGB frames")
    prediction = (
        prediction.permute(0, 2, 1, 3, 4)
        .reshape(batch_size * timesteps, channels, height, width)
        .float()
        .clamp(0, 1)
        .mul(2)
        .sub(1)
    )
    target = (
        target.permute(0, 2, 1, 3, 4)
        .reshape(batch_size * timesteps, channels, height, width)
        .float()
        .clamp(0, 1)
        .mul(2)
        .sub(1)
    )
    return metric(prediction, target, normalize=False).reshape(batch_size, timesteps)


@torch.inference_mode()
def evaluate_batch(
    jepa,
    decoder,
    lpips_metric,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    rollout_steps: int,
    use_amp: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return autoregressive and clean-context pixel LPIPS, both [B, H]."""
    states = states[:, :, : rollout_steps + 1]
    actions = actions[:, :, :rollout_steps]
    with autocast(states.device.type, enabled=use_amp, dtype=dtype):
        autoregressive_latents = autoregressive_latent_rollout(
            jepa,
            states[:, :, :1],
            actions,
            steps=rollout_steps,
        )
        clean_context_latents = clean_context_latent_rollout(
            jepa,
            states,
            actions,
            steps=rollout_steps,
        )
        autoregressive_rgb = decoder(autoregressive_latents[:, :, 1:])
        clean_context_rgb = decoder(clean_context_latents[:, :, 1:])
    targets = states[:, :, 1 : rollout_steps + 1]
    return (
        _lpips_by_horizon(lpips_metric, autoregressive_rgb, targets),
        _lpips_by_horizon(lpips_metric, clean_context_rgb, targets),
    )


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def make_rollout_comparison_video(
    ground_truth: torch.Tensor,
    autoregressive: torch.Tensor,
    clean_context: torch.Tensor,
    *,
    episode_index: int,
) -> torch.Tensor:
    """Render GT, autoregressive, and clean-context rollouts as uint8 TCHW."""
    if not (
        ground_truth.shape == autoregressive.shape == clean_context.shape
        and ground_truth.ndim == 4
        and ground_truth.shape[0] == 3
    ):
        raise ValueError("Visualization expects matching RGB [3,T,H,W] clips")
    _, timesteps, height, width = ground_truth.shape
    separator = 2
    label_height = 20
    columns = (
        ("GROUND TRUTH", ground_truth),
        ("AUTOREGRESSIVE", autoregressive),
        ("CLEAN CONTEXT", clean_context),
    )
    frames = []
    for timestep in range(timesteps):
        canvas = Image.new(
            "RGB",
            (3 * width + 2 * separator, height + label_height),
            color="black",
        )
        draw = ImageDraw.Draw(canvas)
        for column, (label, clip) in enumerate(columns):
            x = column * (width + separator)
            draw.text(
                (x + 4, 4),
                f"{label}  ep {episode_index}  t {timestep}",
                fill="white",
            )
            canvas.paste(_tensor_to_pil(clip[:, timestep]), (x, label_height))
        frames.append(torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1))
    return torch.stack(frames).contiguous()


@torch.inference_mode()
def create_rollout_visualizations(
    jepa,
    decoder,
    selected_samples: list[tuple[int, Any, float]],
    *,
    device: torch.device,
    rollout_steps: int,
    use_amp: bool,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if not selected_samples:
        return []
    states = torch.stack(
        [item[1].states[:, : rollout_steps + 1] for item in selected_samples]
    ).to(device)
    actions = torch.stack(
        [item[1].actions[:, :rollout_steps] for item in selected_samples]
    ).to(device)
    with autocast(device.type, enabled=use_amp, dtype=dtype):
        autoregressive = decoder(
            autoregressive_latent_rollout(
                jepa, states[:, :, :1], actions, steps=rollout_steps
            )
        ).float()
        clean_context = decoder(
            clean_context_latent_rollout(
                jepa, states, actions, steps=rollout_steps
            )
        ).float()

    visualizations = []
    for index, (dataset_index, sample, motion_score) in enumerate(selected_samples):
        visualizations.append(
            {
                "video": make_rollout_comparison_video(
                    states[index].cpu(),
                    autoregressive[index].cpu(),
                    clean_context[index].cpu(),
                    episode_index=int(sample.episode_index),
                ),
                "dataset_index": dataset_index,
                "episode_index": int(sample.episode_index),
                "motion_score": motion_score,
                "first_frame": int(sample.frame_indices[0]),
                "last_frame": int(sample.frame_indices[rollout_steps]),
            }
        )
    return visualizations


def _wandb_directory(checkpoint_path: Path, cfg) -> Path:
    if cfg.logging.get("run_dir"):
        result = Path(cfg.logging.run_dir)
        result.mkdir(parents=True, exist_ok=True)
        return result
    return get_unified_experiment_dir(
        example_name="surgical_jepa_evaluation",
        sweep_name=get_default_dev_name(),
        exp_name=f"evaluation_{checkpoint_path.parent.name}",
        seed=cfg.meta.seed,
    )


def _log_wandb_results(
    wandb_run,
    results: dict[str, Any],
    *,
    sample_fps: int,
    visualizations: list[dict[str, Any]],
    video_fps: int,
) -> None:
    if not wandb_run:
        return
    import wandb

    curve = wandb.Table(
        columns=[
            "horizon",
            "seconds",
            "autoregressive_lpips",
            "clean_context_lpips",
            "compounding_gap",
        ]
    )
    logs = {
        "eval/one_step_lpips": results["one_step_lpips"],
        "eval/autoregressive_lpips_mean": results["autoregressive_lpips_mean"],
        "eval/autoregressive_lpips_final": results[
            "autoregressive_lpips_by_horizon"
        ][-1],
        "eval/clean_context_lpips_mean": results["clean_context_lpips_mean"],
        "eval/compounding_gap_mean": results["compounding_gap_mean"],
        "eval/compounding_gap_final": results["compounding_gap_by_horizon"][-1],
        "eval/num_windows": results["num_windows"],
    }
    for horizon, (ar, clean, gap) in enumerate(
        zip(
            results["autoregressive_lpips_by_horizon"],
            results["clean_context_lpips_by_horizon"],
            results["compounding_gap_by_horizon"],
        ),
        start=1,
    ):
        curve.add_data(horizon, horizon / sample_fps, ar, clean, gap)
        logs[f"eval/autoregressive_lpips/h{horizon}"] = ar
        logs[f"eval/clean_context_lpips/h{horizon}"] = clean
        logs[f"eval/compounding_gap/h{horizon}"] = gap

    logs["charts/lpips_rollout_comparison"] = wandb.plot.line_series(
        xs=list(range(1, len(results["autoregressive_lpips_by_horizon"]) + 1)),
        ys=[
            results["autoregressive_lpips_by_horizon"],
            results["clean_context_lpips_by_horizon"],
        ],
        keys=["autoregressive", "clean context"],
        title="Pixel LPIPS by rollout horizon",
        xname="horizon",
    )
    logs["charts/lpips_compounding_gap"] = wandb.plot.line(
        curve,
        "horizon",
        "compounding_gap",
        title="LPIPS cost of generated context",
    )
    if visualizations:
        combined = torch.cat([item["video"] for item in visualizations], dim=2)
        logs["viz/rollout_comparison"] = wandb.Video(
            combined.numpy(),
            fps=video_fps,
            format="mp4",
            caption="Ground truth | fully autoregressive | clean latent context",
        )
        for index, item in enumerate(visualizations):
            logs[f"viz/rollout_{index:02d}"] = wandb.Video(
                item["video"].numpy(),
                fps=video_fps,
                format="mp4",
                caption=(
                    f"dataset_index={item['dataset_index']}, "
                    f"episode={item['episode_index']}, "
                    f"frames={item['first_frame']}..{item['last_frame']}, "
                    f"motion={item['motion_score']:.5f}"
                ),
            )
    wandb.log(logs, step=0)


@torch.inference_mode()
def run(
    checkpoint: str,
    fname: str = "examples/surgical_jepa/evaluation.yaml",
    cfg=None,
    **overrides,
):
    """Evaluate decoded predictions exclusively in pixel space with LPIPS."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)
    setup_seed(cfg.meta.seed)
    device = setup_device(cfg.meta.get("device", "auto"))
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    jepa, decoder, state = load_system_checkpoint(checkpoint_path, device)
    loader, data_config = _build_evaluation_loader(state, cfg)

    rollout_steps = int(cfg.evaluation.rollout_steps)
    if rollout_steps <= 0:
        raise ValueError("evaluation.rollout_steps must be positive")
    use_amp = bool(cfg.evaluation.use_amp and device.type == "cuda")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(str(cfg.evaluation.dtype).lower(), torch.bfloat16)
    lpips_network = str(cfg.evaluation.lpips_network)
    lpips_metric = lpips.LPIPS(net=lpips_network).to(device).eval()
    for parameter in lpips_metric.parameters():
        parameter.requires_grad_(False)

    visualization_config = cfg.evaluation.visualization
    visualization_steps = int(visualization_config.rollout_steps)
    if visualization_steps <= 0:
        raise ValueError("evaluation.visualization.rollout_steps must be positive")

    run_dir = _wandb_directory(checkpoint_path, cfg)
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config={
            "example": "surgical_jepa_pixel_evaluation",
            "decoder_checkpoint": str(checkpoint_path),
            "holdout_episode_indices": state["holdout_episode_indices"],
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=run_dir,
        run_name=cfg.logging.get("run_name") or run_dir.name,
        tags=["surgical_jepa", "evaluation", "lpips", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", True),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    selected = []
    if (
        wandb_run
        and visualization_config.get("enabled", True)
        and int(visualization_config.num_examples) > 0
    ):
        selected = select_rollout_visualizations(
            _build_visualization_dataset(state, cfg),
            num_examples=int(visualization_config.num_examples),
            selection=str(visualization_config.selection),
            candidate_count=int(visualization_config.candidate_count),
            dataset_indices=(
                list(visualization_config.dataset_indices)
                if visualization_config.get("dataset_indices")
                else None
            ),
        )

    autoregressive_sum = torch.zeros(rollout_steps, dtype=torch.float64)
    clean_context_sum = torch.zeros(rollout_steps, dtype=torch.float64)
    num_windows = 0
    max_batches = cfg.evaluation.get("max_batches")
    start = time()
    progress = tqdm(loader, desc="Evaluating pixel-space rollouts")
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        states = batch.states.to(device, non_blocking=True)
        actions = batch.actions.to(device, non_blocking=True)
        autoregressive_lpips, clean_context_lpips = evaluate_batch(
            jepa,
            decoder,
            lpips_metric,
            states,
            actions,
            rollout_steps=rollout_steps,
            use_amp=use_amp,
            dtype=dtype,
        )
        autoregressive_sum += autoregressive_lpips.double().sum(dim=0).cpu()
        clean_context_sum += clean_context_lpips.double().sum(dim=0).cpu()
        num_windows += states.shape[0]
        progress.set_postfix(
            ar_lpips=f"{autoregressive_lpips.mean().item():.3f}",
            clean_lpips=f"{clean_context_lpips.mean().item():.3f}",
        )
    if num_windows == 0:
        raise ValueError("Evaluation processed no held-out windows")

    autoregressive_curve = (autoregressive_sum / num_windows).tolist()
    clean_context_curve = (clean_context_sum / num_windows).tolist()
    gap_curve = [
        autoregressive - clean
        for autoregressive, clean in zip(
            autoregressive_curve, clean_context_curve
        )
    ]
    results = {
        "one_step_lpips": clean_context_curve[0],
        "autoregressive_lpips_by_horizon": autoregressive_curve,
        "autoregressive_lpips_mean": sum(autoregressive_curve) / rollout_steps,
        "clean_context_lpips_by_horizon": clean_context_curve,
        "clean_context_lpips_mean": sum(clean_context_curve) / rollout_steps,
        "compounding_gap_by_horizon": gap_curve,
        "compounding_gap_mean": sum(gap_curve) / rollout_steps,
        "num_windows": num_windows,
        "rollout_steps": rollout_steps,
        "lpips_network": lpips_network,
        "elapsed_seconds": time() - start,
    }
    visualizations = create_rollout_visualizations(
        jepa,
        decoder,
        selected,
        device=device,
        rollout_steps=visualization_steps,
        use_amp=use_amp,
        dtype=dtype,
    )
    _log_wandb_results(
        wandb_run,
        results,
        sample_fps=data_config.sample_fps,
        visualizations=visualizations,
        video_fps=int(visualization_config.get("video_fps") or data_config.sample_fps),
    )

    logger.info("One-step LPIPS: %.6f", results["one_step_lpips"])
    logger.info(
        "Autoregressive LPIPS by horizon: %s",
        [round(value, 6) for value in autoregressive_curve],
    )
    logger.info(
        "Clean-context LPIPS by horizon: %s",
        [round(value, 6) for value in clean_context_curve],
    )
    logger.info(
        "Compounding gap by horizon: %s",
        [round(value, 6) for value in gap_curve],
    )
    if wandb_run:
        import wandb

        wandb.finish()
    return results


if __name__ == "__main__":
    fire.Fire(run)
