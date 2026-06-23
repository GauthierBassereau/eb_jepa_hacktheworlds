"""Train an RGB decoder on frozen surgical AC-JEPA representations."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any

import fire
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.open_h import OpenHDatasetConfig, OpenHWindowDataset
from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_config,
    log_data_info,
    log_epoch,
    log_model_info,
    setup_device,
    setup_seed,
    setup_wandb,
)

try:
    from examples.surgical_jepa.decoder import (
        build_decoder,
        clean_state_dict,
        load_frozen_jepa,
        normalizer_from_state,
        save_decoder_checkpoint,
    )
except ModuleNotFoundError:  # Direct execution from examples/surgical_jepa.
    from decoder import (
        build_decoder,
        clean_state_dict,
        load_frozen_jepa,
        normalizer_from_state,
        save_decoder_checkpoint,
    )

logger = get_logger(__name__)


def _plain_config(config) -> dict[str, Any]:
    return OmegaConf.to_container(config, resolve=True)


def _decoder_data_config(jepa_config, decoder_config) -> OpenHDatasetConfig:
    values = _plain_config(jepa_config.data)
    for key, value in _plain_config(decoder_config.data).items():
        if value is not None:
            values[key] = value
    values["num_frames"] = 2
    return OpenHDatasetConfig.from_dict(values)


def make_decoder_loaders(
    jepa_config,
    jepa_checkpoint: dict[str, Any],
    decoder_config,
) -> tuple[DataLoader, DataLoader, Any]:
    """Build decoder loaders using the exact split stored by JEPA training."""
    train_episode_indices = tuple(jepa_checkpoint["train_episode_indices"])
    holdout_episode_indices = tuple(jepa_checkpoint["holdout_episode_indices"])
    if set(train_episode_indices) & set(holdout_episode_indices):
        raise ValueError("JEPA checkpoint train and holdout episode sets overlap")

    data_config = _decoder_data_config(jepa_config, decoder_config)
    normalizer = normalizer_from_state(jepa_checkpoint["normalizer_state_dict"])
    train_dataset = OpenHWindowDataset(
        data_config,
        split="train",
        normalizer=normalizer,
        episode_indices=train_episode_indices,
    )
    val_dataset = OpenHWindowDataset(
        data_config,
        split="train_holdout",
        normalizer=normalizer,
        episode_indices=holdout_episode_indices,
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

    generator = torch.Generator().manual_seed(data_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        drop_last=data_config.drop_last,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, data_config


def _select_training_frames(states: torch.Tensor) -> torch.Tensor:
    """Randomly choose one of the two decoded frames for each sample."""
    batch_size, _, timesteps, _, _ = states.shape
    indices = torch.randint(timesteps, (batch_size,), device=states.device)
    return states.permute(0, 2, 1, 3, 4)[
        torch.arange(batch_size, device=states.device), indices
    ]


def _select_validation_frames(states: torch.Tensor) -> torch.Tensor:
    """Use the first frame deterministically for held-out reconstruction."""
    return states[:, :, 0]


def _encode_frames(jepa, frames: torch.Tensor) -> torch.Tensor:
    return jepa.encoder(frames.unsqueeze(2)).squeeze(2)


def _tensor_to_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array)


def make_reconstruction_grid(
    ground_truth: torch.Tensor,
    reconstruction: torch.Tensor,
    max_examples: int = 8,
) -> Image.Image:
    """Create the fixed W&B ``ground truth | decoded`` reconstruction grid."""
    count = min(max_examples, ground_truth.shape[0])
    ground_truth = ground_truth[:count]
    reconstruction = reconstruction[:count]
    sample = _tensor_to_image(ground_truth[0])
    width, height = sample.size
    label_height = 22
    canvas = Image.new(
        "RGB",
        (count * width, 2 * (height + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, (label, values) in enumerate(
        (("Ground truth", ground_truth), ("Decoded real latent", reconstruction))
    ):
        y = row * (height + label_height)
        draw.text((4, y + 3), label, fill="black")
        for column in range(count):
            canvas.paste(
                _tensor_to_image(values[column]),
                (column * width, y + label_height),
            )
    return canvas


def make_sequence_reconstruction_grid(
    ground_truth: torch.Tensor,
    reconstruction: torch.Tensor,
    frame_indices: list[int],
) -> Image.Image:
    """Create a two-row reconstruction grid for one moving sequence."""
    if ground_truth.shape != reconstruction.shape:
        raise ValueError(
            "Ground truth and reconstruction must match, got "
            f"{tuple(ground_truth.shape)} and {tuple(reconstruction.shape)}"
        )
    ground_truth = ground_truth[:, frame_indices]
    reconstruction = reconstruction[:, frame_indices]
    sample = _tensor_to_image(ground_truth[:, 0])
    width, height = sample.size
    label_width = 110
    header_height = 22
    canvas = Image.new(
        "RGB",
        (
            label_width + len(frame_indices) * width,
            header_height + 2 * height,
        ),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, frame_index in enumerate(frame_indices):
        draw.text(
            (label_width + column * width + 4, 4),
            f"t={frame_index}",
            fill="black",
        )
    for row, (label, sequence) in enumerate(
        (("Ground truth", ground_truth), ("Decoded latent", reconstruction))
    ):
        y = header_height + row * height
        draw.text((4, y + 4), label, fill="black")
        for column in range(len(frame_indices)):
            canvas.paste(
                _tensor_to_image(sequence[:, column]),
                (label_width + column * width, y),
            )
    return canvas


def _make_visualization_dataset(
    jepa_config,
    source_checkpoint: dict[str, Any],
    decoder_config,
) -> OpenHWindowDataset:
    """Build full-length held-out windows solely for W&B visualization."""
    values = _plain_config(jepa_config.data)
    for key, value in _plain_config(decoder_config.data).items():
        if value is not None:
            values[key] = value
    values.update(
        {
            "num_frames": int(jepa_config.data.num_frames),
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
        normalizer=normalizer_from_state(source_checkpoint["normalizer_state_dict"]),
        episode_indices=source_checkpoint["holdout_episode_indices"],
    )


def _motion_score(sample) -> float:
    """Score a window by endpoint arm and gripper displacement."""
    proprioception = sample.raw_proprioception
    left_motion = (proprioception[0:3, -1] - proprioception[0:3, 0]).norm()
    right_motion = (proprioception[8:11, -1] - proprioception[8:11, 0]).norm()
    gripper_motion = (proprioception[7, -1] - proprioception[7, 0]).abs() + (
        proprioception[15, -1] - proprioception[15, 0]
    ).abs()
    return float(left_motion + right_motion + 0.1 * gripper_motion)


def select_visualization_samples(
    dataset: OpenHWindowDataset,
    *,
    num_sequences: int,
    selection: str,
    candidate_count: int,
    dataset_indices: list[int] | None = None,
) -> list[tuple[int, Any, float]]:
    """Select fixed windows explicitly, evenly, or by high motion."""
    if dataset_indices:
        invalid = [index for index in dataset_indices if not 0 <= index < len(dataset)]
        if invalid:
            raise ValueError(
                f"visualization dataset_indices are out of range: {invalid}"
            )
        selected = []
        for index in dataset_indices[:num_sequences]:
            sample = dataset[index]
            selected.append((index, sample, _motion_score(sample)))
        return selected

    candidate_count = min(max(candidate_count, num_sequences), len(dataset))
    candidate_indices = (
        torch.linspace(0, len(dataset) - 1, steps=candidate_count)
        .round()
        .to(torch.int64)
        .unique()
        .tolist()
    )
    candidates = [
        (index, dataset[index], _motion_score(dataset[index]))
        for index in candidate_indices
    ]
    if selection == "motion":
        candidates.sort(key=lambda item: item[2], reverse=True)
    elif selection != "even":
        raise ValueError("logging.visualization.selection must be 'motion' or 'even'")

    selected = []
    selected_episodes = set()
    for item in candidates:
        episode_index = int(item[1].episode_index)
        if episode_index in selected_episodes:
            continue
        selected.append(item)
        selected_episodes.add(episode_index)
        if len(selected) == num_sequences:
            return selected
    selected_indices = {item[0] for item in selected}
    selected.extend(item for item in candidates if item[0] not in selected_indices)
    return selected[:num_sequences]


@torch.no_grad()
def validate_decoder(
    jepa,
    decoder,
    loader,
    device: torch.device,
    *,
    use_amp: bool,
    dtype: torch.dtype,
) -> float:
    jepa.eval()
    decoder.eval()
    absolute_error = 0.0
    num_values = 0
    for batch in loader:
        frames = _select_validation_frames(batch.states.to(device, non_blocking=True))
        with autocast(device.type, enabled=use_amp, dtype=dtype):
            latents = _encode_frames(jepa, frames)
            reconstruction = decoder(latents)
        absolute_error += reconstruction.float().sub(frames.float()).abs().sum().item()
        num_values += frames.numel()
    if num_values == 0:
        raise ValueError("Decoder validation loader contains no images")
    return absolute_error / num_values


def _experiment_directory(cfg, folder: str | None, source_path: Path) -> Path:
    if folder is not None:
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    if cfg.meta.get("model_folder"):
        destination = Path(cfg.meta.model_folder)
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    return get_unified_experiment_dir(
        example_name="surgical_jepa_decoder",
        sweep_name=get_default_dev_name(),
        exp_name=f"decoder_{source_path.parent.name}",
        seed=cfg.meta.seed,
    )


def run(
    jepa_checkpoint: str,
    fname: str = "examples/surgical_jepa/decoder_train.yaml",
    cfg=None,
    folder: str | None = None,
    **overrides,
):
    """Train the RGB decoder while keeping the source AC-JEPA frozen."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.get("device", "auto"))
    setup_seed(cfg.meta.seed)
    source_path = Path(jepa_checkpoint).expanduser().resolve()
    experiment_dir = _experiment_directory(cfg, folder, source_path)

    jepa, jepa_config, source_checkpoint = load_frozen_jepa(source_path, device)
    train_loader, val_loader, data_config = make_decoder_loaders(
        jepa_config, source_checkpoint, cfg
    )
    train_episode_indices = list(train_loader.dataset.episode_indices)
    holdout_episode_indices = list(val_loader.dataset.episode_indices)
    if set(train_episode_indices) & set(holdout_episode_indices):
        raise RuntimeError("Decoder training and validation episodes overlap")

    resolved_decoder_config = {
        **_plain_config(cfg.decoder),
        "latent_dim": int(jepa_config.model.latent_dim),
        "image_size": int(jepa_config.data.image_size),
    }
    decoder = build_decoder(resolved_decoder_config).to(device)
    log_model_info(
        decoder,
        {"decoder": sum(parameter.numel() for parameter in decoder.parameters())},
    )
    log_data_info(
        "Open-H Surgical decoder",
        len(train_loader),
        data_config.batch_size,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
    )
    logger.info("Decoder train episodes: %s", train_episode_indices)
    logger.info("Decoder held-out episodes: %s", holdout_episode_indices)

    decoder_training_config = _plain_config(cfg)
    decoder_training_config["decoder"] = resolved_decoder_config
    decoder_training_config["resolved_data"] = {
        field: getattr(data_config, field) for field in data_config.__dataclass_fields__
    }
    OmegaConf.save(
        OmegaConf.create(decoder_training_config),
        experiment_dir / "config.yaml",
    )

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "surgical_jepa_decoder",
            "source_jepa_checkpoint": str(source_path),
            **decoder_training_config,
        },
        run_dir=experiment_dir,
        run_name=experiment_dir.name,
        tags=["surgical_jepa", "decoder", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    optimizer = AdamW(
        decoder.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    total_steps = max(1, cfg.optim.epochs * len(train_loader))
    scheduler = CosineWithWarmup(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg.optim.warmup_ratio,
        min_lr=cfg.optim.min_lr,
    )
    amp_enabled = bool(cfg.training.use_amp and device.type == "cuda")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(str(cfg.training.dtype).lower(), torch.bfloat16)
    scaler = GradScaler(device.type, enabled=amp_enabled)

    visualization_config = cfg.logging.visualization
    frame_indices = list(visualization_config.frame_indices)
    invalid_frame_indices = [
        index
        for index in frame_indices
        if not 0 <= index < int(jepa_config.data.num_frames)
    ]
    if invalid_frame_indices:
        raise ValueError(
            "logging.visualization.frame_indices are outside the JEPA window: "
            f"{invalid_frame_indices}"
        )
    visualization_dataset = _make_visualization_dataset(
        jepa_config, source_checkpoint, cfg
    )
    selected_visualizations = select_visualization_samples(
        visualization_dataset,
        num_sequences=visualization_config.num_sequences,
        selection=visualization_config.selection,
        candidate_count=visualization_config.candidate_count,
        dataset_indices=(
            list(visualization_config.dataset_indices)
            if visualization_config.get("dataset_indices")
            else None
        ),
    )
    fixed_sequences = [
        sample.states.to(device) for _, sample, _ in selected_visualizations
    ]
    for dataset_index, sample, motion_score in selected_visualizations:
        logger.info(
            "Decoder visualization: dataset_index=%d episode=%d frames=%s "
            "motion_score=%.5f",
            dataset_index,
            int(sample.episode_index),
            sample.frame_indices[[0, -1]].tolist(),
            motion_score,
        )

    start_epoch = 0
    global_step = 0
    best_val_l1 = float("inf")
    latest_path = experiment_dir / "latest.pth.tar"
    if cfg.meta.get("load_model", False):
        resume_path = experiment_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            decoder.load_state_dict(clean_state_dict(resume["decoder_state_dict"]))
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            if resume.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume["scaler_state_dict"])
            start_epoch = int(resume["epoch"]) + 1
            global_step = int(resume["step"]) + 1
            best_val_l1 = float(resume["best_val_l1"])
            logger.info("Resumed decoder training from %s", resume_path)

    jepa_config_dict = _plain_config(jepa_config)
    normalizer_state = source_checkpoint["normalizer_state_dict"]

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        jepa.eval()
        decoder.train()
        absolute_error = 0.0
        num_values = 0
        progress = tqdm(
            train_loader,
            desc=f"Decoder epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for batch in progress:
            states = batch.states.to(device, non_blocking=True)
            frames = _select_training_frames(states)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.no_grad(),
                autocast(device.type, enabled=amp_enabled, dtype=dtype),
            ):
                latents = _encode_frames(jepa, frames)
            with autocast(device.type, enabled=amp_enabled, dtype=dtype):
                reconstruction = decoder(latents)
                loss = F.l1_loss(reconstruction, frames)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            absolute_error += (
                reconstruction.detach().float().sub(frames.float()).abs().sum().item()
            )
            num_values += frames.numel()
            progress.set_postfix(l1=f"{loss.item():.4f}")

            if global_step % cfg.logging.log_every == 0 and wandb_run:
                import wandb

                wandb.log(
                    {
                        "train/l1": loss.item(),
                        "optim/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        train_l1 = absolute_error / num_values
        val_l1 = validate_decoder(
            jepa,
            decoder,
            val_loader,
            device,
            use_amp=amp_enabled,
            dtype=dtype,
        )
        log_epoch(
            epoch,
            {"train_l1": train_l1, "val_l1": val_l1},
            total_epochs=cfg.optim.epochs,
            elapsed_time=time() - epoch_start,
        )

        decoder.eval()
        if wandb_run:
            import wandb

            visualization_images = []
            for sequence_number, sequence in enumerate(fixed_sequences):
                with (
                    torch.no_grad(),
                    autocast(device.type, enabled=amp_enabled, dtype=dtype),
                ):
                    latents = jepa.encoder(sequence.unsqueeze(0))
                    reconstruction = decoder(latents)[0]
                dataset_index, sample, motion_score = selected_visualizations[
                    sequence_number
                ]
                visualization_images.append(
                    wandb.Image(
                        make_sequence_reconstruction_grid(
                            sequence,
                            reconstruction,
                            frame_indices,
                        ),
                        caption=(
                            f"dataset_index={dataset_index}, "
                            f"episode={int(sample.episode_index)}, "
                            f"motion={motion_score:.5f}"
                        ),
                    )
                )
            wandb.log(
                {
                    "train_epoch/l1": train_l1,
                    "val/l1": val_l1,
                    "viz/ground_truth_vs_decoded": visualization_images,
                    "epoch": epoch,
                },
                step=global_step,
            )

        checkpoint_kwargs = {
            "decoder": decoder,
            "jepa": jepa,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "step": global_step - 1,
            "best_val_l1": min(best_val_l1, val_l1),
            "source_jepa_checkpoint": source_path,
            "jepa_config": jepa_config_dict,
            "decoder_training_config": decoder_training_config,
            "normalizer_state_dict": normalizer_state,
            "train_episode_indices": train_episode_indices,
            "holdout_episode_indices": holdout_episode_indices,
        }
        save_decoder_checkpoint(latest_path, **checkpoint_kwargs)
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            save_decoder_checkpoint(
                experiment_dir / "best.pth.tar", **checkpoint_kwargs
            )
            logger.info("New best decoder validation L1: %.6f", best_val_l1)
        if (
            cfg.logging.save_every_n_epochs > 0
            and (epoch + 1) % cfg.logging.save_every_n_epochs == 0
        ):
            save_decoder_checkpoint(
                experiment_dir / f"e-{epoch:03d}.pth.tar",
                **checkpoint_kwargs,
            )

    if wandb_run:
        import wandb

        wandb.finish()
    logger.info("Decoder training complete. Best validation L1: %.6f", best_val_l1)


if __name__ == "__main__":
    fire.Fire(run)
