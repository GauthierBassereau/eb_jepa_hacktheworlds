"""Shared decoder and checkpoint utilities for the surgical AC-JEPA example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.datasets.open_h import OpenHNormalizer
from eb_jepa.nn_utils import init_module_weights

try:
    from examples.surgical_jepa.main import build_surgical_jepa
except ModuleNotFoundError:  # Direct execution from examples/surgical_jepa.
    from main import build_surgical_jepa

DECODER_CHECKPOINT_VERSION = 1


def _group_count(channels: int, maximum: int) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DecoderUpsampleBlock(nn.Module):
    """Bilinear 2x upsampling followed by convolutional refinement."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(_group_count(out_channels, norm_groups), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.activation(self.norm(self.conv(value)))


class SurgicalRGBDecoder(nn.Module):
    """Decode global AC-JEPA latents into RGB wrist-camera frames."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 128,
        base_size: int = 4,
        channels: tuple[int, ...] = (512, 256, 128, 64, 32, 16),
        norm_groups: int = 32,
        output_channels: int = 3,
    ):
        super().__init__()
        if image_size < base_size or image_size % base_size:
            raise ValueError(
                f"image_size={image_size} must be divisible by base_size={base_size}"
            )
        scale = image_size // base_size
        if scale & (scale - 1):
            raise ValueError("image_size / base_size must be a power of two")
        num_blocks = scale.bit_length() - 1
        if len(channels) < num_blocks + 1:
            raise ValueError(
                f"Need at least {num_blocks + 1} channel values, got {len(channels)}"
            )

        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.base_size = int(base_size)
        self.channels = tuple(int(value) for value in channels[: num_blocks + 1])
        self.norm_groups = int(norm_groups)
        self.output_channels = int(output_channels)

        initial_channels = self.channels[0]
        self.projection = nn.Linear(
            self.latent_dim,
            initial_channels * self.base_size * self.base_size,
        )
        self.initial_norm = nn.GroupNorm(
            _group_count(initial_channels, self.norm_groups),
            initial_channels,
        )
        self.initial_activation = nn.SiLU(inplace=True)
        self.blocks = nn.ModuleList(
            DecoderUpsampleBlock(in_channels, out_channels, self.norm_groups)
            for in_channels, out_channels in zip(self.channels[:-1], self.channels[1:])
        )
        self.output_conv = nn.Conv2d(
            self.channels[-1],
            self.output_channels,
            kernel_size=3,
            padding=1,
        )
        self.apply(init_module_weights)

    def _decode_flat(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[-2:] != (1, 1):
            raise ValueError(
                "Expected decoder latents [B,D,1,1], got " f"{tuple(latents.shape)}"
            )
        if latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {latents.shape[1]}"
            )
        value = latents.flatten(1)
        value = self.projection(value)
        value = value.view(
            latents.shape[0],
            self.channels[0],
            self.base_size,
            self.base_size,
        )
        value = self.initial_activation(self.initial_norm(value))
        for block in self.blocks:
            value = block(value)
        return torch.sigmoid(self.output_conv(value))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Support both ``[B,D,1,1]`` and ``[B,D,T,1,1]`` latents."""
        if latents.ndim == 4:
            return self._decode_flat(latents)
        if latents.ndim != 5:
            raise ValueError(
                "Expected decoder latents [B,D,1,1] or [B,D,T,1,1], got "
                f"{tuple(latents.shape)}"
            )
        batch_size, latent_dim, timesteps, height, width = latents.shape
        flattened = (
            latents.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * timesteps, latent_dim, height, width)
            .contiguous()
        )
        decoded = self._decode_flat(flattened)
        return (
            decoded.view(
                batch_size,
                timesteps,
                self.output_channels,
                self.image_size,
                self.image_size,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "base_size": self.base_size,
            "channels": list(self.channels),
            "norm_groups": self.norm_groups,
            "output_channels": self.output_channels,
        }


def build_decoder(decoder_config: Any) -> SurgicalRGBDecoder:
    """Build a decoder from an OmegaConf or plain mapping."""
    values = (
        OmegaConf.to_container(decoder_config, resolve=True)
        if OmegaConf.is_config(decoder_config)
        else dict(decoder_config)
    )
    if "channels" in values:
        values["channels"] = tuple(values["channels"])
    return SurgicalRGBDecoder(**values)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove ``torch.compile`` prefixes from a checkpoint state dictionary."""
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def normalizer_from_state(state: dict[str, Any]) -> OpenHNormalizer:
    return OpenHNormalizer(
        proprio_mean=torch.tensor(state["proprio_mean"]),
        proprio_std=torch.tensor(state["proprio_std"]),
    )


def load_frozen_jepa(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a surgical JEPA and freeze it for decoder training."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JEPA checkpoint not found: {path}")
    config_path = path.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected JEPA configuration next to checkpoint: {config_path}"
        )

    config = OmegaConf.load(config_path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    jepa = build_surgical_jepa(
        config.model,
        int(config.data.image_size),
        load_pretrained_encoder=False,
    ).to(device)
    jepa.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
    jepa.eval()
    for parameter in jepa.parameters():
        parameter.requires_grad_(False)
    return jepa, config, checkpoint


def save_decoder_checkpoint(
    path: str | Path,
    *,
    decoder: SurgicalRGBDecoder,
    jepa: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    best_val_l1: float,
    source_jepa_checkpoint: str | Path,
    jepa_config: dict[str, Any],
    decoder_training_config: dict[str, Any],
    normalizer_state_dict: dict[str, Any],
    train_episode_indices: list[int],
    holdout_episode_indices: list[int],
) -> None:
    """Save a decoder checkpoint that is independent of the source JEPA file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": DECODER_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "step": int(step),
        "best_val_l1": float(best_val_l1),
        "source_jepa_checkpoint": str(Path(source_jepa_checkpoint).resolve()),
        "jepa_state_dict": clean_state_dict(jepa.state_dict()),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "jepa_config": jepa_config,
        "decoder_training_config": decoder_training_config,
        "decoder_config": decoder.config_dict(),
        "normalizer_state_dict": normalizer_state_dict,
        "train_episode_indices": [int(value) for value in train_episode_indices],
        "holdout_episode_indices": [int(value) for value in holdout_episode_indices],
    }
    torch.save(payload, destination)


def load_system_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, SurgicalRGBDecoder, dict[str, Any]]:
    """Load the complete JEPA plus decoder system from one checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("format_version")
    if version != DECODER_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported decoder checkpoint format {version!r}; "
            f"expected {DECODER_CHECKPOINT_VERSION}"
        )

    jepa_config = OmegaConf.create(checkpoint["jepa_config"])
    jepa = build_surgical_jepa(
        jepa_config.model,
        int(jepa_config.data.image_size),
        load_pretrained_encoder=False,
    ).to(device)
    jepa.load_state_dict(clean_state_dict(checkpoint["jepa_state_dict"]))
    decoder = build_decoder(checkpoint["decoder_config"]).to(device)
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder_state_dict"]))

    jepa.eval()
    decoder.eval()
    for module in (jepa, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return jepa, decoder, checkpoint


@torch.no_grad()
def autoregressive_latent_rollout(
    jepa: nn.Module,
    initial_frame: torch.Tensor,
    actions: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Return the initial latent followed by ``steps`` autoregressive predictions."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if initial_frame.ndim != 5 or initial_frame.shape[2] != 1:
        raise ValueError(
            "initial_frame must have shape [B,C,1,H,W], got "
            f"{tuple(initial_frame.shape)}"
        )
    if actions.ndim != 3 or actions.shape[2] < steps:
        raise ValueError(
            f"actions must contain at least {steps} transitions, got "
            f"{tuple(actions.shape)}"
        )
    latents, _ = jepa.unroll(
        initial_frame,
        actions[:, :, :steps],
        nsteps=steps,
        unroll_mode="autoregressive",
        ctxt_window_time=max(
            1,
            int(getattr(jepa.predictor, "context_length", 1)),
        ),
        compute_loss=False,
        return_all_steps=False,
    )
    return latents
