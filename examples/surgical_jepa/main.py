"""AC-JEPA training for Open-H surgical wrist-camera videos.

The training objective follows ``examples/ac_video_jepa``: a configurable
image encoder maps RGB frames to latents and an autoregressive predictor models
future latents from 32-D ``[proprio_t, proprio_t+1]`` conditioning vectors. The
predictor can be the original GRU or a LeWorldModel-style causal transformer
with AdaLN-Zero action conditioning.

Validation is deliberately small and decoder-free. Complete episodes are held
out from the official train split, and the only validation metric is
teacher-forced one-step MSE to the next encoded latent.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    ActionConditionedTransformerPredictor,
    ActionSequenceEncoder,
    DINOv3ConvNextEncoder,
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.datasets.open_h import OpenHDatasetConfig, make_open_h_loaders
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)


def _experiment_name(cfg) -> str:
    configured_name = cfg.meta.get("experiment_name")
    if configured_name:
        name = str(configured_name).strip().replace(" ", "_")
        if not name or Path(name).name != name:
            raise ValueError(
                "meta.experiment_name must be a non-empty folder-safe name"
            )
        return name
    predictor_cfg = cfg.model.get("predictor", {})
    predictor_architecture = predictor_cfg.get("architecture", "rnn")
    return (
        f"{cfg.model.get('encoder_architecture', 'impala')}"
        f"_{predictor_architecture}"
        f"_d{cfg.model.latent_dim}"
        f"_n{cfg.model.nsteps}_idm{cfg.model.regularizer.idm_coeff}"
    )


def build_surgical_jepa(
    model_cfg,
    image_size: int,
    *,
    load_pretrained_encoder: bool = True,
) -> JEPA:
    """Build the repo-style AC-JEPA with RGB input and 32-D conditioning."""
    architecture = model_cfg.get("encoder_architecture", "impala")
    if architecture == "impala":
        encoder = ImpalaEncoder(
            width=1,
            stack_sizes=(16, model_cfg.henc, model_cfg.dstc),
            num_blocks=2,
            dropout_rate=None,
            layer_norm=False,
            input_channels=model_cfg.dobs,
            final_ln=True,
            mlp_output_dim=model_cfg.latent_dim,
            input_shape=(model_cfg.dobs, image_size, image_size),
        )
    elif architecture == "dinov3_convnext_tiny":
        encoder_cfg = model_cfg.encoder
        encoder = DINOv3ConvNextEncoder(
            model_name=encoder_cfg.model_name,
            input_shape=(model_cfg.dobs, image_size, image_size),
            latent_dim=model_cfg.latent_dim,
            image_mean=tuple(encoder_cfg.image_mean),
            image_std=tuple(encoder_cfg.image_std),
            frame_batch_size=encoder_cfg.get("frame_batch_size"),
            local_files_only=encoder_cfg.get("local_files_only", False),
            revision=encoder_cfg.get("revision"),
            gradient_checkpointing=encoder_cfg.get("gradient_checkpointing", False),
            load_pretrained=load_pretrained_encoder,
        )
    else:
        raise ValueError(
            "model.encoder_architecture must be 'impala' or "
            f"'dinov3_convnext_tiny', got {architecture!r}"
        )
    predictor_cfg = model_cfg.get("predictor", {})
    predictor_architecture = predictor_cfg.get("architecture", "rnn")
    if predictor_architecture == "rnn":
        action_encoder = nn.Identity()
        predictor = RNNPredictor(
            hidden_size=encoder.mlp_output_dim,
            action_dim=model_cfg.action_dim,
            num_layers=predictor_cfg.get("num_layers", 1),
            final_ln=encoder.final_ln,
        )
    elif predictor_architecture == "transformer":
        condition_dim = int(
            predictor_cfg.get("action_embedding_dim", model_cfg.latent_dim)
        )
        action_encoder = ActionSequenceEncoder(
            action_dim=model_cfg.action_dim,
            embedding_dim=condition_dim,
            smoothed_dim=predictor_cfg.get("action_smoothed_dim"),
            mlp_scale=predictor_cfg.get("action_mlp_scale", 2.0),
        )
        predictor = ActionConditionedTransformerPredictor(
            state_dim=encoder.mlp_output_dim,
            condition_dim=condition_dim,
            hidden_dim=predictor_cfg.get("hidden_dim", model_cfg.latent_dim),
            depth=predictor_cfg.get("depth", 4),
            heads=predictor_cfg.get("heads", 8),
            dim_head=predictor_cfg.get("dim_head", 64),
            mlp_dim=predictor_cfg.get(
                "mlp_dim",
                4 * predictor_cfg.get("hidden_dim", model_cfg.latent_dim),
            ),
            dropout=predictor_cfg.get("dropout", 0.0),
            embedding_dropout=predictor_cfg.get("embedding_dropout", 0.0),
            max_seq_len=predictor_cfg.get("max_seq_len", 17),
            history_size=predictor_cfg.get("history_size", 4),
        )
    else:
        raise ValueError(
            "model.predictor.architecture must be 'rnn' or 'transformer', got "
            f"{predictor_architecture!r}"
        )

    regularizer_cfg = model_cfg.regularizer
    if regularizer_cfg.use_proj:
        projector = Projector(
            f"{model_cfg.latent_dim}-"
            f"{model_cfg.latent_dim * 4}-"
            f"{model_cfg.latent_dim * 4}"
        )
    else:
        projector = None
    idm_state_dim = (
        projector.out_dim
        if projector is not None and regularizer_cfg.idm_after_proj
        else model_cfg.latent_dim
    )

    idm = InverseDynamicsModel(
        state_dim=idm_state_dim,
        hidden_dim=regularizer_cfg.idm_hidden_dim,
        action_dim=model_cfg.action_dim,
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=regularizer_cfg.cov_coeff,
        std_coeff=regularizer_cfg.std_coeff,
        sim_coeff_t=regularizer_cfg.sim_coeff_t,
        idm_coeff=regularizer_cfg.idm_coeff,
        idm=idm,
        first_t_only=regularizer_cfg.first_t_only,
        projector=projector,
        spatial_as_samples=regularizer_cfg.spatial_as_samples,
        idm_after_proj=regularizer_cfg.idm_after_proj,
        sim_t_after_proj=regularizer_cfg.sim_t_after_proj,
    )
    return JEPA(
        encoder=encoder,
        aencoder=action_encoder,
        predictor=predictor,
        regularizer=regularizer,
        predcost=SquareLossSeq(),
    )


def configure_encoder_for_epoch(encoder, model_cfg, epoch: int) -> bool:
    """Freeze a pretrained encoder initially, then fine-tune it completely."""
    if not hasattr(encoder, "set_trainable_backbone_stages"):
        return False
    freeze_epochs = int(model_cfg.encoder.get("freeze_backbone_epochs", 0))
    trainable_stages = 0 if epoch < freeze_epochs else encoder.num_backbone_stages
    return encoder.set_trainable_backbone_stages(trainable_stages)


def build_jepa_optimizer(jepa: JEPA, cfg) -> AdamW:
    """Create separate learning-rate groups for pretrained and new weights."""
    encoder = jepa.encoder
    if not hasattr(encoder, "backbone_parameters"):
        return AdamW(
            jepa.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

    backbone_parameters = list(encoder.backbone_parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    new_parameters = [
        parameter
        for parameter in jepa.parameters()
        if id(parameter) not in backbone_ids
    ]
    return AdamW(
        [
            {
                "params": new_parameters,
                "lr": cfg.optim.lr,
                "weight_decay": cfg.optim.weight_decay,
                "name": "new_weights",
            },
            {
                "params": backbone_parameters,
                "lr": cfg.optim.get("backbone_lr", cfg.optim.lr),
                "weight_decay": cfg.optim.get(
                    "backbone_weight_decay", cfg.optim.weight_decay
                ),
                "name": "pretrained_backbone",
            },
        ]
    )


def teacher_forced_next_latents(
    jepa: JEPA,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict every next latent from the corresponding real current latent.

    Args:
        states: RGB clips ``[B,3,T,H,W]``.
        actions: proprioception pairs ``[B,32,T]``.

    Returns:
        ``(predicted, target)`` with shape ``[B,D,T-1,1,1]``.
    """
    latents = jepa.encoder(states)
    batch_size, latent_dim, timesteps, height, width = latents.shape
    if timesteps < 2:
        raise ValueError("Next-latent evaluation requires at least two frames")
    if height != 1 or width != 1:
        raise ValueError(
            "RNNPredictor expects 1x1 latent maps, got " f"{height}x{width}"
        )
    if actions.shape[0] != batch_size or actions.shape[2] < timesteps - 1:
        raise ValueError(
            "Actions must have shape [B,A,T_actions] with "
            f"T_actions >= {timesteps - 1}; got {tuple(actions.shape)}"
        )

    encoded_actions = jepa.action_encoder(actions[:, :, : timesteps - 1])
    current = (
        latents[:, :, :-1]
        .permute(0, 2, 1, 3, 4)
        .reshape(batch_size * (timesteps - 1), latent_dim, 1, 1, 1)
    )
    conditioning = encoded_actions.permute(0, 2, 1).reshape(
        batch_size * (timesteps - 1), -1, 1
    )
    predicted = jepa.predictor(current, conditioning)
    predicted = (
        predicted.reshape(batch_size, timesteps - 1, latent_dim, 1, 1)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    return predicted, latents[:, :, 1:].contiguous()


def predictor_context_window(jepa: JEPA) -> int:
    """Return the maximum latent history used for autoregressive prediction."""
    return max(1, int(getattr(jepa.predictor, "context_length", 1)))


@torch.no_grad()
def validate_next_latent_mse(
    jepa: JEPA,
    loader,
    device: torch.device,
    *,
    use_amp: bool,
    dtype: torch.dtype,
) -> float:
    """Compute teacher-forced one-step latent MSE on held-out episodes."""
    jepa.eval()
    squared_error = 0.0
    num_values = 0

    for batch in loader:
        states = batch.states.to(device, non_blocking=True)
        actions = batch.actions.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp, dtype=dtype):
            predicted, target = teacher_forced_next_latents(jepa, states, actions)
        squared_error += predicted.float().sub(target.float()).square().sum().item()
        num_values += target.numel()

    if num_values == 0:
        raise ValueError("Validation loader contains no latent targets")
    return squared_error / num_values


def _experiment_directory(cfg, folder: str | None) -> Path:
    if folder is not None:
        result = Path(folder)
        result.mkdir(parents=True, exist_ok=True)
        return result
    if cfg.meta.get("model_folder"):
        result = Path(cfg.meta.model_folder)
        result.mkdir(parents=True, exist_ok=True)
        return result

    experiment_name = _experiment_name(cfg)
    return get_unified_experiment_dir(
        example_name="surgical_jepa",
        sweep_name=get_default_dev_name(),
        exp_name=experiment_name,
        seed=cfg.meta.seed,
    )


def run(
    fname: str = "examples/surgical_jepa/train.yaml",
    cfg=None,
    folder: str | None = None,
    **overrides,
):
    """Train the minimal surgical AC-JEPA baseline."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.get("device", "auto"))
    setup_seed(cfg.meta.seed)
    experiment_dir = _experiment_directory(cfg, folder)

    data_config = OpenHDatasetConfig.from_dict(
        OmegaConf.to_container(cfg.data, resolve=True)
    )
    if cfg.model.action_dim != 32:
        raise ValueError(
            "The Open-H proprioception-pair adapter returns 32-D conditioning; "
            f"got model.action_dim={cfg.model.action_dim}"
        )
    if cfg.model.dobs != 3:
        raise ValueError(
            "The wrist camera adapter returns RGB images; "
            f"got model.dobs={cfg.model.dobs}"
        )
    if cfg.model.nsteps > data_config.num_frames - 1:
        raise ValueError(
            f"model.nsteps={cfg.model.nsteps} exceeds the "
            f"{data_config.num_frames - 1} available transitions"
        )
    train_loader, val_loader, normalizer = make_open_h_loaders(data_config)
    train_episode_indices = list(train_loader.dataset.episode_indices)
    holdout_episode_indices = list(val_loader.dataset.episode_indices)

    logger.info("Train episodes: %s", train_episode_indices)
    logger.info("Held-out episodes: %s", holdout_episode_indices)
    log_data_info(
        "Open-H Surgical",
        len(train_loader),
        data_config.batch_size,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
    )

    resolved_split = {
        "train_episode_indices": train_episode_indices,
        "holdout_episode_indices": holdout_episode_indices,
        "holdout_seed": data_config.holdout_seed,
    }
    (experiment_dir / "episode_split.json").write_text(
        json.dumps(resolved_split, indent=2) + "\n"
    )
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "surgical_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
            "resolved_data_split": resolved_split,
        },
        run_dir=experiment_dir,
        run_name=experiment_dir.name,
        tags=[
            "surgical_jepa",
            "ac_jepa",
            _experiment_name(cfg),
            f"seed_{cfg.meta.seed}",
        ],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    requested_checkpoint_path = experiment_dir / cfg.meta.get(
        "load_checkpoint", "latest.pth.tar"
    )
    resume_from_existing_checkpoint = bool(
        cfg.meta.get("load_model", False) and requested_checkpoint_path.is_file()
    )
    jepa = build_surgical_jepa(
        cfg.model,
        data_config.image_size,
        load_pretrained_encoder=not resume_from_existing_checkpoint,
    ).to(device)
    log_model_info(
        jepa,
        {
            "encoder": sum(p.numel() for p in jepa.encoder.parameters()),
            "action_encoder": sum(p.numel() for p in jepa.action_encoder.parameters()),
            "predictor": sum(p.numel() for p in jepa.predictor.parameters()),
            "regularizer": sum(p.numel() for p in jepa.regularizer.parameters()),
        },
    )
    log_config(cfg)

    total_steps = max(1, cfg.optim.epochs * len(train_loader))
    optimizer = build_jepa_optimizer(jepa, cfg)
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
    logger.info(
        "AMP %s with dtype=%s",
        "enabled" if amp_enabled else "disabled",
        dtype,
    )

    latest_path = experiment_dir / "latest.pth.tar"
    start_epoch = 0
    global_step = 0
    best_val_mse = float("inf")
    if cfg.meta.get("load_model", False):
        checkpoint = load_checkpoint(
            requested_checkpoint_path,
            jepa,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", -1) + int(checkpoint.get("resumed", False))
        best_val_mse = checkpoint.get("best_val_mse", float("inf"))

    if torch.cuda.is_available() and cfg.model.get("compile", False):
        jepa = torch.compile(jepa)

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        encoder = getattr(jepa, "_orig_mod", jepa).encoder
        if configure_encoder_for_epoch(encoder, cfg.model, epoch):
            logger.info(
                "Epoch %d: pretrained backbone trains all %d/%d stages",
                epoch,
                encoder.trainable_backbone_stages,
                encoder.num_backbone_stages,
            )
        jepa.train()
        totals: dict[str, float] = {}
        samples_seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        for batch in progress:
            states = batch.states.to(device, non_blocking=True)
            actions = batch.actions.to(device, non_blocking=True)
            batch_size = states.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=amp_enabled, dtype=dtype):
                _, (
                    total_loss,
                    regularizer_loss,
                    _,
                    regularizer_metrics,
                    prediction_loss,
                ) = jepa.unroll(
                    states,
                    actions,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=predictor_context_window(jepa),
                    compute_loss=True,
                    return_all_steps=False,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                jepa.encoder.parameters(), cfg.optim.grad_clip
            )
            torch.nn.utils.clip_grad_norm_(
                jepa.predictor.parameters(), cfg.optim.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_metrics = {
                "total_loss": total_loss.item(),
                "prediction_loss": prediction_loss.item(),
                "regularizer_loss": regularizer_loss.item(),
                **regularizer_metrics,
            }
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size
            samples_seen += batch_size

            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                pred=f"{prediction_loss.item():.4f}",
            )

            if global_step % cfg.logging.log_every == 0 and wandb_run:
                import wandb

                wandb.log(
                    {
                        **{
                            f"train/{name}": value
                            for name, value in batch_metrics.items()
                        },
                        "optim/lr": optimizer.param_groups[0]["lr"],
                        **(
                            {"optim/backbone_lr": optimizer.param_groups[1]["lr"]}
                            if len(optimizer.param_groups) > 1
                            else {}
                        ),
                        "model/trainable_encoder_parameters": sum(
                            parameter.numel()
                            for parameter in encoder.parameters()
                            if parameter.requires_grad
                        ),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        train_metrics = {name: value / samples_seen for name, value in totals.items()}
        val_mse = validate_next_latent_mse(
            jepa,
            val_loader,
            device,
            use_amp=amp_enabled,
            dtype=dtype,
        )
        epoch_metrics = {
            "train_loss": train_metrics["total_loss"],
            "train_pred": train_metrics["prediction_loss"],
            "val_next_latent_mse": val_mse,
        }
        log_epoch(
            epoch,
            epoch_metrics,
            total_epochs=cfg.optim.epochs,
            elapsed_time=time() - epoch_start,
        )

        if wandb_run:
            import wandb

            wandb.log(
                {
                    **{
                        f"train_epoch/{name}": value
                        for name, value in train_metrics.items()
                    },
                    "val/next_latent_mse": val_mse,
                    "epoch": epoch,
                },
                step=global_step,
            )

        checkpoint_state = {
            "model": jepa,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "step": global_step - 1,
            "best_val_mse": min(best_val_mse, val_mse),
            "normalizer_state_dict": normalizer.state_dict(),
            **resolved_split,
        }
        save_checkpoint(latest_path, **checkpoint_state)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(experiment_dir / "best.pth.tar", **checkpoint_state)
            logger.info("New best validation MSE: %.6f", best_val_mse)

        if (
            cfg.logging.save_every_n_epochs > 0
            and (epoch + 1) % cfg.logging.save_every_n_epochs == 0
        ):
            save_checkpoint(
                experiment_dir / f"e-{epoch:03d}.pth.tar",
                **checkpoint_state,
            )

    if wandb_run:
        import wandb

        wandb.finish()
    logger.info("Training complete. Best validation MSE: %.6f", best_val_mse)


if __name__ == "__main__":
    fire.Fire(run)
