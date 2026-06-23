"""Smoke-test and visualize the Open-H surgical dataset pipeline.

Run from the repository root:

    python -m examples.surgical_jepa.test_dataset

The script uses exactly the tensors that a future surgical AC-JEPA training loop
will consume and writes human-readable previews under
``examples/surgical_jepa/output``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import torch
import yaml
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eb_jepa.datasets.open_h import OpenHDatasetConfig, make_open_h_loaders


ACTION_NAMES = [
    f"{phase}_{arm}_{name}"
    for phase in ("current", "next")
    for arm in ("left", "right")
    for name in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
]


def _to_uint8_frames(states: torch.Tensor) -> list:
    frames = (
        states.detach()
        .cpu()
        .permute(1, 2, 3, 0)
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return list(frames)


def _save_contact_sheet(frames: list, path: Path, columns: int = 5) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    width, height = images[0].size
    label_height = 20
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * width, rows * (height + label_height)), color="white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + height + 2), f"t={index}", fill="black")
    sheet.save(path)


def _save_action_csv(
    path: Path,
    actions: torch.Tensor,
    frame_indices: torch.Tensor,
    timestamps: torch.Tensor,
) -> None:
    values = actions.detach().cpu().transpose(0, 1).numpy()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "frame_index", "timestamp", *ACTION_NAMES])
        for t, row in enumerate(values):
            writer.writerow(
                [
                    t,
                    int(frame_indices[t]),
                    float(timestamps[t]),
                    *[float(value) for value in row],
                ]
            )


def _save_proprioception_plot(path: Path, proprioception: torch.Tensor) -> None:
    values = proprioception.detach().cpu().transpose(0, 1).numpy()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    time = range(values.shape[0])
    for column, (arm, offset) in enumerate((("left", 0), ("right", 8))):
        axes[0, column].plot(time, values[:, offset : offset + 3])
        axes[0, column].set_title(f"{arm.capitalize()} position")
        axes[0, column].legend(["x", "y", "z"])
        axes[0, column].set_ylabel("meters")

        axes[1, column].plot(time, values[:, offset + 3 : offset + 7])
        axes[1, column].set_title(f"{arm.capitalize()} quaternion")
        axes[1, column].legend(["qx", "qy", "qz", "qw"])

        axes[2, column].plot(time, values[:, offset + 7])
        axes[2, column].set_title(f"{arm.capitalize()} gripper")
        axes[2, column].set_xlabel("sample at 5 Hz")
        axes[2, column].set_ylabel("opening")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate_sample(sample, config: OpenHDatasetConfig) -> None:
    expected_states = (3, config.num_frames, config.image_size, config.image_size)
    assert tuple(sample.states.shape) == expected_states
    assert tuple(sample.actions.shape) == (32, config.num_frames)
    assert tuple(sample.proprioception.shape) == (16, config.num_frames)
    assert tuple(sample.raw_actions.shape) == (32, config.num_frames)
    assert tuple(sample.raw_proprioception.shape) == (16, config.num_frames)
    assert tuple(sample.frame_indices.shape) == (config.num_frames,)
    assert tuple(sample.timestamps.shape) == (config.num_frames,)
    assert sample.states.dtype == torch.float32
    assert sample.actions.dtype == torch.float32
    assert torch.isfinite(sample.states).all()
    assert torch.isfinite(sample.actions).all()
    assert 0.0 <= sample.states.min() <= sample.states.max() <= 1.0

    expected_stride = config.source_fps // config.sample_fps
    assert torch.equal(
        sample.frame_indices[1:] - sample.frame_indices[:-1],
        torch.full((config.num_frames - 1,), expected_stride, dtype=torch.int64),
    )
    assert torch.allclose(
        sample.actions[:16, :-1], sample.proprioception[:, :-1]
    )
    assert torch.allclose(
        sample.actions[16:, :-1], sample.proprioception[:, 1:]
    )
    assert torch.allclose(sample.actions[:16, -1], sample.proprioception[:, -1])
    assert torch.allclose(sample.actions[16:, -1], sample.proprioception[:, -1])
    assert (sample.raw_proprioception[6] >= 0).all()
    assert (sample.raw_proprioception[14] >= 0).all()


def _save_preview(sample, output_dir: Path, preview_index: int, fps: int) -> None:
    preview_dir = output_dir / f"preview_{preview_index:02d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    frames = _to_uint8_frames(sample.states)

    imageio.mimsave(preview_dir / "window.gif", frames, fps=fps, loop=0)
    _save_contact_sheet(frames, preview_dir / "contact_sheet.png")
    for name, index in (
        ("first", 0),
        ("middle", len(frames) // 2),
        ("last", len(frames) - 1),
    ):
        Image.fromarray(frames[index]).save(preview_dir / f"{name}.png")

    _save_action_csv(
        preview_dir / "actions_raw.csv",
        sample.raw_actions,
        sample.frame_indices,
        sample.timestamps,
    )
    _save_action_csv(
        preview_dir / "actions_normalized.csv",
        sample.actions,
        sample.frame_indices,
        sample.timestamps,
    )
    _save_proprioception_plot(
        preview_dir / "proprioception_raw.png", sample.raw_proprioception
    )

    metadata = {
        "episode_index": int(sample.episode_index),
        "frame_indices": sample.frame_indices.tolist(),
        "timestamps": sample.timestamps.tolist(),
        "states_shape": list(sample.states.shape),
        "actions_shape": list(sample.actions.shape),
        "states_min": float(sample.states.min()),
        "states_max": float(sample.states.max()),
        "actions_mean": float(sample.actions.mean()),
        "actions_std": float(sample.actions.std()),
    }
    with (preview_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/surgical_jepa/data_config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/surgical_jepa/output"),
    )
    parser.add_argument("--num-previews", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    with args.config.open() as handle:
        values = yaml.safe_load(handle)
    if args.batch_size is not None:
        values["batch_size"] = args.batch_size
    if args.num_workers is not None:
        values["num_workers"] = args.num_workers
    config = OpenHDatasetConfig.from_dict(values)

    train_loader, val_loader, normalizer = make_open_h_loaders(config)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    print(
        f"train={len(train_dataset):,} windows, "
        f"val={len(val_dataset):,} windows, "
        f"frame_stride={train_dataset.frame_stride}, "
        f"clip_seconds={(config.num_frames - 1) / config.sample_fps:.1f}"
    )
    print("proprioception mean:", normalizer.proprio_mean.tolist())
    print("proprioception std: ", normalizer.proprio_std.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "normalization.json").open("w") as handle:
        json.dump(normalizer.state_dict(), handle, indent=2)

    if args.num_previews > 0:
        preview_indices = torch.linspace(
            0, len(train_dataset) - 1, steps=args.num_previews
        ).round().to(torch.int64)
        for preview_index, dataset_index in enumerate(preview_indices.tolist()):
            sample = train_dataset[dataset_index]
            _validate_sample(sample, config)
            print(
                f"preview {preview_index}: dataset_index={dataset_index}, "
                f"episode={int(sample.episode_index)}, "
                f"frames={sample.frame_indices[[0, -1]].tolist()}"
            )
            print("  first raw pair:", sample.raw_actions[:, 0].tolist())
            print("  first normalized pair:", sample.actions[:, 0].tolist())
            _save_preview(
                sample,
                args.output_dir,
                preview_index,
                fps=config.sample_fps,
            )

    batch = next(iter(train_loader))
    assert tuple(batch.states.shape[1:]) == (
        3,
        config.num_frames,
        config.image_size,
        config.image_size,
    )
    assert tuple(batch.actions.shape[1:]) == (32, config.num_frames)
    assert torch.isfinite(batch.states).all()
    assert torch.isfinite(batch.actions).all()
    print("batch states:", tuple(batch.states.shape), batch.states.dtype)
    print("batch actions:", tuple(batch.actions.shape), batch.actions.dtype)
    print("batch proprioception:", tuple(batch.proprioception.shape))
    print(f"Saved previews to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
