"""Windowed Open-H/LeRobot-v2.1 video dataset for action-conditioned JEPA.

The adapter intentionally does not depend on LeRobot. It reads the simple v2.1
storage primitives directly:

* ``meta/info.json`` and ``meta/episodes.jsonl`` for schema and split metadata;
* one Parquet table per episode for proprioception;
* one MP4 per episode/camera for image observations.

Each item is a fixed-rate video window. The action-conditioning vector for a
transition is the concatenation of the normalized current and next
proprioception vectors. This makes the returned tensors directly compatible
with the ``[C,T,H,W]`` / ``[A,T]`` convention used by AC-JEPA in this repo.
"""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only in incomplete environments.
    pq = None


class OpenHSample(NamedTuple):
    """One Open-H training window.

    All time-varying tensors use time as their last non-spatial dimension so
    default PyTorch collation produces the AC-JEPA batch convention.
    """

    states: torch.Tensor  # [3, T, H, W], RGB float32 in [0, 1]
    actions: torch.Tensor  # [32, T], normalized [proprio_t, proprio_t+1]
    proprioception: torch.Tensor  # [16, T], normalized
    raw_actions: torch.Tensor  # [32, T], canonicalized but not normalized
    raw_proprioception: torch.Tensor  # [16, T], canonicalized but not normalized
    episode_index: torch.Tensor  # scalar int64
    frame_indices: torch.Tensor  # [T] source-video frame indices
    timestamps: torch.Tensor  # [T] seconds


@dataclass
class OpenHDatasetConfig:
    """Configuration shared by train and validation Open-H datasets."""

    data_root: str
    camera_key: str = "observation.images.wrist_left"
    state_key: str = "observation.state"
    source_fps: int | None = None
    sample_fps: int = 5
    num_frames: int = 17
    image_size: int = 128

    batch_size: int = 8
    val_batch_size: int = 4
    num_workers: int = 0
    pin_mem: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = True
    seed: int = 0
    holdout_episodes: int = 0
    holdout_seed: int = 0

    table_cache_size: int = 4
    decoder_cache_size: int = 2
    video_backend: str = "auto"
    video_threads: int = 1
    validate_files: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OpenHDatasetConfig":
        """Build a config while rejecting misspelled/unknown keys."""
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"Unknown OpenHDatasetConfig keys: {unknown}")
        return cls(**values)


def canonicalize_quaternions(proprioception: torch.Tensor) -> torch.Tensor:
    """Put both arm quaternions in the deterministic ``qw >= 0`` hemisphere.

    The 16-D state layout is:
    ``left[x,y,z,qx,qy,qz,qw,gripper], right[...]``.
    """
    if proprioception.shape[-1] != 16:
        raise ValueError(
            "Expected 16-D bimanual Cartesian proprioception, got "
            f"shape={tuple(proprioception.shape)}"
        )
    result = proprioception.clone()
    for start, qw_index in ((3, 6), (11, 14)):
        sign = torch.where(
            result[..., qw_index : qw_index + 1] < 0,
            -torch.ones_like(result[..., qw_index : qw_index + 1]),
            torch.ones_like(result[..., qw_index : qw_index + 1]),
        )
        result[..., start : start + 4] *= sign
    return result


def build_proprioception_pairs(proprioception: torch.Tensor) -> torch.Tensor:
    """Return ``[proprio_t, proprio_t+1]`` for every timestep.

    ``proprioception`` is expected as ``[T,16]``. The final transition repeats
    the last state because AC-JEPA keeps an action at every input timestep even
    though the final action has no in-window prediction target.
    """
    if proprioception.ndim != 2 or proprioception.shape[1] != 16:
        raise ValueError(
            f"Expected proprioception [T,16], got {tuple(proprioception.shape)}"
        )
    next_proprioception = torch.cat([proprioception[1:], proprioception[-1:]], dim=0)
    return torch.cat([proprioception, next_proprioception], dim=1)


@dataclass(frozen=True)
class _EpisodeRecord:
    episode_index: int
    length: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class _Window:
    episode_index: int
    start_frame: int


@dataclass
class _EpisodeTable:
    proprioception: torch.Tensor  # [episode_length, 16]
    timestamps: torch.Tensor  # [episode_length]
    frame_indices: torch.Tensor  # [episode_length]


@dataclass
class _DecodedFrames:
    data: torch.Tensor


class _OpenCVVideoDecoder:
    """Small exact-frame decoder used when TorchCodec is unavailable."""

    def __init__(self, path: Path):
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - dependency is project-wide.
            raise ImportError(
                "OpenCV is required for the Open-H video fallback backend"
            ) from error
        self.cv2 = cv2
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        self._num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self) -> int:
        return self._num_frames

    def get_frames_at(self, indices: list[int]) -> _DecodedFrames:
        if not indices:
            return _DecodedFrames(torch.empty(0, 3, 0, 0, dtype=torch.uint8))
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("OpenCV fallback expects sorted, unique frame indices")

        requested = set(indices)
        start, stop = indices[0], indices[-1]
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, start)
        decoded = []
        for frame_index in range(start, stop + 1):
            success, frame = self.capture.read()
            if not success:
                raise RuntimeError(
                    f"OpenCV failed decoding {self.path} at frame {frame_index}"
                )
            if frame_index in requested:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                decoded.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
        if len(decoded) != len(indices):
            raise RuntimeError(
                f"Decoded {len(decoded)} frames from {self.path}, expected {len(indices)}"
            )
        return _DecodedFrames(torch.stack(decoded))

    def __del__(self):
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()


class OpenHNormalizer:
    """Per-coordinate proprioception normalization fitted on train episodes."""

    def __init__(self, proprio_mean: torch.Tensor, proprio_std: torch.Tensor):
        mean = torch.as_tensor(proprio_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(proprio_std, dtype=torch.float32).flatten()
        if mean.shape != (16,) or std.shape != (16,):
            raise ValueError(
                f"Expected 16-D normalization stats, got {mean.shape=} and {std.shape=}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Proprioception normalization stats must be finite")
        self.proprio_mean = mean
        self.proprio_std = torch.where(std < 1e-6, torch.ones_like(std), std)

    def normalize_proprioception(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.proprio_mean.to(device=value.device, dtype=value.dtype)
        std = self.proprio_std.to(device=value.device, dtype=value.dtype)
        return (value - mean) / std

    def unnormalize_proprioception(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.proprio_mean.to(device=value.device, dtype=value.dtype)
        std = self.proprio_std.to(device=value.device, dtype=value.dtype)
        return value * std + mean

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize both 16-D halves of a ``[...,32]`` transition vector."""
        if actions.shape[-1] != 32:
            raise ValueError(f"Expected 32-D actions, got {tuple(actions.shape)}")
        current = self.normalize_proprioception(actions[..., :16])
        following = self.normalize_proprioception(actions[..., 16:])
        return torch.cat([current, following], dim=-1)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "proprio_mean": self.proprio_mean.tolist(),
            "proprio_std": self.proprio_std.tolist(),
        }


def _parse_split(value: str, total_episodes: int) -> range:
    try:
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Open-H split range: {value!r}") from error
    if not (0 <= start <= stop <= total_episodes):
        raise ValueError(f"Split {value!r} is outside [0, {total_episodes}]")
    return range(start, stop)


def _read_metadata(root: Path) -> tuple[dict[str, Any], list[_EpisodeRecord]]:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"Expected {info_path} and {episodes_path} for a LeRobot v2.1 dataset"
        )

    with info_path.open() as handle:
        info = json.load(handle)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "OpenHWindowDataset currently supports LeRobot v2.1 only; "
            f"found {info.get('codebase_version')!r}"
        )

    records = []
    with episodes_path.open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records.append(
                    _EpisodeRecord(
                        episode_index=int(item["episode_index"]),
                        length=int(item["length"]),
                        tasks=tuple(item.get("tasks", ())),
                    )
                )

    expected_count = int(info["total_episodes"])
    if len(records) != expected_count:
        raise ValueError(
            f"Metadata declares {expected_count} episodes but episodes.jsonl has "
            f"{len(records)}"
        )
    for expected_index, record in enumerate(records):
        if record.episode_index != expected_index:
            raise ValueError(
                "episodes.jsonl must be ordered and contiguous: "
                f"expected {expected_index}, found {record.episode_index}"
            )
    return info, records


def _fixed_list_column_to_tensor(column, width: int) -> torch.Tensor:
    array = column.combine_chunks()
    if len(array) == 0:
        return torch.empty((0, width), dtype=torch.float32)
    values = array.values.to_numpy(zero_copy_only=False)
    result = np.asarray(values, dtype=np.float32).reshape(len(array), width)
    return torch.from_numpy(result.copy())


def _scalar_column_to_tensor(column, dtype: torch.dtype) -> torch.Tensor:
    array = column.combine_chunks().to_numpy(zero_copy_only=False)
    return torch.as_tensor(np.asarray(array).copy(), dtype=dtype)


def _load_episode_table(
    path: Path,
    state_key: str,
    expected_length: int | None = None,
) -> _EpisodeTable:
    if pq is None:
        raise ImportError(
            "PyArrow is required for Open-H datasets. Install the project "
            "dependencies or run `uv sync`."
        )
    table = pq.read_table(
        path,
        columns=[state_key, "timestamp", "frame_index"],
        memory_map=True,
    )
    if state_key not in table.column_names:
        raise KeyError(f"{state_key!r} is absent from {path}")
    proprioception = _fixed_list_column_to_tensor(table[state_key], width=16)
    timestamps = _scalar_column_to_tensor(table["timestamp"], torch.float32)
    frame_indices = _scalar_column_to_tensor(table["frame_index"], torch.int64)
    if expected_length is not None and len(proprioception) != expected_length:
        raise ValueError(
            f"{path} has {len(proprioception)} rows, expected {expected_length}"
        )
    return _EpisodeTable(proprioception, timestamps, frame_indices)


def _compute_train_normalizer(
    root: Path,
    info: dict[str, Any],
    records: list[_EpisodeRecord],
    state_key: str,
    episode_indices: Sequence[int] | None = None,
) -> OpenHNormalizer:
    train_indices = (
        tuple(episode_indices)
        if episode_indices is not None
        else tuple(_parse_split(info["splits"]["train"], len(records)))
    )
    count = 0
    running_sum = torch.zeros(16, dtype=torch.float64)
    running_sq_sum = torch.zeros(16, dtype=torch.float64)

    data_template = info["data_path"]
    chunk_size = int(info["chunks_size"])
    for episode_index in train_indices:
        record = records[episode_index]
        relative_path = data_template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        table = _load_episode_table(
            root / relative_path,
            state_key=state_key,
            expected_length=record.length,
        )
        values = canonicalize_quaternions(table.proprioception).to(torch.float64)
        running_sum += values.sum(dim=0)
        running_sq_sum += values.square().sum(dim=0)
        count += values.shape[0]

    if count == 0:
        raise ValueError("The Open-H normalization episode set contains no rows")
    mean = running_sum / count
    variance = (running_sq_sum / count - mean.square()).clamp_min(0)
    return OpenHNormalizer(mean.float(), variance.sqrt().float())


class OpenHWindowDataset(Dataset):
    """Random-access fixed-rate windows from an Open-H v2.1 dataset."""

    proprio_dim = 16
    action_dim = 32
    state_dim = 16

    def __init__(
        self,
        config: OpenHDatasetConfig,
        split: str = "train",
        normalizer: OpenHNormalizer | None = None,
        episode_indices: Sequence[int] | None = None,
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.root = Path(config.data_root).expanduser().resolve()
        self.info, self.episodes = _read_metadata(self.root)

        features = self.info.get("features", {})
        self._validate_feature(features, config.camera_key, expected_dtype="video")
        self._validate_feature(
            features, config.state_key, expected_dtype="float32", expected_shape=[16]
        )

        metadata_fps = int(self.info["fps"])
        source_fps = config.source_fps or metadata_fps
        if source_fps != metadata_fps:
            raise ValueError(
                f"Configured source_fps={source_fps} does not match metadata fps={metadata_fps}"
            )
        if config.sample_fps <= 0 or source_fps % config.sample_fps:
            raise ValueError(
                f"sample_fps={config.sample_fps} must divide source_fps={source_fps}"
            )
        if config.num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        if config.video_backend not in {"auto", "torchcodec", "opencv"}:
            raise ValueError(
                "video_backend must be one of 'auto', 'torchcodec', or 'opencv'"
            )

        self.source_fps = source_fps
        self.frame_stride = source_fps // config.sample_fps
        self.data_template = self.info["data_path"]
        self.video_template = self.info["video_path"]
        self.chunk_size = int(self.info["chunks_size"])

        if episode_indices is None:
            if split not in self.info["splits"]:
                raise KeyError(
                    f"Unknown split {split!r}; "
                    f"available splits: {sorted(self.info['splits'])}"
                )
            selected_indices = tuple(
                _parse_split(self.info["splits"][split], len(self.episodes))
            )
        else:
            selected_indices = tuple(int(index) for index in episode_indices)
            if len(selected_indices) != len(set(selected_indices)):
                raise ValueError("episode_indices must not contain duplicates")
            invalid = [
                index
                for index in selected_indices
                if not 0 <= index < len(self.episodes)
            ]
            if invalid:
                raise ValueError(
                    f"episode_indices contains out-of-range values: {invalid}"
                )
        if not selected_indices:
            raise ValueError(f"Open-H dataset split {split!r} contains no episodes")
        self.episode_indices = selected_indices
        self.windows = self._build_windows()

        self.normalizer = normalizer or _compute_train_normalizer(
            self.root, self.info, self.episodes, config.state_key
        )
        self._table_cache: OrderedDict[int, _EpisodeTable] = OrderedDict()
        self._decoder_cache: OrderedDict[int, Any] = OrderedDict()
        self._resolved_video_backend = (
            None if config.video_backend == "auto" else config.video_backend
        )

        if config.validate_files:
            self._validate_episode_files()

    @staticmethod
    def _validate_feature(
        features: dict[str, Any],
        key: str,
        expected_dtype: str,
        expected_shape: list[int] | None = None,
    ) -> None:
        if key not in features:
            raise KeyError(f"Feature {key!r} is absent from meta/info.json")
        feature = features[key]
        if feature.get("dtype") != expected_dtype:
            raise ValueError(
                f"Feature {key!r} has dtype={feature.get('dtype')!r}, "
                f"expected {expected_dtype!r}"
            )
        if expected_shape is not None and feature.get("shape") != expected_shape:
            raise ValueError(
                f"Feature {key!r} has shape={feature.get('shape')!r}, "
                f"expected {expected_shape!r}"
            )

    def _build_windows(self) -> tuple[_Window, ...]:
        windows = []
        # Reserve one complete source sampling interval after each clip. This
        # mirrors the action-at-every-timestep convention and avoids terminal
        # boundary windows. For the Hamlyn train/val splits this yields the
        # expected 16,262 / 2,103 windows.
        reserved_span = self.config.num_frames * self.frame_stride
        for episode_index in self.episode_indices:
            length = self.episodes[episode_index].length
            last_start = length - reserved_span
            if last_start < 0:
                continue
            windows.extend(
                _Window(episode_index, start)
                for start in range(0, last_start + 1, self.frame_stride)
            )
        return tuple(windows)

    def _data_path(self, episode_index: int) -> Path:
        return self.root / self.data_template.format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
        )

    def _video_path(self, episode_index: int) -> Path:
        return self.root / self.video_template.format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
            video_key=self.config.camera_key,
        )

    def _validate_episode_files(self) -> None:
        missing = []
        for episode_index in self.episode_indices:
            for path in (
                self._data_path(episode_index),
                self._video_path(episode_index),
            ):
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            preview = "\n".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n... and {len(missing)-10} more"
            raise FileNotFoundError(f"Missing Open-H episode files:\n{preview}{suffix}")

    def _load_episode(self, episode_index: int) -> _EpisodeTable:
        cached = self._table_cache.pop(episode_index, None)
        if cached is not None:
            self._table_cache[episode_index] = cached
            return cached
        record = self.episodes[episode_index]
        table = _load_episode_table(
            self._data_path(episode_index),
            state_key=self.config.state_key,
            expected_length=record.length,
        )
        self._table_cache[episode_index] = table
        while len(self._table_cache) > self.config.table_cache_size:
            self._table_cache.popitem(last=False)
        return table

    def _get_decoder(self, episode_index: int):
        cached = self._decoder_cache.pop(episode_index, None)
        if cached is not None:
            self._decoder_cache[episode_index] = cached
            return cached

        video_path = self._video_path(episode_index)
        if self._resolved_video_backend == "opencv":
            decoder = _OpenCVVideoDecoder(video_path)
        else:
            try:
                from torchcodec.decoders import VideoDecoder

                decoder = VideoDecoder(
                    video_path,
                    dimension_order="NCHW",
                    num_ffmpeg_threads=self.config.video_threads,
                    device="cpu",
                    seek_mode="exact",
                )
                self._resolved_video_backend = "torchcodec"
            except Exception as error:
                if self.config.video_backend == "torchcodec":
                    raise RuntimeError(
                        "TorchCodec was explicitly requested but could not be loaded"
                    ) from error
                error_summary = str(error).splitlines()[0]
                warnings.warn(
                    "TorchCodec could not be loaded; falling back to exact OpenCV "
                    f"decoding. Original error: {error_summary}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._resolved_video_backend = "opencv"
                decoder = _OpenCVVideoDecoder(video_path)
        expected_length = self.episodes[episode_index].length
        if len(decoder) < expected_length:
            raise ValueError(
                f"Video for episode {episode_index} has {len(decoder)} frames, "
                f"but metadata declares {expected_length}"
            )
        self._decoder_cache[episode_index] = decoder
        while len(self._decoder_cache) > self.config.decoder_cache_size:
            self._decoder_cache.popitem(last=False)
        return decoder

    def _decode_frames(
        self, episode_index: int, frame_indices: torch.Tensor
    ) -> torch.Tensor:
        decoder = self._get_decoder(episode_index)
        return decoder.get_frames_at(frame_indices.tolist()).data

    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected decoded RGB frames [T,3,H,W], got {tuple(frames.shape)}"
            )
        height, width = frames.shape[-2:]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        frames = frames[:, :, top : top + crop_size, left : left + crop_size].to(
            torch.float32
        )
        frames = F.interpolate(
            frames,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return frames.div_(255.0).clamp_(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> OpenHSample:
        window = self.windows[index]
        offsets = torch.arange(self.config.num_frames, dtype=torch.int64)
        requested_indices = window.start_frame + offsets * self.frame_stride

        table = self._load_episode(window.episode_index)
        stored_frame_indices = table.frame_indices.index_select(0, requested_indices)
        if not torch.equal(stored_frame_indices, requested_indices):
            raise ValueError(
                f"Episode {window.episode_index} frame_index column is not contiguous "
                f"at requested window starting {window.start_frame}"
            )

        raw_proprioception = canonicalize_quaternions(
            table.proprioception.index_select(0, requested_indices)
        )
        proprioception = self.normalizer.normalize_proprioception(raw_proprioception)
        raw_actions = build_proprioception_pairs(raw_proprioception)
        actions = build_proprioception_pairs(proprioception)

        frames = self._decode_frames(window.episode_index, requested_indices)
        states = self._preprocess_frames(frames).permute(1, 0, 2, 3).contiguous()

        return OpenHSample(
            states=states,
            actions=actions.transpose(0, 1).contiguous(),
            proprioception=proprioception.transpose(0, 1).contiguous(),
            raw_actions=raw_actions.transpose(0, 1).contiguous(),
            raw_proprioception=raw_proprioception.transpose(0, 1).contiguous(),
            episode_index=torch.tensor(window.episode_index, dtype=torch.int64),
            frame_indices=requested_indices,
            timestamps=table.timestamps.index_select(0, requested_indices),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_table_cache"] = OrderedDict()
        state["_decoder_cache"] = OrderedDict()
        return state


def make_open_h_loaders(
    config: OpenHDatasetConfig,
) -> tuple[DataLoader, DataLoader, OpenHNormalizer]:
    """Create train/validation loaders sharing leakage-free normalization stats.

    By default, this uses the official ``train`` and ``val`` metadata splits.
    When ``config.holdout_episodes > 0``, that many complete episodes are
    deterministically selected from the official train split for validation.
    They are removed from both the training dataset and normalization fit.
    """
    root = Path(config.data_root).expanduser().resolve()
    info, episodes = _read_metadata(root)
    official_train_indices = tuple(_parse_split(info["splits"]["train"], len(episodes)))

    if config.holdout_episodes < 0:
        raise ValueError("holdout_episodes must be non-negative")
    if config.holdout_episodes >= len(official_train_indices):
        raise ValueError(
            "holdout_episodes must leave at least one official train episode "
            f"for training; got {config.holdout_episodes} holdouts from "
            f"{len(official_train_indices)} episodes"
        )

    if config.holdout_episodes:
        generator = torch.Generator().manual_seed(config.holdout_seed)
        permutation = torch.randperm(
            len(official_train_indices), generator=generator
        ).tolist()
        holdout_positions = set(permutation[: config.holdout_episodes])
        train_episode_indices = tuple(
            episode_index
            for position, episode_index in enumerate(official_train_indices)
            if position not in holdout_positions
        )
        val_episode_indices = tuple(
            sorted(official_train_indices[position] for position in holdout_positions)
        )
        val_split = "train_holdout"
    else:
        train_episode_indices = official_train_indices
        val_episode_indices = tuple(_parse_split(info["splits"]["val"], len(episodes)))
        val_split = "val"

    normalizer = _compute_train_normalizer(
        root,
        info,
        episodes,
        state_key=config.state_key,
        episode_indices=train_episode_indices,
    )
    train_dataset = OpenHWindowDataset(
        config,
        split="train",
        normalizer=normalizer,
        episode_indices=train_episode_indices,
    )
    val_dataset = OpenHWindowDataset(
        config,
        split=val_split,
        normalizer=normalizer,
        episode_indices=val_episode_indices,
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_mem,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
    }
    if config.num_workers > 0 and config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, normalizer
