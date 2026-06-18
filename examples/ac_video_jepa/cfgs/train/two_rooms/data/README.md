# Data pipeline configuration

This directory contains example configs for the three data pipeline modes
available in `two_rooms` training. Select a mode via `data.pipeline.mode`
in your training config.

---

## Modes

### `online` — on-the-fly CPU generation (default)

```yaml
data:
  pipeline:
    mode: online
```

Standard PyTorch `DataLoader` backed by `WallDataset`. Each worker calls
`generate_multistep_sample()` on the fly. Simple, no warm-up needed.
Use this for quick experiments or when CPU generation is fast enough.

---

### `stream` — GPU-resident double-buffer

```yaml
data:
  pipeline:
    mode: stream
    backend: cpu   # or gpu
    chunk_size: 3840
    dtype: bfloat16
    # cpu backend only:
    num_gen_workers: 16
    # gpu backend only:
    gen_batch_size: null
```

Two small chunks of samples live permanently in GPU VRAM (double-buffer).
Every `chunk_size // batch_size` training steps the pipeline swaps:
promotes the pre-fetched next chunk to current, kicks off a new async
generation in the background. The GPU never waits for a full epoch of data.

**Caller contract** (handled automatically by `main.py`):
- call `manager.warm_up()` once after device setup, before training
- call `manager.shutdown()` at the end of training

#### `backend: cpu`

Generation runs in a pool of CPU worker processes (`AsyncChunkGenerator`).
Tune `num_gen_workers` to the number of available CPUs minus a few for the
main process and OS.

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_size` | dataset size | Samples per swap. Must evenly divide `epoch_size // batch_size`. |
| `num_gen_workers` | 16 | CPU cores used for generation. |
| `dtype` | `bfloat16` | Cast samples in workers to halve their RAM footprint. |

#### `backend: gpu`

Generation runs on a dedicated CUDA stream using vectorised GPU kernels
(`GPUWallGenerator`). Much faster than CPU generation on large GPU nodes;
no CPU worker pool needed.

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_size` | dataset size | Samples per swap. |
| `gen_batch_size` | `chunk_size` | Trajectories per GPU kernel call. Larger = fewer launches but more VRAM. |
| `dtype` | `bfloat16` | Output dtype of generated chunk. |

---

### `offline` — pre-generated memmaps from disk

```yaml
data:
  pipeline:
    mode: offline
    stream: true            # default: read through the VRAM stream pipeline
    data_dir: /path/to/dataset
    chunk_size: 9600
    dtype: bfloat16
```

Reads a fixed dataset of numpy memmaps pre-generated once and stored on
disk (e.g. Lustre). No online generation overhead during training.

**Step 1 — generate the dataset (once):**

```bash
python -m eb_jepa.datasets.two_rooms.offline_dataset \
  --data-dir /path/to/dataset \
  --num-samples 1200000 \
  --backend gpu
```

**Step 2 — train** with `pipeline.mode: offline` and `data_dir` pointing at the output.

#### `stream: true` (default) — offline-STREAM

The dataset is fed through the same double-buffered VRAM pipeline as `stream`
mode, but the chunk *source* is `OfflineChunkReader` (large sequential reads
from the memmap) instead of CPU/GPU generation. The dataset is traversed in
stored order across epochs; each epoch consumes `config.size` samples.

> **Keep `chunk_size` small.** The pipeline only hides I/O when there are many
> chunks per epoch to overlap with compute. Setting `chunk_size` to the whole
> epoch (e.g. 100000) degenerates it: one chunk/epoch means zero overlap plus a
> ~29 GB non-overlappable copy+pin+H2D every epoch (~290 s/epoch vs ~28 s with
> `chunk_size=9600`).

| Option | Default | Description |
|--------|---------|-------------|
| `data_dir` | — (required) | Path to the directory produced by `gpu_generator.py`. |
| `chunk_size` | 9600 | Samples per swap (25 batches). Must evenly divide `epoch_size // batch_size`. Keep small. |
| `dtype` | `bfloat16` | Cast states/actions/locations to this dtype on read. |
| `shuffle` | `false` | `false` = in-order traversal; `true` = per-epoch block shuffle (whole chunks permuted + intra-chunk shuffle). |
| `read_workers` | 1 | `>1` fans each chunk read over N threads (concurrent I/O against the OST). Only helps when **cold-cache / disk-bound**; warm page cache already hits the compute floor at 1. |
| `prefetch_depth` | 1 | `>1` switches to `DeepPrefetchManager`, which keeps N chunks reading + staging in VRAM continuously and frees each GPU chunk as it is consumed. Helps hide a slow disk; no benefit once compute-bound. |

#### `stream: false` — legacy random-access DataLoader

Standard `DataLoader` with `OfflineWallDataset` and `shuffle=True`. One epoch =
~1.2M scattered per-sample reads across the memmap — fine on local SSD, **very
slow on Lustre** (~20+ min/epoch). Prefer `stream: true`.

`config.size` is automatically overridden at runtime by the actual dataset length.

---

## Summary

| Mode | Generation | Warm-up needed | When to use |
|------|-----------|----------------|-------------|
| `online` | per-batch CPU | no | quick experiments |
| `stream cpu` | async CPU workers | yes | large CPU nodes, moderate GPU |
| `stream gpu` | async GPU kernels | yes | large GPU nodes, generation bottleneck |
| `offline` | pre-generated disk | no | repeated training runs on same data |

## Parallel eval

All modes support batched MPPI evaluation across multiple environments in lockstep.
Set `meta.n_parallel` in your eval config (e.g. `cfgs/eval/two_rooms/eval.yaml`) to run `n_parallel`
episodes simultaneously — one batched MPPI call instead of N sequential ones.

```yaml
# cfgs/eval/two_rooms/eval.yaml
meta:
  num_eval_episodes: 20
  n_parallel: 20   # 1 = sequential (default), >1 = parallel
```

> **Note:** a single parallel batch of K episodes is one correlated draw, not K
> independent samples. For reliable success-rate estimates use sequential eval
> (`n_parallel: 1`) or average several parallel batches.
