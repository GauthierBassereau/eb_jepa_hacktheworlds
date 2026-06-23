<h1 align="center">
    <p>⚡ <b>EB-JEPA</b></p>
</h1>

<h2 align="center">
    <p><i>Energy-Based Joint-Embedding Predictive Architectures</i></p>
</h2>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/Trick5t3r/eb_jepa" target="_blank" style="margin: 2px;"><img alt="Github" src="https://img.shields.io/badge/Github-Trick5t3r/eb__jepa-black?logo=github" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://arxiv.org/abs/2602.03604" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2602.03604-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<br>

<p align="center">
  <b><a href="https://ai.facebook.com/research/">Meta AI Research, FAIR</a></b>
</p>

<p align="center">
  <a href="https://x.com/BasileTerv987">Basile Terver</a>,
  Randall Balestriero,
  Megi Dervishi,
  David Fan,
  Quentin Garrido,
  Tushar Nagarajan,
  <br>
  Koustuv Sinha,
  Wancong Zhang,
  Mike Rabbat,
  Yann LeCun,
  Amir Bar
</p>

<p align="center">
  An open source library and tutorial for learning representations for<br>
  prediction and planning using joint embedding predictive architectures.
</p>

<p align="center">
  <img src="docs/archi-schema-eb-jepa.png" alt="EB-JEPA Architecture" width="800">
</p>

> Each example is (almost) self-contained and training takes up to a few hours on a single GPU card.

---

## 🩺 Hackathon: Surgical JEPA

For the hackathon, our team tried to turn the AC-JEPA example into a small
surgical world model. The idea was simple: take wrist-camera video from a robot,
encode it into a latent state, condition the predictor on proprioception, and see
whether EB-JEPA can roll the scene forward without reconstructing pixels during
training.

We used the
[`PhysicalAI-Robotics-Open-H-Embodiment`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
dataset, specifically `Surgical/hamlyn/suturing_2`. Each sample is a 17-frame
RGB wrist-camera clip sampled at 5 fps, resized to 128x128. The conditioning
vector is 32-D: `[proprio_t, proprio_t+1]`, with 16-D bimanual Cartesian
proprioception on each side. Complete episodes are held out, and normalization
statistics are fit only on the remaining train episodes.

The work is intentionally split between reusable library changes and the actual
experiment code:

- `eb_jepa/datasets/open_h` adds a LeRobot v2.1 reader without depending on
  LeRobot itself. It reads the episode metadata, Parquet proprioception tables,
  and MP4 videos directly, canonicalizes quaternions, decodes exact frames with
  TorchCodec or OpenCV, and returns tensors in the AC-JEPA `[C,T,H,W]` /
  `[A,T]` convention.
- `eb_jepa/jepa.py` now lets sequence predictors start autoregressive rollout
  from one real frame and grow their context window as predictions are appended.
- `eb_jepa/architectures.py` adds a LeWorldModel-style causal Transformer
  predictor with AdaLN-Zero action conditioning, plus a DINOv3 ConvNeXt encoder
  that projects patch tokens back into the standard `[B,D,T,1,1]` latent format.
- `examples/surgical_jepa` contains the hackathon pipeline: dataset smoke tests,
  JEPA training configs, RGB decoder training, LPIPS evaluation, rollout videos,
  coefficient notes, and an inference benchmark.

<p align="center">
  <img src="docs/eval_ep15.gif" alt="Surgical JEPA rollout at epoch 15" width="776">
  <br>
  <i>Best run at epoch 15: ground truth, autoregressive rollout, and clean-context prediction.</i>
</p>

### What we tried

The baseline is close to the existing AC-JEPA setup: an IMPALA encoder, a GRU
latent predictor, VC/IDM anti-collapse regularization, and 8-step autoregressive
training. We then tried two bigger changes:

- **Transformer predictor:** a compact 4-layer causal Transformer, conditioned
  by embedded proprioception with AdaLN-Zero. During rollout it starts from one
  real latent and can use up to four previous predicted latents.
- **DINOv3 encoder:** `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` as the
  image encoder, with a learned projection from ConvNeXt patch tokens into the
  EB-JEPA latent. The pretrained backbone has its own smaller learning rate.

Pixels are only used after JEPA training. We freeze the JEPA, train a small RGB
decoder on its latents, then evaluate rollouts in pixel space with LPIPS. The
evaluation reports both fully autoregressive predictions and clean-context
predictions; the gap between them is a useful proxy for compounding error.

### Ablations

The table below is copied from [`docs/ablations.xlsx`](docs/ablations.xlsx).
Lower LPIPS is better. `Gap Clean/AR` is the extra LPIPS paid by using generated
latent context instead of clean encoded context, so lower is also better.
`Mean perf. change` is the spreadsheet's relative average vs. the baseline
across AR LPIPS, gap, and the horizon LPIPS columns; positive means better than
baseline.

| Run | AR LPIPS ↓ | Gap Clean/AR ↓ | LPIPS @ 0.2s ↓ | LPIPS @ 1s ↓ | LPIPS @ 2s ↓ | Mean perf. change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Collapse (no reg term) | 0.575 | 0.026 | 0.549 | 0.550 | 0.587 | - |
| Baseline (EB-JEPA) | 0.470 | 0.020 | 0.446 | 0.448 | 0.480 | - |
| `W_Cov=0` | 0.507 | 0.010 | 0.495 | 0.497 | 0.512 | +2.7% |
| `W_Std=0` | 0.530 | 0.009 | 0.519 | 0.520 | 0.536 | -0.4% |
| `W_Sim=0` | 0.465 | 0.026 | 0.435 | 0.438 | 0.478 | -4.8% |
| `W_Idm=0` | 0.476 | 0.021 | 0.450 | 0.454 | 0.488 | -2.0% |
| DINOv3 Encoder | 0.478 | 0.014 | 0.450 | 0.466 | 0.475 | +4.7% |
| Transformer Predictor | 0.476 | 0.007 | 0.435 | 0.438 | 0.450 | +14.9% |

What we take from this:

- The no-regularization run collapses badly enough to make the point: AR LPIPS
  goes from `0.470` to `0.575`.
- Removing temporal similarity gives the best raw mean AR LPIPS (`0.465`) and
  strong short-horizon numbers, but it increases the clean/AR gap. That looks
  like a nicer one-step model, not necessarily a safer rollout model.
- The Transformer predictor is the coolest result. It does not win on mean AR
  LPIPS, but it cuts the compounding gap from `0.020` to `0.007` and improves
  LPIPS at 2 seconds from `0.480` to `0.450`.
- DINOv3 helped the gap but was not a free win on the horizon metrics. With more
  tuning it might still be worth revisiting, but for this hackathon the
  Transformer change is the cleaner story.

To reproduce the pipeline, first update `data.data_root` in the selected config.

```bash
# Check the dataset and generate sample previews
uv run python -m examples.surgical_jepa.test_dataset --num-previews 3

# Train the IMPALA + GRU baseline
uv run python examples/surgical_jepa/main.py \
  --fname examples/surgical_jepa/train.yaml

# Other variants we tested
uv run python examples/surgical_jepa/main.py \
  --fname examples/surgical_jepa/train_transformer.yaml
uv run python examples/surgical_jepa/main.py \
  --fname examples/surgical_jepa/train_ConvNeXt.yaml

# Train the RGB decoder from a JEPA checkpoint
uv run python examples/surgical_jepa/train_decoder.py \
  --jepa_checkpoint /path/to/jepa/best.pth.tar

# Evaluate decoded rollouts
uv run python examples/surgical_jepa/evaluation.py \
  --checkpoint /path/to/decoder/best.pth.tar
```

---

## 📚 Examples

### [Image JEPA](examples/image_jepa/README.md)

Self-supervised representations from unlabeled images on CIFAR-10, evaluated on classification.

![Image JEPA Architecture](examples/image_jepa/assets/arch_figure.png)

### [Video JEPA](examples/video_jepa/README.md)

Predict next image representation in a sequence.

![Moving MNIST](examples/video_jepa/assets/viz.png)

### [AC Video JEPA](examples/ac_video_jepa/README.md)

JEPA for world modeling + planning in Two Rooms environment.

| Planning Episode | Task Definition |
|------------------|-----------------|
| <img src="examples/ac_video_jepa/assets/top_randw_agent_steps_succ.gif" alt="Successful planning episode" width="155" /> | <img src="examples/ac_video_jepa/assets/top_randw_state.png" alt="Episode task definition" width="300" /> |
| *Successful planning episode* | *From init to goal state* |

---

## 🚀 Installation

### HTW cluster — quick start (hackathon only)

> Skip this section unless you are on the HTW hackathon cluster — the generic install below is all you need locally.

Please follow the [setup instructions](setup.md) before starting the project.

---

### Local / generic (start here)

We use [uv](https://docs.astral.sh/uv/guides/projects/) for package management.

```bash
# Install dependencies
uv sync
# Option 1: Activate virtual environment
source .venv/bin/activate
python -m examples.image_jepa.main
# Option 2: Run directly with uv
uv run python -m examples.image_jepa.main
```
If you need conda-specific packages, you can use **Conda + uv**

```bash
# Create conda environment with Python 3.12
conda create -n eb_jepa python=3.12 -y
conda activate eb_jepa
# Install package in editable mode with dev dependencies (pytest, black, isort, autoflake)
uv pip install -e . --group dev
```

Add these to your `~/.bashrc` for persistent configuration.

```bash
# Where datasets are stored / looked up
export EBJEPA_DSETS=/path/to/eb_jepa/datasets
# Optional: Directory for checkpoints and logs
export EBJEPA_CKPTS=/path/to/checkpoints
```

Verify the install with `uv run pytest tests/`.

## 🏋️ Training

### Quick Start

```bash
# Local training
python -m examples.{image_jepa,video_jepa,ac_video_jepa}.main
```
> Our default configs are tuned for H100 GPUs. With older GPUs (e.g., A100, V100), you may need to reduce batch size to fit in memory.

### 📂 Folder Structure

All experiments use a unified folder structure:

```
checkpoints/
└── {example_name}/
    ├── dev_2026-01-16_00-10/                 # Single/local runs (dev_ prefix)
    │   └── {exp_name}_seed1/
    │
    ├── sweep_2026-01-16_00-10/         # Auto-named 3-seed sweep
    │   ├── {exp_name}_seed1/
    │   ├── {exp_name}_seed1000/
    │   └── {exp_name}_seed10000/
    │
    └── sweep_my_experiment/            # Custom-named sweep
        └── ...
```

`{exp_name}` encodes key hyperparameters to avoid folder collisions, e.g.:
- **image_jepa**: `resnet_vicreg_proj_bs256_ep300_ph2048_po2048_std1.0_cov80.0`
- **video_jepa**: `resnet_bs64_lr0.001_std10.0_cov100.0`
- **ac_video_jepa**: `impala_cov8_std16_simt12_idm1`

<details>
<summary><span style="font-size: 1.17em; font-weight: bold;">🖥️ SLURM Launcher (optional)</span></summary>

| Command | Description |
|---------|-------------|
| `--example {name}` | Choose: `image_jepa`, `video_jepa`, `ac_video_jepa`, `maze`, `fintime`, `ltsf`, `eeg`, `audio`, `pointcloud`, `gray_scott`, `intuitive_physics`, `factors_of_variation` |
| `--fname {path}` | Run the sweep specified in the config at `{path}` |
| `--single` | Launch single job (dev mode) |
| `--sweep {name}` | Custom sweep name |
| `--array-parallelism {N}` | Limits the maximum number of concurrent jobs to `N` |
| `--full-sweep` | Full hyperparameter sweep from config |
| `--use-wandb-sweep` | Enable wandb sweep UI |

```bash
# 3 seeds with wandb averaging (recommended)
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml

# Custom sweep name
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --sweep my_experiment

# Single job
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --single

# Full hyperparameter sweep
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --full-sweep

# With wandb sweep UI for hyperparameter analysis
python -m examples.launch_sbatch --example image_jepa --fname examples/image_jepa/cfgs/default.yaml --use-wandb-sweep
```

Replace `image_jepa` with `ac_video_jepa`, `video_jepa`, or `maze` for other examples.

**Full Sweep Configuration:** The `--full-sweep` flag reads the `sweep.param_grid` section from the example's YAML config file (e.g., `examples/image_jepa/cfgs/default.yaml`). Without this flag, only a 3-seed sweep is launched. To customize sweep parameters, edit the `sweep` section in the config:

```yaml
# Example: examples/image_jepa/cfgs/default.yaml
sweep:
  param_grid:
    loss.cov_coeff: [0.1, 1.0, 10.0, 100.0]
    loss.std_coeff: [1.0, 10.0]
    meta.seed: [1, 1000, 10000]
```

### Wandb Seed Averaging

Runs with the same hyperparameters but different seeds share the same wandb run name, enabling automatic averaging:

1. Go to wandb web UI → Runs table
2. Click **"Group by"** → select **"Name"**
   → Groups runs with identical hyperparameters (different seeds) together

To filter runs from a specific sweep:
3. Click **"Filter"** → **"Group"** → select your sweep name

For detailed wandb sweep analysis (parallel coordinates, hyperparameter importance):
1. Use `--use-wandb-sweep` flag when launching
2. Go to wandb web UI → left pane → **"Sweeps"** → click your sweep name

**SLURM Configuration:** SLURM parameters default to the HTW cluster and are read from `EBJEPA_SLURM_*` env vars (set by `env.sh`, which also auto-detects your account/QOS per user). Override per launch with the CLI flags `--partition`/`--account`/`--cpus-per-task`/`--time-min`/`--gpus-per-node`, or export the matching `EBJEPA_SLURM_*` var. The `SLURM_DEFAULTS` dictionary at the top of `examples/launch_sbatch.py` holds the fallbacks.

</details>

## 🧪 Running test cases

Libraries added to eb_jepa [must have their own test cases](/tests/). To run the tests:

```bash
# With uv sync installation
uv run pytest tests/
# With conda + uv installation (no .venv created)
pytest tests/
```

## 👩‍💻 Development

Before contributing, please format your code with the following tools:

```bash
# Remove unused imports
autoflake --remove-all-unused-imports -r --in-place .
# Sort imports
python -m isort eb_jepa examples tests
# Format code
python -m black eb_jepa examples tests
```

## 📚 Citing EB-JEPA

If you find this repository useful, please consider giving a ⭐ and citing:

```bibtex
@misc{terver2026lightweightlibraryenergybasedjointembedding,
      title={A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures},
      author={Basile Terver and Randall Balestriero and Megi Dervishi and David Fan and Quentin Garrido and Tushar Nagarajan and Koustuv Sinha and Wancong Zhang and Mike Rabbat and Yann LeCun and Amir Bar},
      year={2026},
      eprint={2602.03604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03604},
}
```

## 📄 License

EB-JEPA is Apache licensed. See [LICENSE](LICENSE.md).
