# Factors of variation — stress-testing the Two Rooms recipe (Track 13)

**Question.** `eb_jepa`'s Two Rooms world model is deliberately tiny. Does the minimal recipe
(VICReg variance/covariance + temporal-similarity + inverse-dynamics regularizers,
distance-cost MPPI planning) survive realistic perturbations, and *which factor / regularizer
term breaks first* as the world gets harder?

You train the recipe, then sweep planning success as you dial controllable Two Rooms factors
of variation out of distribution — no new environment to build (the perturbations are config
overrides on the existing `two_rooms` env).

## Layout
```
examples/factors_of_variation/
├── main.py        # train the recipe — TODO: build_jepa() (the loop is provided)
├── eval.py        # the stress test: zero-shot planning sweep over a factor grid (provided)
├── make_figure.py # aggregate per-seed sweeps -> success-vs-severity figure + results.json
└── cfgs/
    ├── train.yaml # the recipe (GPU-stream data; set pipeline.mode=online without a strong GPU)
    └── eval.yaml  # the perturbation grid + planner + episode count
```
The dataset (`two_rooms`) and planner (`eb_jepa.planning`) are reused from the core; the eval
sweep calls `ac_video_jepa`'s `launch_plan_eval`.

## What you implement (the `# TODO`)
1. `main.py:build_jepa` — assemble the AC-Video-JEPA recipe from `eb_jepa` parts
   (`ImpalaEncoder` + `RNNPredictor` + identity action encoder + `InverseDynamicsModel` +
   `VC_IDM_Sim_Regularizer` + `JEPA`). The docstring lists the exact pieces;
   `examples/ac_video_jepa/main.py` is the full reference for the same recipe. `eval.py`
   imports this function, so once it is filled both training and the sweep work.

## Factors (controllable from `cfgs/eval.yaml:grid`)
- **dot_std** — Gaussian dot blur (visual appearance).
- **wall_width** — central wall thickness (geometry).
- **door_space** — door half-gap; *smaller = narrower passage* (geometry).
Add any `WallDatasetConfig` field as a new grid point.

## Run
```bash
# train the baseline (3 seeds) via the launcher (HTW SLURM autoconfig)
python -m examples.launch_sbatch --example factors_of_variation --full-sweep
# variants are pure overrides:
python -m examples.launch_sbatch --example factors_of_variation --model.dstc 128      # capacity
python -m examples.launch_sbatch --example factors_of_variation --data.door_space 2   # train-on-perturbed

# stress-test a trained checkpoint, then plot across seeds
python -m examples.factors_of_variation.eval --model_folder <ckpt_dir>
python examples/factors_of_variation/make_figure.py \
    --sweeps <seed1>/fov_sweep.json <seed1000>/fov_sweep.json <seed10000>/fov_sweep.json \
    --out_dir results/factors_of_variation
```

## What to look for
The minimal recipe is robust to appearance (dot blur) and wall thickness, but planning
**collapses as the door narrows** (≈100% → 40% → 0% at door half-gap 4 → 2 → 1). Don't stop at
the curve — calibrate it with a true-state oracle + random policy (what is achievable / chance?),
localize the failure (per-term losses, position-probe MSE, predictor rollout MSE, the encoder's
effective rank), and test whether *strengthening a term* helps or only *capacity* (`model.dstc`)
does. Richer appearance factors (color, background, occlusion, multiple doors) via the
`stable-worldmodel` / `jepa-wms` suites are the ambitious stretch.
