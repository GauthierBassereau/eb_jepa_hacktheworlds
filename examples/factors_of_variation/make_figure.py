"""Aggregate per-seed ``fov_sweep.json`` files into a success-vs-severity figure + results JSON.

    python examples/factors_of_variation/make_figure.py \
        --sweeps run_seed1/fov_sweep.json run_seed1000/fov_sweep.json ... \
        --out_dir results/factors_of_variation
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# factor -> (training/baseline value, x-axis label, "harder" direction)
FACTORS = {
    "dot_std": (1.3, "dot_std (blur)", "larger = blurrier"),
    "wall_width": (3, "wall_width (px)", "larger = thicker"),
    "door_space": (4, "door_space (half-gap px)", "smaller = narrower"),
}


def parse_label(label):
    if label == "baseline":
        return "baseline", None
    m = re.match(r"^(.*)_([0-9.]+)$", label)
    return (m.group(1), float(m.group(2))) if m else (label, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--out_dir", default="results/factors_of_variation")
    args = ap.parse_args()

    data = defaultdict(lambda: defaultdict(list))  # factor -> severity -> [success%]
    base_vals = []
    for path in args.sweeps:
        for row in json.load(open(path))["results"]:
            factor, sev = parse_label(row["label"])
            sr = row["success_rate"] * 100.0
            (base_vals if factor == "baseline" else data[factor][sev]).append(sr)
    base_m = float(np.mean(base_vals)) if base_vals else float("nan")
    base_s = float(np.std(base_vals) / max(1, np.sqrt(len(base_vals)))) if base_vals else 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    factors = [f for f in FACTORS if f in data]
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 4.2), squeeze=False)
    out = {"n_seeds": len(args.sweeps), "baseline_pct": round(base_m, 2),
           "baseline_sem_pct": round(base_s, 2), "factors": {}}
    for ax, factor in zip(axes[0], factors):
        base_v, xlabel, note = FACTORS[factor]
        pts = [(base_v, base_m, base_s)] + [
            (s, float(np.mean(data[factor][s])),
             float(np.std(data[factor][s]) / max(1, np.sqrt(len(data[factor][s])))))
            for s in sorted(data[factor])]
        pts.sort(key=lambda t: t[0])
        xs, ms, ss = zip(*pts)
        ax.errorbar(xs, ms, yerr=ss, marker="o", lw=2, capsize=3)
        ax.axvline(base_v, ls="--", color="gray", alpha=0.5)
        ax.set(xlabel=f"{xlabel}\nbaseline={base_v}; {note}", ylabel="planning success (%)",
               title=factor, ylim=(-3, 103))
        ax.grid(alpha=0.3)
        out["factors"][factor] = {"severities": list(xs),
                                  "success_pct": [round(m, 2) for m in ms],
                                  "sem_pct": [round(s, 2) for s in ss]}
    fig.suptitle(f"Two Rooms planning success vs factor-of-variation severity "
                 f"({len(args.sweeps)} seeds, mean±SEM)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "factors_compare.png"), dpi=110, bbox_inches="tight")
    json.dump(out, open(os.path.join(args.out_dir, "results.json"), "w"), indent=2)
    print("saved", os.path.join(args.out_dir, "factors_compare.png"), "+ results.json")


if __name__ == "__main__":
    main()
