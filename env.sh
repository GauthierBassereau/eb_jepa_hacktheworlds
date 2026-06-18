#!/usr/bin/env bash
# Source this file to set up the EB-JEPA environment variables.
# Usage: source env.sh
# In SLURM scripts: source "$(dirname "$0")/../env.sh"  (adjust path as needed)

# Your personal work directory — defaults to the project work partition under your
# username. Override by setting EBJEPA_WORK before sourcing. WORK IS NOT YOUR HOME:
# clone the repo and run everything from /lustre/work (the home quota blocks git/venvs).
WORK=${EBJEPA_WORK:-/lustre/work/pdl17890/$USER}
export EBJEPA_WORK="$WORK"                  # exported so python (launch_sbatch) sees it
ARCH=$(uname -m)                           # x86_64 on login node, aarch64 on compute nodes
export EBJEPA_COMPUTE_ARCH=${EBJEPA_COMPUTE_ARCH:-aarch64}  # target arch for SLURM jobs

# Repo root (this file lives at the repo root) — SLURM scripts read $EBJEPA_REPO.
export EBJEPA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cluster utility scripts
export PATH="$EBJEPA_REPO/cluster:$PATH"

# uv binary (arch-specific, avoids Exec format error across node types)
export UV_INSTALL_DIR=$WORK/uv_bin/$ARCH
export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$PATH"

# uv cache and venv (arch-specific so x86_64 and aarch64 don't collide)
export UV_CACHE_DIR=$WORK/uv_cache/$ARCH
export UV_PROJECT_ENVIRONMENT=$WORK/venvs/eb_jepa_$ARCH

# Keep ALL caches on /work — the /lustre/home quota is small and fills up fast
# (model/dataset downloads, torch.compile kernels, pip wheels, matplotlib, ...).
export XDG_CACHE_HOME=$WORK/.cache              # catch-all (pip, matplotlib, fontconfig, ...)
export HF_HOME=$WORK/.cache/huggingface         # HuggingFace hub + datasets + transformers
export TORCH_HOME=$WORK/.cache/torch            # torch hub weights
export TRITON_CACHE_DIR=$WORK/.cache/triton     # torch.compile / triton kernels
export PIP_CACHE_DIR=$WORK/.cache/pip
export WANDB_DIR=$WORK/wandb                     # W&B run files + cache
export WANDB_CACHE_DIR=$WORK/.cache/wandb

# EB-JEPA paths
export EBJEPA_CKPTS=${EBJEPA_CKPTS:-$WORK/checkpoints}
# Dataset folder. Defaults to your own $WORK/datasets; set EBJEPA_DSETS to point at
# a shared/provided dataset folder if one is available on your cluster.
export EBJEPA_DSETS=${EBJEPA_DSETS:-$WORK/datasets}

# W&B: export WANDB_DISABLED=true before sourcing to turn off logging cluster-wide
export WANDB_DISABLED=${WANDB_DISABLED:-false}

# --- SLURM defaults for the HTW cluster (GB200 / Grace-Blackwell nodes) --------------
# These feed examples/launch_sbatch.py. Override any of them by exporting the same var
# before sourcing this file (so the launcher stays portable to other clusters).
export EBJEPA_SLURM_PARTITION=${EBJEPA_SLURM_PARTITION:-defq}
export EBJEPA_SLURM_GPUS=${EBJEPA_SLURM_GPUS:-1}
export EBJEPA_SLURM_CPUS=${EBJEPA_SLURM_CPUS:-8}
export EBJEPA_SLURM_TIME_MIN=${EBJEPA_SLURM_TIME_MIN:-120}
# GB200 nodes carry large Grace LPDDR (~480 GB per superchip, ~240 GB/GPU). Request a
# generous slice per GPU; lower it with `scontrol show node <node>` (RealMemory) if jobs
# stay PENDING on memory.
export EBJEPA_SLURM_MEM=${EBJEPA_SLURM_MEM:-220G}

# Per-user SLURM account & QOS.
# Resolution order: value already in the environment  >  the persisted per-user file
# (written once by setup.sh)  >  SLURM auto-detection via sacctmgr. If nothing is found
# they stay empty, and launch_sbatch simply omits --account/--qos (SLURM then uses your
# own defaults). This keeps shells non-interactive; the prompt only happens in setup.sh.
EBJEPA_SLURM_USER_ENV=${EBJEPA_SLURM_USER_ENV:-$WORK/.eb_jepa_slurm.env}
[ -f "$EBJEPA_SLURM_USER_ENV" ] && source "$EBJEPA_SLURM_USER_ENV"

if [ -z "${EBJEPA_SLURM_CONFIGURED:-}" ]; then
    if [ -z "${EBJEPA_SLURM_ACCOUNT:-}" ] && command -v sacctmgr >/dev/null 2>&1; then
        _eb_acct=$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null | grep -v '^$' | head -1)
        [ -z "$_eb_acct" ] && _eb_acct=$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1)
        [ -n "$_eb_acct" ] && export EBJEPA_SLURM_ACCOUNT="$_eb_acct"
    fi
    if [ -z "${EBJEPA_SLURM_QOS:-}" ] && command -v sacctmgr >/dev/null 2>&1; then
        _eb_qos=$(sacctmgr -nP show assoc user="$USER" format=QOS 2>/dev/null | tr ',' '\n' | grep -v '^$' | head -1)
        [ -n "$_eb_qos" ] && export EBJEPA_SLURM_QOS="$_eb_qos"
    fi
fi
# Export whatever we resolved (possibly empty -> omitted by the launcher).
export EBJEPA_SLURM_ACCOUNT=${EBJEPA_SLURM_ACCOUNT:-}
export EBJEPA_SLURM_QOS=${EBJEPA_SLURM_QOS:-}
