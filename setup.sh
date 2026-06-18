#!/usr/bin/env bash
# One-shot setup script for EB-JEPA on the HTW cluster.
#
# Clone the repo ANYWHERE (even your home) and run `bash setup.sh`: it relocates
# itself to your work partition ($WORK/eb_jepa), sets everything up there, and leaves
# only a pointer README where you cloned it. You then just `cd` into the work copy.
# (The /lustre/home quota is too small for git + venvs + model caches.)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Self-relocate to the work partition --------------------------------------
TARGET_WORK="${EBJEPA_WORK:-/lustre/work/pdl17890/$USER}"
TARGET="$TARGET_WORK/eb_jepa"
if [ "$REPO_ROOT" != "$TARGET" ]; then
    echo "=== Relocating EB-JEPA to your work partition ==="
    echo "    from : $REPO_ROOT"
    echo "    to   : $TARGET"
    mkdir -p "$TARGET_WORK"
    if [ -d "$TARGET/.git" ]; then
        echo ">>> $TARGET already exists — reusing it."
    else
        rsync -a "$REPO_ROOT/" "$TARGET/" 2>/dev/null || cp -a "$REPO_ROOT/." "$TARGET/"
    fi
    [ -d "$TARGET/.git" ] || { echo "!! copy to $TARGET failed"; exit 1; }
    echo ">>> continuing setup from $TARGET ..."
    exec env EBJEPA_ORIG_CLONE="$REPO_ROOT" bash "$TARGET/setup.sh"
fi

# If we got here via relocation, reduce the original clone to a pointer README.
if [ -n "${EBJEPA_ORIG_CLONE:-}" ] && [ "$EBJEPA_ORIG_CLONE" != "$TARGET" ] \
   && [ "$EBJEPA_ORIG_CLONE" != "$HOME" ] && [ -d "$EBJEPA_ORIG_CLONE" ]; then
    find "$EBJEPA_ORIG_CLONE" -mindepth 1 -delete 2>/dev/null || true
    cat > "$EBJEPA_ORIG_CLONE/README.md" <<EOF
# EB-JEPA moved to your work partition

This clone was relocated (with its git history) to your work partition, because the
/lustre/home quota is too small for git, virtualenvs and model caches.

Go there and work from it:

    cd $TARGET

Then it is fully set up — \`source env.sh\` (already added to ~/.bashrc) and, to verify,
\`sbatch slurm_test.sh\`.
EOF
    echo ">>> Original clone cleaned — only a pointer README.md remains at $EBJEPA_ORIG_CLONE"
fi

source "$REPO_ROOT/env.sh"

echo "=== EB-JEPA cluster setup ==="
echo "    Arch   : $ARCH"
echo "    Home   : $HOME"
echo "    Work   : $WORK"
echo "    venv   : $UV_PROJECT_ENVIRONMENT"
echo "    cache  : $UV_CACHE_DIR"
echo ""

# 1. Make cluster scripts executable and ensure they're in PATH via env.sh
chmod +x "$REPO_ROOT"/cluster/{sq,qall,log,gpus,users}

# 2. Create required directories in the work partition (venvs, caches, logs, ckpts).
#    All caches live on /work so the small /lustre/home quota never fills up.
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$WORK/venvs" \
         "$WORK/checkpoints" "$WORK/logs" \
         "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$PIP_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# 3. Install uv for the current arch if not already present
if ! "$UV_INSTALL_DIR/uv" --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    echo "    uv installed: $("$UV_INSTALL_DIR/uv" --version)"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# 4. Pin Python version and install dependencies for the current arch
echo ""
echo ">>> Running uv sync for $ARCH (this downloads wheels — may take a few minutes)..."
cd "$REPO_ROOT"
uv sync --dev

# 5. Sync the aarch64 compute-node venv (only needed when running from the x86_64 login node)
#    Submits a short SLURM job so the aarch64 venv gets torch+cu128 instead of torch+cpu.
if [[ "$ARCH" == "x86_64" ]]; then
    COMPUTE_ARCH="${EBJEPA_COMPUTE_ARCH:-aarch64}"
    COMPUTE_UV_DIR="$WORK/uv_bin/$COMPUTE_ARCH"
    mkdir -p "$COMPUTE_UV_DIR" "$WORK/uv_cache/$COMPUTE_ARCH"
    echo ""
    echo ">>> Submitting $COMPUTE_ARCH venv sync job to SLURM..."
    SYNC_JOB=$(sbatch \
        --partition=defq --account=pdl17890 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time=0:30:0 --job-name=eb_jepa_setup \
        --output="$WORK/logs/setup_${COMPUTE_ARCH}_%j.out" \
        --parsable \
        --wrap="set -e
source $REPO_ROOT/env.sh
if ! \$UV_INSTALL_DIR/uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=\$UV_INSTALL_DIR sh
fi
cd $REPO_ROOT && uv sync --dev
echo '$COMPUTE_ARCH venv ready: '\$UV_PROJECT_ENVIRONMENT")
    echo "    Job $SYNC_JOB submitted — monitor with: log $SYNC_JOB"
fi

# 6. W&B login
echo ""
echo ">>> W&B setup"
if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "    W&B already configured in ~/.netrc"
else
    echo -n "    Enter your W&B API key (leave blank to skip): "
    read -r WANDB_KEY
    if [ -n "$WANDB_KEY" ]; then
        uv run wandb login "$WANDB_KEY"
        echo "    W&B key saved to ~/.netrc"
    else
        echo "    Skipped. To enable later: uv run wandb login <key>"
        echo "    To disable cluster-wide: export WANDB_DISABLED=true before sourcing env.sh"
    fi
fi

# 7. SLURM account / QOS — resolve per-user and persist.
#    env.sh (sourced above) already tried to auto-detect them via sacctmgr; we only
#    prompt for whatever it couldn't find, then save it so future shells reuse it.
echo ""
echo ">>> SLURM account / QOS"
if [ -n "${EBJEPA_SLURM_ACCOUNT:-}" ]; then
    echo "    Account: $EBJEPA_SLURM_ACCOUNT (auto-detected)"
else
    echo -n "    SLURM account not auto-detected — enter it (blank = let SLURM use your default): "
    read -r EBJEPA_SLURM_ACCOUNT
fi
if [ -n "${EBJEPA_SLURM_QOS:-}" ]; then
    echo "    QOS: $EBJEPA_SLURM_QOS (auto-detected)"
else
    echo -n "    SLURM QOS not auto-detected — enter it (blank = let SLURM use your default): "
    read -r EBJEPA_SLURM_QOS
fi
# Persist so every future shell (via env.sh) reuses these without re-querying SLURM.
SLURM_ENV_FILE="${EBJEPA_SLURM_USER_ENV:-$WORK/.eb_jepa_slurm.env}"
{
    echo "# EB-JEPA per-user SLURM settings (written by setup.sh). Delete to re-detect."
    echo "export EBJEPA_SLURM_CONFIGURED=1"
    if [ -n "$EBJEPA_SLURM_ACCOUNT" ]; then echo "export EBJEPA_SLURM_ACCOUNT='$EBJEPA_SLURM_ACCOUNT'"; fi
    if [ -n "$EBJEPA_SLURM_QOS" ]; then echo "export EBJEPA_SLURM_QOS='$EBJEPA_SLURM_QOS'"; fi
} > "$SLURM_ENV_FILE"
echo "    Saved to $SLURM_ENV_FILE (sourced automatically by env.sh)."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Add these lines to your ~/.bashrc for persistent configuration:"
echo ""
echo "  # EB-JEPA"
echo "  source $REPO_ROOT/env.sh"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "To verify: uv run pytest tests/ -v"
echo ""
echo "To disable W&B logging: export WANDB_DISABLED=true  (before sourcing env.sh)"
echo "To re-enable:            export WANDB_DISABLED=false (before sourcing env.sh)"
