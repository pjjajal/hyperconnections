#!/bin/bash
#SBATCH -A davisjam
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=a100-80gb
#SBATCH --time=3:59:00
#SBATCH --job-name=kernel_research

# NOTE: intentionally NOT `set -e` — a correctness FAIL or a broken candidate must
# be reported to the forum, never abort the loop.
set -uo pipefail

cd /scratch/gilbreth/neliopou/pj-hyperconnections

# ── Args ──────────────────────────────────────────────────────────────────────
# $1 target  : expm_force | stream_mix    (default expm_force)
# $2 pass    : fwd | bwd                   (default bwd)
# $3 iters   : how many evaluate-and-post iterations (default 1)
TARGET="${1:-expm_force}"
PASS="${2:-bwd}"
ITERS="${3:-1}"

# ── Environment (mirrors scripts/jobs/dispatch_*.sh) ──────────────────────────
python -c "import numpy; import torch; import transformers"
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

echo "=== kernel_research loop: target=$TARGET pass=$PASS iters=$ITERS ==="

# Each iteration evaluates whatever is currently in the sandbox and posts a row.
# For the LLM-driven loop, an agent edits sandbox/<...>_candidate.py between
# iterations (e.g. run this with ITERS=1 after each edit, or salloc this node and
# drive the agent interactively). A bare loop re-measures the same candidate,
# which is still useful as a variance check.
for i in $(seq 1 "$ITERS"); do
    echo ""
    echo "[$i/$ITERS] evaluate + post  ($TARGET $PASS)"
    python -m hyperconnections.kernel_research.auto_research --target "$TARGET" --pass "$PASS"
done

echo ""
echo "=== kernel_research loop complete ==="
echo "Inspect results:  python pj-forum/forum-skill/scripts/forum recent --forum $TARGET"
