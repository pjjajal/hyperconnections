#!/usr/bin/env bash
# Render the full paper figure/table suite.
#
# The three kernels are dispatched as SEPARATE jobs, so their reports live in
# separate roots (benchmark_reports/<kernel>_arxiv_final_*). Each root already
# contains that kernel's own subdir (expm/ | expm_force/ | stream_mix/), exactly
# the layout generate_all.py expects — so we invoke generate_all once per kernel
# and hand the accuracy/correctness scripts every root at once.
#
#   bash scripts/generate_plots.sh [PLOTS_OUT]
#
# Any input root can be overridden via env. To plot a single full_eval tree
# (which holds all three kernels under one root) the roots simply collapse:
#   EXPM_DIR=$R FORCE_DIR=$R STREAM_DIR=$R \
#   NORM_FP32_DIR=$R/expm_norm_fp32 NORM_BF16_DIR=$R/expm_norm_bf16 \
#   bash scripts/generate_plots.sh out/
#
# Each step is non-fatal: a missing input for one figure won't block the others.

set -uo pipefail

# Repo root — the plot scripts `import bench_plot_common` bare and use relative paths.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

# Prefer the active venv's python; fall back to the uvhypercon interpreter.
PY="${PY:-$(command -v python || echo /scratch/gilbreth/neliopou/venvs/uvhypercon/bin/python)}"

BR="benchmark_reports"
EXPM_DIR="${EXPM_DIR:-$BR/expm_arxiv_final_7_8_2026}"
FORCE_DIR="${FORCE_DIR:-$BR/expm_forced_arxiv_final_7_8_2026}"
STREAM_DIR="${STREAM_DIR:-$BR/stream_mix_arxiv_final_7_8_2026}"
NORM_FP32_DIR="${NORM_FP32_DIR:-$BR/expm_norm_arxiv_final_fp32_7_7_2026}"
NORM_BF16_DIR="${NORM_BF16_DIR:-$BR/expm_norm_arxiv_final_bf16_7_7_2026}"

PLOTS_OUT="${1:-${PLOTS_OUT:-plots/plots_arxiv_final_7_8_2026}}"

# Correctness-table rows: expm/expm_force outputs + the four stream_mix rows.
ROWS="expm_fwd ef_E ef_psi ef_grad sm_fwd sm_bwd sm_proj_fwd sm_proj_bwd"

# De-duplicate the roots — they collapse to one when plotting a full_eval tree.
# Each family's glob is name-specific, so handing every root to every script is safe.
ALL_DIRS=$(printf '%s\n' "$EXPM_DIR" "$FORCE_DIR" "$STREAM_DIR" | awk '!seen[$0]++')
MATEXP_DIRS=$(printf '%s\n' "$EXPM_DIR" "$FORCE_DIR" | awk '!seen[$0]++')

mkdir -p "$PLOTS_OUT"
echo "=== generate_plots → $PLOTS_OUT ==="
echo "  expm       : $EXPM_DIR"
echo "  expm_force : $FORCE_DIR"
echo "  stream_mix : $STREAM_DIR"
echo "  norm       : $NORM_FP32_DIR | $NORM_BF16_DIR"

run() { echo ""; echo "+ $*"; "$@" || echo "[warn] step failed (continuing): $*"; }

# ── perf figures — one generate_all pass per kernel root; all write one tree ──
run "$PY" plots/generate_all.py --singles-dir "$EXPM_DIR"   --kernels expm       --out-dir "$PLOTS_OUT" --per-type
run "$PY" plots/generate_all.py --singles-dir "$FORCE_DIR"  --kernels expm_force --out-dir "$PLOTS_OUT" --per-type
run "$PY" plots/generate_all.py --singles-dir "$STREAM_DIR" --kernels stream_mix --out-dir "$PLOTS_OUT" --per-type

# ── main-paper accuracy story (matrix-exponential kernels only) ──────────────
run "$PY" plots/accuracy_figure.py --reports-dir $MATEXP_DIRS \
    --out "$PLOTS_OUT/accuracy_figure.png"
run "$PY" plots/accuracy_summary_table.py --reports-dir $MATEXP_DIRS --dtype all \
    --out "$PLOTS_OUT/accuracy_summary.tex"
run "$PY" plots/expm_norm_s_numeric_plot.py --report-dir "$NORM_FP32_DIR" \
    -o "$PLOTS_OUT/expm_squaring_offset_error_fp32.png"
run "$PY" plots/expm_norm_s_numeric_plot.py --report-dir "$NORM_BF16_DIR" \
    -o "$PLOTS_OUT/expm_squaring_offset_error_bf16.png"

# ── appendix per-configuration correctness tables (all three kernels) ────────
run "$PY" plots/expm_correctness_table.py --reports-dir $ALL_DIRS --dtype fp32 \
    --rows $ROWS --out "$PLOTS_OUT/correctness_fp32.tex"
run "$PY" plots/expm_correctness_table.py --reports-dir $ALL_DIRS --dtype bf16 \
    --rows $ROWS --out "$PLOTS_OUT/correctness_bf16.tex"

echo ""
echo "=== generate_plots done → $PLOTS_OUT ==="
