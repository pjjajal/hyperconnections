#!/usr/bin/env bash
# Render the full paper figure/table suite from a full_eval.sh results tree.
# CPU-only — run on the login node, or automatically at the end of full_eval.sh.
#
#   bash scripts/generate_plots.sh [RESULTS_ROOT] [PLOTS_OUT]
#
# Each step is non-fatal: a missing input for one figure won't block the others.

set -uo pipefail

# Repo root — the plot scripts `import bench_plot_common` bare and use relative paths.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

PY="${PY:-/scratch/gilbreth/neliopou/venvs/uvhypercon/bin/python}"
RESULTS_ROOT="${1:-${RESULTS_ROOT:-benchmark_reports/full_eval_$(date +%Y_%m_%d)}}"
PLOTS_OUT="${2:-${PLOTS_OUT:-plots/plots_arxiv_final_$(date +%Y_%m_%d)}}"

# Correctness-table rows: expm/expm_force outputs + the four stream_mix rows.
ROWS="expm_fwd ef_E ef_psi ef_grad sm_fwd sm_bwd sm_proj_fwd sm_proj_bwd"

mkdir -p "$PLOTS_OUT"
echo "=== generate_plots: RESULTS_ROOT=$RESULTS_ROOT  PLOTS_OUT=$PLOTS_OUT ==="

run() { echo ""; echo "+ $*"; "$@" || echo "[warn] step failed (continuing): $*"; }

# ── perf figures (latency + speedup bars per kernel/dtype/N) ──────────────────
run "$PY" plots/generate_all.py --singles-dir "$RESULTS_ROOT" --out-dir "$PLOTS_OUT" --per-type

# ── main-paper accuracy story ────────────────────────────────────────────────
run "$PY" plots/accuracy_figure.py       --reports-dir "$RESULTS_ROOT" \
    --out "$PLOTS_OUT/accuracy_figure.png"
run "$PY" plots/accuracy_summary_table.py --reports-dir "$RESULTS_ROOT" --dtype all \
    --out "$PLOTS_OUT/accuracy_summary.tex"
run "$PY" plots/expm_norm_s_numeric_plot.py --report-dir "$RESULTS_ROOT/expm_norm_fp32" \
    -o "$PLOTS_OUT/expm_squaring_offset_error_fp32.png"
run "$PY" plots/expm_norm_s_numeric_plot.py --report-dir "$RESULTS_ROOT/expm_norm_bf16" \
    -o "$PLOTS_OUT/expm_squaring_offset_error_bf16.png"

# ── appendix per-configuration correctness tables (expm/force + stream_mix) ───
run "$PY" plots/expm_correctness_table.py --reports-dir "$RESULTS_ROOT" --dtype fp32 \
    --rows $ROWS --out "$PLOTS_OUT/correctness_fp32.tex"
run "$PY" plots/expm_correctness_table.py --reports-dir "$RESULTS_ROOT" --dtype bf16 \
    --rows $ROWS --out "$PLOTS_OUT/correctness_bf16.tex"

echo ""
echo "=== generate_plots done → $PLOTS_OUT ==="
