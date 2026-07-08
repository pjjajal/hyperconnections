#!/bin/bash
#SBATCH -A davisjam
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=a100-80gb
#SBATCH --time=24:00:00
#SBATCH --job-name=full_eval
#
# Big paper sweep for the three Triton kernels (expm, expm_force, stream_mix),
# matching the two experimental-setup tables, plus the correctness + expm_norm
# data feeding the accuracy table/plots. Launch manually:  sbatch scripts/jobs/full_eval.sh
#
# The a100-80gb partition is uncapped (MaxTime=UNLIMITED); one long job is fine.
# The sweep is resumable (skips configs whose output CSV already exists) and
# OOM-tolerant (a config that OOMs/errors is logged and skipped, never aborts).
#
# Phases run PERF (the big grid) then CORRECTNESS then NORM then PLOT. Override
# via env, e.g.:
#     PHASES="perf" KERNELS="stream_mix" sbatch scripts/jobs/full_eval.sh
#     SMOKE=1 bash scripts/jobs/full_eval.sh        # fast wiring test on an interactive node

# NOT `set -e`: a single OOM / correctness FAIL must be reported and skipped,
# never abort the multi-hour sweep (cf. scripts/jobs/dispatch_research.sh).
set -uo pipefail

cd /scratch/gilbreth/neliopou/pj-hyperconnections || exit 1

# Absolute venv python so the job runs unattended regardless of the submitting
# shell's PATH (the stock dispatchers assume an already-activated venv).
PY="${PY:-/scratch/gilbreth/neliopou/venvs/uvhypercon/bin/python}"
BENCH_DIR="benchmarks"

# Dated, isolated result root — keeps this run's CSVs separate from the stale
# benchmark_reports/singles/ files (old check-names) so the tables see clean data.
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_reports/full_eval_$(date +%Y_%m_%d)}"
PLOTS_OUT="${PLOTS_OUT:-plots/plots_arxiv_final_$(date +%Y_%m_%d)}"

PHASES="${PHASES:-perf correctness norm plot}"
KERNELS="${KERNELS:-expm expm_force stream_mix}"
SMOKE="${SMOKE:-0}"

WARMUP=16
REP=128

# ── Sweep grids (from the LaTeX experimental-setup tables) ────────────────────
B_LIST=(256 512 1024 2048 4096 8192 16384 32768 65536 131072)   # 2^8 .. 2^17
N_LIST=(4 8 16 32)
DTYPE_LIST=(fp32 bf16)
E_LIST=(1024 1536)          # stream_mix embed-dim (m=1 => D = E)
PASS_LIST=(fwd bwd)         # each pass is a SEPARATE python invocation (caps peak mem)

# Accuracy is ~batch-independent, so the correctness / norm sub-grids stay small.
CORR_B=4096                 # representative batch for the expm/expm_force correctness pass
CORR_SM_B=2048              # representative batch for the stream_mix correctness pass
NORM_N=(4 8 16)
NORM_B=8192
NORM_NORMS=(0.1 1.0 4.0 10.0 20.0)
NORM_SQUARINGS=(0 1 2 4)

# SMOKE=1: shrink every grid to a couple of points for a fast end-to-end check.
if [[ "$SMOKE" == "1" ]]; then
    B_LIST=(4096); N_LIST=(4 8); DTYPE_LIST=(fp32); E_LIST=(1024); PASS_LIST=(fwd bwd)
    CORR_B=4096; CORR_SM_B=2048
    NORM_N=(4); NORM_B=4096; NORM_NORMS=(1.0 10.0); NORM_SQUARINGS=(0 1 2)
    RESULTS_ROOT="${RESULTS_ROOT}_smoke"
    PLOTS_OUT="${PLOTS_OUT}_smoke"
fi

mkdir -p benchmark_reports/logs "$RESULTS_ROOT"

# ── Environment ──────────────────────────────────────────────────────────────
"$PY" -c "import numpy; import torch; import transformers" \
    || { echo "FATAL: venv import failed ($PY)"; exit 1; }
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

# ── Helpers ──────────────────────────────────────────────────────────────────
want_kernel() { [[ " $KERNELS " == *" $1 "* ]]; }
want_phase()  { [[ " $PHASES "  == *" $1 "* ]]; }

# True if a completed output CSV already exists under <dir> matching <glob>.
have_output() { compgen -G "$1/$2" > /dev/null 2>&1; }

done_count=0; skip_count=0; fail_count=0

# run_cfg <label> <output-dir> <output-glob> <cmd...>
# Resumable (skip if output exists) + OOM-tolerant (log + continue on non-zero).
run_cfg() {
    local label="$1" odir="$2" oglob="$3"; shift 3
    if have_output "$odir" "$oglob"; then
        skip_count=$(( skip_count + 1 ))
        echo "[skip] $label"
        return 0
    fi
    echo ""
    echo "[run] $label"
    "$@"
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        done_count=$(( done_count + 1 ))
    else
        fail_count=$(( fail_count + 1 ))
        echo "[FAIL] $label (exit $rc)"
    fi
    return 0
}

# ── Phase P: performance sweep (the big grid) ────────────────────────────────
sweep_expm_perf() {   # $1 = bench script, $2 = kernel subdir
    local bench="$1" sub="$2"
    echo ""; echo "=== PERF: $sub — B×N×dtype×pass ==="
    for n in "${N_LIST[@]}"; do
      for b in "${B_LIST[@]}"; do
        for dtype in "${DTYPE_LIST[@]}"; do
          for pass in "${PASS_LIST[@]}"; do
            local tag="n${n}_b${b}_${dtype}_${pass}"
            local odir="$RESULTS_ROOT/$sub/$tag"; mkdir -p "$odir"
            run_cfg "$sub perf $tag" "$odir" "*perf*.CSV" \
                "$PY" "$BENCH_DIR/$bench" --mode perf \
                    --n "$n" --b "$b" --dtype "$dtype" \
                    --warmup "$WARMUP" --rep "$REP" "--$pass" --out-dir "$odir"
          done
        done
      done
    done
}

sweep_stream_mix_perf() {
    echo ""; echo "=== PERF: stream_mix — B×N×dtype×E×pass ==="
    for n in "${N_LIST[@]}"; do
      for b in "${B_LIST[@]}"; do
        for e in "${E_LIST[@]}"; do
          for dtype in "${DTYPE_LIST[@]}"; do
            for pass in "${PASS_LIST[@]}"; do
              local tag="n${n}_m1_b${b}_e${e}_${dtype}_${pass}"
              local odir="$RESULTS_ROOT/stream_mix/$tag"; mkdir -p "$odir"
              run_cfg "stream_mix perf $tag" "$odir" "stream_mix_perf.csv" \
                  "$PY" "$BENCH_DIR/stream_mix_bench.py" --mode perf \
                      --n "$n" --m 1 --b "$b" --embed-dim "$e" --dtype "$dtype" \
                      --warmup "$WARMUP" --rep "$REP" "--$pass" \
                      --csv "$odir/stream_mix_perf.csv"
            done
          done
        done
      done
    done
}

# ── Phase C: correctness (one invocation per kernel; loops N+dtype internally) ─
corr_expm() {   # $1 = bench script, $2 = kernel subdir
    local bench="$1" sub="$2"
    local odir="$RESULTS_ROOT/$sub/correctness"; mkdir -p "$odir"
    run_cfg "$sub correctness" "$odir" "*correctness*.CSV" \
        "$PY" "$BENCH_DIR/$bench" --mode correctness \
            --n "${N_LIST[@]}" --b "$CORR_B" --dtype "${DTYPE_LIST[@]}" --out-dir "$odir"
}

corr_stream_mix() {
    local odir="$RESULTS_ROOT/stream_mix/correctness"; mkdir -p "$odir"
    run_cfg "stream_mix correctness" "$odir" "stream_mix_correctness.csv" \
        "$PY" "$BENCH_DIR/stream_mix_bench.py" --mode correctness \
            --n "${N_LIST[@]}" --m 1 --b "$CORR_SM_B" --embed-dim "${E_LIST[@]}" \
            --dtype "${DTYPE_LIST[@]}" --csv "$odir/stream_mix_perf.csv"
}

# ── Phase S: expm_norm data for the S-ablation plot (one run per dtype) ───────
sweep_norm() {
    echo ""; echo "=== NORM: expm_norm S-ablation ==="
    for dtype in "${DTYPE_LIST[@]}"; do
        local odir="$RESULTS_ROOT/expm_norm_$dtype"; mkdir -p "$odir"
        run_cfg "expm_norm $dtype" "$odir" "*norm_correctness*.CSV" \
            "$PY" "$BENCH_DIR/expm_norm_bench.py" --mode correctness \
                --n "${NORM_N[@]}" --b "$NORM_B" --dtype "$dtype" \
                --norms "${NORM_NORMS[@]}" --squarings "${NORM_SQUARINGS[@]}" \
                --out-dir "$odir"
    done
}

# ── Driver ───────────────────────────────────────────────────────────────────
echo "=== full_eval ==="
echo "RESULTS_ROOT=$RESULTS_ROOT  PLOTS_OUT=$PLOTS_OUT"
echo "PHASES=[$PHASES]  KERNELS=[$KERNELS]  SMOKE=$SMOKE"
echo "grid: B=${#B_LIST[@]} N=${#N_LIST[@]} dtype=${#DTYPE_LIST[@]} E=${#E_LIST[@]} pass=${#PASS_LIST[@]}"

if want_phase perf; then
    want_kernel expm       && sweep_expm_perf expm_bench.py       expm
    want_kernel expm_force && sweep_expm_perf expm_force_bench.py expm_force
    want_kernel stream_mix && sweep_stream_mix_perf
fi

if want_phase correctness; then
    echo ""; echo "=== CORRECTNESS ==="
    want_kernel expm       && corr_expm expm_bench.py       expm
    want_kernel expm_force && corr_expm expm_force_bench.py expm_force
    want_kernel stream_mix && corr_stream_mix
fi

if want_phase norm; then
    sweep_norm
fi

if want_phase plot; then
    echo ""; echo "=== PLOT ==="
    bash scripts/generate_plots.sh "$RESULTS_ROOT" "$PLOTS_OUT" \
        || echo "[warn] plotting failed (non-fatal); re-run scripts/generate_plots.sh manually"
fi

echo ""
echo "=== full_eval done: $done_count ran, $skip_count skipped, $fail_count failed ==="
echo "Results under $RESULTS_ROOT ; figures/tables under $PLOTS_OUT"
