#!/bin/bash
#SBATCH -A davisjam
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=a100-80gb
#SBATCH --time=4:59:00
#SBATCH --job-name=expm_forced_bench

set -euo pipefail

cd /scratch/gilbreth/neliopou/pj-hyperconnections

BENCH_DIR="benchmarks"
SINGLES_DIR="benchmark_reports/singles"

mkdir -p "benchmark_reports/logs"

###
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=1

# ── Sweep grid ────────────────────────────────────────────────────────────────
EXPM_FORCE_N=(4 8 16 32)
EXPM_FORCE_B=(1024 4096 8192 16384 32768 65536 120064)
EXPM_FORCE_DTYPE=(fp32 bf16)
EXPM_FORCE_PASS=(fwd bwd)

MODE=all
WARMUP=16
REP=128

# ── Helpers ───────────────────────────────────────────────────────────────────
pass_flags() {
    case "$1" in
        fwd)    echo "--fwd" ;;
        bwd)    echo "--bwd" ;;
        fwdbwd) echo "--fwd --bwd" ;;
        *)      echo "--fwd" ;;
    esac
}

total=$(( ${#EXPM_FORCE_N[@]} * ${#EXPM_FORCE_B[@]} * ${#EXPM_FORCE_DTYPE[@]} * ${#EXPM_FORCE_PASS[@]} ))
done_count=0

# ── Environment ───────────────────────────────────────────────────────────────
python -c "import numpy; import torch; import transformers"
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

# ── Sweep ─────────────────────────────────────────────────────────────────────
echo "=== expm_force_bench sweep ($MODE) — $total configs ==="
for n in "${EXPM_FORCE_N[@]}"; do
  for b in "${EXPM_FORCE_B[@]}"; do
    for dtype in "${EXPM_FORCE_DTYPE[@]}"; do
      for pass in "${EXPM_FORCE_PASS[@]}"; do
        done_count=$(( done_count + 1 ))
        tag="n${n}_b${b}_${dtype}_${pass}"
        out_dir="$SINGLES_DIR/expm_force/$tag"
        mkdir -p "$out_dir"
        echo ""
        echo "[$done_count/$total] expm_force  $tag"
        python "$BENCH_DIR/expm_force_bench.py" \
            --mode  "$MODE"  \
            --n     "$n"     \
            --b     "$b"     \
            --dtype "$dtype" \
            --warmup "$WARMUP" --rep "$REP" \
            --out-dir "$out_dir" \
            $(pass_flags "$pass")
      done
    done
  done
done

echo ""
echo "=== expm_force sweep complete: $total configs ==="
echo "Reports in $SINGLES_DIR/expm_force"
