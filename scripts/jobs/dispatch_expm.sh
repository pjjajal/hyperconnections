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
#SBATCH --job-name=expm_bench

set -euo pipefail

cd /scratch/gilbreth/neliopou/pj-hyperconnections

BENCH_DIR="benchmarks"
SINGLES_DIR="benchmark_reports/singles"

mkdir -p "benchmark_reports/logs"

# ── Sweep grid ────────────────────────────────────────────────────────────────
EXPM_N=(4 8 16 32)
EXPM_B=(1024 4096 8192 16384 32768 65536 120064)
EXPM_DTYPE=(fp32 bf16)
EXPM_PASS=(fwd bwd)

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

total=$(( ${#EXPM_N[@]} * ${#EXPM_B[@]} * ${#EXPM_DTYPE[@]} * ${#EXPM_PASS[@]} ))
done_count=0

# ── Environment ───────────────────────────────────────────────────────────────
python -c "import numpy; import torch; import transformers"
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

# ── Sweep ─────────────────────────────────────────────────────────────────────
echo "=== expm_bench sweep ($MODE) — $total configs ==="
for n in "${EXPM_N[@]}"; do
  for b in "${EXPM_B[@]}"; do
    for dtype in "${EXPM_DTYPE[@]}"; do
      for pass in "${EXPM_PASS[@]}"; do
        done_count=$(( done_count + 1 ))
        tag="n${n}_b${b}_${dtype}_${pass}"
        out_dir="$SINGLES_DIR/expm/$tag"
        mkdir -p "$out_dir"
        echo ""
        echo "[$done_count/$total] expm  $tag"
        python "$BENCH_DIR/expm_bench.py" \
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
echo "=== expm sweep complete: $total configs ==="
echo "Reports in $SINGLES_DIR/expm"
