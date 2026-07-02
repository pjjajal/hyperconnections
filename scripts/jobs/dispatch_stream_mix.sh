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
#SBATCH --job-name=stream_mix_bench

set -euo pipefail

cd /scratch/gilbreth/neliopou/pj-hyperconnections

BENCH_DIR="benchmarks"
SINGLES_DIR="benchmark_reports/singles"

mkdir -p "benchmark_reports/logs"

# ── Sweep grid ────────────────────────────────────────────────────────────────
STREAM_N=(4 8 16 32)
STREAM_M=(1)
STREAM_B=(2048 4096 8192)
STREAM_E=(768 1024 1536)
STREAM_DTYPE=(bf16)
STREAM_PASS=(bwd)

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

total=$(( ${#STREAM_N[@]} * ${#STREAM_M[@]} * ${#STREAM_B[@]} * ${#STREAM_E[@]} * ${#STREAM_DTYPE[@]} * ${#STREAM_PASS[@]} ))
done_count=0

# ── Environment ───────────────────────────────────────────────────────────────
python -c "import numpy; import torch; import transformers"
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

# ── Sweep ─────────────────────────────────────────────────────────────────────
echo "=== stream_mix_bench sweep ($MODE) — $total configs ==="
for n in "${STREAM_N[@]}"; do
  for m in "${STREAM_M[@]}"; do
    for b in "${STREAM_B[@]}"; do
      for e in "${STREAM_E[@]}"; do
        for dtype in "${STREAM_DTYPE[@]}"; do
          for pass in "${STREAM_PASS[@]}"; do
            done_count=$(( done_count + 1 ))
            tag="n${n}_m${m}_b${b}_e${e}_${dtype}_${pass}"
            out_dir="$SINGLES_DIR/stream_mix/$tag"
            mkdir -p "$out_dir"
            csv_path="$out_dir/stream_mix_perf.csv"
            echo ""
            echo "[$done_count/$total] stream_mix  $tag"
            python "$BENCH_DIR/stream_mix_bench.py" \
                --mode  "$MODE"      \
                --n     "$n"         \
                --m     "$m"         \
                --b     "$b"         \
                --embed-dim "$e"     \
                --dtype "$dtype"     \
                --warmup "$WARMUP" --rep "$REP" \
                --csv   "$csv_path"  \
                $(pass_flags "$pass")
          done
        done
      done
    done
  done
done

echo ""
echo "=== stream_mix sweep complete: $total configs ==="
echo "Reports in $SINGLES_DIR/stream_mix"
