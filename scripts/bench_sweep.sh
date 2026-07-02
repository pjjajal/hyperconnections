#!/usr/bin/env bash
# Runs one benchmark invocation per config so torch.compile max-autotune
# doesn't corrupt the inductor cache across configurations.
# Reports are written to benchmark_reports/singles/<bench>/<config tag>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/../benchmarks"
SINGLES_DIR="$SCRIPT_DIR/../benchmark_reports/singles"

# ── Sweep grids ──────────────────────────────────────────────────────────────
# Edit these arrays to choose which configs to sweep.

# EXPM_N=(4 8 16 32)
# EXPM_B=(1024 4096 8192 16384 32768 65536 120064)
# EXPM_DTYPE=(fp32 bf16)
# EXPM_PASS=(fwd bwd)   # fwd | bwd | fwdbwd

# EXPM_FORCE_N=(4 8 16 32)
# EXPM_FORCE_B=(1024 4096 8192 16384 32768 65536 120064)
# EXPM_FORCE_DTYPE=(fp32 bf16)
# EXPM_FORCE_PASS=(fwd bwd)

# STREAM_N=(4 8 16)
# STREAM_M=(1)
# STREAM_B=(1024 4096 8192 16384 32768 65536 120064)
# STREAM_E=(1024 1536 2048)
# STREAM_DTYPE=(bf16)
# STREAM_PASS=(fwd)

### Python cache warming
python -c "import numpy; import torch; import transformers"
unset TORCH_LOGS
export TRITON_ALWAYS_COMPILE=0

EXPM_N=(4 8)
EXPM_B=(16384)
EXPM_DTYPE=(fp32)
EXPM_PASS=(fwd bwd)   # fwd | bwd | fwdbwd

EXPM_FORCE_N=(4 8)
EXPM_FORCE_B=(16384)
EXPM_FORCE_DTYPE=(fp32)
EXPM_FORCE_PASS=(fwd bwd)

STREAM_N=(4 8)
STREAM_M=(1)
STREAM_B=(2048)
STREAM_E=(1536)
STREAM_DTYPE=(fp32)
STREAM_PASS=(fwd bwd)

MODE=all # correctness | perf | all
WARMUP=16
REP=128

# ── Helpers ───────────────────────────────────────────────────────────────────
pass_flags() {
    local pass="$1"
    case "$pass" in
        fwd)    echo "--fwd" ;;
        bwd)    echo "--bwd" ;;
        fwdbwd) echo "--fwd --bwd" ;;
        *)      echo "--fwd" ;;
    esac
}

total=0
done_count=0

count_total() {
    local expm=$(( ${#EXPM_N[@]} * ${#EXPM_B[@]} * ${#EXPM_DTYPE[@]} * ${#EXPM_PASS[@]} ))
    local force=$(( ${#EXPM_FORCE_N[@]} * ${#EXPM_FORCE_B[@]} * ${#EXPM_FORCE_DTYPE[@]} * ${#EXPM_FORCE_PASS[@]} ))
    local stream=$(( ${#STREAM_N[@]} * ${#STREAM_M[@]} * ${#STREAM_B[@]} * ${#STREAM_E[@]} * ${#STREAM_DTYPE[@]} * ${#STREAM_PASS[@]} ))
    total=$(( expm + force + stream ))
}

run_one() {
    done_count=$(( done_count + 1 ))
    echo ""
    echo "[$done_count/$total] $*"
}

count_total

# ── expm_bench ───────────────────────────────────────────────────────────────
echo "=== expm_bench sweep ($MODE) ==="
for n in "${EXPM_N[@]}"; do
  for b in "${EXPM_B[@]}"; do
    for dtype in "${EXPM_DTYPE[@]}"; do
      for pass in "${EXPM_PASS[@]}"; do
        tag="n${n}_b${b}_${dtype}_${pass}"
        out_dir="$SINGLES_DIR/expm/$tag"
        mkdir -p "$out_dir"
        run_one "expm  $tag"
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

# ── expm_force_bench ─────────────────────────────────────────────────────────
echo ""
echo "=== expm_force_bench sweep ($MODE) ==="
for n in "${EXPM_FORCE_N[@]}"; do
  for b in "${EXPM_FORCE_B[@]}"; do
    for dtype in "${EXPM_FORCE_DTYPE[@]}"; do
      for pass in "${EXPM_FORCE_PASS[@]}"; do
        tag="n${n}_b${b}_${dtype}_${pass}"
        out_dir="$SINGLES_DIR/expm_force/$tag"
        mkdir -p "$out_dir"
        run_one "expm_force  $tag"
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

# ── stream_mix_bench ─────────────────────────────────────────────────────────
echo ""
echo "=== stream_mix_bench sweep ($MODE) ==="
for n in "${STREAM_N[@]}"; do
  for m in "${STREAM_M[@]}"; do
    for b in "${STREAM_B[@]}"; do
      for e in "${STREAM_E[@]}"; do
        for dtype in "${STREAM_DTYPE[@]}"; do
          for pass in "${STREAM_PASS[@]}"; do
            tag="n${n}_m${m}_b${b}_e${e}_${dtype}_${pass}"
            out_dir="$SINGLES_DIR/stream_mix/$tag"
            mkdir -p "$out_dir"
            csv_path="$out_dir/stream_mix_perf.csv"
            run_one "stream_mix  $tag"
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
echo "=== sweep complete: $total configs ==="
echo "Reports in $SINGLES_DIR"
