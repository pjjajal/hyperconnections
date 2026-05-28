#!/usr/bin/env bash
# muP hyperparameter transfer sweep
# Two-stage workflow:
#   Stage 1: LR sweep at base_dim to find optimal LR
#   Stage 2: Validate transfer to other widths with optimal LR
# Run from the repo root:
#   bash experiments/synthetic_grid_world/sweep_mup_transfer.sh

set -euo pipefail

BASE_DIM=64
LEVEL=h2_medium
DEPTH=12
EPOCHS=40
LOG_DIR=experiments/synthetic_grid_world/runs_mup
CONFIG_DIR=experiments/synthetic_grid_world/configs/sweep

# Stage 1: LR sweep at base_dim
echo "========================================"
echo "STAGE 1: LR sweep at base_dim=${BASE_DIM}"
echo "========================================"

LRS_STAGE1=(1e-4 3e-4 1e-3 3e-3 1e-2)

for lr in "${LRS_STAGE1[@]}"; do
    run_name="${LEVEL}_mup_base${BASE_DIM}_d${BASE_DIM}_lr${lr}"
    if [ -f "${LOG_DIR}/${run_name}/metrics.json" ]; then
        echo "Skipping (exists): ${run_name}"
        continue
    fi
    echo "Running: ${run_name}"
    uv run --extra experiments -- \
        python -m experiments.synthetic_grid_world.train \
        --config "${CONFIG_DIR}/${LEVEL}.yaml" \
        model.dim="${BASE_DIM}" \
        model.num_layers="${DEPTH}" \
        model.mup.enabled=true \
        model.mup.base_dim="${BASE_DIM}" \
        training.lr="${lr}" \
        training.epochs="${EPOCHS}" \
        logging.log_dir="${LOG_DIR}" \
        logging.run_name="${run_name}"
done

echo ""
echo "========================================"
echo "STAGE 1 COMPLETE - Finding best LR"
echo "========================================"

# Automatically find the best LR from Stage 1 results (highest validation accuracy)
BEST_LR=""
BEST_VAL_ACC=0.0

for lr in "${LRS_STAGE1[@]}"; do
    run_name="${LEVEL}_mup_base${BASE_DIM}_d${BASE_DIM}_lr${lr}"
    metrics_file="${LOG_DIR}/${run_name}/metrics.json"

    if [ -f "$metrics_file" ]; then
        # Extract maximum validation accuracy from metrics
        val_acc=$(python3 -c "
import json
with open('$metrics_file', 'r') as f:
    data = json.load(f)
    val_accs = [m['val_acc'] for m in data if 'val_acc' in m]
    print(max(val_accs) if val_accs else 0.0)
" 2>/dev/null || echo "0.0")

        echo "  LR=${lr}: max_val_acc=${val_acc}"

        # Check if this is the best LR so far
        if (( $(echo "$val_acc > $BEST_VAL_ACC" | bc -l) )); then
            BEST_VAL_ACC=$val_acc
            BEST_LR=$lr
        fi
    else
        echo "  LR=${lr}: metrics not found"
    fi
done

if [ -z "$BEST_LR" ]; then
    echo ""
    echo "ERROR: Could not determine best LR from Stage 1 results."
    echo "Please check ${LOG_DIR} and run Stage 2 manually."
    exit 1
fi

echo ""
echo "Best LR found: ${BEST_LR} (val_acc=${BEST_VAL_ACC})"
echo ""

# Stage 2: Transfer validation

echo "========================================"
echo "STAGE 2: Transfer validation with LR=${BEST_LR}"
echo "========================================"

# Test transfer to different widths
DIMS_STAGE2=(16 32 64 128 256)

for dim in "${DIMS_STAGE2[@]}"; do
    run_name="${LEVEL}_mup_base${BASE_DIM}_d${dim}_lr${BEST_LR}"
    if [ -f "${LOG_DIR}/${run_name}/metrics.json" ]; then
        echo "Skipping (exists): ${run_name}"
        continue
    fi
    echo "Running: ${run_name}"
    uv run --extra experiments -- \
        python -m experiments.synthetic_grid_world.train \
        --config "${CONFIG_DIR}/${LEVEL}.yaml" \
        model.dim="${dim}" \
        model.num_layers="${DEPTH}" \
        model.mup.enabled=true \
        model.mup.base_dim="${BASE_DIM}" \
        training.lr="${BEST_LR}" \
        training.epochs="${EPOCHS}" \
        logging.log_dir="${LOG_DIR}" \
        logging.run_name="${run_name}"
done

echo ""
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"
echo "Results saved to: ${LOG_DIR}"
echo ""
echo "To analyze transfer quality, compare performance across widths:"
echo "  - All widths should achieve similar performance with the same LR"
echo "  - This validates muP hyperparameter transfer"
