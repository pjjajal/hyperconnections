#!/usr/bin/env bash
# Brief LR sweep for the baseline at each backbone dim.
# Stored in runs_lr/.
# Run from the repo root:
#   bash experiments/synthetic_grid_world/sweep_lr.sh

set -euo pipefail

LRS=(1e-4 3e-4 1e-3 3e-3 1e-2)
DIMS=(16 32 64 128)
DEPTH=12
LEVEL=h2_medium
EPOCHS=40
LOG_DIR=experiments/synthetic_grid_world/runs_lr
CONFIG_DIR=experiments/synthetic_grid_world/configs/sweep

run_dims() {
    for dim in "$@"; do
        for lr in "${LRS[@]}"; do
            run_name="${LEVEL}_d${dim}_lr${lr}"
            if [ -f "${LOG_DIR}/${run_name}/metrics.json" ]; then
                echo "Skipping (exists): ${run_name}"
                continue
            fi
            echo "========================================"
            echo "  ${run_name}"
            echo "========================================"
            uv run --extra experiments -- \
                python -m experiments.synthetic_grid_world.train \
                --config "${CONFIG_DIR}/${LEVEL}.yaml" \
                model.dim="${dim}" \
                model.num_layers="${DEPTH}" \
                training.lr="${lr}" \
                training.epochs="${EPOCHS}" \
                logging.log_dir="${LOG_DIR}" \
                logging.run_name="${run_name}"
        done
    done
}

# Run all dimensions sequentially
run_dims 16 32 64 128

echo "Sweep complete."
