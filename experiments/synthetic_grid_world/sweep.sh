#!/usr/bin/env bash
# Baseline (standard transformer) sweep over hardness levels × model dim.
# Run from the repo root:
#   bash experiments/synthetic_grid_world/sweep.sh

set -euo pipefail

DIMS=(16 32 64 128)
DEPTH=12
LEVELS=(h1_easy h2_medium h3_hard h4_veryhard)
CONFIG_DIR=experiments/synthetic_grid_world/configs/sweep

for level in "${LEVELS[@]}"; do
    for dim in "${DIMS[@]}"; do
        run_name="${level}_d${dim}"
        echo "========================================"
        echo "  ${run_name}"
        echo "========================================"
        uv run --extra experiments python -m experiments.synthetic_grid_world.train \
            --config "${CONFIG_DIR}/${level}.yaml" \
            model.dim="${dim}" \
            model.num_layers="${DEPTH}" \
            logging.run_name="${run_name}"
    done
done

echo "Sweep complete."
