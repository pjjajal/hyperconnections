#!/usr/bin/env bash
# Identity-HC sweep over hardness levels × model dim × n_streams.
# Skips runs whose metrics.json already exists (checks both new and legacy naming).
# Run from the repo root:
#   bash experiments/synthetic_grid_world/sweep_ihc.sh

set -euo pipefail

DIMS=(16 32 64 128)
DEPTH=12
LEVELS=(h1_easy h2_medium h3_hard h4_veryhard)
N_STREAMS=(4 8 16)
RUNS_DIR=experiments/synthetic_grid_world/runs
CONFIG_DIR=experiments/synthetic_grid_world/configs/sweep

for level in "${LEVELS[@]}"; do
    for dim in "${DIMS[@]}"; do
        for n in "${N_STREAMS[@]}"; do
            run_name="${level}_ihc_n${n}_d${dim}"

            # n=4 may exist under the legacy name {level}_ihc_d{dim}
            if [ "${n}" -eq 4 ] && [ -f "${RUNS_DIR}/${level}_ihc_d${dim}/metrics.json" ]; then
                echo "Skipping (exists, legacy name): ${level}_ihc_d${dim}"
                continue
            fi

            if [ -f "${RUNS_DIR}/${run_name}/metrics.json" ]; then
                echo "Skipping (exists): ${run_name}"
                continue
            fi

            echo "========================================"
            echo "  ${run_name}"
            echo "========================================"
            uv run --extra experiments -- \
                python -m experiments.synthetic_grid_world.train \
                --config "${CONFIG_DIR}/${level}_ihc.yaml" \
                model.dim="${dim}" \
                model.num_layers="${DEPTH}" \
                model.hc.n="${n}" \
                logging.run_name="${run_name}"
        done
    done
done

echo "Sweep complete."
