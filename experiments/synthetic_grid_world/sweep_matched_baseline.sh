#!/usr/bin/env bash
# Baseline sweep with depth matched to IHC parameter counts.
# Keeps dim fixed so per-layer thinking capacity is identical;
# only depth is increased to match params.
# Skips runs whose metrics.json already exists.
# Run from the repo root:
#   bash experiments/synthetic_grid_world/sweep_matched_baseline.sh

set -euo pipefail

DIMS=(16 32 64 128)
DEPTH=12
LEVELS=(h1_easy h2_medium h3_hard h4_veryhard)
N_STREAMS=(4)
RUNS_DIR=experiments/synthetic_grid_world/runs
CONFIG_DIR=experiments/synthetic_grid_world/configs/sweep

for level in "${LEVELS[@]}"; do
    for dim in "${DIMS[@]}"; do
        for n in "${N_STREAMS[@]}"; do
            # Skip if any completed matched run exists for this (level, n, dim)
            matched_exists=false
            for d in "${RUNS_DIR}/${level}_baseline_n${n}eq_d${dim}_L"*/; do
                if [ -f "${d}metrics.json" ]; then
                    matched_exists=true
                    echo "Skipping (exists): $(basename "${d%/}")"
                    break
                fi
            done
            [ "${matched_exists}" = true ] && continue

            matched_depth=$(uv run --extra experiments -- \
                python -m experiments.synthetic_grid_world.find_matched_dim \
                --config "${CONFIG_DIR}/${level}_ihc.yaml" \
                --n "${n}" --dim "${dim}" --num-layers "${DEPTH}")

            run_name="${level}_baseline_n${n}eq_d${dim}_L${matched_depth}"
            echo "========================================"
            echo "  ${run_name}"
            echo "  (matched to IHC n=${n} d=${dim} L=${DEPTH})"
            echo "========================================"
            uv run --extra experiments -- \
                python -m experiments.synthetic_grid_world.train \
                --config "${CONFIG_DIR}/${level}.yaml" \
                model.dim="${dim}" \
                model.num_layers="${matched_depth}" \
                logging.run_name="${run_name}"
        done
    done
done

echo "Sweep complete."
