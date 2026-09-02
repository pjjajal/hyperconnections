#!/usr/bin/env bash
set -euo pipefail

for task in preservation rotation permutation filtering; do
    for seed in 4 7 533; do
        uv run python -m experiments.synthetic_dynamics.train \
            "$task" cghc \
            --depth 8 \
            --steps 2500 \
            --seed "$seed" \
            --device mps \
            --lr 5e-3
    done
done
