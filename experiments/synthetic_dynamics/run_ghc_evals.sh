#!/usr/bin/env bash
set -euo pipefail

for task in preservation rotation permutation filtering; do
    for seed in 4 7 533; do
        uv run python -m experiments.synthetic_dynamics.train \
            "$task" ghc \
            --depth 8 \
            --steps 2500 \
            --seed "$seed" \
            --device mps
    done
done
