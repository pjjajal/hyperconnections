#!/bin/bash
# Sweep over model types and depths for signal rotation task (orthogonal)

set -e  # Exit on error

# Configuration
TASK="rotation"
TRANSFORM_TYPE="rotation"
N_SAMPLES=10000
N_STREAMS=4
D=32
EPOCHS=100
BATCH_SIZE=64
DEVICE="cuda"
SEED=42

# Model types to test
MODELS=("cghc" "mhc" "ghc" "identity_hc")

# Depths to test
DEPTHS=(1 4 8 16 32 64)

echo "Starting rotation task sweep (orthogonal)..."
echo "Models: ${MODELS[@]}"
echo "Depths: ${DEPTHS[@]}"
echo "Samples: $N_SAMPLES"
echo "Transform: $TRANSFORM_TYPE (random orthogonal matrix)"
echo ""

# Run experiments
for model in "${MODELS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        run_name="rotation_orthogonal_${model}_depth${depth}"
        echo "========================================="
        echo "Running: $run_name"
        echo "========================================="

        uv run python -m experiments.synthetic.train \
            --task $TASK \
            --transform-type $TRANSFORM_TYPE \
            --model $model \
            --n-samples $N_SAMPLES \
            --n-streams $N_STREAMS \
            --d $D \
            --n-layers $depth \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --device $DEVICE \
            --seed $SEED \
            --run-name $run_name

        echo ""
        echo "Completed: $run_name"
        echo ""
    done
done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
echo "Results saved to: experiments/synthetic/runs/"
