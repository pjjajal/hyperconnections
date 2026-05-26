#!/bin/bash
# Sweep over model types and depths for signal filtering task

set -e  # Exit on error

# Configuration
TASK="filtering"
N_SAMPLES=10000
N_STREAMS=4
D=64
EPOCHS=100
BATCH_SIZE=64
DEVICE="cuda"
SEED=42

# Filtering-specific parameters
N_SIGNAL_BASIS=1
N_SIGNAL_MEMORIES=1
N_NOISE_BASIS=2
N_NOISE_MEMORIES=1
NOISE_SCALE=3.0

# Model types to test
MODELS=("cghc" "mhc" "ghc" "identity_hc")

# Depths to test
DEPTHS=(4 8 16 32 64)

echo "Starting filtering task sweep..."
echo "Models: ${MODELS[@]}"
echo "Depths: ${DEPTHS[@]}"
echo "Samples: $N_SAMPLES"
echo "Signal basis: $N_SIGNAL_BASIS, Signal memories: $N_SIGNAL_MEMORIES"
echo "Noise basis: $N_NOISE_BASIS, Noise memories: $N_NOISE_MEMORIES"
echo "Noise scale: $NOISE_SCALE"
echo ""

# Run experiments
for model in "${MODELS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        run_name="filtering_${model}_depth${depth}"
        echo "========================================="
        echo "Running: $run_name"
        echo "========================================="

        uv run python -m experiments.synthetic.train \
            --task $TASK \
            --model $model \
            --n-samples $N_SAMPLES \
            --n-streams $N_STREAMS \
            --d $D \
            --n-layers $depth \
            --n-signal-basis $N_SIGNAL_BASIS \
            --n-signal-memories $N_SIGNAL_MEMORIES \
            --n-noise-basis $N_NOISE_BASIS \
            --n-noise-memories $N_NOISE_MEMORIES \
            --noise-scale $NOISE_SCALE \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --device $DEVICE \
            --seed $SEED \
            --run-name $run_name \
            --dt 0.01 \
            --projection mean \
            --generator-type conservative_diag_diss \
            --lr 1e-3
        echo ""
        echo "Completed: $run_name"
        echo ""
    done
done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
echo "Results saved to: experiments/synthetic/runs/"
