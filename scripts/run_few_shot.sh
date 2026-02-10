#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
echo "Running few-shot adaptation experiments with model: $MODEL"

# Basic aTLAS for different shot settings
echo "Running basic aTLAS experiments..."
for SHOT in 1 2 4 8 16; do
    echo "  Training with $SHOT shots..."
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT
done

# aTLAS with adapters (LP++ and Tip)
echo "Running aTLAS with adapters..."
for SHOT in 1 2 4 8 16; do
    echo "  Training with $SHOT shots + Tip adapter..."
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter tip
    echo "  Training with $SHOT shots + LP++ adapter..."
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter lpp
done

echo "Few-shot experiments complete!"
