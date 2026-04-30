#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL=${1:-ViT-B-32}
BACKEND=${2:-clip}
CKPT_ROOT=${3:-checkpoints_${BACKEND}}

COMMON=(--model="$MODEL" --clip-backend="$BACKEND" --checkpoint-root="$CKPT_ROOT")

echo "Running few-shot adaptation: model=$MODEL backend=$BACKEND root=$CKPT_ROOT"

echo "Running basic aTLAS experiments..."
for SHOT in 1 2 4 8 16; do
    echo "  Training with $SHOT shots..."
    python src/learn_few_shots.py "${COMMON[@]}" --blockwise-coef --subsample $SHOT
done

echo "Running aTLAS with adapters..."
for SHOT in 1 2 4 8 16; do
    echo "  Training with $SHOT shots + Tip adapter..."
    python src/learn_few_shots.py "${COMMON[@]}" --blockwise-coef --subsample $SHOT --adapter tip
    echo "  Training with $SHOT shots + LP++ adapter..."
    python src/learn_few_shots.py "${COMMON[@]}" --blockwise-coef --subsample $SHOT --adapter lpp
done

echo "Few-shot experiments complete!"
