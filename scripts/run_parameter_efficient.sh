#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL=${1:-ViT-B-32}
BACKEND=${2:-clip}
CKPT_ROOT=${3:-checkpoints_${BACKEND}}
PARTITION=${4:-10}

COMMON=(--model="$MODEL" --clip-backend="$BACKEND" --checkpoint-root="$CKPT_ROOT")

echo "Running parameter-efficient fine-tuning (aTLAS x K): model=$MODEL backend=$BACKEND root=$CKPT_ROOT partitions=$PARTITION"

for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0; do
    echo "  Training with ${PERC} data percentage..."
    python src/learn_few_shots.py "${COMMON[@]}" --partition "$PARTITION" --subsample "$PERC"
done

echo "Parameter-efficient experiments complete!"
