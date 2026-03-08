#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
PARTITION=${2:-10}

echo "Running parameter-efficient fine-tuning (aTLAS x K)"
echo "Model: $MODEL, Partitions: $PARTITION"

for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0; do
    echo "  Training with ${PERC} data percentage..."
    python src/learn_few_shots.py --model=$MODEL --partition $PARTITION --subsample $PERC
done

echo "Parameter-efficient experiments complete!"
