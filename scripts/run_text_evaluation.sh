#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
DATASET=${2:-Cars}
APPROACH=${3:-hypernetwork}

echo "Evaluating text-based adaptation"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Approach: $APPROACH"

CHECKPOINT="checkpoints/$MODEL/hypernetworks/text_to_coef/meta_trained.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Hypernetwork checkpoint not found at $CHECKPOINT"
    echo "Please run scripts/run_hypernetwork_training.sh first"
    exit 1
fi

python src/eval_text_adaptation.py \
    --model $MODEL \
    --dataset $DATASET \
    --approach $APPROACH \
    --hypernetwork-checkpoint $CHECKPOINT \
    --text-source manual

echo "Evaluation complete!"
