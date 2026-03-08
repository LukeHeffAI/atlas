#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
echo "Running task addition experiment with model: $MODEL"
python src/learn_task_addition.py --model=$MODEL --blockwise-coef
echo "Results saved to checkpoints/$MODEL/learned_additions.json"
