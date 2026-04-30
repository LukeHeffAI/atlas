#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL=${1:-ViT-B-32}
BACKEND=${2:-clip}
CKPT_ROOT=${3:-checkpoints_${BACKEND}}

echo "Running task addition: model=$MODEL backend=$BACKEND root=$CKPT_ROOT"
python src/learn_task_addition.py \
    --model="$MODEL" \
    --clip-backend="$BACKEND" \
    --checkpoint-root="$CKPT_ROOT" \
    --blockwise-coef
echo "Results saved to $CKPT_ROOT/$MODEL/learned_additions.json"
