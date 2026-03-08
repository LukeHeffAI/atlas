#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
echo "Running test-time adaptation (UFM) with model: $MODEL"
python src/learn_ufm.py --model=$MODEL --blockwise-coef
echo "Test-time adaptation complete!"
