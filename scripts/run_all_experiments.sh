#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}

echo "========================================"
echo "Running ALL aTLAS experiments"
echo "Model: $MODEL"
echo "========================================"

./scripts/run_task_negation.sh $MODEL
./scripts/run_task_addition.sh $MODEL
./scripts/run_test_time_adaptation.sh $MODEL
./scripts/run_few_shot.sh $MODEL
./scripts/run_parameter_efficient.sh $MODEL

echo "========================================"
echo "All core experiments complete!"
echo "========================================"
