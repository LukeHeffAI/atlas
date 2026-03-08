#!/bin/bash
set -e

echo "Generating few-shot adaptation experiments"

python src/generate_synthetic_data.py \
    --datasets CUB200 \
    --text-source manual \
    --t2i-backend stable_diffusion \
    --t2i-model stabilityai/stable-diffusion-xl-base-1.0 \
    --num-images-per-class 4 \
    --output-dir data/synthetic_images \
    --force-regenerate
echo "Synthetic data generation for CUB images complete!"

MODEL=${1:-ViT-B-32}
echo "Running few-shot adaptation experiments with model: $MODEL"


echo "Running basic aTLAS experiments with synthetic data..."
python src/learn_few_shots.py \
    --model=$MODEL \
    --save checkpoints \
    --task-vector-source synthetic \
    --synthetic-data-location data/synthetic_images \
    --t2i-backend stable_diffusion \
    --blockwise-coef \
    --subsample 1,2,4


echo "Running basic aTLAS experiments with mixed real and synthetic data..."
python src/learn_few_shots.py \
    --model=$MODEL \
    --save checkpoints \
    --task-vector-source mixed \
    --synthetic-data-location data/synthetic_images \
    --t2i-backend stable_diffusion \
    --blockwise-coef \
    --subsample 1,2,4


python scripts/analyze_synthetic_benchmark.py