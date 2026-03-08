#!/bin/bash
set -e

MODEL=${1:-ViT-B-32}
ARCH=${2:-medium}
EPOCHS=${3:-50}

echo "Meta-training text-to-coefficient hypernetwork"
echo "Model: $MODEL"
echo "Architecture: $ARCH"
echo "Meta-epochs: $EPOCHS"

python src/learn_text_to_coef.py \
    --model $MODEL \
    --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
    --meta-val-datasets Caltech101,Flowers102 \
    --hypernetwork-arch $ARCH \
    --text-source manual \
    --meta-epochs $EPOCHS \
    --episodes-per-epoch 10 \
    --blockwise-coef

echo "Hypernetwork training complete!"
echo "Checkpoint saved to: checkpoints/$MODEL/hypernetworks/text_to_coef/meta_trained.pt"
