#!/bin/bash
# Removes duplicate/unused datasets from ~/data/
# EuroSAT excluded per user request.

set -e

DATA_DIR="$HOME/data"

targets=(
    "BREEDS-Benchmarks"
    "Caltech256"
    "country211 (2)"
    "country211.tgz"
    "CUB"
    "fgvc-aircraft"
    "ImageNetV2"
    "Oxford_flowers"
    "VISSL_Food_101"
    "GTSRB"
    "Oxford_Pets"
    "Places365"
    "places_devkit"
    "SVHN"
    "tmp"
)

for target in "${targets[@]}"; do
    path="$DATA_DIR/$target"
    if [ -e "$path" ]; then
        echo "Removing: $path"
        rm -rf "$path"
    else
        echo "Not found (skipping): $path"
    fi
done

echo "Done."
