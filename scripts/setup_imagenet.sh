#!/usr/bin/env bash
# =============================================================================
# setup_imagenet.sh
#
# Extracts and organizes ImageNet ILSVRC2012 archives into the directory
# structure expected by src/datasets/imagenet.py:
#
#   ~/data/imagenet/
#       train/<synset_id>/*.JPEG   (1000 class dirs, ~1.28M images)
#       val/<synset_id>/*.JPEG     (1000 class dirs, 50K images)
#
# Prerequisites:
#   Download from https://image-net.org/ (requires account):
#     - ILSVRC2012_img_train.tar  (~138 GB, classification training images)
#     - ILSVRC2012_img_val.tar    (~6.3 GB, validation images)
#
# Usage:
#   ./scripts/setup_imagenet.sh [DOWNLOAD_DIR] [DATA_LOCATION]
#
#   DOWNLOAD_DIR   Where the .tar files are (default: ~/Downloads)
#   DATA_LOCATION  Root data directory (default: ~/data)
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${1:-$HOME/Downloads}"
DATA_LOCATION="${2:-$HOME/data}"
IMAGENET_DIR="${DATA_LOCATION}/imagenet"
LOG_FILE="${IMAGENET_DIR}/setup.log"

TRAIN_TAR="${DOWNLOAD_DIR}/ILSVRC2012_img_train.tar"
VAL_TAR="${DOWNLOAD_DIR}/ILSVRC2012_img_val.tar"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ---------------------------------------------------------------------------
# 1. Preflight checks
# ---------------------------------------------------------------------------
if [ ! -f "$TRAIN_TAR" ]; then
    echo "Error: Training archive not found at $TRAIN_TAR"
    exit 1
fi
if [ ! -f "$VAL_TAR" ]; then
    echo "Error: Validation archive not found at $VAL_TAR"
    exit 1
fi

mkdir -p "$IMAGENET_DIR"
log "Starting ImageNet setup"
log "  Download dir: $DOWNLOAD_DIR"
log "  Target dir:   $IMAGENET_DIR"

# ---------------------------------------------------------------------------
# 2. Extract training set
# ---------------------------------------------------------------------------
TRAIN_DIR="${IMAGENET_DIR}/train"
if [ -d "$TRAIN_DIR" ] && [ "$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 999 ]; then
    log "Train directory already has $(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) class dirs — skipping extraction"
else
    mkdir -p "$TRAIN_DIR"
    log "Extracting training archive (this takes a while)..."
    tar xf "$TRAIN_TAR" -C "$TRAIN_DIR"
    log "Training archive extracted"

    # The train tar contains 1000 nested .tar files (one per class synset).
    # Extract each into its own directory.
    log "Extracting per-class tarballs (1000 classes)..."
    cd "$TRAIN_DIR"
    count=0
    for class_tar in n*.tar; do
        [ -f "$class_tar" ] || continue
        synset="${class_tar%.tar}"
        mkdir -p "$synset"
        tar xf "$class_tar" -C "$synset"
        rm "$class_tar"
        count=$((count + 1))
        if [ $((count % 100)) -eq 0 ]; then
            log "  Extracted $count / 1000 class tarballs"
        fi
    done
    log "Extracted $count class tarballs"
fi

# ---------------------------------------------------------------------------
# 3. Extract validation set
# ---------------------------------------------------------------------------
VAL_DIR="${IMAGENET_DIR}/val"
if [ -d "$VAL_DIR" ] && [ "$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 999 ]; then
    log "Val directory already organized into $(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) class dirs — skipping"
else
    mkdir -p "$VAL_DIR"
    log "Extracting validation archive..."
    tar xf "$VAL_TAR" -C "$VAL_DIR"
    log "Validation archive extracted"

    # Reorganize flat val images into class subdirectories
    log "Reorganizing validation images into class folders..."
    if [ -x "${SCRIPT_DIR}/imagenet_valprep.sh" ]; then
        bash "${SCRIPT_DIR}/imagenet_valprep.sh" "$DATA_LOCATION"
    else
        log "Error: imagenet_valprep.sh not found at ${SCRIPT_DIR}/imagenet_valprep.sh"
        exit 1
    fi
    log "Validation images reorganized"
fi

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
log "Verifying setup..."

train_classes="$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
train_images="$(find "$TRAIN_DIR" -name '*.JPEG' | wc -l)"
val_classes="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
val_images="$(find "$VAL_DIR" -name '*.JPEG' | wc -l)"

log "  Train: $train_classes classes, $train_images images"
log "  Val:   $val_classes classes, $val_images images"

ok=true
if [ "$train_classes" -ne 1000 ]; then
    log "WARNING: Expected 1000 train classes, got $train_classes"
    ok=false
fi
if [ "$val_classes" -ne 1000 ]; then
    log "WARNING: Expected 1000 val classes, got $val_classes"
    ok=false
fi
if [ "$val_images" -ne 50000 ]; then
    log "WARNING: Expected 50000 val images, got $val_images"
    ok=false
fi

if $ok; then
    log "ImageNet setup complete and verified!"
else
    log "ImageNet setup complete with warnings (see above)"
fi
