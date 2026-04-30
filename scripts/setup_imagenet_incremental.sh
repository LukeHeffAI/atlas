#!/usr/bin/env bash
# =============================================================================
# setup_imagenet_incremental.sh
#
# Space-efficient ImageNet extraction. Processes class tars one at a time,
# deleting each after extracting its images, so we never need the full
# 138GB of class tars on disk simultaneously.
#
# Phase 1: Process any class .tar files already in train/ (from partial extraction)
# Phase 2: Extract remaining class tars from the main archive one at a time
# Phase 3: Extract and organize validation set
# Phase 4: Verify
# =============================================================================
set -euo pipefail

DOWNLOAD_DIR="${1:-$HOME/Downloads}"
DATA_LOCATION="${2:-$HOME/data}"
IMAGENET_DIR="${DATA_LOCATION}/imagenet"
TRAIN_DIR="${IMAGENET_DIR}/train"
VAL_DIR="${IMAGENET_DIR}/val"
TRAIN_TAR="${DOWNLOAD_DIR}/ILSVRC2012_img_train.tar"
VAL_TAR="${DOWNLOAD_DIR}/ILSVRC2012_img_val.tar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Phase 1: Process existing class tars in train/
# ---------------------------------------------------------------------------
existing_tars="$(find "$TRAIN_DIR" -maxdepth 1 -name 'n*.tar' 2>/dev/null | wc -l)"
if [ "$existing_tars" -gt 0 ]; then
    log "Phase 1: Processing $existing_tars existing class tars..."
    count=0
    for class_tar in "$TRAIN_DIR"/n*.tar; do
        [ -f "$class_tar" ] || continue
        synset="$(basename "${class_tar%.tar}")"
        if [ -d "$TRAIN_DIR/$synset" ] && [ "$(find "$TRAIN_DIR/$synset" -name '*.JPEG' | head -1)" ]; then
            # Already extracted, just delete the tar
            rm "$class_tar"
        else
            mkdir -p "$TRAIN_DIR/$synset"
            tar xf "$class_tar" -C "$TRAIN_DIR/$synset" 2>/dev/null && rm "$class_tar"
        fi
        count=$((count + 1))
        if [ $((count % 50)) -eq 0 ]; then
            log "  Processed $count / $existing_tars"
        fi
    done
    log "  Phase 1 done: processed $count class tars"
else
    log "Phase 1: No existing class tars to process"
fi

# ---------------------------------------------------------------------------
# Phase 2: Extract remaining classes from main archive
# ---------------------------------------------------------------------------
# Build set of already-completed synsets (have directories with images)
log "Phase 2: Checking which classes still need extraction..."
completed=0
declare -A done_synsets
for d in "$TRAIN_DIR"/n*/; do
    [ -d "$d" ] || continue
    synset="$(basename "$d")"
    # Consider done if directory has at least 1 JPEG
    if [ "$(find "$d" -maxdepth 1 -name '*.JPEG' -print -quit 2>/dev/null)" ]; then
        done_synsets["$synset"]=1
        completed=$((completed + 1))
    fi
done
log "  $completed classes already complete"

if [ "$completed" -lt 1000 ]; then
    remaining=$((1000 - completed))
    log "  Extracting $remaining remaining classes from $TRAIN_TAR (one at a time)..."

    # List all class tars in the archive
    tar tf "$TRAIN_TAR" | grep '^n.*\.tar$' | while read -r entry; do
        synset="${entry%.tar}"
        # Skip if already done
        if [ -d "$TRAIN_DIR/$synset" ] && [ "$(find "$TRAIN_DIR/$synset" -maxdepth 1 -name '*.JPEG' -print -quit 2>/dev/null)" ]; then
            continue
        fi
        # Extract this single class tar from the main archive
        tar xf "$TRAIN_TAR" -C "$TRAIN_DIR" "$entry"
        # Extract images from the class tar
        mkdir -p "$TRAIN_DIR/$synset"
        tar xf "$TRAIN_DIR/$entry" -C "$TRAIN_DIR/$synset"
        # Delete the class tar immediately
        rm -f "$TRAIN_DIR/$entry"
    done
    log "  Phase 2 done"
else
    log "  All 1000 classes present, skipping Phase 2"
fi

# ---------------------------------------------------------------------------
# Phase 3: Validation set
# ---------------------------------------------------------------------------
val_classes="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
if [ "$val_classes" -ge 999 ]; then
    log "Phase 3: Val already organized ($val_classes class dirs) — skipping"
else
    mkdir -p "$VAL_DIR"
    # Check if val images exist (flat or organized)
    val_jpegs="$(find "$VAL_DIR" -maxdepth 1 -name '*.JPEG' 2>/dev/null | wc -l)"
    if [ "$val_jpegs" -eq 0 ] && [ "$val_classes" -eq 0 ]; then
        log "Phase 3: Extracting validation archive..."
        tar xf "$VAL_TAR" -C "$VAL_DIR"
    fi
    log "Phase 3: Reorganizing validation images into class folders..."
    bash "${SCRIPT_DIR}/imagenet_valprep.sh" "$DATA_LOCATION"
    log "  Phase 3 done"
fi

# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------
log "Phase 4: Verifying..."
train_classes="$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
train_images="$(find "$TRAIN_DIR" -name '*.JPEG' | wc -l)"
val_classes="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
val_images="$(find "$VAL_DIR" -name '*.JPEG' | wc -l)"
remaining_tars="$(find "$TRAIN_DIR" -maxdepth 1 -name '*.tar' | wc -l)"

log "  Train: $train_classes classes, $train_images images, $remaining_tars leftover tars"
log "  Val:   $val_classes classes, $val_images images"

ok=true
[ "$train_classes" -eq 1000 ] || { log "WARNING: Expected 1000 train classes, got $train_classes"; ok=false; }
[ "$val_classes" -eq 1000 ]   || { log "WARNING: Expected 1000 val classes, got $val_classes"; ok=false; }
[ "$val_images" -eq 50000 ]   || { log "WARNING: Expected 50000 val images, got $val_images"; ok=false; }
[ "$remaining_tars" -eq 0 ]   || { log "WARNING: $remaining_tars leftover class tars in train/"; ok=false; }

if $ok; then
    log "ImageNet setup complete and verified!"
    log ""
    log "You can now safely delete the source archives to free ~144GB:"
    log "  rm $TRAIN_TAR $VAL_TAR"
else
    log "ImageNet setup complete with warnings (see above)"
fi
