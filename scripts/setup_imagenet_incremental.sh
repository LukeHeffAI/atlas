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

# ---------------------------------------------------------------------------
# Preflight checks (mirrors setup_imagenet.sh — gives clearer errors than the
# terse `tar: ... Cannot open` we'd otherwise hit under `set -e`).
# Phase 1 only needs class tars already on disk; the main TRAIN_TAR is
# optional in that case but required for Phase 2. VAL_TAR is required if
# Phase 3 needs to extract.
# ---------------------------------------------------------------------------
existing_class_tars="$(find "$TRAIN_DIR" -maxdepth 1 -name 'n*.tar' 2>/dev/null | wc -l)"
if [ ! -f "$TRAIN_TAR" ] && [ "$existing_class_tars" -eq 0 ]; then
    echo "Error: Training archive not found at $TRAIN_TAR (and no class tars in $TRAIN_DIR)" >&2
    exit 1
fi
if [ -f "$TRAIN_TAR" ] && [ ! -r "$TRAIN_TAR" ]; then
    echo "Error: Training archive at $TRAIN_TAR is not readable" >&2
    exit 1
fi
if [ ! -f "$VAL_TAR" ] && [ ! -d "$VAL_DIR" ]; then
    echo "Error: Validation archive not found at $VAL_TAR (and $VAL_DIR does not exist)" >&2
    exit 1
fi
if [ -f "$VAL_TAR" ] && [ ! -r "$VAL_TAR" ]; then
    echo "Error: Validation archive at $VAL_TAR is not readable" >&2
    exit 1
fi

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Phase 1: Process existing class tars in train/
# ---------------------------------------------------------------------------
if [ "$existing_class_tars" -gt 0 ]; then
    log "Phase 1: Processing $existing_class_tars existing class tars..."
    count=0
    for class_tar in "$TRAIN_DIR"/n*.tar; do
        [ -f "$class_tar" ] || continue
        synset="$(basename "${class_tar%.tar}")"
        if [ -d "$TRAIN_DIR/$synset" ] && [ "$(find "$TRAIN_DIR/$synset" -name '*.JPEG' | head -1)" ]; then
            # Already extracted, just delete the tar
            rm "$class_tar"
        else
            mkdir -p "$TRAIN_DIR/$synset"
            # Let tar's own stderr through so corrupt class tars produce a
            # diagnosable error message (previously suppressed by `2>/dev/null`,
            # which left `set -e` exits unexplained).
            if tar xf "$class_tar" -C "$TRAIN_DIR/$synset"; then
                rm "$class_tar"
            else
                echo "Error: failed to extract class tar: $class_tar" >&2
                exit 1
            fi
        fi
        count=$((count + 1))
        if [ $((count % 50)) -eq 0 ]; then
            log "  Processed $count / $existing_class_tars"
        fi
    done
    log "  Phase 1 done: processed $count class tars"
else
    log "Phase 1: No existing class tars to process"
fi

# ---------------------------------------------------------------------------
# Phase 2: Extract remaining classes from main archive
# ---------------------------------------------------------------------------
# Note: this script writes each class tar to disk briefly (then deletes it),
# rather than using extract_imagenet_stream.py's pure in-memory streaming.
# That trade-off lets us pluck specific entries out of the outer archive on
# resume, at the cost of ~2x peak disk per class. Use
# extract_imagenet_stream.py instead if you want the stream-only flow.
#
# Build set of already-completed synsets (have directories with images) so we
# can skip them without re-stating the filesystem inside the inner loop.
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
    if [ ! -f "$TRAIN_TAR" ]; then
        echo "Error: $remaining classes still need extraction but $TRAIN_TAR is missing" >&2
        exit 1
    fi
    log "  Extracting $remaining remaining classes from $TRAIN_TAR (one at a time)..."

    # List all class tars in the archive. Some `tar` implementations emit
    # entries with a leading `./`, so we strip it before matching. We also
    # capture the listing into a temp file first so a `grep` non-match (or
    # any later pipeline failure under `pipefail`) cannot abort the script
    # silently before extraction begins.
    listing="$(mktemp)"
    trap 'rm -f "$listing"' EXIT
    if ! tar tf "$TRAIN_TAR" > "$listing"; then
        echo "Error: could not list contents of $TRAIN_TAR" >&2
        exit 1
    fi
    # Normalise leading `./` and keep only `nNNNNNNNN.tar` entries.
    class_entries="$(sed 's|^\./||' "$listing" | grep -E '^n[0-9]+\.tar$' || true)"
    if [ -z "$class_entries" ]; then
        echo "Error: no class tars (n*.tar) found inside $TRAIN_TAR" >&2
        exit 1
    fi

    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        synset="${entry%.tar}"
        # Skip via the cached set populated above (avoids re-walking the FS).
        if [ "${done_synsets[$synset]:-0}" = "1" ]; then
            continue
        fi
        # Defensive: also skip if the dir already has JPEGs (e.g., a class
        # finished mid-run and wasn't in the cache).
        if [ -d "$TRAIN_DIR/$synset" ] && [ "$(find "$TRAIN_DIR/$synset" -maxdepth 1 -name '*.JPEG' -print -quit 2>/dev/null)" ]; then
            done_synsets["$synset"]=1
            continue
        fi
        # Extract this single class tar from the main archive
        tar xf "$TRAIN_TAR" -C "$TRAIN_DIR" "$entry"
        # Extract images from the class tar
        mkdir -p "$TRAIN_DIR/$synset"
        tar xf "$TRAIN_DIR/$entry" -C "$TRAIN_DIR/$synset"
        # Delete the class tar immediately
        rm -f "$TRAIN_DIR/$entry"
        done_synsets["$synset"]=1
    done <<< "$class_entries"
    rm -f "$listing"
    trap - EXIT
    log "  Phase 2 done"
else
    log "  All 1000 classes present, skipping Phase 2"
fi

# ---------------------------------------------------------------------------
# Phase 3: Validation set
# ---------------------------------------------------------------------------
val_classes=0
val_jpegs_total=0
if [ -d "$VAL_DIR" ]; then
    val_classes="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    val_jpegs_total="$(find "$VAL_DIR" -name '*.JPEG' | wc -l)"
fi
# Require both the canonical 1000 class dirs and 50,000 JPEGs before
# declaring val complete — a directory-only check can wave through a
# half-organised tree and let the script proceed without rerunning
# imagenet_valprep.sh.
if [ "$val_classes" -eq 1000 ] && [ "$val_jpegs_total" -eq 50000 ]; then
    log "Phase 3: Val already organized ($val_classes class dirs, $val_jpegs_total images) — skipping"
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
