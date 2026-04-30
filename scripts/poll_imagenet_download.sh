#!/usr/bin/env bash
# =============================================================================
# poll_imagenet_download.sh
#
# Intended to be run by cron. Checks whether ILSVRC2012_img_train.tar has
# finished downloading (file exists and hasn't grown in 3 minutes), then
# runs setup_imagenet.sh and removes itself from crontab.
#
# Usage (cron entry, every 15 minutes):
#   */15 * * * * /home/luke/Documents/GitHub/atlas/scripts/poll_imagenet_download.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${HOME}/Downloads"
TRAIN_TAR="${DOWNLOAD_DIR}/ILSVRC2012_img_train.tar"
POLL_LOG="${HOME}/data/imagenet_poll.log"

mkdir -p "$(dirname "$POLL_LOG")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$POLL_LOG"; }

# --- Check if file exists ---
if [ ! -f "$TRAIN_TAR" ]; then
    log "Train tar not found yet at $TRAIN_TAR"
    exit 0
fi

# --- Check if file is still being written (size unchanged for 3 min) ---
size1="$(stat -c%s "$TRAIN_TAR" 2>/dev/null || echo 0)"
sleep 180
size2="$(stat -c%s "$TRAIN_TAR" 2>/dev/null || echo 0)"

if [ "$size1" != "$size2" ]; then
    log "Train tar still downloading (${size1} -> ${size2} bytes)"
    exit 0
fi

# --- Minimum size check (~100GB, the train tar should be ~138GB) ---
min_size=$((100 * 1024 * 1024 * 1024))
if [ "$size2" -lt "$min_size" ]; then
    log "Train tar looks too small (${size2} bytes < 100GB) — may be incomplete"
    exit 0
fi

log "Download appears complete (${size2} bytes, stable). Running setup..."

# --- Run setup ---
if bash "${SCRIPT_DIR}/setup_imagenet.sh" >> "$POLL_LOG" 2>&1; then
    log "Setup completed successfully!"
else
    log "Setup failed with exit code $?"
    exit 1
fi

# --- Remove this cron entry ---
log "Removing poll cron job..."
crontab -l 2>/dev/null | grep -v "poll_imagenet_download" | crontab - 2>/dev/null || true
log "Cron job removed. All done."
