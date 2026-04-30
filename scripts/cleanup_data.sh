#!/bin/bash
# =============================================================================
# cleanup_data.sh
#
# Removes a hand-picked list of duplicate / unused dataset directories from
# $DATA_DIR (default: ~/data). EuroSAT is intentionally excluded.
#
# Safety: this script performs `rm -rf` on user data, so it requires explicit
# confirmation. It will print the deletion plan and exit unless `--yes` (or
# `-y`) is passed. Pass `--dry-run` to print the plan without deleting.
#
# Usage:
#   ./scripts/cleanup_data.sh [--dry-run | --yes] [--data-dir PATH]
# =============================================================================
set -e

DATA_DIR="${HOME}/data"
DRY_RUN=0
CONFIRMED=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -y|--yes)
            CONFIRMED=1
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --data-dir=*)
            DATA_DIR="${1#--data-dir=}"
            shift
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            echo "Run with --help to see usage." >&2
            exit 2
            ;;
    esac
done

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

# Print the full deletion plan up front so the user can review it.
echo "Deletion plan (root: $DATA_DIR):"
plan_count=0
for target in "${targets[@]}"; do
    path="$DATA_DIR/$target"
    if [ -e "$path" ]; then
        echo "  WILL DELETE: $path"
        plan_count=$((plan_count + 1))
    else
        echo "  not present: $path"
    fi
done

if [ "$plan_count" -eq 0 ]; then
    echo "Nothing to delete."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "(--dry-run set — no files removed)"
    exit 0
fi

if [ "$CONFIRMED" -ne 1 ]; then
    echo
    echo "Refusing to delete $plan_count item(s) without confirmation."
    echo "Re-run with --yes (or -y) to actually delete, or --dry-run to preview."
    exit 1
fi

for target in "${targets[@]}"; do
    path="$DATA_DIR/$target"
    if [ -e "$path" ]; then
        echo "Removing: $path"
        rm -rf "$path"
    fi
done

echo "Done."
