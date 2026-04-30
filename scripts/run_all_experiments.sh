#!/bin/bash
set -e

# Project root on PYTHONPATH so `from src.*` imports resolve when running the
# scripts in isolation (outside run_full_experiment_suite.sh).
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

MODEL=${1:-ViT-B-32}
# Second arg can restrict to a single backend: `clip`, `openclip`, or `both` (default).
BACKENDS_ARG=${2:-both}

case "$BACKENDS_ARG" in
    both) BACKENDS=("clip" "openclip") ;;
    clip) BACKENDS=("clip") ;;
    openclip) BACKENDS=("openclip") ;;
    *) echo "Unknown backend selector '$BACKENDS_ARG' (use clip | openclip | both)"; exit 1 ;;
esac

for BACKEND in "${BACKENDS[@]}"; do
    CKPT_ROOT="checkpoints_${BACKEND}"
    echo "========================================"
    echo "Running ALL aTLAS experiments"
    echo "Model:    $MODEL"
    echo "Backend:  $BACKEND"
    echo "Ckpt root: $CKPT_ROOT"
    echo "========================================"

    ./scripts/run_task_negation.sh        "$MODEL" "$BACKEND" "$CKPT_ROOT"
    ./scripts/run_task_addition.sh        "$MODEL" "$BACKEND" "$CKPT_ROOT"
    ./scripts/run_test_time_adaptation.sh "$MODEL" "$BACKEND" "$CKPT_ROOT"
    ./scripts/run_few_shot.sh             "$MODEL" "$BACKEND" "$CKPT_ROOT"
    ./scripts/run_parameter_efficient.sh  "$MODEL" "$BACKEND" "$CKPT_ROOT"

    echo "[$BACKEND] core experiments complete."
done

echo "========================================"
echo "All core experiments complete (${BACKENDS[*]})"
echo "========================================"
