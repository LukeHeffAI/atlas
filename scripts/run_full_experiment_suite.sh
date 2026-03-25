#!/bin/bash
# =============================================================================
# Full aTLAS Experiment Suite Runner
#
# Runs ALL experiments sequentially for both HuggingFace CLIP and OpenCLIP
# backends, saving checkpoints to separate directories to avoid collisions.
#
# Usage:
#   ./scripts/run_full_experiment_suite.sh              # Run everything
#   ./scripts/run_full_experiment_suite.sh --dry-run     # Print commands only
#   ./scripts/run_full_experiment_suite.sh --rerun       # Rerun failed steps
#   ./scripts/run_full_experiment_suite.sh --backend clip # Only HF backend
#
# Backends & checkpoint separation:
#   HuggingFace CLIP: checkpoints/{model}/     results/{model}/
#   OpenCLIP:         checkpoints_openclip/{model}/  results_openclip/{model}/
# =============================================================================

set -uo pipefail

# Ensure project root is in PYTHONPATH for reliable imports
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CLIP_MODELS="ViT-B-32 ViT-B-16 ViT-L-14"
# OPENCLIP_MODELS="ViT-B-32 ViT-B-16 ViT-L-14"
CLIP_MODELS="ViT-B-32"
OPENCLIP_MODELS="ViT-B-32"

# Hypernetwork meta-learning configuration
META_TRAIN_DATASETS="CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101"
META_VAL_DATASETS="Caltech101,Flowers102"
HYPERNETWORK_ARCH="small"
META_EPOCHS=3
EPISODES_PER_EPOCH=3

# Few-shot settings
FEW_SHOT_SETTINGS="2"

# Parameter-efficient settings
PARTITION=10
PERC_VALUES="0.25"

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

DRY_RUN=false
RERUN_FAILURES=false
BACKEND_FILTER=""  # empty = both

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rerun)
            RERUN_FAILURES=true
            shift
            ;;
        --backend)
            BACKEND_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--rerun] [--backend clip|openclip]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging & failure tracking
# ---------------------------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiment_suite_${TIMESTAMP}"
FAILURE_LOG="${LOG_DIR}/failures.log"
RERUN_SCRIPT="${LOG_DIR}/rerun_failures.sh"
SUMMARY_LOG="${LOG_DIR}/summary.log"

# For --rerun mode, find the most recent failure log
if [ "$RERUN_FAILURES" = true ]; then
    LATEST_LOG_DIR=$(ls -dt logs/experiment_suite_* 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG_DIR" ] || [ ! -f "${LATEST_LOG_DIR}/failures.log" ]; then
        echo "No previous failure log found. Run without --rerun first."
        exit 1
    fi
    PREV_FAILURE_LOG="${LATEST_LOG_DIR}/failures.log"
    if [ ! -s "$PREV_FAILURE_LOG" ]; then
        echo "No failures recorded in ${PREV_FAILURE_LOG}. Nothing to rerun."
        exit 0
    fi
    echo "Rerunning $(wc -l < "$PREV_FAILURE_LOG") failed steps from ${PREV_FAILURE_LOG}"
fi

mkdir -p "$LOG_DIR"

TOTAL_STEPS=0
PASSED_STEPS=0
FAILED_STEPS=0
SKIPPED_STEPS=0
SUITE_START=$(date +%s)

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

# Run a single experiment step.
# Args: step_id description command...
run_step() {
    local step_id="$1"
    local description="$2"
    shift 2
    local cmd="$*"

    TOTAL_STEPS=$((TOTAL_STEPS + 1))

    # In --rerun mode, skip steps that didn't fail previously
    if [ "$RERUN_FAILURES" = true ]; then
        if ! grep -qF "$step_id" "$PREV_FAILURE_LOG" 2>/dev/null; then
            SKIPPED_STEPS=$((SKIPPED_STEPS + 1))
            return 0
        fi
        log_info "RERUN: ${description}"
    fi

    local step_log="${LOG_DIR}/${step_id}.log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ${description}"
        echo "  CMD: ${cmd}"
        SKIPPED_STEPS=$((SKIPPED_STEPS + 1))
        return 0
    fi

    log_info "START: ${description} (step ${step_id})"
    local step_start=$(date +%s)

    if eval "$cmd" > "$step_log" 2>&1; then
        local step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        log_info "PASS:  ${description} (${elapsed}s)"
        PASSED_STEPS=$((PASSED_STEPS + 1))
    else
        local exit_code=$?
        local step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        log_info "FAIL:  ${description} (${elapsed}s, exit ${exit_code})"
        echo "${step_id}|${description}|${cmd}" >> "$FAILURE_LOG"
        FAILED_STEPS=$((FAILED_STEPS + 1))
    fi
}

# ---------------------------------------------------------------------------
# Experiment functions
# ---------------------------------------------------------------------------

run_finetune() {
    local model="$1" backend="$2" ckpt_root="$3"

    run_step "${backend}_${model}_finetune" \
        "[${backend}] ${model}: Fine-tune on all datasets" \
        "python src/finetune.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root}"
}

run_eval_single_task() {
    local model="$1" backend="$2" ckpt_root="$3" mode="$4"

    run_step "${backend}_${model}_eval_${mode}" \
        "[${backend}] ${model}: Evaluate single-task (${mode})" \
        "python src/eval_single_task.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --finetuning-mode=${mode}"
}

run_task_negation() {
    local model="$1" backend="$2" ckpt_root="$3"

    run_step "${backend}_${model}_task_negation" \
        "[${backend}] ${model}: Learn task negation" \
        "python src/learn_task_negation.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --blockwise-coef"
}

run_task_addition() {
    local model="$1" backend="$2" ckpt_root="$3"

    run_step "${backend}_${model}_task_addition" \
        "[${backend}] ${model}: Learn task addition" \
        "python src/learn_task_addition.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --blockwise-coef"
}

run_test_time_adaptation() {
    local model="$1" backend="$2" ckpt_root="$3" logdir="$4"

    run_step "${backend}_${model}_ufm" \
        "[${backend}] ${model}: Test-time adaptation (UFM)" \
        "python src/learn_ufm.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --logdir=${logdir} --blockwise-coef"
}

run_few_shot() {
    local model="$1" backend="$2" ckpt_root="$3" logdir="$4"

    run_step "${backend}_${model}_few_shot" \
        "[${backend}] ${model}: Few-shot adaptation (${FEW_SHOT_SETTINGS} shots)" \
        "python src/learn_few_shots.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --logdir=${logdir} --blockwise-coef --subsample ${FEW_SHOT_SETTINGS}"
}

run_few_shot_adapter() {
    local model="$1" backend="$2" ckpt_root="$3" logdir="$4" adapter="$5"

    run_step "${backend}_${model}_few_shot_${adapter}" \
        "[${backend}] ${model}: Few-shot + ${adapter} adapter (${FEW_SHOT_SETTINGS} shots)" \
        "python src/learn_few_shots.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --logdir=${logdir} --blockwise-coef --subsample ${FEW_SHOT_SETTINGS} --adapter ${adapter}"
}

run_parameter_efficient() {
    local model="$1" backend="$2" ckpt_root="$3" logdir="$4"

    for perc in $PERC_VALUES; do
        run_step "${backend}_${model}_atlas_x_k_${perc}" \
            "[${backend}] ${model}: aTLAS x K (partition=${PARTITION}, subsample=${perc})" \
            "python src/learn_few_shots.py --model=${model} --clip-backend=${backend} --checkpoint-root=${ckpt_root} --logdir=${logdir} --partition ${PARTITION} --subsample ${perc}"
    done
}

run_text_hypernetwork() {
    local model="$1" backend="$2" ckpt_root="$3"

    run_step "${backend}_${model}_text_hypernetwork" \
        "[${backend}] ${model}: Meta-train text-to-coefficient hypernetwork" \
        "python src/learn_text_to_coef.py --model=${model} --clip-backend=${backend} --save=${ckpt_root}/${model} --hypernetwork-arch=${HYPERNETWORK_ARCH} --text-source=manual --meta-train-datasets=${META_TRAIN_DATASETS} --meta-val-datasets=${META_VAL_DATASETS} --meta-epochs=${META_EPOCHS} --episodes-per-epoch=${EPISODES_PER_EPOCH} --blockwise-coef"
}

run_multimodal_hypernetwork() {
    local model="$1" backend="$2" ckpt_root="$3"

    run_step "${backend}_${model}_multimodal_hypernetwork" \
        "[${backend}] ${model}: Meta-train multi-modal hypernetwork" \
        "python src/learn_multimodal_to_coef.py --model=${model} --clip-backend=${backend} --save=${ckpt_root}/${model} --hypernetwork-arch=${HYPERNETWORK_ARCH} --fusion-mode=concat --num-shots=4 --image-pooling=mean --text-input-mode=dataset --variable-shots --meta-train-datasets=${META_TRAIN_DATASETS} --meta-val-datasets=${META_VAL_DATASETS} --meta-epochs=${META_EPOCHS} --episodes-per-epoch=${EPISODES_PER_EPOCH} --blockwise-coef"
}

# ---------------------------------------------------------------------------
# Run all experiments for one model + backend combination
# ---------------------------------------------------------------------------

run_all_for_model() {
    local model="$1" backend="$2" ckpt_root="$3" logdir="$4"

    log_info "========================================================"
    log_info "Starting experiments: model=${model}, backend=${backend}"
    log_info "  Checkpoints: ${ckpt_root}/${model}/"
    log_info "  Results:     ${logdir}${model}/"
    log_info "========================================================"

    # Step 1: Fine-tune CLIP on all datasets
    run_finetune "$model" "$backend" "$ckpt_root"

    # Step 2-3: Evaluate to generate accuracy JSONs (required by task neg/add)
    run_eval_single_task "$model" "$backend" "$ckpt_root" "none"
    run_eval_single_task "$model" "$backend" "$ckpt_root" "standard"

    # Step 4-5: Task arithmetic experiments (require accuracy JSONs)
    run_task_negation "$model" "$backend" "$ckpt_root"
    run_task_addition "$model" "$backend" "$ckpt_root"

    # Step 6: Test-time adaptation
    run_test_time_adaptation "$model" "$backend" "$ckpt_root" "$logdir"

    # Step 7: Few-shot adaptation (all shots in one run)
    run_few_shot "$model" "$backend" "$ckpt_root" "$logdir"

    # Step 8-9: Few-shot with adapters
    run_few_shot_adapter "$model" "$backend" "$ckpt_root" "$logdir" "tip"
    run_few_shot_adapter "$model" "$backend" "$ckpt_root" "$logdir" "lpp"

    # Step 10: Parameter-efficient fine-tuning (aTLAS x K)
    run_parameter_efficient "$model" "$backend" "$ckpt_root" "$logdir"

    # Step 11: Text hypernetwork (before multimodal, per user request)
    run_text_hypernetwork "$model" "$backend" "$ckpt_root"

    # Step 12: Multi-modal hypernetwork (LAST, per user request)
    run_multimodal_hypernetwork "$model" "$backend" "$ckpt_root"

    log_info "Finished experiments: model=${model}, backend=${backend}"
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

log_info "========================================================"
log_info "aTLAS Full Experiment Suite"
log_info "Started: $(date)"
log_info "Log directory: ${LOG_DIR}"
if [ "$DRY_RUN" = true ]; then
    log_info "Mode: DRY RUN (no commands will be executed)"
fi
if [ "$RERUN_FAILURES" = true ]; then
    log_info "Mode: RERUN FAILURES from ${PREV_FAILURE_LOG}"
fi
log_info "========================================================"

# Phase 1: HuggingFace CLIP backend
if [ -z "$BACKEND_FILTER" ] || [ "$BACKEND_FILTER" = "clip" ]; then
    log_info ""
    log_info "############################################################"
    log_info "# PHASE 1: HuggingFace CLIP Backend                       #"
    log_info "############################################################"
    log_info ""

    for MODEL in $CLIP_MODELS; do
        run_all_for_model "$MODEL" "clip" "checkpoints" "results/"
    done
fi

# Phase 2: OpenCLIP backend
if [ -z "$BACKEND_FILTER" ] || [ "$BACKEND_FILTER" = "openclip" ]; then
    log_info ""
    log_info "############################################################"
    log_info "# PHASE 2: OpenCLIP Backend                                #"
    log_info "############################################################"
    log_info ""

    for MODEL in $OPENCLIP_MODELS; do
        run_all_for_model "$MODEL" "openclip" "checkpoints_openclip" "results_openclip/"
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))
SUITE_HOURS=$(( SUITE_ELAPSED / 3600 ))
SUITE_MINS=$(( (SUITE_ELAPSED % 3600) / 60 ))

log_info ""
log_info "========================================================"
log_info "EXPERIMENT SUITE COMPLETE"
log_info "========================================================"
log_info "Total time:   ${SUITE_HOURS}h ${SUITE_MINS}m"
log_info "Total steps:  ${TOTAL_STEPS}"
log_info "Passed:       ${PASSED_STEPS}"
log_info "Failed:       ${FAILED_STEPS}"
log_info "Skipped:      ${SKIPPED_STEPS}"
log_info "Log directory: ${LOG_DIR}"

if [ "$FAILED_STEPS" -gt 0 ] && [ "$DRY_RUN" = false ]; then
    log_info ""
    log_info "FAILURES (${FAILED_STEPS}):"
    while IFS='|' read -r step_id description cmd; do
        log_info "  - ${description}"
        log_info "    Log: ${LOG_DIR}/${step_id}.log"
    done < "$FAILURE_LOG"

    # Generate rerun script
    {
        echo "#!/bin/bash"
        echo "# Auto-generated script to rerun failed experiments"
        echo "# Generated: $(date)"
        echo "set -uo pipefail"
        echo ""
        while IFS='|' read -r step_id description cmd; do
            echo "echo \"Rerunning: ${description}\""
            echo "${cmd}"
            echo ""
        done < "$FAILURE_LOG"
    } > "$RERUN_SCRIPT"
    chmod +x "$RERUN_SCRIPT"

    log_info ""
    log_info "To rerun failed steps:"
    log_info "  Option 1: ./scripts/run_full_experiment_suite.sh --rerun"
    log_info "  Option 2: ${RERUN_SCRIPT}"
fi

log_info "========================================================"
