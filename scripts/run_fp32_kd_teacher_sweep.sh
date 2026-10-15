#!/bin/bash
set -euo pipefail

# FP32 KD teacher sweep.
#
# Defaults compare TA-KD vs direct R50 distillation across paired seeds.
# Override any loop from the shell, for example:
#   SEEDS="42 43 44" TRIM_RESOLUTIONS="160 192" SLIM_RESOLUTIONS="160" \
#     bash scripts/run_fp32_kd_teacher_sweep.sh
#
# Extra Hydra overrides are forwarded to every training call:
#   bash scripts/run_fp32_kd_teacher_sweep.sh batch_size=32

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-kd_teacher_sweep_$(date +%Y%m%d_%H%M%S)}"
TEACHERS="${TEACHERS:-r18_ta r50_direct}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TRIM_RESOLUTIONS="${TRIM_RESOLUTIONS:-160 192}"
SLIM_RESOLUTIONS="${SLIM_RESOLUTIONS:-160}"
RUN_TRIM="${RUN_TRIM:-1}"
RUN_SLIM="${RUN_SLIM:-1}"
LOG_ROOT="${LOG_ROOT:-logs/${EXPERIMENT_TAG}}"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo " FP32 KD Teacher Sweep"
echo "============================================================"
echo "Experiment tag:    $EXPERIMENT_TAG"
echo "Teachers:          $TEACHERS"
echo "Seeds:             $SEEDS"
echo "Trim resolutions:  $TRIM_RESOLUTIONS"
echo "Slim resolutions:  $SLIM_RESOLUTIONS"
echo "Logs:              $LOG_ROOT"
echo "Extra Hydra args:  $*"
echo "============================================================"

run_training() {
    local family="$1"
    local script_path="$2"
    local resolution="$3"
    local teacher="$4"
    local seed="$5"
    shift 5

    local log_file="${LOG_ROOT}/${family}_trim${resolution}_${teacher}_seed${seed}.log"
    local cmd=(
        "$PYTHON_BIN"
        "$script_path"
        "RANDOM_SEED=${seed}"
        "++student_resolution=${resolution}"
        "++teacher_mode=${teacher}"
        "++experiment_tag=${EXPERIMENT_TAG}"
    )

    if [ "$#" -gt 0 ]; then
        cmd+=("$@")
    fi

    echo
    echo "== ${family} | trim${resolution} | ${teacher} | seed ${seed} =="
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    echo
    "${cmd[@]}" 2>&1 | tee "$log_file"
}

if [ "$RUN_TRIM" = "1" ]; then
    for resolution in $TRIM_RESOLUTIONS; do
        for teacher in $TEACHERS; do
            for seed in $SEEDS; do
                run_training "test_resnet" "src/train_test_resnet_trim.py" \
                    "$resolution" "$teacher" "$seed" "$@"
            done
        done
    done
fi

if [ "$RUN_SLIM" = "1" ]; then
    for resolution in $SLIM_RESOLUTIONS; do
        for teacher in $TEACHERS; do
            for seed in $SEEDS; do
                run_training "slim_test_resnet" "src/train_test_resnet_slim.py" \
                    "$resolution" "$teacher" "$seed" "$@"
            done
        done
    done
fi

echo
echo "Sweep complete."
echo "Models:  models/${EXPERIMENT_TAG}"
echo "Results: runs/${EXPERIMENT_TAG}"
echo "Logs:    ${LOG_ROOT}"
