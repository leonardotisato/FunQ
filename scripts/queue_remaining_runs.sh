#!/bin/bash
# Remaining training runs for the paper, one GPU job at a time; stops at the first failure.
#
# UWF: students of seeds 43-46 are rerun on the split the teachers were trained on (split_seed=42).
# REFUGE: full chain on the official split (region-of-interest ResNet50 -> full-image ResNet50 ->
# ResNet18 TA -> students -> QAT), as on UWF.
#
#   tmux new -d -s queue 'bash scripts/queue_remaining_runs.sh 2>&1 | tee -a logs/queue_remaining_runs.log'
#
# A stage is skipped when its marker exists (logs/queue_markers/<stage>.done), so the queue can be
# restarted after a failure. Within a stage, the QAT runner also skips completed runs.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-$PWD/venv/bin/python}"
UWF_FP32_TA=kd_teacher_sweep_split42_20261120      # started separately on 2026-11-20
UWF_FP32_DIRECT=kd_teacher_sweep_split42_direct
REFUGE_FP32=refuge_kd_teacher_sweep
MARKERS=logs/queue_markers
mkdir -p "$MARKERS"
log() { echo "[$(date '+%F %T')] $*"; }

stage() {
    local name="$1"; shift
    if [ -f "$MARKERS/$name.done" ]; then log "skip $name (done)"; return; fi
    log "START $name"
    "$@"
    touch "$MARKERS/$name.done"
    log "DONE $name"
}

fp32_sweep() {  # tag teachers seeds extra-hydra-args...
    local tag="$1" teachers="$2" seeds="$3"; shift 3
    PYTHON_BIN="$PY" EXPERIMENT_TAG="$tag" TEACHERS="$teachers" SEEDS="$seeds" \
        TRIM_RESOLUTIONS="160 192 224" SLIM_RESOLUTIONS="160 192 224" \
        bash scripts/run_fp32_kd_teacher_sweep.sh "$@"
}

# The FP32 sweep script does not resume, so a stage only counts as done when every report exists.
check_reports() {  # results-root expected-count
    local found
    found=$(ls "$1"/*/*_report.json 2>/dev/null | wc -l)
    [ "$found" -eq "$2" ] || { log "expected $2 reports in $1, found $found"; return 1; }
}

# 1. UWF, TA students already running in tmux uwf_fp32: wait for them.
while tmux has-session -t uwf_fp32 2>/dev/null; do sleep 600; done
stage uwf_fp32_ta check_reports "runs/$UWF_FP32_TA" 24

# 2. UWF QAT from the TA students, 8/6/4 bits.
stage uwf_qat_ta "$PY" scripts/run_qat_kd_teacher_sweep.py --source-sweep "runs/$UWF_FP32_TA" \
    --teachers r18_ta --bits 8 6 4

# 3. REFUGE teachers: ResNet50 on optic-disc crops -> full-image ResNet50, then the ResNet18 TA.
stage refuge_teachers_r50 "$PY" src/main.py --config-name refuge
stage refuge_teacher_r18 "$PY" src/train_resnet18_from_resnet50_kd.py --config-name refuge

# 4. REFUGE FP32 students, TA and direct, seeds 42-46 (official split, seeds vary training only).
stage refuge_fp32 fp32_sweep "$REFUGE_FP32" "r18_ta r50_direct" "42 43 44 45 46" --config-name=refuge
stage refuge_fp32_check check_reports "runs/refuge/$REFUGE_FP32" 60

# 5. REFUGE QAT from the TA students, 8/6/4 bits.
stage refuge_qat_ta "$PY" scripts/run_qat_kd_teacher_sweep.py --source-sweep "runs/refuge/$REFUGE_FP32" \
    --teachers r18_ta --bits 8 6 4 --config-name refuge --output-subdir refuge \
    --data-dir Data/refuge --csv-file refuge.csv

# 6. UWF direct-teacher students on the fixed split (TA vs direct comparison).
stage uwf_fp32_direct fp32_sweep "$UWF_FP32_DIRECT" r50_direct "43 44 45 46" ++split_seed=42
stage uwf_fp32_direct_check check_reports "runs/$UWF_FP32_DIRECT" 24

log "queue finished"
