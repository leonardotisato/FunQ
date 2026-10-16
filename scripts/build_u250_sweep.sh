#!/bin/bash
# Build the swept QAT configurations (Base/Slim x 160/192/224 px x 6/8-bit) that have no U250 build
# yet, at --target-fps 5000, i.e. each configuration's maximum throughput. One build at a time (~9 h each).
#
# Weights come from the matched QAT sweep, seed 42, ResNet18 teacher-assistant run; throughput and
# area do not depend on the weights, board accuracy can be checked against that run's report.
#
# Run on the training VM from the repo root, inside tmux:
#   tmux new -s hwbuild 'bash scripts/build_u250_sweep.sh export build 2>&1 | tee -a logs/hw_sweep/queue.log'
# Steps: "export" writes models/hw_sweep/<config>.onnx; "build" runs the FINN queue.
# Finished builds (bitfile present) are skipped, so the queue can be restarted.

set -u
cd "$(dirname "$0")/.."

SWEEP=qat_kd_teacher_sweep_20260703_121750_v2
RUN_SUFFIX=r18_ta_seed42
STAGE=models/hw_sweep
OUT=build/u250/sweep
LOGS=logs/hw_sweep
TMP_ROOT=/tmp/finn_hw_sweep
TARGET_FPS=5000

# The other six configurations already have 5000fps builds in build/u250/ (pre-sweep checkpoints;
# the Base 224 ones untrimmed). Add them here only to rebuild them from sweep weights.
CONFIGS=(
  test_resnet_trim160_8w8a
  test_resnet_trim192_8w8a
  test_resnet_slim128x64_trim192_6w6a
  test_resnet_slim128x64_trim192_8w8a
  test_resnet_slim128x64_trim224_6w6a
  test_resnet_slim128x64_trim224_8w8a
)

mkdir -p "$STAGE" "$OUT" "$LOGS"
log() { echo "[$(date '+%F %T')] $*"; }

export_all() {
  source venv/bin/activate
  local manifest="$STAGE/MANIFEST.csv"
  echo "config,checkpoint,checkpoint_sha256,onnx_sha256" > "$manifest"
  for c in "${CONFIGS[@]}"; do
    local ckpt="models/$SWEEP/${c}_${RUN_SUFFIX}_qat.pth"
    [ -f "$ckpt" ] || { log "MISSING checkpoint $ckpt"; return 1; }
    # The export scripts look for <models_dir>/<config>_qat.pth and write <models_dir>/<config>.onnx.
    ln -sfn "../$SWEEP/${c}_${RUN_SUFFIX}_qat.pth" "$STAGE/${c}_qat.pth"
    local res bits script
    res=$(sed -E 's/.*_trim([0-9]+)_.*/\1/' <<< "$c")
    bits=$(sed -E 's/.*_([0-9])w[0-9]a$/\1/' <<< "$c")
    if [[ "$c" == *slim* ]]; then script=src/export_test_resnet_slim.py; else script=src/export_test_resnet_trim.py; fi
    if [ ! -f "$STAGE/$c.onnx" ]; then
      log "export $c"
      python "$script" ++student_resolution="$res" ++weight_bits="$bits" ++act_bits="$bits" \
        ++models_dir="$STAGE" > "$LOGS/export_$c.log" 2>&1 || { log "EXPORT FAILED $c (see $LOGS/export_$c.log)"; return 1; }
    fi
    echo "$c,$ckpt,$(sha256sum < "$ckpt" | cut -c1-64),$(sha256sum < "$STAGE/$c.onnx" | cut -c1-64)" >> "$manifest"
  done
  deactivate
  log "exports done: $(ls "$STAGE"/*.onnx | wc -l) ONNX files"
}

build_all() {
  for c in "${CONFIGS[@]}"; do
    local dir="$OUT/build_finn_${c}_u250_${TARGET_FPS}fps"
    local tmp="$TMP_ROOT/$c"
    if [ -f "$dir/bitfile/finn-accel.xclbin" ]; then log "skip $c (bitfile present)"; continue; fi
    [ -f "$STAGE/$c.onnx" ] || { log "skip $c (no ONNX)"; continue; }
    log "START $c  (free on /tmp: $(df -h --output=avail /tmp | tail -1 | tr -d ' '))"
    FINN_HOST_BUILD_DIR="$tmp" FINN_SKIP_DEP_REPOS=1 bash run_finn.sh \
      python3 src/finn_build/build_test_resnet.py --board U250 --onnx "$STAGE/$c.onnx" \
      --output-dir "$dir" --target-fps "$TARGET_FPS" > "$LOGS/build_$c.log" 2>&1
    if [ -f "$dir/bitfile/finn-accel.xclbin" ]; then
      rm -rf "$tmp"  # this build's own FINN scratch; reports and bitfile are already in $dir
      log "DONE $c"
    else
      # Stop the queue: failures so far were environmental (e.g. HLS IP export) and would repeat.
      log "FAILED $c (log $LOGS/build_$c.log, scratch kept in $tmp); queue stopped"
      return 1
    fi
  done
  log "queue finished"
}

for step in "$@"; do
  case "$step" in
    export) export_all || exit 1 ;;
    build) build_all || exit 1 ;;
    *) echo "usage: $0 [export] [build]"; exit 2 ;;
  esac
done
