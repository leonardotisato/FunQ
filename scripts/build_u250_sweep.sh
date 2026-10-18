#!/bin/bash
# Build swept QAT configurations (Base/Slim x 160/192/224 px x 6/8-bit) for the U250 at their maximum
# throughput: --target-fps 5000, above every configuration's limit, except the overrides below.
# One build at a time (~12 h each).
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

# The remaining configurations already reach their limit in the 5000fps builds in build/u250/
# (pre-sweep checkpoints; the Base 224 ones untrimmed). Base 160 6-bit is listed again because its
# 5000fps build stopped short of the limit (see TARGET_FPS_OVERRIDE).
CONFIGS=(
  test_resnet_trim160_8w8a
  test_resnet_trim192_8w8a
  test_resnet_slim128x64_trim192_6w6a
  test_resnet_slim128x64_trim192_8w8a
  test_resnet_slim128x64_trim224_6w6a
  test_resnet_slim128x64_trim224_8w8a
  test_resnet_trim160_6w6a
)

# At 5000 fps (20000 cycles/frame) FINN keeps the last max-pool at PE=1 (19200 cycles), and the
# depthwise window generator feeding it inherits SIMD=1 and becomes the bottleneck (28199 cycles,
# 3546 fps). 5300 fps (18868 cycles) makes FINN raise that PE, so Base 160 reaches its limit,
# the first layer at one input pixel per cycle (26246 cycles, 3810 fps).
declare -A TARGET_FPS_OVERRIDE=(
  [test_resnet_trim160_8w8a]=5300
  [test_resnet_trim160_6w6a]=5300
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
    local fps="${TARGET_FPS_OVERRIDE[$c]:-$TARGET_FPS}"
    local dir="$OUT/build_finn_${c}_u250_${fps}fps"
    local tmp="$TMP_ROOT/${c}_${fps}fps"
    if [ -f "$dir/bitfile/finn-accel.xclbin" ]; then log "skip $c (bitfile present)"; continue; fi
    [ -f "$STAGE/$c.onnx" ] || { log "skip $c (no ONNX)"; continue; }
    log "START $c  (free on /tmp: $(df -h --output=avail /tmp | tail -1 | tr -d ' '))"
    FINN_HOST_BUILD_DIR="$tmp" FINN_SKIP_DEP_REPOS=1 bash run_finn.sh \
      python3 src/finn_build/build_test_resnet.py --board U250 --onnx "$STAGE/$c.onnx" \
      --output-dir "$dir" --target-fps "$fps" > "$LOGS/build_${c}_${fps}fps.log" 2>&1
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
