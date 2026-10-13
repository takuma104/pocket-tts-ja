#!/usr/bin/env bash
# D1: distill the 6-layer student from a teacher snapshot, from the repo root:
#   nohup setsid scripts/distill/run_d1.sh 660000 > runs/D1_6l.out 2>&1 < /dev/null &
# Merges the teacher's EMA snapshot into full weights and trains, resuming after a crash, until
# the snapshot of the last step exists. The teacher is the one the config names:
# distill_teacher_weights must be <teacher run>/weights_<STEP>.safetensors (runs/B1_4k for the
# main line, runs/BCC_4k for the CC BY line) and distill_teacher_config its model config.
# Evaluate the student without CFG (eval_queue.py --cfg 1.0): the guidance is baked in.
set -u
cd "$(dirname "$0")/../.."
STEP=$1
CONFIG=${2:-configs/train/D1_6l.yaml}
RUN=$(grep -E "^run_dir:" "$CONFIG" | awk '{print $2}')
LAST=$(grep -E "^max_steps:" "$CONFIG" | grep -oE "[0-9]+" | head -1)
W=$(grep -E "^distill_teacher_weights:" "$CONFIG" | awk '{print $2}')
TEACHER_CONFIG=$(grep -E "^distill_teacher_config:" "$CONFIG" | awk '{print $2}')
TEACHER_RUN=$(dirname "$W")
[ "$W" = "$TEACHER_RUN/weights_$STEP.safetensors" ] || { echo "$CONFIG does not use step $STEP of its teacher ($W)"; exit 1; }
[ -f "$W" ] || uv run python scripts/merge_snapshot_weights.py "$TEACHER_CONFIG" \
  "$TEACHER_RUN/snapshots/step_$(printf %08d "$STEP").safetensors" "$W" || exit 1
echo "$(date) teacher $W, $LAST steps into $RUN"
for try in $(seq 1 30); do
  uv run python scripts/train_ja.py "$CONFIG" >> "$RUN.log" 2>&1
  [ -f "$RUN/snapshots/step_$(printf %08d "$LAST").safetensors" ] && break
  echo "$(date) training stopped (try $try): $(tr '\r' '\n' < "$RUN.log" | grep -a 'INFO train\] step' | tail -1 | cut -c1-70)"
  sleep 120
done
[ -f "$RUN/snapshots/step_$(printf %08d "$LAST").safetensors" ] && echo "$(date) D1_DONE"
