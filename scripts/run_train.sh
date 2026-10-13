#!/usr/bin/env bash
# Train a config to its last step, resuming after a crash, from the repo root:
#   nohup setsid scripts/run_train.sh configs/train/BCC_4k.yaml > runs/BCC_4k.out 2>&1 < /dev/null &
# The trainer continues from the newest checkpoint of run_dir, so a later config with the same
# run_dir (another mix, the LR decay) can take the run over: write its path to <run_dir>/next_config.
# The trainer then stops right after its next checkpoint and this script goes on with that config.
set -u
cd "$(dirname "$0")/.."
use() {
  CONFIG=$1
  RUN=$(grep -E "^run_dir:" "$CONFIG" | awk '{print $2}')
  LAST=$(grep -E "^max_steps:" "$CONFIG" | grep -oE "[0-9]+" | head -1)
  DONE=$RUN/snapshots/step_$(printf %08d "$LAST").safetensors
  echo "$(date) $CONFIG: $LAST steps into $RUN"
}
use "$1"
echo $$ > "$RUN.launcher.pid"
for try in $(seq 1 50); do
  [ -f "$DONE" ] && [ ! -f "$RUN/next_config" ] && break
  if [ -f "$RUN/next_config" ]; then
    NEXT=$(head -1 "$RUN/next_config")
    mv "$RUN/next_config" "$RUN/next_config.taken"
    echo "$(date) handover at: $(ls "$RUN/snapshots" | tail -1)"
    use "$NEXT"
    continue
  fi
  uv run python scripts/train_ja.py "$CONFIG" >> "$RUN.log" 2>&1
  [ -f "$RUN/next_config" ] && continue
  [ -f "$DONE" ] && break
  echo "$(date) training stopped (try $try): $(tr '\r' '\n' < "$RUN.log" | grep -a 'INFO train\] step' | tail -1 | cut -c1-70)"
  sleep 120
done
[ -f "$DONE" ] && echo "$(date) TRAIN_DONE $CONFIG"
