#!/usr/bin/env bash
# D1 pilot on galleria2 (6-layer student distilled from the 24-layer teacher), from the repo root:
#   nohup setsid scripts/distill/run_d1_pilot_remote.sh > runs/D1_6l_pilot.out 2>&1 < /dev/null &
# The config, the mix manifest and the teacher weights are put in place by runs/run_d1_pilot.sh
# on the main machine. Waits for the C2 synthesis (run_c2.sh) to end, holds the YODAS3 shard
# runner that it restarts (the GPU is shared with the B1 evals), trains (resuming after a crash),
# evaluates the 20k / 40k / 60k snapshots without CFG (the guidance is baked into the student)
# and restarts YODAS3. runs/D1_6l_pilot/PILOT_DONE marks the end.
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
CONFIG=configs/train/D1_6l_pilot.yaml
RUN=runs/D1_6l_pilot
LAST=$(grep -E "^max_steps:" $CONFIG | grep -oE "[0-9]+")
runner="yodas3/run_""shards.sh"; shard="yodas3/process_""shard.py"

until grep -q C2_DONE data/synth/c2_run.log 2>/dev/null; do sleep 120; done
pkill -f "$runner"
while pgrep -f "$shard" >/dev/null; do sleep 60; done
echo "$(date) C2 finished, YODAS3 runner held"

for try in $(seq 1 30); do
  $UV run python scripts/train_ja.py $CONFIG >> $RUN.log 2>&1
  [ -f $RUN/snapshots/step_$(printf %08d "$LAST").safetensors ] && break
  echo "$(date) training stopped (try $try): $(tr '\r' '\n' < $RUN.log | grep -a ' step ' | tail -1 | cut -c1-70)"
  sleep 300
done
echo "$(date) training done: $(tr '\r' '\n' < $RUN.log | grep -a ' step ' | tail -1 | cut -c1-70)"

for step in $LAST $((LAST * 2 / 3)) $((LAST / 3)); do
  for set in jkyb_mini jvs_bal; do
    for try in $(seq 1 12); do
      $UV run python scripts/eval_tts.py $RUN --snapshot $step --set $set --cfg 1.0 \
        > $RUN/eval_${step}_$set.log 2>&1 && break
      sleep 300
    done
    echo "$(date) eval $step $set: $(grep -ohE '"(accuracy|kana_cer@1)": [0-9.]+' $RUN/eval/step$(printf %08d "$step")/${set}_rotation_cfg1.0/metrics.json 2>/dev/null | head -1)"
  done
done

rm -f data/yodas3/.claim_*
nohup scripts/yodas3/run_shards.sh data/yodas3 --workers 8 --batch 16 --max-padded-sec 60 \
  --tts-blocklist data/yodas3/tts_blocklist.npy > /dev/null 2>&1 < /dev/null &
touch $RUN/PILOT_DONE
echo "$(date) YODAS3 runner resumed; PILOT_DONE"
