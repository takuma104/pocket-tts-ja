#!/usr/bin/env bash
# usage: scripts/release/wheel_eval.sh RUN STEP CONFIG SUFFIX SET...
# Drop-in evaluation of a packaged model (plan D3): every SET is generated on this machine's CPU
# with the stock wheel (scripts/release/wheel_generate.py) and scored on the eval host, next to
# the GPU evaluation of the same snapshot. Results: RUN/eval/step<STEP>/<set>_rotation_cfg1.0_<SUFFIX>.
#   scripts/package_release.py configs/model/ja_6l_4k.yaml runs/D1_6l_b/snapshots/step_00120000.safetensors \
#       --name japanese --out exp/release_test/d1b_120k --dtype bfloat16
#   scripts/release/wheel_eval.sh runs/D1_6l_b 120000 exp/release_test/d1b_120k/japanese.yaml wheel jkyb_mini jvs_bal short jsut
cd "$(dirname "$0")/../.." || exit 1
RUN=$1; STEP=$2; CONFIG=$3; SUFFIX=$4; shift 4
WHEEL=${WHEEL:-pocket-tts==3.3.0}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"; H=${EVAL_HOST:-galleria2-direct}; R=co/pocket-tts-ja
E=$RUN/eval/step$(printf %08d "$STEP")
for s in "$@"; do
  tag=${s}_rotation_cfg1.0_$SUFFIX
  uv run python scripts/eval_tts.py "$RUN" --snapshot "$STEP" --set "$s" --cfg 1.0 --tag-suffix "$SUFFIX" --export-items || exit 1
  uvx --from "$WHEEL" python scripts/release/wheel_generate.py "$CONFIG" "$E/$tag" --workers "${WORKERS:-6}" || exit 1
  $SSH $H "mkdir -p $R/$E/$tag" || exit 1
  rsync -a -e "$SSH" --exclude items.json "$E/$tag/" "$H:$R/$E/$tag/" || exit 1
  ok=0
  for try in 1 2 3 4 5 6; do  # the eval host's GPU is shared: retry when the ASR runs out of memory
    $SSH $H "cd $R && ~/.local/bin/uv run python scripts/eval_tts.py $RUN --snapshot $STEP --set $s --cfg 1.0 \
      --tag-suffix $SUFFIX --from-audio --no-wandb" && { ok=1; break; }
    sleep 300
  done
  [ "$ok" = 1 ] || exit 1
  rsync -a --exclude audio --exclude "asr_audio/*.wav" -e "$SSH" "$H:$R/$E/$tag/" "$E/$tag/"
  echo "$(date) $tag: $(tr -d '\n ' < "$E/$tag/metrics.json" | cut -c1-300)"
done
echo WHEEL_EVAL_DONE
