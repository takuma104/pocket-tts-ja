#!/usr/bin/env bash
# usage: scripts/release/wheel_eval.sh RUN STEP CONFIG SUFFIX SET...
# Drop-in evaluation of a packaged model (plan D3): every SET is generated on this machine's CPU
# with the stock wheel (scripts/release/wheel_generate.py) and scored on the eval host, next to
# the GPU evaluation of the same snapshot. Results: RUN/eval/step<STEP>/<set>_rotation_cfg1.0_<SUFFIX>.
#   scripts/package_release.py configs/model/ja_6l_4k.yaml runs/D1_6l_b/snapshots/step_00120000.safetensors \
#       --name japanese --out exp/release_test/d1b_120k --dtype bfloat16
#   scripts/release/wheel_eval.sh runs/D1_6l_b 120000 exp/release_test/d1b_120k/japanese.yaml wheel jkyb_mini jvs_bal short jsut
# REMOTE_PREFIX wraps the scoring command on the eval host (e.g. "flock runs/extra_gpu.lock", so
# that only one of several extra jobs is on its GPU at a time).
# GEN_REMOTE=1 generates on the eval host's CPU instead of this machine's (six workers take about
# half the speed of a training run here); the package is copied there under the same path, which
# its config refers to.
# EVAL_HOST=local generates and scores on this machine (no ssh): the CPU for the wheel, the GPU for
# the ASR. Generation and scoring of one set alternate, so the GPU is free while the wheel runs;
# a set that already has its metrics under the tag is skipped.
cd "$(dirname "$0")/../.." || exit 1
RUN=$1; STEP=$2; CONFIG=$3; SUFFIX=$4; shift 4
WHEEL=${WHEEL:-pocket-tts==3.3.0}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"; H=${EVAL_HOST:-galleria2-direct}; R=co/pocket-tts-ja
E=$RUN/eval/step$(printf %08d "$STEP")
PKG=$(dirname "$CONFIG")
if [ "$H" = local ]; then
  for s in "$@"; do
    tag=${s}_rotation_cfg1.0_$SUFFIX
    if [ ! -f "$E/$tag/metrics.json" ]; then
      uv run python scripts/eval_tts.py "$RUN" --snapshot "$STEP" --set "$s" --cfg 1.0 --tag-suffix "$SUFFIX" --export-items || exit 1
      uvx --from "$WHEEL" python scripts/release/wheel_generate.py "$CONFIG" "$E/$tag" --workers "${WORKERS:-6}" || exit 1
      ${REMOTE_PREFIX:-} uv run python scripts/eval_tts.py "$RUN" --snapshot "$STEP" --set "$s" --cfg 1.0 \
        --tag-suffix "$SUFFIX" --from-audio --no-wandb || exit 1
    fi
    echo "$(date) $tag: $(tr -d '\n ' < "$E/$tag/metrics.json" | cut -c1-300)"
  done
  echo WHEEL_EVAL_DONE
  exit 0
fi
if [ -n "${GEN_REMOTE:-}" ]; then
  $SSH $H "mkdir -p $R/$PKG" && rsync -a -e "$SSH" "$PKG/" "$H:$R/$PKG/" || exit 1
fi
for s in "$@"; do
  tag=${s}_rotation_cfg1.0_$SUFFIX
  uv run python scripts/eval_tts.py "$RUN" --snapshot "$STEP" --set "$s" --cfg 1.0 --tag-suffix "$SUFFIX" --export-items || exit 1
  $SSH $H "mkdir -p $R/$E/$tag" || exit 1
  if [ -n "${GEN_REMOTE:-}" ]; then
    rsync -a -e "$SSH" "$E/$tag/items.json" "$H:$R/$E/$tag/" || exit 1
    ok=0
    # after a dropped connection the next try waits for the lock of the one still running there,
    # then finds every clip done
    for try in 1 2 3; do
      $SSH -o ServerAliveInterval=30 $H "cd $R && flock $E/$tag/gen.lock ~/.local/bin/uvx --from $WHEEL python \
        scripts/release/wheel_generate.py $CONFIG $E/$tag --workers ${WORKERS:-6}" < /dev/null && { ok=1; break; }
      sleep 60
    done
    [ "$ok" = 1 ] || exit 1
  else
    uvx --from "$WHEEL" python scripts/release/wheel_generate.py "$CONFIG" "$E/$tag" --workers "${WORKERS:-6}" || exit 1
    rsync -a -e "$SSH" --exclude items.json "$E/$tag/" "$H:$R/$E/$tag/" || exit 1
  fi
  ok=0
  for try in 1 2 3 4 5 6; do  # the eval host's GPU is shared: retry when the ASR runs out of memory
    $SSH $H "cd $R && ${REMOTE_PREFIX:-} ~/.local/bin/uv run python scripts/eval_tts.py $RUN --snapshot $STEP --set $s --cfg 1.0 \
      --tag-suffix $SUFFIX --from-audio --no-wandb" && { ok=1; break; }
    sleep 300
  done
  [ "$ok" = 1 ] || exit 1
  rsync -a --exclude audio --exclude "asr_audio/*.wav" -e "$SSH" "$H:$R/$E/$tag/" "$E/$tag/"
  echo "$(date) $tag: $(tr -d '\n ' < "$E/$tag/metrics.json" | cut -c1-300)"
done
echo WHEEL_EVAL_DONE
