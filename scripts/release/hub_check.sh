#!/usr/bin/env bash
# usage: scripts/release/hub_check.sh REPO REVISION LOCAL_DIR [VOICE]
# After a push to the Hugging Face repo: every <name>.yaml of LOCAL_DIR (the packaged files that
# were evaluated) is loaded from the hub at REVISION with the stock wheel, in a fresh cache, and
# generates one sentence; the downloaded weights and tokenizer must have the sha256 of the local
# files. VOICE defaults to the repo's voices/female_03.wav (none: the CLI's default voice).
#   scripts/release/hub_check.sh takuma104/pocket-tts-ja-cc <commit> exp/release_test/release_cc
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO=$1; REV=$2; LOCAL=$3; VOICE=${4:-hf://$REPO/voices/female_03.wav@$REV}
WHEEL=${WHEEL:-pocket-tts==3.3.0}
OUT=exp/hub_check/$(basename "$REPO")_${REV:0:8}
mkdir -p "$OUT"
export HF_HUB_CACHE=$PWD/$OUT/hf_cache  # a fresh cache: the files really come from the hub
fail=0
for y in "$LOCAL"/*.yaml; do
  name=$(basename "$y" .yaml)
  ARGS=(--config "hf://$REPO/$name.yaml@$REV" --text "こんにちは。今日はいい天気ですね。" --output-path "$OUT/$name.wav")
  [ "$VOICE" != none ] && ARGS+=(--voice "$VOICE")
  if uvx --from "$WHEEL" pocket-tts generate "${ARGS[@]}" > "$OUT/$name.log" 2>&1; then
    echo "$name: generated $(python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(f'{w.getnframes()/w.getframerate():.2f} s')" "$OUT/$name.wav")"
  else
    echo "$name: generation FAILED (see $OUT/$name.log)"; fail=1
  fi
done
# every file of the local package against the blob the hub served
while IFS= read -r f; do
  rel=${f#"$LOCAL"/}
  case "$rel" in *.yaml|README.md) continue ;; esac
  hub=$(find "$HF_HUB_CACHE" -path "*/snapshots/*/$rel" | head -1)
  if [ -z "$hub" ]; then echo "$rel: not downloaded"; continue; fi
  a=$(sha256sum "$f" | cut -c1-16); b=$(sha256sum "$(readlink -f "$hub")" | cut -c1-16)
  [ "$a" = "$b" ] && echo "$rel: sha256 $a (same)" || { echo "$rel: sha256 local $a, hub $b DIFFERENT"; fail=1; }
done < <(find "$LOCAL" -type f \( -name "*.safetensors" -o -name "tokenizer.model" \))
[ "$fail" = 0 ] && echo HUB_CHECK_OK || echo HUB_CHECK_FAILED
