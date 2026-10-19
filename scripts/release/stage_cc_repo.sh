#!/usr/bin/env bash
# usage: scripts/release/stage_cc_repo.sh OUT NAME [DECODER] [REVISION]
# Writes the files of the CC BY line's Hugging Face repo ($REPO, default takuma104/pocket-tts-ja-cc)
# into OUT (the local clone of that repo, or a test directory) from the checkpoints that
# runs/run_release_cc.sh NAME [DECODER] evaluated:
#   japanese/ + japanese.yaml          D1-CC 160k, bfloat16
#   japanese_24l/ + japanese_24l.yaml  D0-CC 50k, 3 frames after EOS
#   tokenizer.model, voices/ (exp/release_cc/voices), README.md (docs/release/cc_model_card.md
#   without its comments)
# The yaml point at hf://$REPO/...; with REVISION (the commit that holds the weights) they are
# pinned to it. Checks: weights and tokenizer have the sha256 of the evaluated packages
# (exp/release_test/{d1cc_160k,d0cc_50k}_NAME), the yaml differ from theirs only in the paths, no
# audio besides voices/*.wav, no TBD left in the card. Release (the user's go-ahead first):
#   1. stage without REVISION, commit weights + tokenizer + voices, push: commit W
#   2. stage with REVISION=W, commit the yaml + README, push: commit C
#   3. scripts/release/hub_check.sh $REPO C OUT
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=$1; NAME=$2; DEC=${3:-}; REV=${4:-}
REPO=${REPO:-takuma104/pocket-tts-ja-cc}
LIC="CC BY 4.0 (trained on CC BY data only)"
P6=exp/release_test/d1cc_160k_$NAME; P24=exp/release_test/d0cc_50k_$NAME
DECARG=(); [ -n "$DEC" ] && DECARG=(--mimi-decoder "$DEC")
REVARG=(); [ -n "$REV" ] && REVARG=(--revision "$REV")
mkdir -p "$OUT"
uv run python scripts/package_release.py configs/model/ja_cc_6l_4k.yaml runs/D1_cc_6l/snapshots/step_00160000.safetensors \
  --name japanese --out "$OUT" --dtype bfloat16 --license "$LIC" --repo "$REPO" "${REVARG[@]}" "${DECARG[@]}" || exit 1
uv run python scripts/package_release.py configs/model/ja_cc_24l_4k.yaml runs/D0_cc_24l/snapshots/step_00050000.safetensors \
  --name japanese_24l --out "$OUT" --frames-after-eos 3 --license "$LIC" --repo "$REPO" "${REVARG[@]}" "${DECARG[@]}" || exit 1
mkdir -p "$OUT/voices"
cp exp/release_cc/voices/*.wav exp/release_cc/voices/voices.json "$OUT/voices/"
# the card: drop the leading comment (notes for the release) and the inline DECODER markers
python3 - docs/release/cc_model_card.md "$OUT/README.md" <<'EOF'
import re
import sys

text = open(sys.argv[1]).read()
text = re.sub(r"\A<!--.*?-->\n", "", text, flags=re.S)
text = re.sub(r" ?<!-- DECODER.*?-->", "", text)
open(sys.argv[2], "w").write(text)
EOF
if [ ! -f "$OUT/.gitattributes" ]; then
  printf '%s filter=lfs diff=lfs merge=lfs -text\n' '*.safetensors' '*.model' '*.wav' > "$OUT/.gitattributes"
fi

fail=0
for pair in "$P6/japanese/model.safetensors:$OUT/japanese/model.safetensors" \
            "$P24/japanese_24l/model.safetensors:$OUT/japanese_24l/model.safetensors" \
            "$P6/tokenizer.model:$OUT/tokenizer.model"; do
  a=$(sha256sum "${pair%%:*}" | cut -c1-16); b=$(sha256sum "${pair#*:}" | cut -c1-16)
  [ "$a" = "$b" ] && echo "${pair#*:}: sha256 $b (as evaluated)" || { echo "${pair#*:}: sha256 $b, evaluated $a DIFFERENT"; fail=1; }
done
for p in "$P6/japanese.yaml" "$P24/japanese_24l.yaml"; do
  # only the header's usage line and the two paths may differ
  d=$(diff <(grep -v -E "^#   pocket-tts|weights_path:|tokenizer_path:" "$p") \
           <(grep -v -E "^#   pocket-tts|weights_path:|tokenizer_path:" "$OUT/$(basename "$p")"))
  [ -z "$d" ] && echo "$(basename "$p"): as evaluated but the paths" || { echo "$(basename "$p") DIFFERS:"; echo "$d"; fail=1; }
  grep -E "weights_path:|tokenizer_path:" "$OUT/$(basename "$p")"
done
extra=$(find "$OUT" -path "$OUT/.git" -prune -o -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.mp3" -o -iname "*jvs*" \) -print \
  | grep -v -E "^$OUT/voices/(fe)?male_0[1-5]\.wav$")
[ -z "$extra" ] || { echo "unexpected files: $extra"; fail=1; }
grep -n "TBD" "$OUT/README.md" && { echo "README.md: TBD left"; fail=1; }
head -1 "$OUT/README.md" | grep -q '^---$' || { echo "README.md: does not start with the metadata"; fail=1; }
[ "$fail" = 0 ] && echo STAGE_OK || echo STAGE_FAILED
