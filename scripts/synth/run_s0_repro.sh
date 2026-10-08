#!/usr/bin/env bash
# S0 reproduction check (plan §4.4): synthesize JKYB-mini with the Yomi teacher
# under the release JKYB settings (jvs001 VOICEACTRESS100_001, seed 0) and score it.
# Run from the repo root on galleria2:
#
#   bash scripts/synth/run_s0_repro.sh [out-dir] [extra irodori_synth.py args...]
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:-exp/synth/s0_jkyb_mini}
shift || true
UV=${UV:-$HOME/.local/bin/uv}
ENV=${SYNTH_ENV:-$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS}
CKPT=${CKPT:-$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors}
REF=data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition

mkdir -p "$OUT/logs"
python3 scripts/synth/make_synth_input.py data/eval/jkyb_mini.jsonl "$OUT/input.jsonl" \
  --id-field key --ref-wav "$REF" --seed 0

/usr/bin/time -v -o "$OUT/logs/synth.time" env PYTHONPATH="$ENV" \
  "$UV" run --project "$ENV" --no-sync python scripts/synth/irodori_synth.py \
  --input "$OUT/input.jsonl" --output-dir "$OUT/audio" --log-dir "$OUT/synth_logs" \
  --checkpoint "$CKPT" "$@" > "$OUT/logs/synth.log" 2>&1

# Python 3.13 like the local env: under 3.14 the ASR DataLoader fails to pickle its
# workers (forkserver) and jkyb-eval aborts with "ASR failed after transcribing 0".
/usr/bin/time -v -o "$OUT/logs/eval.time" \
  "$UV" run --python 3.13 --project "$JKYB" --extra asr jkyb-eval tts "$OUT/audio" \
  --dataset data/eval/jkyb_mini.jsonl --device cuda --output-dir "$OUT/results" \
  > "$OUT/logs/eval.log" 2>&1
tail -8 "$OUT/logs/eval.log"
