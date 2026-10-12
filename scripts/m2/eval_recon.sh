#!/usr/bin/env bash
# usage: scripts/m2/eval_recon.sh OUT NAME=DECODER_FILE...
# M2 reconstruction check: JVS recordings (5 male + 5 female speakers, parallel100) as recorded,
# through the stock Mimi and through each fine-tuned decoder; kana-whisper on all of them; then
# Kana-CER and the first-kana error rates side by side (scripts/analyze_first_kana.py).
#   scripts/m2/eval_recon.sh exp/m2/recon mel10k=runs/M2_mel/decoder_step_00010000.safetensors
cd "$(dirname "$0")/../.." || exit 1
UV=$(command -v uv || echo ~/.local/bin/uv)
OUT=$1; shift
decoders=(); dirs=("$OUT/orig" "$OUT/mimi"); specs=("orig=$OUT/asr_orig/kana.jsonl" "mimi=$OUT/asr_mimi/kana.jsonl")
for spec in "$@"; do
  name=${spec%%=*}
  decoders+=(--decoder "$spec"); dirs+=("$OUT/$name"); specs+=("$name=$OUT/asr_$name/kana.jsonl")
done
$UV run python scripts/m2/recon_jvs.py --out "$OUT" "${decoders[@]}" || exit 1
$UV run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
  python scripts/asr_transcribe.py "${dirs[@]}" --skip-text > "$OUT/asr.log" 2>&1 || { tail -5 "$OUT/asr.log"; exit 1; }
$UV run python scripts/analyze_first_kana.py --dataset "$OUT/items.jsonl" --reference text "${specs[@]}"
