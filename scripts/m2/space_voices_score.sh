#!/usr/bin/env bash
# usage: [OUT=exp/m2/space_voices] [STUDENTS="nc6 cc6"] [DECS="mimi keep4k ..."] [N=200] scripts/m2/space_voices_score.sh
# (waits until every <student>__<decoder> directory of OUT/tts holds N clips, then ASR and MOS; MOS runs are serialized
# through runs/.mos.lock)
# Score scripts/m2/space_voices.py's output: kana-whisper on every TTS cell, MOS (hp40) on TTS and reconstruction.
cd "$(dirname "$0")/../.." || exit 1
UV=$(command -v uv || echo ~/.local/bin/uv)
OUT=${OUT:-exp/m2/space_voices}
STUDENTS=${STUDENTS:-"nc6 cc6"}
DECS=${DECS:-"mimi keep4k ja_release soup12hp M2CH2_soup4 cc_release"}
N=${N:-200}
# wait until every cell of the students has all 200 clips
for s in $STUDENTS; do for d in $DECS; do
  until [ "$(ls $OUT/tts/${s}__$d 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 30; done
done; done
dirs=(); names=()
for s in $STUDENTS; do for d in $DECS; do dirs+=("$OUT/tts/${s}__$d"); names+=("${s}__$d"); done; done
$UV run --python 3.13 --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
  python scripts/asr_transcribe.py "${dirs[@]}" --skip-text > $OUT/asr_${STUDENTS// /_}.log 2>&1 || { tail -5 $OUT/asr_${STUDENTS// /_}.log; exit 1; }
echo "$(date) ASR done"
flock runs/.mos.lock $UV run python scripts/m2/mos_eval.py --highpass 40 --workers 12 $OUT/tts "${names[@]}" > $OUT/mos_tts_${STUDENTS// /_}.log 2>&1
echo "$(date) MOS tts done"
[ -d $OUT/recon ] && flock runs/.mos.lock $UV run python scripts/m2/mos_eval.py --highpass 40 --workers 12 $OUT/recon orig $DECS > $OUT/mos_recon.log 2>&1
echo "$(date) MOS recon done"
echo SPACE_VOICES_SCORE_DONE
