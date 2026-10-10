#!/usr/bin/env bash
# S5 (kana repair of the S3 rejects, see make_s5_input.py) on galleria2, from the repo root:
#   nohup scripts/synth/run_s5.sh > data/synth/s5/run.log 2>&1 &
# Waits for the Emilia ASR recheck to free its GPU memory, lets the current YODAS3 shard finish
# and holds the shard runner (the GPU is shared with the evals), pilots hiragana against katakana
# on ~1,000 sentences (the QC pass rate picks the form), then synthesizes, QCs and encodes the
# full set and resumes YODAS3.
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
CKPT=$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
S5=data/synth/s5
mkdir -p $S5
recheck="recheck_""asr.py"; runner="yodas3/run_""shards.sh"; shard="yodas3/process_""shard.py"
while pgrep -f "$recheck" >/dev/null; do sleep 120; done
pkill -f "$runner"
while pgrep -f "$shard" >/dev/null; do sleep 60; done
echo "$(date) recheck finished, YODAS3 runner held"

synth() {  # input jsonl, set dir; re-run until every row has audio (the GPU may be full for a while)
  local rows; rows=$(wc -l < "$1")
  mkdir -p "$2"
  for try in 1 2 3 4 5 6; do
    env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
      --input "$1" --output-dir "$2/audio" --log-dir "$2/logs" --checkpoint $CKPT --max-batch-frames 4096 \
      >> "$2/synth.log" 2>&1
    [ "$(ls "$2/audio" 2>/dev/null | wc -l)" -ge "$rows" ] && return 0
    echo "$(date) $2: try $try left $(( rows - $(ls "$2/audio" 2>/dev/null | wc -l) )) rows"; sleep 300
  done
  return 0  # rows that never synthesize are simply missing from the QC
}
post_qc() {  # set dir: kana-whisper transcripts, then target exactness and Sentence Kana-CER
  for try in 1 2 3 4; do
    $UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py "$1/audio" --skip-text && break
    sleep 300
  done
  $UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir "$1" \
    --texts data/synth/texts.jsonl
}
accepted() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['accepted'])" "$1/qc_stats.json"; }

for k in hira kata; do
  synth data/synth/s5_pilot_$k.jsonl data/synth/s5_pilot_$k
  post_qc data/synth/s5_pilot_$k > data/synth/s5_pilot_$k/post.log 2>&1 || exit 1
done
h=$(accepted data/synth/s5_pilot_hira); k=$(accepted data/synth/s5_pilot_kata)
form=hira; [ "$k" -gt "$h" ] && form=kata
echo "$(date) pilot accepted of $(wc -l < data/synth/s5_pilot_hira.jsonl): hiragana $h, katakana $k -> $form"
cp data/synth/s5_input_$form.jsonl data/synth/s5_input.jsonl
mkdir -p $S5/audio && cp -n data/synth/s5_pilot_$form/audio/*.wav $S5/audio/

synth data/synth/s5_input.jsonl $S5
echo "$(date) synthesis done: $(ls $S5/audio | wc -l) clips"
post_qc $S5 > $S5/post.log 2>&1 || exit 1
$UV run python scripts/synth/finalize_s3.py --encode --dir $S5 --texts data/synth/texts.jsonl \
  --input data/synth/s5_input.jsonl >> $S5/post.log 2>&1 || exit 1
touch $S5/POST_DONE
echo "$(date) S5 done: $(cat $S5/qc_stats.json | tr -d '\n ')"
rm -f data/yodas3/.claim_*
nohup scripts/yodas3/run_shards.sh data/yodas3 --workers 8 --batch 16 --max-padded-sec 60 \
  --tts-blocklist data/yodas3/tts_blocklist.npy > /dev/null 2>&1 < /dev/null &
echo "$(date) YODAS3 runner resumed"
