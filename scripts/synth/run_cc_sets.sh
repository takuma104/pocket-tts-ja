#!/usr/bin/env bash
# CC-BY line (plan B-CC) on galleria2: the accepted NC-voice sentences of every synthetic set again
# with CC BY voices (inputs from make_cc_input.py), from the repo root:
#   nohup setsid scripts/synth/run_cc_sets.sh > data/synth/cc_run.log 2>&1 < /dev/null &
# Sets run in the order given (default: the small ones that carry the short texts and the hard
# readings first). Each is synthesized (an Irodori run takes ~7 GB), QC'd and encoded like the
# original set; <set>_cc/POST_DONE marks it ready to fetch. Start it again after a stop: finished
# sets are skipped and an interrupted synthesis continues from the clips that exist.
set -u
cd "$(dirname "$0")/../.."
echo $$ > data/synth/cc_run.pid
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
CKPT=$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
D=data/synth
LOCK="flock runs/extra_gpu.lock"
S1_TAGGED=texts_tagged_jkyb.jsonl,texts_tagged_jkyb_extra.jsonl
S6_TAGGED=s6_tagged_jkyb.jsonl,s6_tagged_jkyb_extra.jsonl

run_set() {  # name, texts, tagged file names, extra QC options, extra encode options
  local S=$D/$1_cc INPUT=$D/$1_cc_input.jsonl
  [ -f $S/POST_DONE ] && return 0
  mkdir -p $S/audio $S/logs
  echo "$(date) $1_cc: $(wc -l < $INPUT) rows"
  for try in $(seq 1 40); do
    rm -f $S/logs/summary_shard0.json
    env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
      --input $INPUT --output-dir $S/audio --log-dir $S/logs --checkpoint $CKPT --max-batch-frames 4096 \
      >> $S/synth.log 2>&1 < /dev/null
    [ -f $S/logs/summary_shard0.json ] && break  # ran to the end; rows that failed stay missing
    echo "$(date) $1_cc: try $try stopped at $(ls $S/audio | wc -l) clips"; sleep 300
  done
  echo "$(date) $1_cc synthesized: $(ls $S/audio | wc -l) clips"
  for try in $(seq 1 12); do
    $LOCK $UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py $S/audio --skip-text \
      > $S/post.log 2>&1 < /dev/null && break
    sleep 300
  done
  # shellcheck disable=SC2086
  $UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir $S \
    --texts $2 --tagged "$3" $4 >> $S/post.log 2>&1 || { echo "$(date) $1_cc QC failed"; return 1; }
  for try in 1 2 3 4; do
    # shellcheck disable=SC2086
    $LOCK $UV run python scripts/synth/finalize_s3.py --encode --dir $S --texts $2 --input $INPUT $5 \
      >> $S/post.log 2>&1 < /dev/null && break
    sleep 300
  done
  [ -f $S/manifest.jsonl ] || { echo "$(date) $1_cc encode failed"; return 1; }
  touch $S/POST_DONE
  echo "$(date) $1_cc done: $(tr -d '\n ' < $S/qc_stats.json | cut -c1-400)"
}

for name in ${*:-s9 s4 s7 s6a s5 s6b s3}; do
  case $name in
    s3) run_set s3 $D/texts.jsonl $S1_TAGGED "" "" ;;
    s4) run_set s4 $D/short_texts.jsonl "" "" "" ;;
    s5) run_set s5 $D/texts.jsonl $S1_TAGGED "" "" ;;
    s6a | s6b) run_set $name $D/s6_texts.jsonl $S6_TAGGED "" "" ;;
    s7) run_set s7 $D/s7_texts.jsonl s7_tagged_jkyb.jsonl "" "" ;;
    s9) run_set s9 $D/s9_texts.jsonl "" "--max-sentence-cer 0.0" "--min-frames 7" ;;
    *) echo "unknown set $name"; exit 1 ;;
  esac
done
echo "$(date) CC_SETS_DONE"
