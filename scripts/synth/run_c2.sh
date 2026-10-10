#!/usr/bin/env bash
# C2 hard-mining synthesis on galleria2, from the repo root:
#   nohup scripts/synth/run_c2.sh > data/synth/c2_run.log 2>&1 &
# Waits for the Emilia ASR recheck to free its GPU memory, lets the current YODAS3 shard finish
# and holds the shard runner (the GPU is shared with the evals). Then:
#   pilot  ~1,000 S5 sentences with the misread words in hiragana and in katakana; the QC pass
#          rate picks the kana form for everything below
#   s6a    S6 sentences that bring every kanji reading towards 100 exposures (build_s6_texts.py)
#   s5     the S3 sentences whose target the teacher misread (make_s5_input.py)
#   s6b    the rest of S6 (towards 300 exposures)
# in that order of expected gain per GPU hour. Each set is synthesized, QC'd (targets exact and
# Sentence Kana-CER <= 0.08 against the original yomi) and encoded; <set>/POST_DONE marks it
# ready to fetch. YODAS3 resumes at the end.
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
CKPT=$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
D=data/synth
recheck="recheck_""asr.py"; runner="yodas3/run_""shards.sh"; shard="yodas3/process_""shard.py"
while pgrep -f "$recheck" >/dev/null; do sleep 120; done
pkill -f "$runner"
while pgrep -f "$shard" >/dev/null; do sleep 60; done
echo "$(date) recheck finished, YODAS3 runner held"

synth() {  # input jsonl, set dir; re-run after a crash (the GPU may be full for a while) until it runs to the end
  local rows; rows=$(wc -l < "$1")
  mkdir -p "$2/audio" "$2/logs"
  for try in $(seq 1 40); do
    rm -f "$2/logs/summary_shard0.json"
    env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
      --input "$1" --output-dir "$2/audio" --log-dir "$2/logs" --checkpoint $CKPT --max-batch-frames 4096 \
      >> "$2/synth.log" 2>&1
    # the summary is written when the run reaches the end; rows that failed are logged and stay missing
    [ -f "$2/logs/summary_shard0.json" ] && return 0
    echo "$(date) $2: try $try stopped at $(ls "$2/audio" | wc -l) of $rows clips"; sleep 300
  done
  return 1
}
post_qc() {  # set dir, texts, tagged file names: kana-whisper transcripts, then the QC
  for try in $(seq 1 12); do
    $UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py "$1/audio" --skip-text && break
    sleep 300
  done
  $UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir "$1" \
    --texts "$2" --tagged "$3"
}
accepted() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['accepted'])" "$1/qc_stats.json"; }
run_set() {  # name, input, texts, tagged file names
  synth "$2" $D/$1 || { echo "$(date) $1 synthesis kept failing"; return 1; }
  echo "$(date) $1 synthesized: $(ls $D/$1/audio | wc -l) clips"
  post_qc $D/$1 "$3" "$4" > $D/$1/post.log 2>&1 || { echo "$(date) $1 QC failed"; return 1; }
  for try in 1 2 3 4; do
    $UV run python scripts/synth/finalize_s3.py --encode --dir $D/$1 --texts "$3" --input "$2" >> $D/$1/post.log 2>&1 && break
    sleep 300
  done
  [ -f $D/$1/manifest.jsonl ] || { echo "$(date) $1 encode failed"; return 1; }
  touch $D/$1/POST_DONE
  echo "$(date) $1 done: $(tr -d '\n ' < $D/$1/qc_stats.json)"
}
S1_TAGGED=texts_tagged_jkyb.jsonl,texts_tagged_jkyb_extra.jsonl
S6_TAGGED=s6_tagged_jkyb.jsonl,s6_tagged_jkyb_extra.jsonl

for k in hira kata; do
  synth $D/s5_pilot_$k.jsonl $D/s5_pilot_$k || exit 1
  post_qc $D/s5_pilot_$k $D/texts.jsonl $S1_TAGGED > $D/s5_pilot_$k/post.log 2>&1 || exit 1
done
h=$(accepted $D/s5_pilot_hira); k=$(accepted $D/s5_pilot_kata)
form=hira; [ "$k" -gt "$h" ] && form=kata
echo "$(date) pilot accepted of $(wc -l < $D/s5_pilot_hira.jsonl): hiragana $h, katakana $k -> $form"
echo $form > $D/c2_kana_form

run_set s6a $D/s6a_input_$form.jsonl $D/s6_texts.jsonl $S6_TAGGED
mkdir -p $D/s5/audio && cp -n $D/s5_pilot_$form/audio/*.wav $D/s5/audio/
run_set s5 $D/s5_input_$form.jsonl $D/texts.jsonl $S1_TAGGED
run_set s6b $D/s6b_input_$form.jsonl $D/s6_texts.jsonl $S6_TAGGED

rm -f data/yodas3/.claim_*
nohup scripts/yodas3/run_shards.sh data/yodas3 --workers 8 --batch 16 --max-padded-sec 60 \
  --tts-blocklist data/yodas3/tts_blocklist.npy > /dev/null 2>&1 < /dev/null &
echo "$(date) YODAS3 runner resumed; C2_DONE"
