#!/usr/bin/env bash
# One synthetic set on galleria2, alongside run_c2.sh (an Irodori run takes ~7 GB at
# --max-batch-frames 4096). From the repo root:
#   nohup scripts/synth/run_set.sh s7 data/synth/s7_texts.jsonl s7_tagged_jkyb.jsonl > data/synth/s7_run.log 2>&1 &
# Reads data/synth/<name>_input_<form>.jsonl with the kana form the run_c2.sh pilot chose
# (data/synth/c2_kana_form), synthesizes, QCs (targets exact and Sentence Kana-CER <= 0.08) and
# encodes; <name>/POST_DONE marks the set ready to fetch.
set -u
NAME=$1; TEXTS=$2; TAGGED=$3
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
CKPT=$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
D=data/synth
until [ -f $D/c2_kana_form ]; do sleep 60; done
INPUT=$D/${NAME}_input_$(cat $D/c2_kana_form).jsonl
S=$D/$NAME
mkdir -p $S/audio $S/logs
echo "$(date) $NAME: $(wc -l < $INPUT) rows from $INPUT"
for try in $(seq 1 40); do
  rm -f $S/logs/summary_shard0.json
  env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
    --input $INPUT --output-dir $S/audio --log-dir $S/logs --checkpoint $CKPT --max-batch-frames 4096 \
    >> $S/synth.log 2>&1
  [ -f $S/logs/summary_shard0.json ] && break  # ran to the end; rows that failed stay missing
  echo "$(date) $NAME: try $try stopped at $(ls $S/audio | wc -l) clips"; sleep 300
done
echo "$(date) $NAME synthesized: $(ls $S/audio | wc -l) clips"
for try in $(seq 1 12); do
  $UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py $S/audio --skip-text > $S/post.log 2>&1 && break
  sleep 300
done
$UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir $S \
  --texts $TEXTS --tagged $TAGGED >> $S/post.log 2>&1 || { echo "$(date) $NAME QC failed"; exit 1; }
for try in $(seq 1 4); do
  $UV run python scripts/synth/finalize_s3.py --encode --dir $S --texts $TEXTS --input $INPUT >> $S/post.log 2>&1 && break
  sleep 300
done
[ -f $S/manifest.jsonl ] || { echo "$(date) $NAME encode failed"; exit 1; }
touch $S/POST_DONE
echo "$(date) $NAME done: $(tr -d '\n ' < $S/qc_stats.json)"
