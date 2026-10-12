#!/usr/bin/env bash
# S9 (very short utterances) on galleria2, from the repo root:
#   nohup scripts/synth/run_s9.sh > data/synth/s9_run.log 2>&1 &
# Builds the texts (scripts/synth_text/build_s9_texts.py; each row carries its clip length),
# holds the YODAS3 shard runner (the current shard runs on), synthesizes, QCs (the kana-whisper
# reading must equal the yomi) and encodes (clips under 7 frames are continued with silence:
# the stock inference ignores an EOS before frame 6); s9/POST_DONE marks the set ready to fetch.
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
CKPT=$HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
D=data/synth
S=$D/s9
mkdir -p $S/audio $S/logs
[ -f $D/s9_texts.jsonl ] || $UV run python scripts/synth_text/build_s9_texts.py --out $D/s9_texts.jsonl || exit 1
python3 scripts/synth/make_s3_input.py $D/s9_texts.jsonl $D/refs.jsonl $D/s9_input.jsonl || exit 1
runner="yodas3/run_""shards.sh"
pkill -f "$runner"
echo "$(date) s9: $(wc -l < $D/s9_input.jsonl) rows, YODAS3 runner held"
for try in $(seq 1 40); do
  rm -f $S/logs/summary_shard0.json
  env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
    --input $D/s9_input.jsonl --output-dir $S/audio --log-dir $S/logs --checkpoint $CKPT --max-batch-frames 4096 \
    --min-seconds 0.2 >> $S/synth.log 2>&1
  [ -f $S/logs/summary_shard0.json ] && break  # ran to the end; rows that failed stay missing
  echo "$(date) s9: try $try stopped at $(ls $S/audio | wc -l) clips"; sleep 300
done
echo "$(date) s9 synthesized: $(ls $S/audio | wc -l) clips"
for try in $(seq 1 12); do
  $UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py $S/audio --skip-text > $S/post.log 2>&1 && break
  sleep 300
done
$UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir $S \
  --texts $D/s9_texts.jsonl --tagged "" --max-sentence-cer 0.0 >> $S/post.log 2>&1 || { echo "$(date) s9 QC failed"; exit 1; }
for try in $(seq 1 4); do
  $UV run python scripts/synth/finalize_s3.py --encode --dir $S --texts $D/s9_texts.jsonl --input $D/s9_input.jsonl --min-frames 7 >> $S/post.log 2>&1 && break
  sleep 300
done
[ -f $S/manifest.jsonl ] || { echo "$(date) s9 encode failed"; exit 1; }
touch $S/POST_DONE
echo "$(date) s9 done: $(tr -d '\n ' < $S/qc_stats.json)"
rm -f data/yodas3/.claim_*
nohup scripts/yodas3/run_shards.sh data/yodas3 --workers 8 --batch 16 --max-padded-sec 60 \
  --tts-blocklist data/yodas3/tts_blocklist.npy > /dev/null 2>&1 < /dev/null &
echo "$(date) YODAS3 runner resumed"
