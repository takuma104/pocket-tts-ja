#!/usr/bin/env bash
# S4 (short sentences) on galleria2, from the repo root:  nohup scripts/synth/run_s4.sh > data/synth/s4/run.log 2>&1 &
# Waits for the S3 post-processing, lets the current YODAS3 shard finish and holds the shard
# runner until S4 is encoded (the GPU is shared with the Emilia ASR recheck and the evals),
# then QCs (Sentence Kana-CER only) and encodes.
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
IRODORI_ENV=$HOME/co/pocket-tts-ja-synth-env/Irodori-TTS
mkdir -p data/synth/s4
until [ -f data/synth/s3/POST_DONE ]; do sleep 120; done
runner="yodas3/run_""shards.sh"; shard="yodas3/process_""shard.py"
pkill -f "$runner"
while pgrep -f "$shard" >/dev/null; do sleep 60; done
echo "$(date) YODAS3 runner held; synthesizing S4"
env PYTHONPATH=$IRODORI_ENV $UV run --project $IRODORI_ENV --no-sync python scripts/synth/irodori_synth.py \
  --input data/synth/s4_input.jsonl --output-dir data/synth/s4/audio --log-dir data/synth/s4/logs \
  --checkpoint $HOME/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors --max-batch-frames 4096 \
  >> data/synth/s4/synth.log 2>&1
echo "$(date) synthesis exit $?: $(ls data/synth/s4/audio | wc -l) clips"
$UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py data/synth/s4/audio --skip-text || exit 1
$UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc --dir data/synth/s4 \
  --texts data/synth/short_texts.jsonl --tagged "" || exit 1
$UV run python scripts/synth/finalize_s3.py --encode --dir data/synth/s4 --texts data/synth/short_texts.jsonl \
  --input data/synth/s4_input.jsonl || exit 1
touch data/synth/s4/POST_DONE
echo "$(date) S4 done"
rm -f data/yodas3/.claim_*
nohup scripts/yodas3/run_shards.sh data/yodas3 --workers 8 --batch 16 --max-padded-sec 60 \
  --tts-blocklist data/yodas3/tts_blocklist.npy > /dev/null 2>&1 < /dev/null &
echo "$(date) YODAS3 runner resumed"
