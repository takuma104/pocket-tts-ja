#!/usr/bin/env bash
# Wait for S3 synthesis to finish, then QC (kana-whisper) and encode the accepted clips.
# Runs on galleria2 from the repo root:  nohup scripts/synth/run_s3_post.sh > data/synth/s3/post.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
pat="synth/irodori_""synth.py"
while pgrep -f "$pat" >/dev/null; do sleep 120; done
echo "$(date) synthesis finished: $(ls data/synth/s3/audio | wc -l) clips"
$UV run --python 3.13 --project $JKYB --extra asr python scripts/asr_transcribe.py data/synth/s3/audio --skip-text || exit 1
echo "$(date) ASR done"
$UV run --python 3.13 --project $JKYB --extra asr python scripts/synth/finalize_s3.py --qc || exit 1
echo "$(date) QC done"
$UV run python scripts/synth/finalize_s3.py --encode || exit 1
echo "$(date) encode done"
touch data/synth/s3/POST_DONE
