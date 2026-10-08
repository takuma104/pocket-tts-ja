#!/usr/bin/env bash
# M0 evaluation after scripts/m0_mimi_recon.py: ASR on JVS orig/recon, JKYB on the
# reconstructed benchmark audio, then scoring. Uses the benchmark's own environment.
set -euo pipefail
ROOT=${1:-exp/m0_mimi_recon}
JKYB=third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition
RUN="uv run --project $JKYB --extra asr"
$RUN python scripts/asr_transcribe.py "$ROOT/jvs/orig" "$ROOT/jvs/recon"
$RUN jkyb-eval tts "$ROOT/jkyb/recon" --device cuda --output-dir "$ROOT/jkyb/results"
$RUN python scripts/m0_score.py "$ROOT"
