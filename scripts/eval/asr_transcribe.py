"""Transcribe every wav in a directory with the JKYB-Parakeet ASR setup.

Runs in the benchmark's own environment (it pins transformers<5):

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/asr_transcribe.py exp/irodori_distill/p0/recon_jvs/mimi

Writes <dir>/../asr_<dirname>/{kana,text}.jsonl with the same models, batch size and
max_new_tokens as `jkyb-eval tts`, so CERs computed from them match the benchmark's.
Transcripts are cached: rerunning only transcribes missing files.
"""

import argparse
from pathlib import Path

from jkyb_eval.asr import AsrConfiguration, transcribe_audio
from jkyb_eval.constants import (
    ASR_MAX_NEW_TOKENS,
    DEFAULT_ASR_BATCH_SIZE,
    DEFAULT_KANA_MODEL,
    DEFAULT_TEXT_MODEL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dirs", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ASR_BATCH_SIZE)
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    jobs = [("kana", "yomi", DEFAULT_KANA_MODEL)]
    if not args.skip_text:
        jobs.append(("text", "text", DEFAULT_TEXT_MODEL))
    for name, field, model in jobs:
        config = AsrConfiguration(
            model=model,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=ASR_MAX_NEW_TOKENS,
        )
        for audio_dir in args.audio_dirs:
            paths = sorted(audio_dir.glob("*.wav"))
            out = audio_dir.parent / f"asr_{audio_dir.name}" / f"{name}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"{model}: {len(paths)} files in {audio_dir} -> {out}", flush=True)
            transcribe_audio(
                audio_paths=paths, output_path=out, value_field=field, configuration=config
            )


if __name__ == "__main__":
    main()
