"""M2 data: clips of a studio corpus (a zip on the NAS) as 24 kHz mono wavs with a manifest.

The output has the layout scripts/m2/train_mimi_decoder.py reads with --data NAME=MANIFEST:AUDIO_DIR:

    <out>/manifest.jsonl    {"id", "duration", "member"} per clip
    <out>/audio/<id>.wav    24 kHz mono 16-bit (resampled with soxr HQ, level unchanged)

    uv run python scripts/m2/extract_corpus.py --out data/m2/hfc_f \
        --zip /mnt/datasets/speech/free_distribute/hi-fi-captain/origin/hfc_ja-JP_F.zip \
        --prefix hi-fi-captain/ja-JP/female/wav/train --id-prefix hfcf_

--count picks that many members at random (seeded); without it every member under --prefix is
written. Clips outside --seconds are skipped. Existing wavs are kept (the run can be resumed).
"""

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 24000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--prefix", default="", help="only members under this path")
    parser.add_argument("--exclude", action="append", default=[], help="skip members containing this text")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-prefix", default="")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seconds", type=float, nargs=2, default=(1.0, 20.0), metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.startswith(args.prefix) and n.lower().endswith(".wav") and not any(x in n for x in args.exclude)
        )
        if args.count:
            names = random.Random(args.seed).sample(names, min(args.count, len(names)))
        for name in names:
            utt_id = args.id_prefix + Path(name).stem
            path = audio / f"{utt_id}.wav"
            if path.exists():
                rows.append({"id": utt_id, "duration": sf.info(str(path)).duration, "member": name})
                continue
            wav, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            duration = len(wav) / sr
            if not (args.seconds[0] <= duration <= args.seconds[1]):
                continue
            if sr != SR:
                wav = soxr.resample(wav, sr, SR, quality="HQ").astype(np.float32)
            sf.write(str(path), np.clip(wav, -1.0, 1.0), SR, subtype="PCM_16")
            rows.append({"id": utt_id, "duration": round(len(wav) / SR, 3), "member": name})
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
