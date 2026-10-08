"""S3: assign a reference voice and a seed to every synthetic-corpus sentence.

Each of the S2 reference speakers gets about len(texts) / len(refs) sentences (~100), so a
synthetic speaker label `syn:<ref speaker>` has many utterances to draw voice prompts from.
The assignment and the seed are hashes of the sentence id, so the input is reproducible
and stable when sentences are added.

    python3 scripts/synth/make_s3_input.py data/synth/texts.jsonl data/synth/refs.jsonl \
        data/synth/s3_input.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def h(s: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{s}".encode()).hexdigest(), 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", type=Path)
    parser.add_argument("refs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    refs = [json.loads(line) for line in args.refs.open(encoding="utf-8") if line.strip()]
    refs.sort(key=lambda r: r["speaker"])
    n = 0
    with args.texts.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            t = json.loads(line)
            ref = refs[h(t["id"], "ref") % len(refs)]
            row = {
                "id": t["id"],
                "text": t["text"],
                "ref_wav": f"data/synth/refs/{ref['speaker']}.wav",
                "ref_speaker": ref["speaker"],
                "ref_id": ref["id"],
                "seed": h(t["id"], "seed") % (2**31),
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows over {len(refs)} reference speakers to {args.output}")


if __name__ == "__main__":
    main()
