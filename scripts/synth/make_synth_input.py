#!/usr/bin/env python3
"""Convert a jsonl (e.g. JKYB rows) into the ``irodori_synth.py`` input schema.

    python scripts/synth/make_synth_input.py data/eval/jkyb_mini.jsonl out.jsonl \
        --id-field key --ref-wav data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav

Each output row is {"id", "text", "ref_wav", "seed"}; ``--ref-wav`` omitted means
null (no reference). All rows get the same ``--seed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--id-field", default="key")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--ref-wav", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw = json.loads(line)
                rows.append(
                    {
                        "id": raw[args.id_field],
                        "text": raw[args.text_field],
                        "ref_wav": args.ref_wav,
                        "seed": args.seed,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
