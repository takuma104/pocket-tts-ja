"""P1 QC by agreement: is the regenerated clip the one main's QC accepted?

Every row of rows_<set>.jsonl was accepted on main (every target reading exact, sentence
Kana-CER <= 0.08) with the kana-whisper transcript it carries as ``old_kana``. The clip is
regenerated with the same text, reference and seed, so it should be the same clip up to the
numerics of the batch. A row passes when the new transcript (asr_transcribe.py on the 16 kHz QC
audio) is within --max-cer of the old one: it then inherits main's QC. The others (a different
clip came out) are dropped rather than re-checked.

    uv run python scripts/teacher/qc.py data/irodori_distill/teacher s3
    -> <root>/<set>/qc.jsonl (id, cer_old, pass) and a summary
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("set")
    parser.add_argument("--max-cer", type=float, default=0.02)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.root / f"rows_{args.set}.jsonl", encoding="utf-8")]
    d = args.root / args.set
    asr = {r["key"]: r["yomi"] for r in map(json.loads, open(d / "asr_audio16k" / "kana.jsonl", encoding="utf-8"))}
    cers: list[float] = []
    missing = 0
    with open(d / "qc.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] not in asr:
                missing += 1
                continue
            old = canonical_kana(r["old_kana"])
            new = canonical_kana(asr[r["id"]])
            cer = Levenshtein.distance(old, new) / max(len(old), 1)
            cers.append(cer)
            f.write(json.dumps({"id": r["id"], "cer_old": round(cer, 4), "pass": cer <= args.max_cer}) + "\n")
    c = np.array(cers)
    print(f"{args.set}: {len(rows)} rows, {missing} without a transcript (not generated)")
    print(f"  CER vs the accepted transcript: exact {np.mean(c == 0):.1%}, <=0.02 {np.mean(c <= 0.02):.1%}, "
          f"<=0.05 {np.mean(c <= 0.05):.1%}, <=0.08 {np.mean(c <= 0.08):.1%}, mean {c.mean():.4f}")
    print(f"  pass (<= {args.max_cer}): {int(np.sum(c <= args.max_cer))} of {len(c)}")


if __name__ == "__main__":
    main()
