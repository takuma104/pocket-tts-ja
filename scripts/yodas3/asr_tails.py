"""Share of training clips whose kana-whisper transcript goes on after the end of the label.

    uv run python scripts/yodas3/asr_tails.py NAME=MANIFEST:RECHECK[:N] ...

Each clip's label reading (pyopenjtalk, canonical kana) is aligned to its ASR transcript; kana
the ASR heard after the last label kana are the clip's tail. Fillers are kept (the trailing
"ン" / "エー" the model produces are what is measured). Transcripts at the ASR token limit are
skipped.
"""

import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pocket_tts_ja.kana import canonical_kana, reading  # noqa: E402

TRUNCATED_AT = 115


def tail(ref: str, hyp: str) -> str:
    """ASR kana after the last label kana (trailing inserts of the alignment)."""
    ops = list(Levenshtein.opcodes(ref, hyp))
    out = ""
    for o in reversed(ops):
        if o.tag == "insert" and o.src_start == len(ref):
            out = hyp[o.dest_start : o.dest_end] + out
        else:
            break
    return out


def main() -> None:
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        manifest, recheck = Path(parts[0]), Path(parts[1])
        n = int(parts[2]) if len(parts) > 2 else 3000
        hyps = {(r := json.loads(line))["id"]: r["kana"] for line in open(recheck)}
        rows = [r for line in open(manifest) if (r := json.loads(line))["id"] in hyps]
        rows = [r for r in rows if len(hyps[r["id"]]) < TRUNCATED_AT]
        random.seed(0)
        rows = random.sample(rows, min(n, len(rows)))
        with Pool(16) as pool:
            refs = pool.map(reading, [r["transcript"] for r in rows], chunksize=64)
        tails = [tail(canonical_kana(ref), canonical_kana(hyps[r["id"]])) for r, ref in zip(rows, refs, strict=True)]
        has = [t for t in tails if t]
        print(f"{name}: {len(rows)} clips, tail >= 1 kana {100 * len(has) / len(rows):.1f}%, "
              f">= 2 kana {100 * sum(len(t) >= 2 for t in tails) / len(rows):.1f}%")
        print("   most common tails:", Counter(has).most_common(15))


if __name__ == "__main__":
    main()
