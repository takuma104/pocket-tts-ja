"""P1 QC by agreement: is the regenerated clip the one main's QC accepted?

Every row of rows_<set>.jsonl was accepted on main (every target reading exact, sentence
Kana-CER <= 0.08) with the kana-whisper transcript it carries as ``old_kana``. The clip is
regenerated with the same text, reference and seed, so it should be the same clip up to the
numerics of the batch. A row passes when the new transcript (asr_transcribe.py on the 16 kHz QC
audio) is within --max-cer of the old one: it then inherits main's QC. The others (a different
clip came out) are dropped rather than re-checked.

--alt-field FIELD also passes a row whose transcript is within --alt-max-cer of the row's FIELD
(latin: ``rec_kana``, the reading of the original recording, for the words pyopenjtalk spells
letter by letter in ``old_kana``). --spell-letters turns the Latin letters pyopenjtalk leaves in
``old_kana`` as they are (N, A, E1) into their names (エヌ, エー) before the comparison.

    uv run python scripts/teacher/qc.py data/irodori_distill/teacher s3
    -> <root>/<set>/qc.jsonl (id, cer_old, pass) and a summary
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana

LETTER_NAMES = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "エー ビー シー ディー イー エフ ジー エイチ アイ ジェー ケー エル エム "
                        "エヌ オー ピー キュー アール エス ティー ユー ブイ ダブリュー エックス ワイ ゼット".split()))


def spell_letters(kana: str) -> str:
    return re.sub(r"[A-Za-z]", lambda m: LETTER_NAMES[m.group().upper()], kana)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("set")
    parser.add_argument("--max-cer", type=float, default=0.02)
    parser.add_argument("--alt-field", default=None, help="a second reference reading in the rows")
    parser.add_argument("--alt-max-cer", type=float, default=0.08)
    parser.add_argument("--spell-letters", action="store_true", help="read the Latin letters left in old_kana")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.root / f"rows_{args.set}.jsonl", encoding="utf-8")]
    d = args.root / args.set
    asr = {r["key"]: r["yomi"] for r in map(json.loads, open(d / "asr_audio16k" / "kana.jsonl", encoding="utf-8"))}
    cers: list[float] = []
    passed = alt_only = missing = 0
    with open(d / "qc.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] not in asr:
                missing += 1
                continue
            old = canonical_kana(spell_letters(r["old_kana"]) if args.spell_letters else r["old_kana"])
            new = canonical_kana(asr[r["id"]])
            cer = Levenshtein.distance(old, new) / max(len(old), 1)
            cers.append(cer)
            qc = {"id": r["id"], "cer_old": round(cer, 4), "pass": cer <= args.max_cer}
            if args.alt_field and r.get(args.alt_field):
                alt = canonical_kana(r[args.alt_field])
                qc["cer_alt"] = round(Levenshtein.distance(alt, new) / max(len(alt), 1), 4)
                if not qc["pass"] and qc["cer_alt"] <= args.alt_max_cer:
                    qc["pass"] = True
                    alt_only += 1
            passed += qc["pass"]
            f.write(json.dumps(qc) + "\n")
    c = np.array(cers)
    print(f"{args.set}: {len(rows)} rows, {missing} without a transcript (not generated)")
    print(f"  CER vs the accepted transcript: exact {np.mean(c == 0):.1%}, <=0.02 {np.mean(c <= 0.02):.1%}, "
          f"<=0.05 {np.mean(c <= 0.05):.1%}, <=0.08 {np.mean(c <= 0.08):.1%}, mean {c.mean():.4f}")
    print(f"  pass (<= {args.max_cer}): {int(np.sum(c <= args.max_cer))} of {len(c)}")
    if args.alt_field:
        print(f"  pass with {args.alt_field} (<= {args.alt_max_cer}) as well: {passed} of {len(c)} ({alt_only} by it alone)")


if __name__ == "__main__":
    main()
