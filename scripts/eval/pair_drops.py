"""pair_probe (scripts/eval/make_chunk_sets.py): the items whose transcript lost one of the two sentences.

An item counts as lost when its Kana errors exceed --frac of the reference; it is then "first only"
when the transcript is within --frac of the first sentence's reading alone (the student stopped on
EOS after it), "second only" likewise, else "partial". The sentences' readings come from the set
the pairs were made from (--from, the jsut set by default).

    uv run python scripts/eval/pair_drops.py exp/irodori_distill/eval/pair_probe release P3j_20k_wheel
"""

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana


def readings(path: Path) -> dict[str, str]:
    return {it["key"]: canonical_kana(it["yomi"]) for it in map(json.loads, open(path / "items.jsonl", encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the pair_probe set")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--from", dest="source", type=Path, default=None, help="the set of the single sentences")
    parser.add_argument("--frac", type=float, default=0.3)
    args = parser.parse_args()

    single = readings(args.source or args.root.parent / "jsut")
    keys = [it["key"] for it in map(json.loads, open(args.root / "items.jsonl", encoding="utf-8"))]
    print("| result | lost a sentence | first only | second only | partial |")
    print("|---|---:|---:|---:|---:|")
    for name in args.names:
        hyp = {r["key"]: canonical_kana(r["yomi"])
               for r in map(json.loads, open(args.root / f"asr_{name}" / "kana.jsonl", encoding="utf-8"))}
        first = second = partial = 0
        for key in keys:
            ka, suffix = key.split("+")
            a, b = single[ka], single[ka.rsplit("_", 1)[0] + "_" + suffix]
            h = hyp.get(key, "")
            if Levenshtein.distance(a + b, h) <= args.frac * len(a + b):
                continue
            if Levenshtein.distance(a, h) <= args.frac * len(a):
                first += 1
            elif Levenshtein.distance(b, h) <= args.frac * len(b):
                second += 1
            else:
                partial += 1
        lost = first + second + partial
        print(f"| {name} | {lost}/{len(keys)} ({lost / len(keys):.1%}) | {first} | {second} | {partial} |")


if __name__ == "__main__":
    main()
