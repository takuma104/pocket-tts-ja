"""Break a `short` eval (scripts/eval_tts.py --set short) down by token count and kind.

Per group: rows, rows that ran to the length budget without EOS, rows whose kana-whisper
reading differs from the text's, rows with extra speech after the text (kana left over after
the last reference kana), and the mean Kana-CER.

    uv run python scripts/analyze_short_eval.py teacher=runs/B1_4k/eval/step00660000/short_rotation_cfg2.0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana


def trailing_extra(ref: str, hyp: str) -> int:
    """Kana of the prediction after the last reference kana in a minimum edit alignment."""
    ops = Levenshtein.editops(ref, hyp)
    extra = 0
    for op in reversed(ops):
        if op.tag == "insert" and op.src_pos == len(ref) and op.dest_pos == len(hyp) - 1 - extra:
            extra += 1
        else:
            break
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="NAME=result directory")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/short_ja.jsonl"))
    parser.add_argument("--examples", type=int, default=0, help="print this many wrong rows per result")
    args = parser.parse_args()

    rows = {json.loads(line)["key"]: json.loads(line) for line in open(args.dataset)}
    for spec in args.results:
        name, path = spec.split("=", 1)
        result = Path(path)
        gen = {json.loads(line)["key"]: json.loads(line) for line in open(result / "generation.jsonl")}
        asr = {json.loads(line)["key"]: json.loads(line)["yomi"] for line in open(result / "asr_audio" / "kana.jsonl")}
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        wrong = []
        for key, row in rows.items():
            ref, hyp = canonical_kana(row["kana"]), canonical_kana(asr[key])
            cer = min(1.0, Levenshtein.distance(ref, hyp) / max(len(ref), 1))
            stat = {
                "no_eos": float(gen[key]["no_eos"]),
                "wrong": float(ref != hyp),
                "extra": float(trailing_extra(ref, hyp) >= 2),
                "cer": cer,
            }
            bucket = "2-3" if row["tokens"] <= 3 else "8+" if row["tokens"] >= 8 else str(row["tokens"])
            for group in ("all", f"kind {row['kind']}", f"tokens {bucket}"):
                groups[group].append(stat)
            if ref != hyp:
                wrong.append(f"    {row['text']}  ->  {asr[key]}" + ("  [no EOS]" if gen[key]["no_eos"] else ""))
        print(f"### {name}\n\n| group | rows | no EOS | wrong | extra speech | Kana-CER |\n|---|---:|---:|---:|---:|---:|")
        for group in sorted(groups, key=lambda g: (g != "all", g)):
            stats = groups[group]
            n = len(stats)
            total = {k: sum(s[k] for s in stats) for k in stats[0]}
            print(f"| {group} | {n} | {total['no_eos']:.0f} | {total['wrong']:.0f} | {total['extra']:.0f} | {total['cer'] / n * 100:.1f}% |")
        print("\n".join(wrong[: args.examples]) + "\n")


if __name__ == "__main__":
    main()
