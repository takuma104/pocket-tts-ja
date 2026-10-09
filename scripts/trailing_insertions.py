"""Count extra speech after the end of the text ("…ハチガツデス、ゲツ") in JKYB eval details.

Aligns each prediction_yomi_normalized to reference_yomi_normalized (Levenshtein) and counts the
predicted kana left over after the last reference kana. Reported by reference length because
listening suggested the problem is mostly in short sentences.

    uv run python scripts/trailing_insertions.py A5_4k=runs/A5_4k/eval/step00015000/jkyb_mini_rotation_cfg2.0
"""

import argparse
import json
from pathlib import Path

import numpy as np

BINS = ((0, 15), (15, 30), (30, 60), (60, 10_000))


def trailing_extra(ref: str, hyp: str) -> int:
    """Predicted characters after the last reference character in a minimum edit alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    # the earliest hyp position where the whole reference can end with the optimal total cost,
    # i.e. trailing hyp characters are pure insertions
    best = d[n, m]
    for j in range(m + 1):
        if d[n, j] + (m - j) == best:
            return m - j
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+", help="LABEL=jkyb eval dir (with results/details/all.jsonl)")
    parser.add_argument("--min-extra", type=int, default=2)
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()
    head = " | ".join(f"ref {lo}-{hi if hi < 10_000 else ''} kana" for lo, hi in BINS)
    print(f"share of sentences with >= {args.min_extra} extra kana after the end of the text\n")
    print(f"| set | all | {head} |")
    print("|---|---:|" + "---:|" * len(BINS))
    for spec in args.sets:
        label, d = spec.split("=", 1)
        rows = [json.loads(line) for line in open(Path(d) / "results" / "details" / "all.jsonl")]
        flags: list[tuple[int, bool, dict]] = []
        for r in rows:
            ref, hyp = r["reference_yomi_normalized"], r.get("prediction_yomi_normalized") or ""
            extra = trailing_extra(ref, hyp)
            flags.append((len(ref), extra >= args.min_extra, r))
        cells = []
        for lo, hi in BINS:
            sel = [f for n, f, _ in flags if lo <= n < hi]
            cells.append(f"{100 * np.mean(sel):.1f}% ({len(sel)})" if sel else "–")
        print(f"| {label} | {100 * np.mean([f for _, f, _ in flags]):.1f}% | " + " | ".join(cells) + " |")
        for _, _, r in [x for x in flags if x[1]][: args.examples]:
            print(f"    {r['text']}  ->  {r.get('prediction_yomi')}")


if __name__ == "__main__":
    main()
