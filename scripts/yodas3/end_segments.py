"""Share of training clips that end in a short sound after a pause (from end_levels.py output).

A frame is voiced when it is within --quiet-db of the clip's loud level (95th percentile; the same
rule as trim_tails.py). From the end of the fed clip: skip the quiet frames, take the last voiced
run (its length L, letting dips of one frame through) and the quiet run before it (G). A clip ends
in a short sound after a pause when G >= --min-gap and L <= --max-seg: "…ですね。[pause] うん", or
a hum, which the kana-whisper recheck does not always write down.

    uv run python scripts/yodas3/end_segments.py yodas3=exp/end_levels/yodas3.jsonl:data/yodas3_kept/train_punct_trim_rechecked.jsonl ...

--drop-out writes the ids of the flagged clips (one per line) for the first set.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

BINS = ((0, 10), (10, 20), (20, 40), (40, 10_000))  # transcript characters


def end_segment(tail: list[int], p95: float, quiet_db: float) -> tuple[int, int]:
    """(L, G): frames of the last voiced run and of the quiet run before it."""
    voiced = [x >= p95 + quiet_db for x in tail]
    for k in range(1, len(voiced) - 1):  # a one-frame dip inside a sound is not a pause
        if not voiced[k] and voiced[k - 1] and voiced[k + 1]:
            voiced[k] = True
    i = len(voiced) - 1
    while i >= 0 and not voiced[i]:
        i -= 1
    end = i
    while i >= 0 and voiced[i]:
        i -= 1
    seg = end - i
    gap = 0
    while i >= 0 and not voiced[i]:
        gap += 1
        i -= 1
    if i < 0:  # the whole window is the segment or silence: no pause before it in view
        gap = 0
    return seg, gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="NAME=LEVELS:MANIFEST")
    parser.add_argument("--quiet-db", type=float, default=-30.0)
    parser.add_argument("--min-gap", type=int, default=3)
    parser.add_argument("--max-seg", type=int, default=6)
    parser.add_argument("--drop-out", type=Path, default=None)
    args = parser.parse_args()

    head = " | ".join(f"{lo}-{hi if hi < 10_000 else ''} chars" for lo, hi in BINS)
    print(f"clips ending in <= {args.max_seg} voiced frames after >= {args.min_gap} quiet frames "
          f"(voiced: within {-args.quiet_db:.0f} dB of the loud level)\n")
    print(f"| set | clips | all | {head} |")
    print("|---|---:|---:|" + "---:|" * len(BINS))
    for n, spec in enumerate(args.sets):
        name, rest = spec.split("=", 1)
        levels, manifest = rest.split(":")
        lv = {(r := json.loads(line))["id"]: r for line in open(levels)}
        chars = {}
        for line in open(manifest):
            r = json.loads(line)
            if r["id"] in lv:
                chars[r["id"]] = len(re.sub(r"[\s.,!?、。．，！？]", "", r["transcript"]))
        flagged = []
        stats: list[tuple[int, bool]] = []
        for cid, r in lv.items():
            seg, gap = end_segment(r["tail"], r["p95"], args.quiet_db)
            f = gap >= args.min_gap and seg <= args.max_seg
            stats.append((chars.get(cid, 0), f))
            if f:
                flagged.append(cid)
        cells = []
        for lo, hi in BINS:
            sel = [f for c, f in stats if lo <= c < hi]
            cells.append(f"{100 * np.mean(sel):.1f}% ({len(sel)})" if sel else "–")
        print(f"| {name} | {len(stats)} | {100 * np.mean([f for _, f in stats]):.1f}% | " + " | ".join(cells) + " |")
        if n == 0 and args.drop_out:
            args.drop_out.write_text("".join(c + "\n" for c in flagged))
            print(f"    wrote {len(flagged)} ids to {args.drop_out}")


if __name__ == "__main__":
    main()
