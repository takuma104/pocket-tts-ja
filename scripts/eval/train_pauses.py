"""Long pauses in a teacher training set, by the voice's caption (hi2) or overall.

S1h / S2h's 6L made female_04's sentences stall (clips with a pause of 1 s or more: S1 60k 5 of 20,
S1h 15k 12, HS_s2h 13; the teacher none). This measures the teacher's own clips the students learn
from: per clip, pauses.py's longest pause inside the speech, on the 16 kHz audio the QC's ASR heard
(audio16k/<id>.wav), for the clips that passed the QC. For hi2, grouped by the caption of the
designed voice (synth_src/hi_design/refs.jsonl). On 10/25 hi2 had 15.5% of its clips with a pause
of 0.7 s or more (hi1 4.9%; the caption 「落ち着いて、丁寧に読み上げている」 21.5%). --long-out writes
the ids of the clips with a pause of --long-s or more, one per line, to drop from a split.

    uv run python scripts/eval/train_pauses.py data/irodori_distill/teacher/hi2 --refs data/irodori_distill/synth_src/hi_design/refs.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pauses import FRAME_MS, pauses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set_dir", type=Path, help="teacher set: manifest.jsonl, qc.jsonl, audio16k/")
    parser.add_argument("--refs", type=Path, help="designed voices (id, caption) to group by")
    parser.add_argument("--limit", type=int, default=0, help="at most this many clips (0: all)")
    parser.add_argument("--long-out", type=Path, help="write the ids of the clips with a pause of --long-s or more")
    parser.add_argument("--long-s", type=float, default=0.7)
    args = parser.parse_args()

    passed = {r["id"] for r in map(json.loads, open(args.set_dir / "qc.jsonl", encoding="utf-8")) if r["pass"]}
    rows = [r for r in map(json.loads, open(args.set_dir / "manifest.jsonl", encoding="utf-8")) if r["id"] in passed]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    caption = {}
    if args.refs:
        caption = {r["id"]: r["caption"] for r in map(json.loads, open(args.refs, encoding="utf-8"))}
    groups: dict[str, list[float]] = defaultdict(list)
    long_ids: list[str] = []
    for r in rows:
        path = args.set_dir / "audio16k" / f"{r['id']}.wav"
        if not path.exists():
            continue
        _, p = pauses(path, 35.0, 150 // FRAME_MS)
        longest = max(p, default=0.0)
        if longest >= args.long_s:
            long_ids.append(r["id"])
        groups[caption.get(r["ref_id"], "all")].append(longest)
        if caption:
            groups["all"].append(longest)
    print(f"{args.set_dir}: clips with a pause of 1 s or more inside the speech (QC passed)")
    print("| group | clips | >= 1 s | >= 0.7 s | mean longest s |\n|---|---:|---:|---:|---:|")
    for g in sorted(groups, key=lambda g: (g != "all", g)):
        x = np.array(groups[g])
        print(f"| {g[:60]} | {len(x)} | {np.mean(x >= 1.0):.1%} | {np.mean(x >= 0.7):.1%} | {x.mean():.2f} |")
    if args.long_out:
        args.long_out.write_text("".join(f"{i}\n" for i in long_ids))
        print(f"{args.long_out}: {len(long_ids)} clips with a pause of {args.long_s} s or more")


if __name__ == "__main__":
    main()
