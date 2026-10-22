"""Pauses inside the speech: a student that stalls at punctuation (user's listening of P2 60k:
jvs031's "一方で、漁業と商業で、…" lasts 8 s against the teacher's 4.9 s).

Per clip, 20 ms frames are silent when their energy is more than --floor-db below the clip's 95th
percentile frame; the speech runs from the first to the last loud frame, and a pause is a run of
silent frames of --min-ms or more inside it. Per result: the mean speech duration and its ratio
to the first result's on the same items, the mean total and longest pause per clip, and the share
of clips with a pause of 1 s or more.

    uv run python scripts/eval/pauses.py exp/irodori_distill/eval/jvs_bal teacher release P2_60k P2_60k_cfg1.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_MS = 20


def pauses(path: Path, floor_db: float, min_frames: int) -> tuple[float, list[float]]:
    """(speech duration in s, pause lengths in s)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop = sr * FRAME_MS // 1000
    n = len(x) // hop
    if n == 0:
        return 0.0, []
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = e > np.percentile(e, 95) - floor_db
    idx = np.flatnonzero(loud)
    if len(idx) == 0:
        return 0.0, []
    a, b = idx[0], idx[-1] + 1
    out, run = [], 0
    for v in loud[a:b]:
        if v:
            if run >= min_frames:
                out.append(run * FRAME_MS / 1000)
            run = 0
        else:
            run += 1
    return (b - a) * FRAME_MS / 1000, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("names", nargs="+", help="result directories; the first is the reference for the ratio")
    parser.add_argument("--floor-db", type=float, default=35.0)
    parser.add_argument("--min-ms", type=int, default=150)
    args = parser.parse_args()

    keys = [json.loads(line)["key"] for line in open(args.eval_dir / "items.jsonl", encoding="utf-8")]
    keys = [k for k in keys if all((args.eval_dir / n / f"{k}.wav").exists() for n in args.names)]
    stats = {n: [pauses(args.eval_dir / n / f"{k}.wav", args.floor_db, args.min_ms // FRAME_MS) for k in keys] for n in args.names}
    ref = np.array([d for d, _ in stats[args.names[0]]])
    print(f"{len(keys)} clips in every directory; pauses of {args.min_ms} ms or more inside the speech")
    print("| result | speech s | ratio | pause total s | longest s | clips with >= 1 s |\n|---|---:|---:|---:|---:|---:|")
    for n in args.names:
        dur = np.array([d for d, _ in stats[n]])
        total = np.array([sum(p) for _, p in stats[n]])
        longest = np.array([max(p, default=0.0) for _, p in stats[n]])
        print(f"| {n} | {dur.mean():.2f} | {np.mean(dur / np.maximum(ref, 1e-3)):.2f} | {total.mean():.2f} | {longest.mean():.2f}"
              f" | {np.mean(longest >= 1.0):.1%} |")


if __name__ == "__main__":
    main()
