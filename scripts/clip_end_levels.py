"""Clip ends of eval_tts.py audio by level, not ASR: short sounds after a pause, and clips cut while sounding.

A clip "ends in a short sound" when its last voiced run (frames within --quiet-db of the clip's
95th-percentile frame level, one-frame dips bridged; scripts/yodas3/end_segments.py) is at most
--max-frames long and follows at least two quiet frames. ASR mostly does not write these sounds
down (breaths, small voices), so they are counted here. "Cut" is the share whose last frame is
still within 30 dB of that level (what too few frames after EOS gives).

    uv run python scripts/clip_end_levels.py D1-CC=runs/D1_cc_6l/eval/step00160000/jkyb_mini_rotation_cfg1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "yodas3"))
from end_segments import end_segment  # noqa: E402

FRAME_RATE = 12.5
WINDOW = 20


def frame_db(path: Path) -> np.ndarray:
    w, sr = sf.read(path)
    hop = int(sr / FRAME_RATE)
    f = len(w) // hop
    return 10 * np.log10((w[: f * hop].reshape(f, hop) ** 2).mean(1) + 1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="LABEL=eval dir (with audio/)")
    parser.add_argument("--quiet-db", type=float, default=-30.0)
    parser.add_argument("--max-frames", type=int, default=6)
    args = parser.parse_args()
    print("| set | clips | short sound at the end | its peak (median, dB) | quiet frames after it: 0 / 1-4 / 5+ | cut |")
    print("|---|---:|---:|---:|---:|---:|")
    for spec in args.sets:
        label, d = spec.split("=", 1)
        n, peaks, after, cut = 0, [], [], 0
        for p in sorted(Path(d, "audio").glob("*.wav")):
            db = frame_db(p)
            ref = float(np.percentile(db, 95))
            tail = [float(x) for x in db[-WINDOW:]]
            n += 1
            cut += tail[-1] > ref - 30
            seg, gap = end_segment(tail, ref, args.quiet_db)
            if gap < 2 or seg > args.max_frames:
                continue
            voiced = np.flatnonzero(np.array(tail) >= ref + args.quiet_db)
            end = int(voiced[-1])
            peaks.append(max(tail[end - seg + 1 : end + 1]) - ref)
            after.append(len(tail) - 1 - end)
        a = np.array(after)
        share = [100 * np.mean(a == 0), 100 * np.mean((a >= 1) & (a <= 4)), 100 * np.mean(a >= 5)] if len(a) else [0, 0, 0]
        print(f"| {label} | {n} | {100 * len(peaks) / max(n, 1):.1f}% | {np.median(peaks) if peaks else float('nan'):.0f} "
              f"| {share[0]:.0f} / {share[1]:.0f} / {share[2]:.0f}% | {100 * cut / max(n, 1):.1f}% |")


if __name__ == "__main__":
    main()
