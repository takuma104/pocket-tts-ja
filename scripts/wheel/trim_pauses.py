"""Shorten the pauses inside voice prompts to at most --max-pause seconds.

The release's ten voice prompts are 23-39% silence (female_04 39%), against a median of 10% (p90 24%)
for the teacher's reference clips the students were trained with. The students carry the prompt's
long pauses into their speech: female_04's pauses run 1.8x the teacher's through the wheel (HSnl,
median 0.73 vs 0.40 s), female_01's (35%) 1.3x.

Silence as in scripts/eval/pauses.py: 40 ms frames more than --floor-db below the clip's 95th
percentile frame. Each run of silent frames inside the speech longer than --max-pause keeps its
first and last max-pause / 2 and loses the middle, joined with a 10 ms crossfade. The leading and
trailing silence are left as they are.

    uv run python scripts/wheel/trim_pauses.py exp/release_cc/voices exp/irodori_distill/voices_trim
    -> <out>/<name>.wav for every <in>/<name>.wav, same rate and subtype
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

FRAMES_PER_S = 25


def silent_runs(w: np.ndarray, sr: int, floor_db: float) -> list[tuple[int, int]]:
    """(start, end) samples of the silent runs inside the speech."""
    hop = sr // FRAMES_PER_S
    n = len(w) // hop
    db = 20 * np.log10(np.sqrt((w[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    sil = db < np.percentile(db, 95) - floor_db
    loud = np.flatnonzero(~sil)
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(loud[0], loud[-1] + 1):
        if sil[i] and start is None:
            start = i
        elif not sil[i] and start is not None:
            runs.append((start * hop, i * hop))
            start = None
    return runs


def trim(w: np.ndarray, sr: int, max_pause: float, floor_db: float) -> tuple[np.ndarray, float]:
    keep = int(max_pause * sr / 2)
    fade = int(0.01 * sr)
    ramp = np.linspace(0.0, 1.0, fade)
    out: list[np.ndarray] = []
    pos = 0
    removed = 0
    for a, b in silent_runs(w, sr, floor_db):
        if b - a <= 2 * keep + fade:
            continue
        head = w[pos : a + keep + fade].copy()
        tail = w[b - keep : b - keep + fade]
        head[-fade:] = head[-fade:] * (1 - ramp) + tail * ramp
        out.append(head)
        pos = b - keep + fade
        removed += (b - keep) - (a + keep)
    out.append(w[pos:])
    return np.concatenate(out), removed / sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-pause", type=float, default=0.35)
    parser.add_argument("--floor-db", type=float, default=35.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for p in sorted(args.src.glob("*.wav")):
        info = sf.info(p)
        w, sr = sf.read(p)
        mono = w if w.ndim == 1 else w.mean(1)
        y, removed = trim(mono, sr, args.max_pause, args.floor_db)
        sf.write(args.out / p.name, y, sr, subtype=info.subtype)
        print(f"{p.stem}: {len(mono) / sr:.1f} -> {len(y) / sr:.1f} s ({removed:.1f} s of pauses removed)")


if __name__ == "__main__":
    main()
