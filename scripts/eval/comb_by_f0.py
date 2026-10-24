"""The harmonics of voiced frames by the frame's own F0: where the students fall short of the teacher.

The user hears female_02 / 05 crack and double where the intonation lifts the pitch, and less so
with the high band cut away (10/24, female_02's 自国の言葉について). tts_comb.py's comb per voice
says these two voices lose harmonics, not where: here every voiced frame of the female voices
(43 ms, F0 from the band below 1 kHz refined on the comb of 1-2.5 kHz, as tts_comb.py) gives the
comb of 2.5-6 kHz (mean level at the multiples of F0 minus halfway between them, dB), pooled by its
F0. On 10/24 the students matched the teacher below 250 Hz and fell 2 dB short at 300-400 Hz and
3-4 dB at 400-450 Hz, the F0 the training references hardly reach.

    uv run python scripts/eval/comb_by_f0.py exp/irodori_distill/eval/voices teacher S1_60k_t07s8_cfg1.5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import hf_structure as hf

hf.N_FFT, hf.HOP = 1024, 512  # 43 ms: short enough for the high voices' gliding harmonics
EDGES = (0, 250, 300, 350, 400, 450, 2000)
BAND = (2500.0, 6000.0)


def frame_combs(path: Path) -> list[tuple[float, float]]:
    """(F0, comb of BAND in dB) per voiced frame."""
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    step = sr / (hf.N_FFT * hf.PAD)
    out = []
    for start, f0 in hf.voiced_f0(x, sr):
        log = np.log(hf.spectrum(x[start : start + hf.N_FFT]) + 1e-9)
        f0 = hf.refine_f0(log, f0, step, span=0.04, lo=1000, hi=2500)
        out.append((f0, hf.contrast(log, f0, BAND[0], BAND[1], step)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="evaluation directory (items.jsonl, one subdirectory per result)")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--gender", default="F", help="the items' gender to pool")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    bins = [f"{EDGES[i]}-{EDGES[i + 1]} Hz" if EDGES[i + 1] < 2000 else f"{EDGES[i]}+ Hz" for i in range(len(EDGES) - 1)]
    print(f"{args.gender} voices: comb {BAND[0] / 1000:g}-{BAND[1] / 1000:g} kHz (dB, 43 ms frames) by the frame's F0 "
          "(frames in brackets)")
    print("| result | " + " | ".join(bins) + " |")
    print("|---|" + "---:|" * len(bins))
    for name in args.names:
        acc: dict[int, list[float]] = defaultdict(list)
        for it in items:
            path = args.root / name / f"{it['key']}.wav"
            if it["gender"] != args.gender or not path.exists():
                continue
            for f0, comb in frame_combs(path):
                acc[int(np.searchsorted(EDGES, f0)) - 1].append(comb)
        cells = [f"{np.mean(acc[b]):.1f} ({len(acc[b])})" if acc[b] else "-" for b in range(len(bins))]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
