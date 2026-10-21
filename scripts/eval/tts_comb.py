"""P2: are the harmonics above 4 kHz there in generated speech? The unaligned hf_structure.

hf_structure.py compares a decoder's output with the recording frame by frame. Generated
speech has no recording to align with, so here every result is measured on its own: its voiced
frames (85 ms, F0 from the autocorrelation of the band below 1 kHz), the F0 refined within 4%
on the comb of the band 1-2.5 kHz alone, and the comb (mean level at the multiples of F0 minus
halfway between them, dB) measured in the bands above it. Noise gives 0 dB. Per gender, and per
speaker for the voices set.

    uv run python scripts/eval/tts_comb.py exp/irodori_distill/eval/voices teacher P2_20k
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from hf_structure import BANDS, N_FFT, PAD, band_stats, refine_f0, spectrum, voiced_f0


def comb_frames(path: Path) -> list[list[float]]:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    step = sr / (N_FFT * PAD)
    out = []
    for start, f0 in voiced_f0(x, sr):
        spec = spectrum(x[start : start + N_FFT])
        f0 = refine_f0(np.log(spec + 1e-9), f0, step, span=0.04, lo=1000, hi=2500)
        out.append([c for c, _ in band_stats(spec, f0, step)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="evaluation directory (items.jsonl, one subdirectory per result)")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--per-speaker", action="store_true", help="also a table per speaker")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    labels = [f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS[1:]]
    results: dict[str, dict[str, list[list[float]]]] = {}
    for name in args.names:
        by: dict[str, list[list[float]]] = defaultdict(list)
        for it in items:
            path = args.root / name / f"{it['key']}.wav"
            if not path.exists():
                continue
            frames = comb_frames(path)
            by[it["gender"]].extend(frames)
            by[f"spk:{it['speaker']}"].extend(frames)
        results[name] = by
    groups = ["F", "M"] + (sorted({f"spk:{it['speaker']}" for it in items}) if args.per_speaker else [])
    for g in groups:
        if not any(results[n].get(g) for n in args.names):
            continue
        print(f"\n{g.removeprefix('spk:')}: comb dB at the result's own F0 (voiced frames)")
        print("| result | frames | " + " | ".join(labels) + " |")
        print("|---|---:|" + "---:|" * len(labels))
        for n in args.names:
            f = np.array(results[n].get(g, []))
            if len(f) == 0:
                continue
            print(f"| {n} | {len(f)} | " + " | ".join(f"{v:.1f}" for v in f[:, 1:].mean(axis=0)) + " |")


if __name__ == "__main__":
    main()
