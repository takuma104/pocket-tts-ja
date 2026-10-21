"""M2 high band: are the decoder's 4-8 kHz components pitch pulses without a coherent waveform, or noise?

A voice's band above 4 kHz repeats with the pitch in two ways. Its harmonics make the waveform of
the band repeat (high correlation of the band-passed signal with itself one pitch period later);
the glottal pulses make its energy repeat (high correlation of the band's Hilbert envelope one
period later), which also holds for breath noise that the pulses modulate. For the voiced frames
of the recording (F0 from scripts/m2/hf_structure.py), per band and result directory this prints

    wave  correlation of the band-passed waveform with itself shifted by one period (T0)
    env   the same for its envelope (mean removed)

Stationary noise gives about 0 for both. Pulses at the right times with a waveform that changes
from one pulse to the next give a high env and a low wave: such a decoder knows when the pulses
come but cannot repeat their waveform, which a harmonic source (sine excitation from F0) would
give it. Low env as well means the band is not even pitch-synchronous.

    uv run python scripts/m2/hf_coherence.py exp/m2/recon orig mimi keep4k gs2_80k
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import firwin, hilbert

from hf_structure import N_FFT, refine_f0, spectrum, voiced_f0

BANDS = ((1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000))
TAPS = 255


def lag_corr(x: npt.NDArray[np.float64], lag: float) -> float:
    """Correlation of x with itself `lag` samples later (linear interpolation for the fraction)."""
    whole, frac = int(lag), lag - int(lag)
    n = len(x) - whole - 1
    a = x[:n]
    b = (1 - frac) * x[whole : whole + n] + frac * x[whole + 1 : whole + 1 + n]
    return float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("names", nargs="+", help="result directories; the first one is the recording")
    parser.add_argument("--per-speaker", type=int, default=6)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.recon / "items.jsonl", encoding="utf-8")]
    seen: dict[str, int] = defaultdict(int)
    wave: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    env: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    kernels: list[npt.NDArray[np.float64]] = []
    for it in items:
        seen[it["speaker"]] += 1
        if seen[it["speaker"]] > args.per_speaker:
            continue
        ref, sr = sf.read(str(args.recon / args.names[0] / f"{it['key']}.wav"), dtype="float32")
        if not kernels:
            kernels = [firwin(TAPS, (lo, hi), fs=sr, pass_zero=False) for lo, hi in BANDS]
        step = sr / (N_FFT * 4)
        frames = [(start, refine_f0(np.log(spectrum(ref[start : start + N_FFT]) + 1e-9), f0, step))
                  for start, f0 in voiced_f0(ref, sr)]
        for name in args.names:
            x, _ = sf.read(str(args.recon / name / f"{it['key']}.wav"), dtype="float32")
            x = np.pad(x, (0, max(0, len(ref) - len(x))))[: len(ref)].astype(np.float64)
            bands = [np.convolve(x, k, "same") for k in kernels]
            envs = [np.abs(hilbert(b)) for b in bands]
            for start, f0 in frames:
                lag = sr / f0
                w_row, e_row = [], []
                for b, e in zip(bands, envs, strict=True):
                    seg_b = b[start : start + N_FFT]
                    seg_e = e[start : start + N_FFT]
                    w_row.append(lag_corr(seg_b, lag))
                    e_row.append(lag_corr(seg_e - seg_e.mean(), lag))
                wave[name, it["gender"]].append(w_row)
                env[name, it["gender"]].append(e_row)
    head = " | ".join(f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS)
    for group in ("F", "M"):
        if not wave[args.names[0], group]:
            continue
        print(f"\n{group}: {len(wave[args.names[0], group])} voiced frames. correlation one period later: waveform / envelope"
              f"\n| result | {head} |\n|---|" + "---|" * len(BANDS))
        for name in args.names:
            w = np.mean(wave[name, group], axis=0)
            e = np.mean(env[name, group], axis=0)
            print(f"| {name} | " + " | ".join(f"{a:.2f} / {b:.2f}" for a, b in zip(w, e)) + " |")


if __name__ == "__main__":
    main()
