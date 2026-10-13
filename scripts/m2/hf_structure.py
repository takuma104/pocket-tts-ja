"""M2: is the high band of voiced speech harmonic or noise? Per decoder, on the JVS reconstructions.

A listener hears a decoder that fills the band above 4 kHz of a voice with noise instead of
harmonics as a hiss that follows the voicing (worst for high voices); the Kana-CER and the mel
loss do not see it. For every voiced frame of the recording (85 ms, F0 from the autocorrelation
of its band below 1 kHz) and every band this prints, per decoder:

    comb   how far the harmonics stand out: mean level of the spectrum at the multiples of F0
           minus the mean level halfway between them, in dB. F0 is refined on the recording
           (the value within 2% that maximizes this contrast between 1 and 4 kHz), so the
           recording's harmonics are met exactly; noise gives 0 dB, and so do harmonics that
           are not where the recording has them.
    level  power of the band relative to the recording's, in dB.

A high level with a low comb is noise in place of harmonics. The last table is the level in
the frames of unvoiced consonants (loud enough, zero crossings like a fricative), where noise
is what the recording has too.

    uv run python scripts/m2/hf_structure.py exp/m2/recon orig mimi mel10k gan60k gs2_80k
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import firwin

N_FFT = 2048
HOP = 1024
PAD = 4  # zero padding factor of the spectrum
BANDS = ((1000, 2500), (2500, 4000), (4000, 6000), (6000, 8000), (8000, 10000))


def voiced_f0(x: npt.NDArray[np.float32], sr: int) -> list[tuple[int, float]]:
    """(frame start, F0) of the frames whose low band is clearly periodic."""
    low = np.convolve(x, firwin(255, 1000, fs=sr), "same")
    lo, hi = sr // 450, sr // 80
    out = []
    for start in range(0, len(x) - N_FFT, HOP):
        frame = low[start : start + N_FFT]
        if np.sqrt(np.mean(frame**2)) < 0.01:
            continue
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, "full")[N_FFT - 1 :]
        ac = ac / (ac[0] + 1e-12)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] < 0.75 or lag in (lo, hi - 1):
            continue
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        out.append((start, sr / (lag + shift)))
    return out


def fricative_frames(x: npt.NDArray[np.float32]) -> list[int]:
    starts = np.arange(0, len(x) - N_FFT, HOP)
    frames = np.stack([x[s : s + N_FFT] for s in starts]) if len(starts) else np.zeros((0, N_FFT), np.float32)
    rms = np.sqrt((frames**2).mean(axis=1))
    zcr = (np.diff(np.signbit(frames), axis=1) != 0).mean(axis=1)
    return [int(s) for s in starts[(rms > 0.05 * rms.max(initial=1e-9)) & (zcr > 0.3)]]


def spectrum(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    return np.abs(np.fft.rfft(frame * np.hanning(N_FFT), N_FFT * PAD))


def contrast(log: npt.NDArray[np.float64], f0: float, lo: float, hi: float, step: float) -> float:
    """Mean log magnitude at the harmonics of f0 minus that halfway between them, in dB."""
    k = np.arange(int(np.ceil(lo / f0)), int(hi / f0))
    if len(k) == 0:
        return 0.0
    peaks = log[np.round(k * f0 / step).astype(int)]
    valleys = log[np.round((k + 0.5) * f0 / step).astype(int)]
    return float(20 / np.log(10) * (peaks.mean() - valleys.mean()))


def refine_f0(log: npt.NDArray[np.float64], f0: float, step: float) -> float:
    candidates = f0 * (1 + np.linspace(-0.02, 0.02, 81))
    return float(candidates[int(np.argmax([contrast(log, c, 1000, 4000, step) for c in candidates]))])


def band_stats(spec: npt.NDArray[np.float64], f0: float, step: float) -> list[tuple[float, float]]:
    """(comb in dB, band power) per band of one frame."""
    log = np.log(spec + 1e-9)
    return [(contrast(log, f0, lo, hi, step), float(np.sum(spec[round(lo / step) : round(hi / step)] ** 2)))
            for lo, hi in BANDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("names", nargs="+", help="result directories; the first one is the recording")
    parser.add_argument("--per-speaker", type=int, default=20)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.recon / "items.jsonl", encoding="utf-8")]
    seen: dict[str, int] = defaultdict(int)
    comb: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    power: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    noise: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for it in items:
        seen[it["speaker"]] += 1
        if seen[it["speaker"]] > args.per_speaker:
            continue
        ref, sr = sf.read(str(args.recon / args.names[0] / f"{it['key']}.wav"), dtype="float32")
        step = sr / (N_FFT * PAD)
        frames = [(start, refine_f0(np.log(spectrum(ref[start : start + N_FFT]) + 1e-9), f0, step))
                  for start, f0 in voiced_f0(ref, sr)]
        for name in args.names:
            x, _ = sf.read(str(args.recon / name / f"{it['key']}.wav"), dtype="float32")
            x = np.pad(x, (0, max(0, len(ref) - len(x))))
            for start, f0 in frames:
                stats = band_stats(spectrum(x[start : start + N_FFT]), f0, step)
                for group in ("all", it["gender"]):
                    comb[name, group].append([s[0] for s in stats])
                    power[name, group].append([s[1] for s in stats])
            for start in fricative_frames(ref):
                stats = band_stats(spectrum(x[start : start + N_FFT]), 200.0, step)
                for group in ("all", it["gender"]):
                    noise[name, group].append([s[1] for s in stats])
    head = " | ".join(f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS)
    for group in ("F", "M", "all"):
        n = len(comb[args.names[0], group])
        print(f"\n{group}: {n} voiced frames. comb dB / level dB re {args.names[0]}\n| decoder | {head} |")
        print("|---|" + "---|" * len(BANDS))
        ref_power = np.mean(power[args.names[0], group], axis=0)
        for name in args.names:
            c = np.mean(comb[name, group], axis=0)
            level = 10 * np.log10(np.mean(power[name, group], axis=0) / ref_power)
            print(f"| {name} | " + " | ".join(f"{a:.1f} / {b:+.1f}" for a, b in zip(c, level)) + " |")
    for group in ("F", "M"):
        print(f"\n{group}: {len(noise[args.names[0], group])} frames of unvoiced consonants. level dB re {args.names[0]}"
              f"\n| decoder | {head} |\n|---|" + "---|" * len(BANDS))
        ref_power = np.mean(noise[args.names[0], group], axis=0)
        for name in args.names:
            level = 10 * np.log10(np.mean(noise[name, group], axis=0) / ref_power)
            print(f"| {name} | " + " | ".join(f"{b:+.1f}" for b in level) + " |")


if __name__ == "__main__":
    main()
