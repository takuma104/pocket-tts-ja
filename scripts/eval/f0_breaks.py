"""Cracks of the voice: sudden pitch jumps and frames out of the clip's register.

The user hears female_02 / 05's students "flip" (裏返る) and sound doubled, less so at 8 LSD steps
than at 1, never in the teacher; the comb barely follows it. Here each clip's F0 is tracked every
10 ms (40 ms frames of the band below 1.5 kHz, the normalized cross-correlation over lags of
70-900 Hz; of the lags within 10% of the best correlation the shortest, so a clean voice is not
read an octave low, while a voice with a strong subharmonic is) and, per result:

    jump    share of the pairs of adjacent voiced frames whose F0 differs by more than 5 semitones
            (a voice glides far slower: 5 semitones in 10 ms is a break, or a tracker error, which
            the teacher's figure shows the rate of)
    out     share of the voiced frames more than 7 semitones from the clip's median F0 (a flip
            into another register, or a subharmonic: the doubled voice)
    voiced  share of the speech frames (loud enough) that are voiced
    wobble  the pitch contour's movement at the student's frame rate: in every voiced run of at
            least 0.2 s, the RMS (cents) of the contour band-passed to 15-35 Hz (25 Hz is one
            DACVAE frame, 40 ms). Frames drawn one by one could set the pitch in 40 ms steps
            (a stepped, auto-tune-like voice) where the teacher glides. It is tracked separately
            with 20 ms frames every 5 ms (F0 of 100-900 Hz: the female voices), short enough
            not to smooth a 25 Hz movement away.

    uv run python scripts/eval/f0_breaks.py exp/irodori_distill/eval/voices teacher P3m_20k_t07_cfg1.0_eos-1 --per-speaker
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr
from scipy.signal import firwin

SR = 16000
HOP = SR // 100
WIN = SR * 40 // 1000
MIN_LAG, MAX_LAG = SR // 900, SR // 70
FFT = 2048
WOBBLE_HOP, WOBBLE_WIN, WOBBLE_MAX_LAG = SR // 200, SR * 20 // 1000, SR // 100
WOBBLE_BAND = (15.0, 35.0)
WOBBLE_MIN_RUN = 40  # 0.2 s of 5 ms frames


def track(
    x: npt.NDArray[np.float32], hop: int = HOP, win: int = WIN, max_lag: int = MAX_LAG
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """(F0 in Hz per frame, NaN where unvoiced; loud frames)."""
    low = np.convolve(x, firwin(255, 1500, fs=SR), "same")
    n = (len(low) - win - max_lag) // hop
    if n <= 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    starts = np.arange(n) * hop
    seg = np.stack([low[s : s + win + max_lag] for s in starts])  # [n, win + max_lag]
    frame = seg[:, :win]
    # sum_k frame[k] * seg[k + lag] for every lag, and the energy of seg[lag : lag + win]
    cross = np.fft.irfft(np.conj(np.fft.rfft(frame, FFT)) * np.fft.rfft(seg, FFT), FFT)[:, : max_lag + 1]
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg**2, axis=1)], axis=1)
    energy = csum[:, win : win + max_lag + 1] - csum[:, : max_lag + 1]
    nccf = cross / np.sqrt(energy[:, :1] * energy + 1e-12)
    band = nccf[:, MIN_LAG:]
    best = band.max(axis=1)
    lag = MIN_LAG + np.argmax(band >= 0.9 * best[:, None], axis=1)  # the shortest lag near the best
    rms = np.sqrt((frame**2).mean(axis=1))
    loud = 20 * np.log10(rms + 1e-9) > 20 * np.log10(rms.max(initial=1e-9) + 1e-9) - 35
    f0 = np.where(loud & (best >= 0.7), SR / lag, np.nan)
    return f0, loud


def wobble(f0: npt.NDArray[np.float64]) -> tuple[float, int]:
    """(sum over the runs of the band power in cents^2 x samples, samples) of a 5 ms contour."""
    cents = 1200 * np.log2(f0 / 100)
    voiced = ~np.isnan(cents)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    power, count = 0.0, 0
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a < WOBBLE_MIN_RUN:
            continue
        run = cents[a:b]
        t = np.arange(len(run))
        run = run - np.polyval(np.polyfit(t, run, 1), t)
        spec = np.fft.rfft(run)
        freq = np.fft.rfftfreq(len(run), WOBBLE_HOP / SR)
        band = (freq >= WOBBLE_BAND[0]) & (freq <= WOBBLE_BAND[1])
        # Parseval: the band's share of the mean square, times the run's length
        power += 2 * float((np.abs(spec[band]) ** 2).sum()) / len(run)
        count += len(run)
    return power, count


def stats(f0: npt.NDArray[np.float64], loud: npt.NDArray[np.bool_]) -> tuple[int, int, int, int, int]:
    """(voiced pairs, jumps, voiced frames, out of register, loud frames)."""
    st = 12 * np.log2(f0 / 100)
    pair = ~np.isnan(st[1:]) & ~np.isnan(st[:-1])
    jumps = int((np.abs(np.diff(st))[pair] > 5).sum())
    voiced = st[~np.isnan(st)]
    out = int((np.abs(voiced - np.median(voiced)) > 7).sum()) if len(voiced) else 0
    return int(pair.sum()), jumps, len(voiced), out, int(loud.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="evaluation directory (items.jsonl, one subdirectory per result)")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--per-speaker", action="store_true", help="also a table per speaker")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    groups: dict[str, list[str]] = defaultdict(list)
    for it in items:
        groups[it["gender"]].append(it["key"])
        if args.per_speaker:
            groups[it["speaker"]].append(it["key"])
    totals: dict[tuple[str, str], npt.NDArray[np.int64]] = {}
    for name in args.names:
        per_key = {}
        for it in items:
            path = args.root / name / f"{it['key']}.wav"
            if not path.exists():
                continue
            x, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = soxr.resample(x.mean(axis=1), sr, SR)
            fine, _ = track(y, WOBBLE_HOP, WOBBLE_WIN, WOBBLE_MAX_LAG)
            per_key[it["key"]] = np.array([*stats(*track(y)), *wobble(fine)])
        for g, keys in groups.items():
            totals[(name, g)] = sum((per_key[k] for k in keys if k in per_key), np.zeros(7))
    for g in sorted(groups, key=lambda g: (len(g) > 1, g)):
        print(f"{g}: F0 breaks (10 ms frames)")
        print("| result | voiced frames | jump > 5 st | out of register > 7 st | voiced share | wobble 15-35 Hz (cents) |")
        print("|---|---:|---:|---:|---:|---:|")
        for name in args.names:
            pairs, jumps, voiced, out, loud, power, count = totals[(name, g)]
            print(f"| {name} | {int(voiced)} | {100 * jumps / max(pairs, 1):.2f}% | {100 * out / max(voiced, 1):.2f}% | "
                  f"{100 * voiced / max(loud, 1):.1f}% | {np.sqrt(power / max(count, 1)):.1f} |")
        print()


if __name__ == "__main__":
    main()
