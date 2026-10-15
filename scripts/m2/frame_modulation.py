"""M2 high band: amplitude modulation of the decoded bands at the decoder's fixed frame rates.

The decoder runs its transformer at 200 Hz and upsamples by 6, 5 and 4 (1.2 kHz, 6 kHz, 24 kHz).
A gain that repeats with one of those periods puts a line into the spectrum of a band's envelope
at that rate, whatever the voice's pitch: heard as a buzz under the voice, a mechanical quality.
periodicity.py looks for such a gain in the unvoiced consonants; this looks at all loud frames,
voiced ones included, per band.

For every band, the Hilbert envelope of the band-passed signal is cut into 170 ms segments, and
the spectra of (envelope - mean) / mean are averaged over the segments and files. The pitch of the
voices varies between and within utterances and spreads over a wide range of modulation rates;
a fixed rate stays a line. Reported: the level at each fixed rate over the median of its
neighbours (±40 Hz, the central ±4 Hz left out), in dB, and the total modulation depth of the
band (the envelope's relative standard deviation), both next to the recording's.

The first table is the direct measure (as codec_ft.GridModulationLoss computes it in training):
the relative envelope (over its 10 ms mean), weighted by the recording's band level, folded on
the decoder's 120-sample frame and averaged over the utterances; the rms of that profile in %
(the recordings: about 1%, the noise floor of the average), and how well the single
utterances' profiles correlate with it (a pattern every utterance shares: near 1).

    uv run python scripts/m2/frame_modulation.py exp/m2/recon orig keep4k M2H_H2 --gender F
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

BANDS = ((1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000))
RATES = (12.5, 25.0, 200.0, 400.0, 600.0, 1200.0, 2400.0)
PERIOD = 120  # samples of one 200 Hz decoder frame at 24 kHz
SEG = 4096
NFFT = 16384


def band_envelope(x: npt.NDArray[np.float64], sr: int, lo: float, hi: float) -> npt.NDArray[np.float64]:
    spec = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(freq < lo) | (freq > hi)] = 0
    # analytic signal: the positive frequencies doubled, back to the full length
    full = np.zeros(len(x), dtype=complex)
    full[: len(spec)] = spec
    full[1 : len(spec) - (1 if len(x) % 2 == 0 else 0)] *= 2
    return np.abs(np.fft.ifft(full))


def smooth(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    return np.convolve(x, np.ones(n) / n, mode="same")


def grid_profile(env: npt.NDArray[np.float64], ref_env: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Relative envelope folded on the decoder's frame, weighted by the recording's band level."""
    n = len(env) // PERIOD * PERIOD
    rel = env[:n] / (smooth(env[:n], 2 * PERIOD) + 1e-9) - 1
    weight = smooth(ref_env[:n], 2 * PERIOD)
    weight[: 2 * PERIOD] = 0
    weight[-2 * PERIOD :] = 0
    return (rel * weight).reshape(-1, PERIOD).sum(0) / max(weight.reshape(-1, PERIOD).sum(0).mean(), 1e-9)


def loud_segments(ref: npt.NDArray[np.float64]) -> list[int]:
    starts = range(0, len(ref) - SEG, SEG // 2)
    rms = np.array([np.sqrt(np.mean(ref[s : s + SEG] ** 2)) for s in starts])
    return [s for s, r in zip(starts, rms, strict=True) if r > 0.1 * rms.max()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("names", nargs="+", help="result directories; the first is the reference (the recording)")
    parser.add_argument("--gender", choices=("F", "M", "all"), default="F")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    if args.gender != "all":
        items = [it for it in items if it["gender"] == args.gender]
    items = items[:: max(1, len(items) // args.limit)][: args.limit]
    window = np.hanning(SEG)
    freq = np.fft.rfftfreq(NFFT, 1)
    acc = {name: np.zeros((len(BANDS), len(freq))) for name in args.names}
    depth = {name: np.zeros(len(BANDS)) for name in args.names}
    profiles: dict[str, list[list[npt.NDArray[np.float64]]]] = {name: [[] for _ in BANDS] for name in args.names}
    count = 0
    for it in items:
        wavs = {}
        for name in args.names:
            x, sr = sf.read(args.root / name / f"{it['key']}.wav", dtype="float64")
            wavs[name] = x
        n = min(len(x) for x in wavs.values())
        segs = loud_segments(wavs[args.names[0]][:n])
        if not segs:
            continue
        count += len(segs)
        ref_envs = [band_envelope(wavs[args.names[0]][:n], sr, lo, hi) for lo, hi in BANDS]
        for name, x in wavs.items():
            for b, (lo, hi) in enumerate(BANDS):
                env = band_envelope(x[:n], sr, lo, hi)
                profiles[name][b].append(grid_profile(env, ref_envs[b]))
                for s in segs:
                    e = env[s : s + SEG]
                    m = e.mean() + 1e-9
                    depth[name][b] += e.std() / m
                    acc[name][b] += np.abs(np.fft.rfft((e - m) / m * window, NFFT)) ** 2
    hz = freq * sr
    print(f"{args.gender}: {len(items)} utterances, {count} segments of {SEG / sr * 1000:.0f} ms")
    print("\nenvelope folded on the 5 ms decoder frame: rms of the mean profile (%) / mean correlation of the utterances' profiles with it")
    print("| | " + " | ".join(f"{lo}-{hi}" for lo, hi in BANDS) + " |")
    print("|---" * (len(BANDS) + 1) + "|")
    for name in args.names:
        cells = []
        for b in range(len(BANDS)):
            prof = np.array(profiles[name][b])
            mean = prof.mean(0)
            corr = np.mean([np.corrcoef(p, mean)[0, 1] for p in prof])
            cells.append(f"{100 * np.sqrt(np.mean((mean - mean.mean()) ** 2)):.1f} / {corr:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nline at a fixed modulation rate over its neighbours (dB) | modulation depth (std / mean of the envelope)")
    for b, (lo, hi) in enumerate(BANDS):
        print(f"\n{lo}-{hi} Hz | " + " | ".join(f"{r:g}" for r in RATES) + " | depth |")
        print("|---" * (len(RATES) + 2) + "|")
        for name in args.names:
            spec = acc[name][b] / count
            cells = []
            for r in RATES:
                i = int(np.argmin(np.abs(hz - r)))
                peak = spec[max(0, i - 2) : i + 3].max()
                near = (np.abs(hz - r) < 40) & (np.abs(hz - r) > 4)
                cells.append(f"{10 * np.log10(peak / np.median(spec[near])):.1f}")
            print(f"| {name} | " + " | ".join(cells) + f" | {depth[name][b] / count:.3f} |")


if __name__ == "__main__":
    main()
