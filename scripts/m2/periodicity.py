"""M2 diagnosis: periodic components in the unvoiced consonants of reconstructed JVS recordings.

A listener hears a metallic, periodic noise on consonants with the fine-tuned decoders, which the
ASR and speaker-similarity metrics do not show. A fricative is noise: the autocorrelation of its
frames is near zero at every lag beyond the spectral envelope. This script picks the frames that
are unvoiced consonants in the recording (high zero-crossing rate, not silent), takes the same
frames from every result directory of scripts/m2/recon_jvs.py, and reports

  - the mean, over frames, of the largest normalized autocorrelation at lags of 1.25-20 ms
    (the pitch range, 50-800 Hz): how periodic the consonants are
  - the lags where the autocorrelation averaged over all frames peaks: a fixed period shared by
    all frames (the decoder's strides are 4, 20, 120 and 1920 samples) shows up here
  - narrow lines in the mean spectrum of the frames (tones at fixed frequencies)
  - the same for the magnitude of the frames (a gain that repeats with a fixed period)

The averaged curves go to <out>/periodicity.npz.

    uv run python scripts/m2/periodicity.py exp/m2/recon orig mimi mel10k gan60k
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

FRAME = 960  # 40 ms at 24 kHz
HOP = 480
LAG_LO, LAG_HI = 30, 480  # 1.25-20 ms


def frames_of(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n = 1 + (len(x) - FRAME) // HOP
    index = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[index]


def unvoiced_consonants(x: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """Frames of the recording that are loud enough and cross zero like a fricative."""
    f = frames_of(x)
    rms = np.sqrt((f**2).mean(axis=1))
    zcr = (np.diff(np.signbit(f), axis=1) != 0).mean(axis=1)
    return (rms > 0.05 * rms.max()) & (zcr > 0.3)


def autocorrelation(f: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Normalized autocorrelation of each frame after a first difference (flattens the spectrum)."""
    d = np.diff(f.astype(np.float64), axis=1)
    d = (d - d.mean(axis=1, keepdims=True)) * np.hanning(d.shape[1])
    spec = np.abs(np.fft.rfft(d, n=2 * FRAME, axis=1)) ** 2
    r = np.fft.irfft(spec, axis=1)[:, :FRAME]
    return r / np.maximum(r[:, :1], 1e-12)


def envelope_autocorrelation(f: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Normalized autocorrelation of each frame's magnitude: a periodic gain (noise that is
    louder at a fixed phase of some period) does not show in the waveform's autocorrelation."""
    e = np.abs(np.diff(f.astype(np.float64), axis=1))
    e = (e - e.mean(axis=1, keepdims=True)) * np.hanning(e.shape[1])
    spec = np.abs(np.fft.rfft(e, n=2 * FRAME, axis=1)) ** 2
    r = np.fft.irfft(spec, axis=1)[:, :FRAME]
    return r / np.maximum(r[:, :1], 1e-12)


def shared_peaks(r: npt.NDArray[np.float64], top: int = 4) -> str:
    inner = r[LAG_LO:LAG_HI]
    is_peak = (inner[1:-1] > inner[:-2]) & (inner[1:-1] > inner[2:])
    lags = np.flatnonzero(is_peak) + LAG_LO + 1
    best = lags[np.argsort(r[lags])[::-1][:top]]
    return ", ".join(f"{lag}: {r[lag]:+.3f}" for lag in best)


def spectral_lines(r: npt.NDArray[np.float64]) -> tuple[float, int]:
    """Narrow lines of the mean spectrum of the frames (the Fourier transform of their mean
    autocorrelation, 12.5 Hz bins) between 1 and 11.5 kHz: the largest excess in dB over the
    median of the bins within 300 Hz, and the number of bins more than 2 dB above it."""
    sym = np.concatenate([r, [0.0], r[:0:-1]])
    db = 10 * np.log10(np.maximum(np.fft.rfft(sym).real, 1e-9))
    half = 24
    excess = np.array([db[i] - np.median(db[i - half : i + half + 1]) for i in range(80, 921)])
    return float(excess.max()), int((excess > 2.0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("names", nargs="+", help="result directories; frames are chosen on orig/")
    parser.add_argument("--limit", type=int, default=0, help="first N clips only")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.out / "items.jsonl", encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    peak: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in args.names}
    mean_r = {name: np.zeros(FRAME) for name in args.names}
    mean_env = {name: np.zeros(FRAME) for name in args.names}
    count = 0
    for it in items:
        orig, sr = sf.read(str(args.out / "orig" / f"{it['key']}.wav"), dtype="float32")
        assert sr == 24000, sr
        keep = unvoiced_consonants(orig)
        if not keep.any():
            continue
        count += int(keep.sum())
        for name in args.names:
            x, _ = sf.read(str(args.out / name / f"{it['key']}.wav"), dtype="float32")
            x = np.pad(x, (0, max(0, len(orig) - len(x))))[: len(orig)]
            f = frames_of(x)[keep]
            r = autocorrelation(f)
            mean_env[name] += envelope_autocorrelation(f).sum(axis=0)
            top = r[:, LAG_LO:LAG_HI].max(axis=1)
            peak[name]["all"].extend(top.tolist())
            peak[name][it["gender"]].extend(top.tolist())
            mean_r[name] += r.sum(axis=0)
    groups = ["all"] + sorted(g for g in peak[args.names[0]] if g != "all")
    strides = (20, 120, 240, 360)
    print(f"{count} unvoiced consonant frames in {len(items)} clips")
    print("| result | " + " | ".join(f"periodicity {g}" for g in groups) + " | "
          + " | ".join(f"r[{lag}]" for lag in strides) + " | line max dB | lines > 2 dB | strongest shared lags | "
          + " | ".join(f"env r[{lag}]" for lag in strides) + " | strongest shared lags of the envelope |")
    print("|---|" + "---:|" * (len(groups) + len(strides) + 2) + "---|" + "---:|" * len(strides) + "---|")
    for name in args.names:
        r, env = mean_r[name] / count, mean_env[name] / count
        cells = [f"{np.mean(peak[name][g]):.3f}" for g in groups] + [f"{r[lag]:+.3f}" for lag in strides]
        line_max, line_count = spectral_lines(r)
        cells += [f"{line_max:.1f}", str(line_count), shared_peaks(r)]
        cells += [f"{env[lag]:+.3f}" for lag in strides] + [shared_peaks(env)]
        print(f"| {name} | " + " | ".join(cells) + " |")
    np.savez(args.out / "periodicity.npz", **{f"r_{n}": mean_r[n] / count for n in args.names},
             **{f"env_{n}": mean_env[n] / count for n in args.names})

if __name__ == "__main__":
    main()
