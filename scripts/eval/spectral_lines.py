"""Fixed-frequency lines in the long-term spectrum: a decoder's frame-rate buzz.

A decoder that leaves a trace of its frame grid (the SEANet's 200 Hz, the latent rate) puts lines
at the multiples of that rate into the spectrum averaged over a whole clip, where the voice's own
harmonics, moving with F0, average out. Per result: the mean level at the multiples of each rate
minus the level halfway between them, 2-10 kHz, in dB (a recording gives ~0.3, the DACVAE decode
~0.65, the stock Mimi ~1.6 at 200 Hz; P1-side's first probe 4.7 after 11k GAN steps).

    uv run python scripts/eval/spectral_lines.py exp/irodori_distill/p0/recon_jvs orig dacvae mimi W1_30k
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

RATES = (200.0, 100.0, 25.0, 12.5)
N = 8192


def ltas(files: list[Path]) -> tuple[npt.NDArray[np.float64], int]:
    acc, sr = None, 0
    for f in files:
        x, sr = sf.read(str(f), dtype="float32")
        frames = [x[i : i + N] * np.hanning(N) for i in range(0, len(x) - N, N // 2)]
        if not frames:
            continue
        p = np.mean([np.abs(np.fft.rfft(fr)) ** 2 for fr in frames], axis=0)
        acc = p if acc is None else acc + p
    assert acc is not None, "no clip long enough"
    return 10 * np.log10(acc / len(files) + 1e-12), sr


def contrast(db: npt.NDArray[np.float64], sr: int, rate: float, lo: float = 2000, hi: float = 10000) -> float:
    step = sr / N
    k = np.arange(int(np.ceil(lo / rate)), int(hi / rate))
    on = db[np.round(k * rate / step).astype(int)]
    off = db[np.round((k + 0.5) * rate / step).astype(int)]
    return float(np.mean(on - off))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--clips", type=int, default=200)
    args = parser.parse_args()
    print("| result | " + " | ".join(f"{r:g} Hz" for r in RATES) + " |")
    print("|---|" + "---:|" * len(RATES))
    for name in args.names:
        files = sorted((args.root / name).glob("*.wav"))[: args.clips]
        db, sr = ltas(files)
        print(f"| {name} | " + " | ".join(f"{contrast(db, sr, r):.2f}" for r in RATES) + " |")


if __name__ == "__main__":
    main()
