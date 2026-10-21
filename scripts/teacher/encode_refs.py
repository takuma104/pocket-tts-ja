"""P1: DACVAE latents of the teacher's speaker references, the students' voice prompts.

Irodori encodes a reference after audiotools' loudness normalisation to -16 dB (the runtime's
codec default), so the prompt latents here are made the same way: the student sees the voice as
the teacher did. One float16 ``(frames, 32)`` file per reference speaker.

    uv run python scripts/teacher/encode_refs.py data/irodori_distill/synth_src/refs_cc \
        data/irodori_distill/synth_src/refs_yodas --out data/irodori_distill/teacher/refs
"""

import argparse
from pathlib import Path

import numpy as np

from pocket_tts_ja.dacvae import DACVAE, REF_LOUDNESS_DB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+", help="directories of reference wavs (<speaker>.wav)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    codec = DACVAE.load()
    n = 0
    for d in args.dirs:
        for wav in sorted(d.glob("*.wav")):
            dst = args.out / f"{wav.stem}.npy"
            if dst.exists():
                continue
            z = codec.encode_file(wav, loudness_db=REF_LOUDNESS_DB)
            np.save(dst, z.numpy().astype(np.float16))
            n += 1
    print(f"{args.out}: {n} new reference latents, {len(list(args.out.glob('*.npy')))} in all")


if __name__ == "__main__":
    main()
