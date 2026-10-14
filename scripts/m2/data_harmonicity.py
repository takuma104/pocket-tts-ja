"""M2: do the candidate training sources have harmonics above 4 kHz in voiced speech?

A decoder can only learn to put harmonics into the band above 4 kHz from audio that has them
there. This measures, per source, the same "comb" as scripts/m2/hf_structure.py (level at the
multiples of F0 minus the level halfway between them, dB; noise gives 0) on each source's own
voiced frames, and the level of each band relative to the 1-2.5 kHz band. Frames are grouped by
their F0 (a high voice spaces its harmonics widely, which is where a missing comb is heard).

Sources (all resampled to 24 kHz, each clip scaled to the same RMS before the analysis):

    jvs_F / jvs_M   JVS parallel100 of the 5 + 5 speakers of recon_jvs.py (evaluation only)
    emilia_hq       data/m2/emilia_hq/manifest.jsonl, read from the Emilia tar shards on the NAS
    hfc_F / hfc_M   Hi-Fi-CAPTAIN ja-JP (studio, 48 kHz)
    jsut            JSUT 1.1 (one female speaker, 48 kHz)
    jvnv            JVNV (emotional speech, 2 female + 2 male, 48 kHz)
    amitaro         ITA corpus read by amitaro (one female speaker)
    irodori_jvs001  Irodori-TTS v4.1-small output with the jvs001 (male) reference, a stand-in for
                    the synthetic S3 set (DACVAE codec); compare with jvs001's own recordings

    uv run python scripts/m2/data_harmonicity.py --emilia-manifest data/m2_emilia_hq_manifest.jsonl
"""

import argparse
import io
import json
import random
import sys
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_structure import BANDS, N_FFT, PAD, band_stats, refine_f0, spectrum, voiced_f0  # noqa: E402

SR = 24000
TARGET_RMS = 0.05
NAS = Path("/mnt/datasets/speech/free_distribute")
EMILIA_ROOT = Path("/mnt/ext11/Emilia-Dataset/Emilia/JA")
IRODORI_JVS001 = Path("/mnt/artifacts/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001/audio")
F0_GROUPS = (("F0<165", 0.0, 165.0), ("F0>=165", 165.0, 1000.0))

Wave = npt.NDArray[np.float32]


def prepare(wav: npt.NDArray[np.float32], sr: int) -> Wave:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR, quality="HQ").astype(np.float32)
    rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
    return (wav * (TARGET_RMS / rms)).astype(np.float32)


def from_zip(path: Path, prefix: str, count: int, rng: random.Random) -> Iterator[Wave]:
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.lower().endswith(".wav"))
        for name in rng.sample(names, min(count, len(names))):
            wav, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32")
            yield prepare(wav, sr)


def from_files(paths: list[Path], count: int, rng: random.Random) -> Iterator[Wave]:
    for path in rng.sample(sorted(paths), min(count, len(paths))):
        wav, sr = sf.read(str(path), dtype="float32")
        yield prepare(wav, sr)


def from_emilia(manifest: Path, count: int, rng: random.Random) -> Iterator[Wave]:
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    for row in sorted(rng.sample(rows, min(count, len(rows))), key=lambda r: (r["shard"], r["offset"])):
        with open(EMILIA_ROOT / row["shard"], "rb") as f:
            f.seek(row["offset"])
            data = f.read(row["size"])
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        yield prepare(wav, sr)[: int(row["duration"] * SR)]


def jvs_speakers(jvs_root: Path, gender: str, per_gender: int = 5) -> list[str]:
    from pocket_tts_ja.jvs import read_genders

    genders = read_genders(jvs_root)
    return sorted(s for s, info in genders.items() if info[0] == gender)[:per_gender]


def analyze(wav: Wave) -> list[tuple[float, list[float], list[float]]]:
    """(F0, comb per band, power per band) of each voiced frame."""
    step = SR / (N_FFT * PAD)
    out = []
    for start, f0 in voiced_f0(wav, SR):
        spec = spectrum(wav[start : start + N_FFT])
        f0 = refine_f0(np.log(spec + 1e-9), f0, step)
        stats = band_stats(spec, f0, step)
        out.append((f0, [s[0] for s in stats], [s[1] for s in stats]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--emilia-manifest", type=Path, default=Path("data/m2/emilia_hq/manifest.jsonl"))
    parser.add_argument("--clips", type=int, default=150, help="clips per source (emilia: 3x)")
    parser.add_argument("--only", nargs="*", default=None, help="names of the sources to run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    jvs_wavs: dict[str, list[Path]] = {
        g: [p for s in jvs_speakers(args.jvs_root, g) for p in (args.jvs_root / s / "parallel100" / "wav24kHz16bit").glob("*.wav")]
        for g in ("F", "M")
    }
    jvs001 = sorted((args.jvs_root / "jvs001").glob("*/wav24kHz16bit/*.wav"))
    sources: dict[str, Callable[[], Iterator[Wave]]] = {
        "jvs_F": lambda: from_files(jvs_wavs["F"], args.clips, rng),
        "jvs_M": lambda: from_files(jvs_wavs["M"], args.clips, rng),
        "emilia_hq": lambda: from_emilia(args.emilia_manifest, 3 * args.clips, rng),
        "hfc_F": lambda: from_zip(NAS / "hi-fi-captain/origin/hfc_ja-JP_F.zip", "hi-fi-captain/ja-JP/female/wav/train", args.clips, rng),
        "hfc_M": lambda: from_zip(NAS / "hi-fi-captain/origin/hfc_ja-JP_M.zip", "hi-fi-captain/ja-JP/male/wav/train", args.clips, rng),
        "jsut": lambda: from_zip(NAS / "jsut/origin/jsut_ver1.1.zip", "jsut_ver1.1/", args.clips, rng),
        "jvnv": lambda: from_zip(NAS / "jvnv/origin/jvnv_ver1.zip", "jvnv_v1/", args.clips, rng),
        "amitaro": lambda: from_zip(NAS / "amitaro/origin/ITAcorpus_amitaro_2.1.zip", "", args.clips, rng),
        "jvs001": lambda: from_files(jvs001, args.clips, rng),
        "irodori_jvs001": lambda: from_files(sorted(IRODORI_JVS001.glob("*.wav")), args.clips, rng),
    }
    comb: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    power: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for name, make in sources.items():
        if args.only and name not in args.only:
            continue
        clips = 0
        for wav in make():
            clips += 1
            for f0, c, p in analyze(wav):
                for group, lo, hi in F0_GROUPS:
                    if lo <= f0 < hi:
                        comb[name, group].append(c)
                        power[name, group].append(p)
        print(f"{name}: {clips} clips", file=sys.stderr, flush=True)
    head = " | ".join(f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS)
    for group, _, _ in F0_GROUPS:
        print(f"\n{group}: comb dB / band level dB re the 1-2.5k band\n| source | frames | {head} |")
        print("|---|---:|" + "---|" * len(BANDS))
        for name in sources:
            if (name, group) not in comb:
                continue
            c = np.mean(comb[name, group], axis=0)
            p = np.mean(power[name, group], axis=0)
            level = 10 * np.log10(p / p[0])
            print(f"| {name} | {len(comb[name, group])} | " + " | ".join(f"{a:.1f} / {b:+.1f}" for a, b in zip(c, level)) + " |")


if __name__ == "__main__":
    main()
