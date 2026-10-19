"""M2 CC data: per-clip measures of the YODAS3 clips (extract_yodas3.py) for choosing the decoder's training audio.

For each clip of <dir>/manifest.jsonl with its wav in <dir>/audio, appends a row to <dir>/scores.jsonl:

    f0           median F0 of the voiced frames (hf_structure.voiced_f0, refined on the 1-4 kHz harmonics), Hz
    voiced       number of voiced frames (2048 samples, hop 1024)
    comb         per band of hf_structure.BANDS (1-2.5, 2.5-4, 4-6, 6-8, 8-10 kHz): the mean over the voiced frames
                 of the level at the harmonics minus that halfway between them, dB (noise gives 0; the recordings of
                 JVS's women have 4.7 dB at 4-6 kHz, keep4k's reconstructions 1.3)
    level        per band, the voiced frames' power relative to the 1-2.5 kHz band, dB
    top          the clip's long-term spectrum at 8-11 and 11-12 kHz relative to 1-4 kHz, dB (a low-pass or a
                 band-limited re-encode shows as a drop at the top)
    clipped      share of the samples at 0.99 of full scale or more
    rms          the clip's level, dBFS

Clips are scaled to the same RMS before the voiced-frame analysis (data_harmonicity.prepare).

    uv run python scripts/m2/score_yodas3.py data/m2/yodas3 --workers 20
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from data_harmonicity import SR, analyze, prepare
from hf_structure import BANDS


def band_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    return float(10 * np.log10(power[(freqs >= lo) & (freqs < hi)].mean() + 1e-20))


def score(job: tuple[str, Path]) -> dict[str, Any]:
    clip_id, path = job
    raw, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    out: dict[str, Any] = {"id": clip_id, "clipped": float(np.mean(np.abs(raw) >= 0.99)),
                           "rms": float(20 * np.log10(np.sqrt(np.mean(raw**2)) + 1e-9))}
    n = len(raw) // 2048 * 2048
    frames = raw[:n].reshape(-1, 2048) * np.hanning(2048)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    ref = band_db(power, freqs, 1000, 4000)
    out["top"] = [round(band_db(power, freqs, 8000, 11000) - ref, 2), round(band_db(power, freqs, 11000, 12000) - ref, 2)]
    voiced = analyze(prepare(raw, sr))
    out["voiced"] = len(voiced)
    if voiced:
        f0 = np.array([v[0] for v in voiced])
        comb = np.array([v[1] for v in voiced])
        level = np.array([v[2] for v in voiced]).sum(axis=0)
        out["f0"] = round(float(np.median(f0)), 1)
        out["comb"] = [round(float(c), 2) for c in comb.mean(axis=0)]
        out["level"] = [round(float(10 * np.log10(p / level[0] + 1e-20)), 2) for p in level]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out_path = args.dir / "scores.jsonl"
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    jobs = []
    for line in open(args.dir / "manifest.jsonl"):
        clip_id = json.loads(line)["id"]
        path = args.dir / "audio" / f"{clip_id}.wav"
        if clip_id not in done and path.exists():
            jobs.append((clip_id, path))
    print(f"{len(jobs)} clips to score ({len(done)} done), bands {BANDS}", flush=True)
    with open(out_path, "a") as f, mp.get_context("fork").Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(score, jobs, chunksize=16)):
            f.write(json.dumps(row) + "\n")
            if k % 5000 == 0:
                print(f"{k} scored", flush=True)


if __name__ == "__main__":
    main()
