"""ViSQOL (audio mode, 48 kHz) of decoded audio against the recordings, per directory, by gender.

ViSQOL v3 (google/visqol, here the pure Python port visqol-python, which matches the C++ scores) compares each
decoded file with its recording on a gammatone spectrogram (32 bands from 50 Hz to 24 kHz, ERB spaced) by a
structural similarity (NSIM) of patches, and maps the per-band similarities to a MOS-LQO (1 to about 4.73, the
recording against itself) by an SVR trained on codec listening tests. The 24 kHz files are upsampled to 48 kHz,
so the bands above 12 kHz are empty in both and near 1. Besides the MOS, the mean similarity of the bands below
4 kHz, at 4-8 kHz and at 8-12 kHz is shown: unlike UTMOS22 / UTMOSv2 (16 kHz) it sees the band above 8 kHz.

It measures closeness to the recording, not naturalness: a decoder with plausible but different detail loses.
Use it next to mos_eval.py, as a guard. Both signals are high-passed at 40 Hz like mos_eval.py --highpass 40.
Clips too short for one patch (under about 2 s) are skipped. Each directory's scores are cached in
<root>/visqol_<name>.json and reused.

    uv run python scripts/m2/visqol_eval.py exp/m2/recon keep4k M2H_H2gh M2H_H2ghSG4 --ref keep4k
"""

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import butter, resample_poly, sosfiltfilt  # noqa: E402
from visqol import VisqolApi  # noqa: E402
from visqol.gammatone import calc_center_freqs  # noqa: E402

BANDS = {"nsim <4k": (0.0, 4000.0), "nsim 4-8k": (4000.0, 8000.0), "nsim 8-12k": (8000.0, 12000.0)}
CENTER_FREQS = np.sort(calc_center_freqs(50.0, 24000.0, 32))  # the order of fvnsim (low to high)
_api: VisqolApi | None = None


def load(path: Path) -> npt.NDArray[np.float64]:
    x, sr = sf.read(path, dtype="float64")
    x = sosfiltfilt(butter(4, 40, "highpass", fs=sr, output="sos"), x)
    return resample_poly(x, 48000 // sr, 1) if sr != 48000 else x


def score_one(paths: tuple[Path, Path]) -> tuple[str, dict[str, float | list[float]] | None]:
    global _api
    if _api is None:
        logging.disable(logging.WARNING)
        _api = VisqolApi()
        _api.create(mode="audio")
    ref, deg = load(paths[0]), load(paths[1])
    n = min(len(ref), len(deg))
    try:
        res = _api.measure_from_arrays(ref[:n], deg[:n], sample_rate=48000)
    except ValueError:  # shorter than one patch
        return paths[1].stem, None
    return paths[1].stem, {"visqol": float(res.moslqo), "fvnsim": [float(v) for v in res.fvnsim]}


def metric(s: dict[str, float | list[float]], name: str) -> float:
    if name == "visqol":
        return float(s["visqol"])  # type: ignore[arg-type]
    lo, hi = BANDS[name]
    return float(np.mean(np.asarray(s["fvnsim"])[(CENTER_FREQS >= lo) & (CENTER_FREQS < hi)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory holding orig/ and one subdirectory of wavs per decoder")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--ref", default=None, help="also print the paired difference from this name")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.root / "items.jsonl", encoding="utf-8") as f:
        items = {it["key"]: it for it in map(json.loads, f)}
    names = list(dict.fromkeys(args.names + ([args.ref] if args.ref else [])))
    scores: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for name in names:
        cache = args.root / f"visqol_{name}.json"
        if cache.exists():
            scores[name] = json.loads(cache.read_text())
            continue
        paths = sorted((args.root / name).glob("*.wav"))
        print(f"{name}: {len(paths)} files", flush=True)
        jobs = [(args.root / "orig" / p.name, p) for p in paths]
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            scores[name] = {k: s for k, s in pool.imap_unordered(score_one, jobs, chunksize=8) if s is not None}
        cache.write_text(json.dumps(scores[name]))

    common = sorted(set.intersection(*(set(s) for s in scores.values())))
    groups = {"all": common}
    for g in ("F", "M"):
        sel = [k for k in common if items.get(k, {}).get("gender") == g]
        if sel:
            groups[g] = sel
    print(f"{len(common)} utterances in every directory")
    for m in ("visqol", *BANDS):
        header = f"| {m} | " + " | ".join(groups) + (" | Δ" + args.ref + " (all) | Δ (F) |" if args.ref else " |")
        print("\n" + header)
        print("|---" * (header.count("|") - 1) + "|")
        for name in names:
            row = [f"{np.mean([metric(scores[name][k], m) for k in sel]):.3f}" for sel in groups.values()]
            if args.ref:
                for g in ("all", "F"):
                    sel = groups.get(g, [])
                    d = np.array([metric(scores[name][k], m) - metric(scores[args.ref][k], m) for k in sel])
                    row.append(f"{d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}" if len(d) > 1 else "")
            print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
