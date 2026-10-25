"""The longest pause inside each training row's teacher latent, from the latent alone.

The QC audio of most teacher sets is gone, so silence is read off the latents: a frame is silent
when it lies within --threshold of the silence centroid, both fitted on sets that still have their
16 kHz QC audio (frames more than 35 dB below the clip's 95th percentile, as scripts/eval/pauses.py;
on num: accuracy 0.92, precision 0.84, recall 0.67, so the runs come out a little short). A pause
is a run of silent frames inside speech_start..speech_end.

(10/26: female_04's pauses through S1 grow with every fine-tune: median 0.58 s at S1_60k, 0.91 at
S1h, 1.30 at S1n, the teacher's 0.40; num's and latin's teacher clips pause 1 s or more 7-8% of
the time, the other sets 0.3-1.4%.)

    uv run python scripts/train/latent_pauses.py data/irodori_distill/mix/S1nl_all_train.jsonl --calib num latin
    -> <manifest>.pauses.json {row id: longest pause s}, and the share of rows over a few lengths per set
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

TEACHER = Path("data/irodori_distill/teacher")
FRAME_RATE = 25
QC_HOP = 640  # 16 kHz samples per latent frame


class Latents:
    def __init__(self) -> None:
        self.shards: dict[Path, np.ndarray] = {}

    def __call__(self, path: Path, start: int, length: int) -> np.ndarray:
        if path not in self.shards:
            self.shards[path] = np.load(path, mmap_mode="r")
        return np.asarray(self.shards[path][start : start + length], dtype=np.float32)


def calibrate(sets: list[str], rows_per_set: int, lat: Latents) -> tuple[np.ndarray, float]:
    xs, ys = [], []
    for s in sets:
        man = [json.loads(line) for line in open(TEACHER / s / "manifest.jsonl", encoding="utf-8")]
        for r in random.Random(0).sample(man, min(rows_per_set, len(man))):
            w, _ = sf.read(TEACHER / s / "audio16k" / f"{r['id']}.wav")
            z = lat(TEACHER / s / r["lat_file"], r["lat_start"], r["lat_len"])
            n = min(len(z), len(w) // QC_HOP)
            db = 20 * np.log10(np.sqrt((w[: n * QC_HOP].reshape(n, QC_HOP) ** 2).mean(1)) + 1e-9)
            xs.append(z[:n])
            ys.append(db < np.percentile(db, 95) - 35)
    x, y = np.concatenate(xs), np.concatenate(ys)
    centroid = x[y].mean(0)
    d = np.linalg.norm(x - centroid, axis=1)
    threshold = float(max(np.linspace(0.5, 6.0, 111), key=lambda t: ((d < t) == y).mean()))
    sil = d < threshold
    print(f"calibration on {sets}: {len(y)} frames, silent {y.mean():.2f}; threshold {threshold:.2f}: accuracy "
          f"{(sil == y).mean():.3f}, precision {y[sil].mean():.3f}, recall {sil[y].mean():.3f}")
    return centroid, threshold


def longest_pause(z: np.ndarray, centroid: np.ndarray, threshold: float) -> float:
    sil = np.linalg.norm(z - centroid, axis=1) < threshold
    best = run = 0
    for x in sil:
        run = run + 1 if x else 0
        best = max(best, run)
    return best / FRAME_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--calib", nargs="+", default=["num", "latin"])
    parser.add_argument("--calib-rows", type=int, default=300)
    args = parser.parse_args()

    lat = Latents()
    centroid, threshold = calibrate(args.calib, args.calib_rows, lat)
    out: dict[str, float] = {}
    per_set: dict[str, list[float]] = defaultdict(list)
    for line in open(args.manifest, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in out:  # rows listed twice
            continue
        z = lat(args.manifest.parent / r["lat_file"], r["lat_start"], r["lat_len"])[r["speech_start"] : r["speech_end"]]
        out[r["id"]] = longest_pause(z, centroid, threshold)
        per_set[r["set"]].append(out[r["id"]])
    path = args.manifest.with_suffix(".pauses.json")
    path.write_text(json.dumps(out))
    print(f"{len(out)} rows -> {path}")
    print("| set | rows | longest pause >= 0.6 s | >= 0.8 s | >= 1.0 s |")
    for s, v in sorted(per_set.items()):
        a = np.array(v)
        print(f"| {s} | {len(a)} | {(a >= 0.6).mean():.1%} | {(a >= 0.8).mean():.1%} | {(a >= 1.0).mean():.1%} |")


if __name__ == "__main__":
    main()
