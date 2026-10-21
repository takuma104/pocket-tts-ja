"""P2: training / validation manifests from packed teacher sets, and the latent statistics.

Validation holds out whole voices (--val-voices reference speakers, chosen by hash), so neither
the voice nor the text of a validation row is ever trained on (every s3 sentence is used once).
The rows get paths relative to the output directory (lat_file -> the shard, ref_file -> the voice
prompt latent), as pocket_tts_ja.dv_loader reads them.

The FlowLM normalises latents with fixed per-dimension statistics (plan §2): mean and std over
the speech frames of the training rows (--stats-rows of them), written to <name>_stats.npz.

    uv run python scripts/train/make_split.py --sets s3 --name P2_s3
    -> data/irodori_distill/mix/P2_s3_{train,valid}.jsonl, P2_s3_stats.npz
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

TEACHER = Path("data/irodori_distill/teacher")
OUT = Path("data/irodori_distill/mix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--val-voices", type=int, default=50)
    parser.add_argument("--max-val-rows", type=int, default=2000)
    parser.add_argument("--stats-rows", type=int, default=20000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in args.sets:
        d = TEACHER / s
        for r in map(json.loads, open(d / "manifest.jsonl", encoding="utf-8")):
            r["lat_file"] = os.path.relpath(d / r["lat_file"], OUT)
            r["ref_file"] = os.path.relpath(TEACHER / "refs" / f"{r['ref_speaker']}.npy", OUT)
            rows.append(r)
    speakers = sorted({r["ref_speaker"] for r in rows})
    order = sorted(speakers, key=lambda s: hashlib.md5(s.encode()).hexdigest())
    val_voices = set(order[: args.val_voices])
    train = [r for r in rows if r["ref_speaker"] not in val_voices]
    valid = [r for r in rows if r["ref_speaker"] in val_voices]
    random.Random(0).shuffle(valid)
    valid = valid[: args.max_val_rows]
    for split, part in (("train", train), ("valid", valid)):
        with open(OUT / f"{args.name}_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sample = random.Random(1).sample(train, min(args.stats_rows, len(train)))
    shards: dict[str, np.ndarray] = {}
    frames = []
    for r in sample:
        path = str(OUT / r["lat_file"])
        if path not in shards:
            shards[path] = np.load(path, mmap_mode="r")
        z = shards[path][r["lat_start"] : r["lat_start"] + r["lat_len"]]
        frames.append(np.asarray(z[r["speech_start"] : r["speech_end"]], dtype=np.float32))
    x = np.concatenate(frames)
    mean, std = x.mean(axis=0), x.std(axis=0)
    np.savez(OUT / f"{args.name}_stats.npz", mean=mean, std=std, frames=len(x))

    hours = sum(r["duration"] for r in train) / 3600
    print(f"{args.name}: train {len(train)} rows / {hours:.1f} h / {len(speakers) - len(val_voices)} voices, "
          f"valid {len(valid)} rows / {len(val_voices)} voices")
    print(f"stats over {len(x)} speech frames: mean |{np.abs(mean).mean():.3f}| (max {np.abs(mean).max():.3f}), "
          f"std {std.mean():.3f} (min {std.min():.3f}, max {std.max():.3f})")


if __name__ == "__main__":
    main()
