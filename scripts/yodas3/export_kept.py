"""Y2 -> training machine: copy the latents of the utterances that passed finalize_yodas3.py.

The per-shard latent files hold every utterance of the shard (23 GB for the 108 shards), the
manifests keep about 7% of them. This writes the kept ones back to back into <out>/latents/
part_NNN.npy (float16, about 1 GB each) and <out>/{train,valid}.jsonl pointing at them, so only
<out> has to be copied.

    uv run python scripts/yodas3/export_kept.py --root data/yodas3 --out data/yodas3_kept
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PART_FRAMES = 16_000_000  # 32 float16 per frame: 1.02 GB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/yodas3"))
    parser.add_argument("--out", type=Path, default=Path("data/yodas3_kept"))
    args = parser.parse_args()

    (args.out / "latents").mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {
        split: [json.loads(line) for line in open(args.root / f"{split}.jsonl", encoding="utf-8")]
        for split in ("train", "valid")
    }
    rows = sorted((r for part in splits.values() for r in part), key=lambda r: (r["lat_file"], r["lat_start"]))
    sources: dict[str, npt.NDArray[np.float16]] = {}
    buffer: list[npt.NDArray[np.float16]] = []
    pending: list[dict[str, Any]] = []
    filled = 0
    parts = 0

    def flush() -> None:
        nonlocal buffer, pending, filled, parts
        if not buffer:
            return
        name = f"latents/part_{parts:03d}.npy"
        np.save(args.out / name, np.concatenate(buffer))
        for r in pending:
            r["lat_file"] = name
        buffer, pending, filled, parts = [], [], 0, parts + 1

    for r in rows:
        if r["lat_file"] not in sources:
            sources = {r["lat_file"]: np.load(args.root / r["lat_file"], mmap_mode="r")}  # one shard at a time
        chunk = np.asarray(sources[r["lat_file"]][r["lat_start"] : r["lat_start"] + r["lat_len"]], dtype=np.float16)
        assert chunk.shape == (r["lat_len"], 32), (r["id"], chunk.shape)
        buffer.append(chunk)
        pending.append(r)
        r["lat_start"] = filled
        filled += r["lat_len"]
        if filled >= PART_FRAMES:
            flush()
    flush()
    for split, part in splits.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hours = sum(r["duration"] for r in part) / 3600
        print(f"{split}: {len(part)} utterances, {hours:.1f} h")
    print(f"{parts} latent files in {args.out / 'latents'}")


if __name__ == "__main__":
    main()
