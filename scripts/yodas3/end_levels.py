"""Frame levels at the end of each training clip, as the loader feeds it, for end-of-clip checks.

Decodes every clip from its latents up to speech_end + 3 frames (src/pocket_tts_ja/loader.py) and
writes, per clip, the level of the clip's loud frames (95th percentile, dB) and the levels of its
last TAIL frames (80 ms each, rounded dB). With them a short sound after a pause at the very end
(another speaker's "うん", a hum) can be looked for without decoding again: scripts/yodas3/
end_segments.py.

    uv run python scripts/yodas3/end_levels.py data/yodas3_kept/train_punct_trim_rechecked.jsonl \
        --out data/yodas3_kept/end_levels.jsonl

Resumable: clips already in --out are skipped. --sample N decodes a random N clips only.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402

from pocket_tts_ja.mimi_utils import DECODE_CONTEXT_FRAMES, load_mimi, load_pocket_config  # noqa: E402

TRAIL_FRAMES = 3  # src/pocket_tts_ja/loader.py
TAIL = 30


def frame_db(wav: np.ndarray, hop: int) -> np.ndarray:
    f = len(wav) // hop
    return 10 * np.log10((wav[: f * hop].reshape(f, hop) ** 2).mean(axis=1) + 1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    if args.sample:
        random.seed(0)
        rows = random.sample(rows, min(args.sample, len(rows)))
    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in open(args.out)}
    todo = [r for r in rows if r["id"] not in done]
    for r in todo:
        r["_frames"] = min(int(r.get("speech_end", r["lat_len"])) + TRAIL_FRAMES, int(r["lat_len"]))
    todo.sort(key=lambda r: r["_frames"])
    print(f"{len(todo)} clips to decode ({len(done)} already done)", flush=True)
    if not todo:
        return

    device = torch.device("cuda")
    mimi = load_mimi(load_pocket_config(), device)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    hop = int(round(mimi.sample_rate / mimi.frame_rate))
    base = args.manifest.parent
    mm: dict[str, np.ndarray] = {}
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]
            T = max(r["_frames"] for r in batch)
            lat = np.zeros((len(batch), T, 32), dtype=np.float32)
            for k, r in enumerate(batch):
                arr = mm.setdefault(r["lat_file"], np.load(base / r["lat_file"], mmap_mode="r"))
                lat[k, : r["_frames"]] = arr[r["lat_start"] : r["lat_start"] + r["_frames"]]
            with torch.no_grad():
                state = init_states(mimi, len(batch), (T + DECODE_CONTEXT_FRAMES) * ratio)
                wav = mimi.decode_from_latent(torch.from_numpy(lat).to(device), state)[:, 0].float().cpu().numpy()
            for k, r in enumerate(batch):
                db = frame_db(wav[k, : r["_frames"] * hop], hop)
                out.write(json.dumps({"id": r["id"], "fed": r["_frames"], "p95": round(float(np.percentile(db, 95)), 1),
                                      "tail": [round(float(x)) for x in db[-TAIL:]]}) + "\n")
            if (b // args.batch_size) % 500 == 0:
                out.flush()
                n = b + len(batch)
                el = time.time() - t0
                print(f"{n}/{len(todo)} {n / el:.0f} clips/s eta {(len(todo) - n) / (n / el) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
