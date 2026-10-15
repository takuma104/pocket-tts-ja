"""Move the speech end of YODAS3 clips to where the speech ends, as in Emilia.

The loader feeds each clip up to speech_end + 3 frames, and EOS is trained on the frame after.
speech_end is the last 80 ms frame within 35 dB of the clip's peak; on YouTube audio the
background (music, room noise) often stays above that, so the fed YODAS3 clip goes on ~240 ms
after the last speech (median; 74% have 3 or more quiet frames), while Emilia and the synthetic
sets stop ~80 ms after it (3+ quiet frames in 6-11%). Trained on both, B-CC fires EOS early and
keeps talking in the frames generated after it ("、ン", "、エー").

This decodes every clip from its latents exactly as the loader feeds it, finds the last frame
within 30 dB of the clip's loud level (95th percentile of the frame levels) and sets speech_end
so that one quiet frame follows it in the fed clip (Emilia's median). The old value is kept as
speech_end_orig.

    uv run python scripts/yodas3/trim_tails.py data/yodas3_kept/train_punct.jsonl \
        --levels data/yodas3_kept/tail_levels.jsonl --out data/yodas3_kept/train_punct_trim.jsonl

Resumable: clips already in --levels are not decoded again.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402

from pocket_tts_ja.mimi_utils import DECODE_CONTEXT_FRAMES, load_mimi, load_pocket_config  # noqa: E402

TRAIL_FRAMES = 3  # src/pocket_tts_ja/loader.py
QUIET_DB = -30.0
QUIET_KEEP = 1


def last_loud_frame(wav: np.ndarray, hop: int) -> int:
    """Index of the last frame within QUIET_DB of the clip's loud level."""
    f = len(wav) // hop
    db = 10 * np.log10((wav[: f * hop].reshape(f, hop) ** 2).mean(axis=1) + 1e-10)
    loud = np.nonzero(db >= np.percentile(db, 95) + QUIET_DB)[0]
    return int(loud[-1]) if len(loud) else f - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--levels", type=Path, required=True, help="per-clip results (resumable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    done: dict[str, int] = {}
    if args.levels.exists():
        done = {(r := json.loads(line))["id"]: r["last_loud"] for line in open(args.levels)}
    todo = [r for r in rows if r["id"] not in done]
    for r in todo:
        r["_frames"] = min(int(r.get("speech_end", r["lat_len"])) + TRAIL_FRAMES, int(r["lat_len"]))
    todo.sort(key=lambda r: r["_frames"])
    print(f"{len(todo)} clips to decode ({len(done)} already done)", flush=True)

    if todo:
        device = torch.device("cuda")
        mimi = load_mimi(load_pocket_config(), device)
        ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
        hop = int(round(mimi.sample_rate / mimi.frame_rate))
        base = args.manifest.parent
        mm: dict[str, np.ndarray] = {}
        t0 = time.time()
        with open(args.levels, "a", encoding="utf-8") as out:
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
                    ll = last_loud_frame(wav[k, : r["_frames"] * hop], hop)
                    done[r["id"]] = ll
                    out.write(json.dumps({"id": r["id"], "fed": r["_frames"], "last_loud": ll}) + "\n")
                if (b // args.batch_size) % 500 == 0:
                    out.flush()
                    n = b + len(batch)
                    el = time.time() - t0
                    print(f"{n}/{len(todo)} {n / el:.0f} clips/s eta {(len(todo) - n) / (n / el) / 3600:.2f}h", flush=True)

    moved, cut = 0, []
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            fed = min(int(r.get("speech_end", r["lat_len"])) + TRAIL_FRAMES, int(r["lat_len"]))
            new_fed = min(fed, done[r["id"]] + 1 + QUIET_KEEP)
            if new_fed < fed:
                moved += 1
                cut.append(fed - new_fed)
                r["speech_end_orig"] = r.get("speech_end", r["lat_len"])
                r["speech_end"] = new_fed - TRAIL_FRAMES
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"moved the speech end of {moved} / {len(rows)} clips "
          f"(frames cut: median {np.median(cut) if cut else 0:.0f}, mean {np.mean(cut) if cut else 0:.1f}); wrote {args.out}")


if __name__ == "__main__":
    main()
