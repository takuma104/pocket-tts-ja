"""How training clips end: level of the last frames and length of the trailing quiet part.

Decodes each clip from its Mimi latents exactly as the loader feeds it (up to speech_end + 3
frames) and measures the 80 ms frame levels against the clip's loud level (95th percentile).

    uv run python scripts/yodas3/clip_ends.py NAME=MANIFEST[:N] ...
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402

from pocket_tts_ja.mimi_utils import DECODE_CONTEXT_FRAMES, load_mimi, load_pocket_config  # noqa: E402

TRAIL_FRAMES = 3
BATCH = 16


def main() -> None:
    device = torch.device("cuda")
    mimi = load_mimi(load_pocket_config(), device)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    hop = int(round(mimi.sample_rate / mimi.frame_rate))
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        manifest = Path(parts[0])
        n = int(parts[1]) if len(parts) > 1 else 1000
        rows = [json.loads(line) for line in open(manifest)]
        random.seed(0)
        rows = random.sample(rows, min(n, len(rows)))
        for r in rows:
            r["_frames"] = min(int(r.get("speech_end", r["lat_len"])) + TRAIL_FRAMES, int(r["lat_len"]))
        base = manifest.parent
        mm: dict[str, np.ndarray] = {}
        last_db, quiet_frames, cut_frames = [], [], []
        for b in range(0, len(rows), BATCH):
            batch = rows[b : b + BATCH]
            T = max(r["_frames"] for r in batch)
            lat = np.zeros((len(batch), T, 32), dtype=np.float32)
            for k, r in enumerate(batch):
                arr = mm.setdefault(r["lat_file"], np.load(base / r["lat_file"], mmap_mode="r"))
                lat[k, : r["_frames"]] = arr[r["lat_start"] : r["lat_start"] + r["_frames"]]
            with torch.no_grad():
                state = init_states(mimi, len(batch), (T + DECODE_CONTEXT_FRAMES) * ratio)
                wav = mimi.decode_from_latent(torch.from_numpy(lat).to(device), state)[:, 0].float().cpu().numpy()
            for k, r in enumerate(batch):
                f = r["_frames"]
                w = wav[k, : f * hop].reshape(f, hop)
                db = 10 * np.log10((w**2).mean(axis=1) + 1e-10)
                ref = np.percentile(db, 95)
                rel = db - ref
                last_db.append(float(rel[-2:].max()))
                q = 0
                while q < f and rel[f - 1 - q] < -30:
                    q += 1
                quiet_frames.append(q)
                cut_frames.append(int(r["lat_len"]) - f)
        last_db_a, quiet = np.array(last_db), np.array(quiet_frames)
        print(f"{name}: {len(rows)} clips | last 2 frames vs loud level: median {np.median(last_db_a):.1f} dB, "
              f"> -20 dB {100 * (last_db_a > -20).mean():.1f}%, > -30 dB {100 * (last_db_a > -30).mean():.1f}% | "
              f"trailing quiet (< -30 dB) median {np.median(quiet) * 80:.0f} ms, 0 frames {100 * (quiet == 0).mean():.1f}%, "
              f">= 3 frames {100 * (quiet >= 3).mean():.1f}% | frames cut after speech_end+3: median {np.median(cut_frames):.0f}",
              flush=True)


if __name__ == "__main__":
    main()
