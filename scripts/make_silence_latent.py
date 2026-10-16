"""The Mimi latents of digital silence, for supervising the frames after EOS (loader post_eos_frames).

Encodes a few seconds of zeros with the stock Mimi and keeps frames from the steady middle, as
float16 [FRAMES, 32] in src/pocket_tts_ja/silence_latent.npy. Prints how far they are from the
quiet last frames of training clips and from their speech frames (normalized with the model's
latent statistics), as a check that silence lands near the clips' own quiet ends.

    uv run python scripts/make_silence_latent.py --manifest data/mix/BCC_train_v8.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "pocket_tts_ja" / "silence_latent.npy"
FRAMES = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("runs/BCC_4k_tail/snapshots/step_00550000.safetensors"),
                        help="flow_lm weights with emb_mean / emb_std, for the distance check")
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    mimi = load_mimi(load_pocket_config(), torch.device("cpu"))
    with torch.no_grad():
        lat = mimi.encode_to_latent(torch.zeros(1, 1, mimi.sample_rate * 4))[0].numpy()
    mid = lat.shape[0] // 2 - FRAMES // 2
    sil = lat[mid : mid + FRAMES]
    print(f"{lat.shape[0]} frames encoded; spread over the kept {FRAMES}: {np.abs(sil - sil.mean(0)).max():.4f}, "
          f"first frame vs the kept: {np.abs(lat[0] - sil.mean(0)).max():.3f}")
    np.save(OUT, sil.astype(np.float16))
    print(f"wrote {OUT}")

    state = safetensors.torch.load_file(str(args.weights))
    mean = state["flow_lm.emb_mean"].float().numpy()
    std = state["flow_lm.emb_std"].float().numpy()
    rows = [json.loads(line) for line in open(args.manifest)]
    random.seed(0)
    rows = random.sample(rows, args.sample)
    base = args.manifest.parent
    mm: dict[str, np.ndarray] = {}
    last, speech = [], []
    for r in rows:
        arr = mm.setdefault(r["lat_file"], np.load(base / r["lat_file"], mmap_mode="r"))
        n = min(int(r.get("speech_end", r["lat_len"])) + 3, int(r["lat_len"]))
        x = np.asarray(arr[r["lat_start"] : r["lat_start"] + n], dtype=np.float32)
        last.append(x[-1])
        speech.append(x[len(x) // 2])
    z = lambda a: (np.asarray(a) - mean) / std  # noqa: E731
    s = z(sil.mean(0))
    d_last = np.linalg.norm(z(last) - s, axis=1)
    d_speech = np.linalg.norm(z(speech) - s, axis=1)
    d_ls = np.linalg.norm(z(last) - z(speech), axis=1)
    print(f"normalized distance to silence: last frames median {np.median(d_last):.2f} "
          f"(p10 {np.percentile(d_last, 10):.2f}, p90 {np.percentile(d_last, 90):.2f}), "
          f"mid-clip frames median {np.median(d_speech):.2f}; last vs mid-clip {np.median(d_ls):.2f}")


if __name__ == "__main__":
    main()
