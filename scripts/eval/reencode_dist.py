"""How far generated latents lie from what DACVAE's encoder makes of their own decode.

Every latent clip z of a result is decoded and encoded again; the per-frame RMS of z - enc(dec(z))
(over the 32 dimensions) says how far z is from latents the codec itself would produce. The
teacher's clips sit at 0.23-0.38 by voice. On 10/24 the students sat below the teacher in eight
of the ten voices (temperature 0.7 keeps them typical) and above it only in female_02 / 05 (ratio
1.12-1.24), the two voices whose cracks the user hears: the only measure so far that singles them
out. Per speaker: the mean and 95th percentile over frames.

    uv run python scripts/eval/reencode_dist.py exp/irodori_distill/eval/voices teacher S1_60k_t07s8_cfg1.5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pocket_tts_ja.dacvae import DACVAE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="evaluation directory (items.jsonl, <name>_latents/<key>.npy)")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    codec = DACVAE.load(args.device)
    dist: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for name in args.names:
        for it in items:
            path = args.root / f"{name}_latents" / f"{it['key']}.npy"
            if not path.exists():
                continue
            z = torch.from_numpy(np.load(path).astype(np.float32))[None]
            z2 = codec.encode(codec.decode(z))[0].cpu().numpy()
            n = min(len(z2), z.shape[1])
            dist[(name, it["speaker"])].append(np.sqrt(((z[0, :n].numpy() - z2[:n]) ** 2).mean(axis=1)))
    speakers = sorted({it["speaker"] for it in items})
    print("RMS of z - enc(dec(z)) per frame: mean / 95th percentile")
    print("| speaker | " + " | ".join(args.names) + " |")
    print("|---|" + "---:|" * len(args.names))
    for sp in speakers:
        cells = []
        for name in args.names:
            d = np.concatenate(dist[(name, sp)]) if dist[(name, sp)] else np.zeros(0)
            cells.append(f"{d.mean():.3f} / {np.percentile(d, 95):.3f}" if len(d) else "-")
        print(f"| {sp} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
