"""M2 evaluation: speaker similarity of reconstructed JVS recordings to the recordings themselves.

For every result directory of scripts/m2/recon_jvs.py (stock Mimi, fine-tuned decoders) it gives
the mean ECAPA cosine similarity between a reconstructed clip and the recording it came from, the
ceiling of the similarity a TTS with that decoder can reach in `eval_tts.py --set jvs_bal`.

    uv run python scripts/m2/recon_similarity.py exp/m2/recon mimi mel10k gan60k
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("names", nargs="+", help="result directories to compare with orig/")
    args = parser.parse_args()

    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
        savedir=str(REPO / "data/models/spkrec-ecapa-voxceleb"),
    )

    def emb(path: Path) -> torch.Tensor:
        wav, sr = sf.read(str(path), dtype="float32")
        x = AF.resample(torch.from_numpy(wav)[None].cuda(), sr, 16000)
        return torch.nn.functional.normalize(enc.encode_batch(x)[0, 0], dim=-1)

    items = [json.loads(line) for line in open(args.out / "items.jsonl", encoding="utf-8")]
    sims: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in args.names}
    with torch.no_grad():
        for it in items:
            ref = emb(args.out / "orig" / f"{it['key']}.wav")
            for name in args.names:
                sim = float(emb(args.out / name / f"{it['key']}.wav") @ ref)
                sims[name]["all"].append(sim)
                sims[name][it["gender"]].append(sim)
    groups = ["all"] + sorted(g for g in sims[args.names[0]] if g != "all")
    print("| result | clips | " + " | ".join(groups) + " |")
    print("|---|---:|" + "---:|" * len(groups))
    for name in args.names:
        cells = " | ".join(f"{np.mean(sims[name][g]):.3f}" for g in groups)
        print(f"| {name} | {len(sims[name]['all'])} | {cells} |")


if __name__ == "__main__":
    main()
