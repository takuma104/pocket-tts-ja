"""ECAPA similarity of generated clips to their voice prompt, per prompt (e.g. the voices/ of a release).

On an eval directory with audio/<key>.wav; the prompt of each item comes from --items (relative to
the directory), e.g. the voices assignment that runs/run_voices_cc.sh keeps in items_voices.json
(eval_tts.py --from-audio writes items.json again with the set's own prompts):

    uv run python scripts/release/voice_similarity.py \
        runs/D1_cc_6l/eval/step00160000/jsut_rotation_cfg1.0_wheelvoices --items items_voices.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--items", default="items.json", help="file in each directory with key / prompt")
    args = parser.parse_args()

    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
        savedir=str(REPO / "data/models/spkrec-ecapa-voxceleb"),
    )
    cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def emb(path: Path) -> torch.Tensor:
        if str(path) not in cache:
            wav, sr = sf.read(str(path), dtype="float32")
            x = AF.resample(torch.from_numpy(wav)[None].cuda(), sr, 16000)
            e = torch.nn.functional.normalize(enc.encode_batch(x)[0, 0], dim=-1)
            if "/audio/" not in str(path):
                cache[str(path)] = e
            return e
        return cache[str(path)]

    for d in args.dirs:
        items = json.loads((d / args.items).read_text())
        sims: dict[str, list[float]] = defaultdict(list)
        for it in items:
            sims[Path(it["prompt"]).stem].append(float(emb(d / "audio" / f"{it['key']}.wav") @ emb(Path(it["prompt"]))))
        print(f"## {d} ({args.items})")
        print("| prompt | items | similarity |")
        print("|---|---:|---:|")
        for name, v in sorted(sims.items()):
            print(f"| {name} | {len(v)} | {sum(v) / len(v):.3f} |")
        allv = [x for v in sims.values() for x in v]
        print(f"| all | {len(allv)} | {sum(allv) / len(allv):.3f} |")


if __name__ == "__main__":
    main()
