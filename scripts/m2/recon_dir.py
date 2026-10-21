"""Reconstruct every wav of a directory through the Mimi encoder and one or more decoders (like recon_jvs.py, any audio).

    uv run python scripts/m2/recon_dir.py data/m2/anime/dev --out exp/m2/recon_anime \
        --decoder keep4k=runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors --decoder soup4=runs/M2H_soups/M2CH2_soup4.safetensors

writes <out>/items.jsonl (key, speaker, gender, text), <out>/orig/<key>.wav, <out>/mimi/<key>.wav and <out>/<NAME>/<key>.wav,
the layout hf_structure.py, hf_coherence.py, mos_eval.py and recon_similarity.py read. --gender and --speaker-from (a
jsonl with id and a field) fill the item fields; without them every clip is one speaker of the given gender.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pocket_tts_ja.codec_ft import load_decoder
from pocket_tts_ja.mimi_utils import decode, encode, load_mimi, load_pocket_config, to_mono_float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="directory of wavs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder", action="append", default=[], help="NAME=decoder safetensors")
    parser.add_argument("--gender", default="F")
    parser.add_argument("--speaker-from", type=Path, default=None, help="jsonl with id and --speaker-field")
    parser.add_argument("--speaker-field", default="family")
    args = parser.parse_args()

    speakers: dict[str, str] = {}
    if args.speaker_from is not None:
        for row in map(json.loads, open(args.speaker_from, encoding="utf-8")):
            speakers[row["id"]] = str(row[args.speaker_field])
    paths = sorted(args.src.glob("*.wav"))
    items = [{"key": p.stem, "speaker": speakers.get(p.stem, "all"), "gender": args.gender, "text": ""} for p in paths]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")

    device = torch.device("cuda")
    mimi = load_mimi(load_pocket_config(), device)
    decoders = {"mimi": mimi}
    for spec in args.decoder:
        name, path = spec.split("=", 1)
        m = load_mimi(load_pocket_config(), device)
        load_decoder(m, Path(path))
        decoders[name] = m
    for name in ["orig", *decoders]:
        (args.out / name).mkdir(exist_ok=True)
    for p in paths:
        wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = to_mono_float(wav, sr, mimi.sample_rate)
        sf.write(str(args.out / "orig" / p.name), x, mimi.sample_rate, subtype="PCM_16")
        latents = encode(mimi, x, device)
        for name, m in decoders.items():
            dst = args.out / name / p.name
            if dst.exists():
                continue
            y = decode(m, latents, len(x)).float().cpu().numpy()
            sf.write(str(dst), np.clip(y, -1, 1), mimi.sample_rate, subtype="PCM_16")
    print(f"{args.out}: {len(paths)} clips x {len(decoders) + 1} directories")


if __name__ == "__main__":
    main()
