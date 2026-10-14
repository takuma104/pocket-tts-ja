"""M2 evaluation: JVS recordings through the Mimi encoder and one or more decoders.

JVS is never trained on. For 5 male and 5 female speakers (the first of each by id) it writes
the parallel100 recordings as they are, through the stock Mimi, and through each fine-tuned
decoder:

    <out>/items.jsonl           key, text, speaker, gender
    <out>/orig/<key>.wav        the recording (24 kHz)
    <out>/mimi/<key>.wav        stock encoder -> stock decoder
    <out>/<NAME>/<key>.wav      stock encoder -> the decoder given as --decoder NAME=FILE

    uv run python scripts/m2/recon_jvs.py --out exp/m2/recon \
        --decoder pilot5k=runs/M2_pilot/decoder_step_00005000.safetensors
    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/asr_transcribe.py exp/m2/recon/orig exp/m2/recon/mimi exp/m2/recon/pilot5k --skip-text
    uv run python scripts/analyze_first_kana.py --dataset exp/m2/recon/items.jsonl --reference text \
        orig=exp/m2/recon/asr_orig/kana.jsonl mimi=exp/m2/recon/asr_mimi/kana.jsonl \
        pilot5k=exp/m2/recon/asr_pilot5k/kana.jsonl

Directories that already hold every clip are skipped. A decoder file with a harmonic head
(pocket_tts_ja.harmonic, trained with --harmonic oracle) is driven by the recording's F0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch

from pocket_tts_ja.codec_ft import load_decoder
from pocket_tts_ja.harmonic import decode_harmonic, f0_track, harmonic_source, load_harmonic_head, refine_f0_track
from pocket_tts_ja.jvs import read_genders
from pocket_tts_ja.mimi_utils import decode, encode, load_mimi, load_pocket_config, to_mono_float


def write_wav(path: Path, wav: npt.NDArray[np.float32], sample_rate: int) -> None:
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sample_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--out", type=Path, default=Path("exp/m2/recon"))
    parser.add_argument("--per-gender", type=int, default=5)
    parser.add_argument("--decoder", action="append", default=[], help="NAME=decoder file (safetensors)")
    args = parser.parse_args()

    genders = read_genders(args.jvs_root)
    speakers: list[str] = []
    for gender in ("M", "F"):
        speakers += sorted(s for s, info in genders.items() if info[0] == gender)[: args.per_gender]
    items: list[dict[str, str]] = []
    sources: dict[str, Path] = {}
    for speaker in sorted(speakers):
        base = args.jvs_root / speaker / "parallel100"
        for line in (base / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines():
            utt, text = line.split(":", 1)
            wav_path = base / "wav24kHz16bit" / f"{utt}.wav"
            if wav_path.exists():
                key = f"{speaker}_{utt}"
                items.append({"key": key, "text": text, "speaker": speaker, "gender": genders[speaker][0]})
                sources[key] = wav_path
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))

    device = torch.device("cuda")
    mimi = load_mimi(load_pocket_config(), device)
    sample_rate = mimi.sample_rate
    jobs: list[tuple[str, Path | None]] = [("orig", None), ("mimi", None)]
    jobs += [(name, Path(path)) for name, path in (spec.split("=", 1) for spec in args.decoder)]
    todo = [(name, path) for name, path in jobs
            if not all((args.out / name / f"{it['key']}.wav").exists() for it in items)]
    if not todo:
        print("nothing to do")
        return
    waves = {}
    latents = {}
    for it in items:
        wav, sr = sf.read(str(sources[it["key"]]), dtype="float32", always_2d=False)
        waves[it["key"]] = to_mono_float(wav, sr, sample_rate)
        latents[it["key"]] = encode(mimi, waves[it["key"]], device)
    for name, path in todo:
        (args.out / name).mkdir(exist_ok=True)
        head = None
        if path is not None:
            mimi = load_mimi(load_pocket_config(), device)
            load_decoder(mimi, path)
            hop = round(sample_rate / mimi.encoder_frame_rate)
            loaded = load_harmonic_head(path, mimi.decoder.dimension, hop, device)
            if loaded is not None:
                head, config = loaded
        for it in items:
            key = it["key"]
            if name == "orig":
                wav = waves[key]
            elif head is not None:
                lat = latents[key]
                f0 = f0_track(waves[key], sample_rate, lat.shape[0] * 16, config.allowed_range, config.fill_frames)
                f0_t = torch.from_numpy(f0)[None].to(device)
                if config.refine:
                    f0_t = refine_f0_track(torch.from_numpy(waves[key])[None].to(device), f0_t, hop, sample_rate)
                with torch.no_grad():
                    out = decode_harmonic(mimi, head, lat[None], 0, harmonic_source(f0_t, hop, sample_rate, config))
                wav = out[0, 0, : len(waves[key])].float().cpu().numpy()
            else:
                wav = decode(mimi, latents[key], len(waves[key])).float().cpu().numpy()
            write_wav(args.out / name / f"{key}.wav", wav, sample_rate)
        print(f"{name}: {len(items)} clips", flush=True)


if __name__ == "__main__":
    main()
