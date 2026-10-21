"""Codec round trips for P0b: recordings through Mimi (stock and fine-tuned decoders) and DACVAE.

Writes the layout every scripts/eval tool reads (as mimi_ft's recon_jvs.py / recon_dir.py):

    <out>/items.jsonl         key, text, speaker, gender
    <out>/orig/<key>.wav      the source at 24 kHz
    <out>/mimi/<key>.wav      stock Mimi encoder -> stock decoder
    <out>/<NAME>/<key>.wav    stock Mimi encoder -> the decoder given as --decoder NAME=FILE
    <out>/dacvae/<key>.wav    DACVAE (Semantic-DACVAE-Japanese-32dim) at 48 kHz, resampled to 24 kHz
    <out>/dacvae48/<key>.wav  the same at 48 kHz (with --dacvae48; for listening)

Sources: --jvs N takes the parallel100 clips of the first N female and N male JVS speakers;
--src DIR takes every wav of a directory, with --meta (jsonl: id and optional text, speaker,
gender) for the item fields. All comparisons are made at 24 kHz, the rate of every source here.
DACVAE gets the source resampled to 48 kHz (torchaudio, as Irodori) without loudness
normalisation. Directories that already hold every clip are skipped.

    uv run python scripts/eval/recon.py exp/irodori_distill/p0/recon_jvs --jvs 5 \
        --decoder keep4k=runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr
import torch

from pocket_tts_ja.dacvae import SAMPLE_RATE as DACVAE_SR
from pocket_tts_ja.dacvae import DACVAE, load_wav
from pocket_tts_ja.jvs import read_genders
from pocket_tts_ja.mimi_utils import decode, encode, load_mimi, load_pocket_config, to_mono_float

SR = 24000


def write_wav(path: Path, wav: npt.NDArray[np.float32], sample_rate: int) -> None:
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sample_rate, subtype="PCM_16")


def jvs_items(root: Path, per_gender: int) -> tuple[list[dict[str, str]], dict[str, Path]]:
    genders = read_genders(root)
    speakers: list[str] = []
    for gender in ("M", "F"):
        speakers += sorted(s for s, info in genders.items() if info[0] == gender)[:per_gender]
    items: list[dict[str, str]] = []
    sources: dict[str, Path] = {}
    for speaker in sorted(speakers):
        base = root / speaker / "parallel100"
        for line in (base / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines():
            utt, text = line.split(":", 1)
            wav_path = base / "wav24kHz16bit" / f"{utt}.wav"
            if wav_path.exists():
                key = f"{speaker}_{utt}"
                items.append({"key": key, "text": text, "speaker": speaker, "gender": genders[speaker][0]})
                sources[key] = wav_path
    return items, sources


def dir_items(src: Path, meta: Path | None, gender: str) -> tuple[list[dict[str, str]], dict[str, Path]]:
    rows: dict[str, dict[str, str]] = {}
    if meta is not None:
        rows = {r["id"]: r for r in map(json.loads, open(meta, encoding="utf-8"))}
    items: list[dict[str, str]] = []
    sources: dict[str, Path] = {}
    for path in sorted(src.glob("*.wav")):
        row = rows.get(path.stem, {})
        items.append({
            "key": path.stem,
            "text": str(row.get("text", "")),
            "speaker": str(row.get("speaker", "all")),
            "gender": str(row.get("gender", gender)),
        })
        sources[path.stem] = path
    return items, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--jvs", type=int, default=0, help="speakers per gender from JVS parallel100")
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--src", type=Path, default=None, help="directory of wavs")
    parser.add_argument("--meta", type=Path, default=None, help="jsonl with id and optional text / speaker / gender")
    parser.add_argument("--gender", default="F", help="gender of --src clips without one in --meta")
    parser.add_argument("--decoder", action="append", default=[], help="NAME=Mimi decoder file (safetensors)")
    parser.add_argument("--dacvae48", action="store_true", help="also keep DACVAE's 48 kHz output")
    args = parser.parse_args()

    if args.jvs:
        items, sources = jvs_items(args.jvs_root, args.jvs)
    elif args.src is not None:
        items, sources = dir_items(args.src, args.meta, args.gender)
    else:
        parser.error("give --jvs N or --src DIR")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8"
    )

    device = torch.device("cuda")
    mimi_jobs: list[tuple[str, Path | None]] = [("mimi", None)]
    mimi_jobs += [(name, Path(path)) for name, path in (spec.split("=", 1) for spec in args.decoder)]
    names = ["orig", "dacvae", *(["dacvae48"] if args.dacvae48 else []), *(n for n, _ in mimi_jobs)]

    def done(name: str) -> bool:
        return all((args.out / name / f"{it['key']}.wav").exists() for it in items)

    todo = [n for n in names if not done(n)]
    if not todo:
        print("nothing to do")
        return
    for name in todo:
        (args.out / name).mkdir(exist_ok=True)
    waves: dict[str, npt.NDArray[np.float32]] = {}
    for it in items:
        wav, sr = sf.read(str(sources[it["key"]]), dtype="float32", always_2d=False)
        waves[it["key"]] = to_mono_float(wav, sr, SR)
        if "orig" in todo:
            write_wav(args.out / "orig" / f"{it['key']}.wav", waves[it["key"]], SR)

    if "dacvae" in todo or "dacvae48" in todo:
        codec = DACVAE.load(str(device))
        for it in items:
            x48 = load_wav(sources[it["key"]], DACVAE_SR)
            y48 = codec.decode(codec.encode(x48))[0, : len(x48)].cpu().numpy()
            if "dacvae48" in todo:
                write_wav(args.out / "dacvae48" / f"{it['key']}.wav", y48, DACVAE_SR)
            y = soxr.resample(y48, DACVAE_SR, SR, quality="HQ").astype(np.float32)[: len(waves[it["key"]])]
            write_wav(args.out / "dacvae" / f"{it['key']}.wav", y, SR)
        del codec
        print(f"dacvae: {len(items)} clips", flush=True)

    stock = load_mimi(load_pocket_config(), device)
    latents = {key: encode(stock, wav, device) for key, wav in waves.items()}
    for name, path in mimi_jobs:
        if name not in todo:
            continue
        mimi = stock if path is None else load_mimi(load_pocket_config(), device, decoder=path)
        for it in items:
            key = it["key"]
            wav = decode(mimi, latents[key], len(waves[key])).float().cpu().numpy()
            write_wav(args.out / name / f"{key}.wav", wav, SR)
        print(f"{name}: {len(items)} clips", flush=True)


if __name__ == "__main__":
    main()
