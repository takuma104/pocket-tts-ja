"""W1 encoder: (24 kHz audio, DACVAE latent) pairs for a stock-wheel encoder of DACVAE latents.

The wheel clones a voice by running its Mimi encoder on the prompt audio (resampled to 24 kHz,
at whatever loudness the user gives) and feeding the latents to the FlowLM through
speaker_proj. The students were trained on prompts made by DACVAE: the reference at 48 kHz,
loudness-normalised to -16 dB (scripts/teacher/encode_refs.py, as Irodori). So the target of a
clip is that latent (pocket_tts_ja.dacvae: torchaudio resampling, which the encoder is sensitive
to), and the input is the clip at 24 kHz at its own loudness (train_encoder.py varies the gain).

Sources (CC BY only):
  --wav-dir DIR       every <id>.wav (the teacher's references: synth_src/refs_cc, refs_yodas)
  --emilia-tar NAME   an Emilia-YODAS JA shard (e.g. JA-B000000), downloaded from
                      amphion/Emilia-Dataset into the Hugging Face cache; its mp3s (24 kHz)
Clips outside --min-sec / --max-sec are skipped (the wheel's prompts are a few seconds to 30 s).

    uv run python scripts/wheel/encode_targets.py data/irodori_distill/w1_enc_targets \
        --wav-dir data/irodori_distill/synth_src/refs_cc --emilia-tar JA-B000000

Writes out/lat/<id>.npy (float16 [frames, 32], 25 Hz) and out/wav/<id>.npy (int16, 24 kHz), the
wav last (renamed when complete), and out/<source>.done when a source is finished.
"""

import argparse
import io
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download

from pocket_tts_ja.dacvae import DACVAE, REF_LOUDNESS_DB, normalize_loudness
from pocket_tts_ja.dacvae import SAMPLE_RATE as DV_SR

SR = 24000


def wav_dir_clips(d: Path) -> Iterator[tuple[str, np.ndarray, int]]:
    for p in sorted(d.glob("*.wav")):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        yield p.stem, x.mean(axis=1), sr


def emilia_clips(shard: str) -> Iterator[tuple[str, np.ndarray, int]]:
    path = hf_hub_download("amphion/Emilia-Dataset", f"Emilia-YODAS/JA/{shard}.tar", repo_type="dataset")
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.name.endswith(".mp3"):
                continue
            f = tar.extractfile(member)
            assert f is not None
            x, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
            yield Path(member.name).stem, x.mean(axis=1), sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--wav-dir", type=Path, action="append", default=[])
    parser.add_argument("--emilia-tar", action="append", default=[])
    parser.add_argument("--min-sec", type=float, default=3.0)
    parser.add_argument("--max-sec", type=float, default=20.0)
    args = parser.parse_args()

    (args.out / "lat").mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    codec = DACVAE.load("cuda")
    sources = [(d.name, wav_dir_clips(d)) for d in args.wav_dir] + [(s, emilia_clips(s)) for s in args.emilia_tar]
    for name, clips in sources:
        if (args.out / f"{name}.done").exists():
            continue
        done = skipped = 0
        seconds = 0.0
        t0 = time.time()
        for key, x, sr in clips:
            wav_path = args.out / "wav" / f"{key}.npy"
            if wav_path.exists():
                continue
            if not args.min_sec <= len(x) / sr <= args.max_sec:
                skipped += 1
                continue
            t = torch.from_numpy(x)[None]
            x48 = AF.resample(t, sr, DV_SR)[0].numpy() if sr != DV_SR else x
            z = codec.encode(normalize_loudness(x48, DV_SR, REF_LOUDNESS_DB))[0]
            x24 = AF.resample(t, sr, SR)[0] if sr != SR else t[0]
            np.save(args.out / "lat" / f"{key}.npy", z.cpu().numpy().astype(np.float16))
            part = wav_path.with_suffix(".part")
            with open(part, "wb") as f:
                np.save(f, (x24.clamp(-1, 1) * 32767).round().short().numpy())
            part.replace(wav_path)
            done += 1
            seconds += len(x) / sr
            if done % 2000 == 0:
                print(f"{name}: {done} clips ({seconds / 3600:.1f} h, {seconds / (time.time() - t0):.0f}x real time)", flush=True)
        (args.out / f"{name}.done").touch()
        print(f"{name}: ENCODED {done} clips, {seconds / 3600:.1f} h; {skipped} outside {args.min_sec}-{args.max_sec} s", flush=True)
    print("ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()
