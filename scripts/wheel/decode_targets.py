"""W1: the DACVAE decode of the teacher latents, once, as the decoder distillation targets.

DACVAE decodes at about 270x real time on a 5090 (batching does not help: the padding costs as
much as it saves), so decoding the windows at every step or every clip at every run would take a
large share of the decoder training. Every clip gets --silence-frames frames of DACVAE's latent
of digital silence after it (dv_silence_latent.npy): what a student emits after EOS and the wheel
still decodes (model_recommended_frames_after_eos), so a delayed decoder (train_decoder.py
--delay) learns to finish the speech on those frames.

The clips are taken in the order of a hash of their names across all directories, so a partial
output is an even mix and training can start on it (train_decoder.py rescans). --shard I/N lets
several processes share the GPU.

    uv run python scripts/wheel/decode_targets.py data/irodori_distill/w1_targets \
        --latents data/irodori_distill/teacher/s4/latents --latents data/irodori_distill/teacher/s5/latents

Writes out/lat/<id>.npy (float16 [frames + silence, 32]) and out/wav/<id>.npy (int16, 24 kHz,
960 samples per frame), the wav last, so a clip with a wav is complete.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

from pocket_tts_ja.dacvae import DACVAE
from pocket_tts_ja.dacvae import HOP as DV_HOP
from pocket_tts_ja.dacvae import SAMPLE_RATE as DV_SR
from pocket_tts_ja.dv_loader import SILENCE_LATENT

SR = 24000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--latents", type=Path, action="append", required=True, help="directory of <id>.npy")
    parser.add_argument("--silence-frames", type=int, default=12)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="the first clips in hash order")
    parser.add_argument("--shard", default="0/1", help="I/N: this process takes every N-th clip from the I-th")
    args = parser.parse_args()

    shard, shards = map(int, args.shard.split("/"))
    (args.out / "lat").mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    paths = sorted((p for d in args.latents for p in d.glob("*.npy")), key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
    paths = paths[: args.limit][shard::shards]
    silence = np.load(SILENCE_LATENT)[: args.silence_frames].astype(np.float16)
    codec = DACVAE.load("cuda")
    hop = DV_HOP * SR // DV_SR
    done = skipped = 0
    seconds = 0.0
    t0 = time.time()
    for i, p in enumerate(paths):
        wav_path = args.out / "wav" / p.name
        if wav_path.exists():
            continue
        z = np.load(p)
        if z.shape[0] < args.min_frames:
            skipped += 1
            continue
        z = np.concatenate([z.astype(np.float16), silence])
        y = AF.resample(codec.decode(torch.from_numpy(z.astype(np.float32))[None])[0], DV_SR, SR)
        y = torch.nn.functional.pad(y[: len(z) * hop], (0, max(0, len(z) * hop - len(y))))
        np.save(args.out / "lat" / p.name, z)
        np.save(wav_path, (y.clamp(-1, 1) * 32767).round().short().cpu().numpy())
        done += 1
        seconds += len(z) / 25
        if done % 2000 == 0:
            print(f"{i + 1}/{len(paths)} decoded {done} ({seconds / 3600:.1f} h, {seconds / (time.time() - t0):.0f}x real time)",
                  flush=True)
    print(f"DECODE_DONE {done} decoded, {skipped} shorter than {args.min_frames} frames, {seconds / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
