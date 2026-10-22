"""W1: a student's free generation decoded by a stock-wheel decoder instead of DACVAE.

student_synth.py keeps the generated latents (<eval dir>/<name>_latents/<key>.npy) beside its
DACVAE decode (<eval dir>/<name>/). This decodes the same latents with a decoder from
scripts/wheel/train_decoder.py, streaming as the wheel does, so the two can be scored side by
side with the usual tools (asr_transcribe, kana_cer, similarity, nisqa_eval, tts_comb). A decoder
trained with --delay K gets K frames of the silence latent after the clip and its output is
taken K frames late; the wheel instead plays the K frames of silence first.

    uv run python scripts/wheel/decode_with.py exp/irodori_distill/eval/voices P2_60k_cfg1.5 \
        runs/irodori_distill/W1_main_d2/decoder_step_00150000.safetensors:2 --name P2_60k_cfg1.5_W1

Writes <eval dir>/<--name>/<key>.wav (24 kHz).
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pocket_tts_ja.dv_loader import SILENCE_LATENT
from pocket_tts_ja.mimi_utils import decode
from pocket_tts_ja.wheel_codec import load_dv_decoder

SR = 24000
HOP = 960


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("latents", help="result name whose <name>_latents to decode")
    parser.add_argument("decoder", help="decoder safetensors[:delay in frames]")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    file, _, delay = args.decoder.partition(":")
    k = int(delay or 0)
    device = torch.device("cuda")
    mimi = load_dv_decoder(Path(file), device)
    silence = torch.from_numpy(np.load(SILENCE_LATENT).astype(np.float32)).to(device)
    out = args.eval_dir / args.name
    out.mkdir(exist_ok=True)
    paths = sorted((args.eval_dir / f"{args.latents}_latents").glob("*.npy"))
    for p in paths:
        z = torch.from_numpy(np.load(p).astype(np.float32)).to(device)
        n = z.shape[0] * HOP
        with torch.no_grad():
            y = decode(mimi, torch.cat([z, silence[:k]]), n + k * HOP)[k * HOP :].float().cpu().numpy()
        sf.write(str(out / f"{p.stem}.wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
    print(f"{out}: {len(paths)} clips")


if __name__ == "__main__":
    main()
