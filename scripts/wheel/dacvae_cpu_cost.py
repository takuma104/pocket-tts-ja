"""CPU cost of the DACVAE decoder (Snake, 48 kHz) next to a W1 decoder (the stock SEANet, 24 kHz).

Both decode the same 10 s of latents in one call on --threads torch threads (no streaming), so the
figure is the decoder's own compute per second of audio. It bounds what a W2 decoder of DACVAE's
kind would cost on the wheel's CPU path:

    uv run python scripts/wheel/dacvae_cpu_cost.py --w1 runs/irodori_distill/W1w2/decoder_step_00020000.safetensors
"""

import argparse
import time
from pathlib import Path

import torch

from pocket_tts_ja.dacvae import DACVAE, FRAME_RATE, LATENT_DIM
from pocket_tts_ja.mimi_utils import decode
from pocket_tts_ja.wheel_codec import load_dv_decoder


def timed(fn, repeats: int) -> float:
    fn()  # warm-up
    start = time.monotonic()
    for _ in range(repeats):
        fn()
    return (time.monotonic() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w1", type=Path, required=True, help="W1 decoder safetensors")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    frames = round(args.seconds * FRAME_RATE)
    latents = torch.randn(1, frames, LATENT_DIM)
    cpu = torch.device("cpu")

    dv = DACVAE.load(device="cpu")
    for m in dv.model.decoder.modules():  # fold the weight norm, as an exported decoder would
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    n_dv = sum(p.numel() for p in dv.model.decoder.parameters())
    with torch.inference_mode():
        s_dv = timed(lambda: dv.decode(latents), args.repeats)

    mimi = load_dv_decoder(args.w1, cpu).eval()
    n_w1 = sum(p.numel() for n, p in mimi.named_parameters() if n.startswith(("decoder", "upsample", "quantizer")))

    with torch.inference_mode():
        s_w1 = timed(lambda: decode(mimi, latents[0]), args.repeats)
    print(f"{args.threads} threads, {args.seconds:.0f} s of latents in one call")
    print(f"  DACVAE decoder ({n_dv / 1e6:.1f} M params, 48 kHz): {s_dv / args.seconds:.3f} s per s of audio")
    print(f"  W1 {args.w1.parent.name} ({n_w1 / 1e6:.1f} M params, 24 kHz): {s_w1 / args.seconds:.3f} s per s of audio")


if __name__ == "__main__":
    main()
