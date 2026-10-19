"""CPU speed of the decode side of Mimi, frame by frame as pocket-tts streams it (1 thread by default).

    uv run python scripts/m2/decoder_speed.py r1 r3 w96 r3w96 --seconds 20

Each spec is rN (residual blocks per stage) and / or wN (SEANet base channels), built from the stock decoder (the
weights do not change the speed). Prints the real-time factor (decode time / audio time) of 12.5 Hz frames decoded
one at a time, the median of --repeats runs.
"""

import argparse
import re
import time

import torch

from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts_ja.codec_ft import expand_residual_layers, widen_decoder
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    for spec in args.specs:
        match = re.fullmatch(r"(?:r(\d+))?(?:w(\d+))?", spec)
        assert match and spec, f"{spec}: expected rN, wN or rNwN"
        mimi = load_mimi(load_pocket_config(), device)
        expand_residual_layers(mimi, int(match.group(1) or 1))
        widen_decoder(mimi, int(match.group(2) or 64))
        mimi.eval()
        steps = round(mimi.encoder_frame_rate / mimi.frame_rate)
        frames = round(args.seconds * mimi.frame_rate)
        latents = torch.randn(1, frames, 32)  # the quantizer-space latent of the flow LM
        times = []
        with torch.no_grad():
            for _ in range(args.repeats):
                state = init_states(mimi, 1, (frames + 8) * steps)
                start = time.perf_counter()
                for t in range(frames):
                    mimi.decode_from_latent(latents[:, t : t + 1], state)
                    increment_steps(mimi, state, increment=steps)
                times.append(time.perf_counter() - start)
        params = sum(p.numel() for p in mimi.decoder.parameters()) / 1e6
        rtf = sorted(times)[len(times) // 2] / (frames / mimi.frame_rate)
        print(f"{spec}: SEANet {params:.2f}M parameters, real-time factor {rtf:.4f} ({args.threads} thread(s))", flush=True)


if __name__ == "__main__":
    main()
