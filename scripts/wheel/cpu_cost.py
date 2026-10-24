"""CPU cost of a package's options on the stock wheel: LSD steps and the size of the decoder.

Runs in the stock wheel's own environment, one process with --threads torch threads (the
"per worker" figure of wheel_generate.py):

    uvx --from pocket-tts==4.0.1 python scripts/wheel/cpu_cost.py \
        exp/irodori_distill/package/p3m_20k_w1x/japanese_dv.yaml exp/irodori_distill/eval/jsut \
        --steps 1 4 8 --decoders 64x3 96x3 128x3

- generation: TTSModel.generate_audio (FlowLM and the decoder thread together) on the first
  --items texts of the set with one prompt, seconds of compute per second of audio, for each
  --steps (sampler_decode_steps) with the package's decoder;
- the decoder alone: the package's latents of a 10 s utterance decoded frame by frame as the wheel
  streams them, for each --decoders NFxK (decoder_seanet n_filters x n_residual_layers; random
  weights for a size the package does not have: the cost does not depend on the values).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from pocket_tts import TTSModel
from pocket_tts.models.mimi import build_mimi
from pocket_tts.models.tts_model import stamp_state_names
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.config import MimiConfig


def generation_rtf(config: str, items: list[dict], steps: int, seed: int) -> tuple[float, float]:
    model = TTSModel.load_model(config=config, sampler_decode_steps=steps)
    state = model.get_state_for_audio_prompt(Path(items[0]["prompt"]))
    model.generate_audio(state, items[0]["text"])  # warm-up
    torch.manual_seed(seed)
    gen, audio = 0.0, 0.0
    for it in items:
        start = time.monotonic()
        wav = model.generate_audio(state, it["text"])
        gen += time.monotonic() - start
        audio += wav.shape[-1] / model.sample_rate
    return gen / audio, audio


def decoder_rtf(config: str, size: str, frames: int) -> float:
    model = TTSModel.load_model(config=config)
    n_filters, k = (int(v) for v in size.split("x"))
    cfg = model.config.mimi.model_dump()
    dec = dict(cfg["decoder_seanet"] or cfg["seanet"])
    if (dec["n_filters"], dec["n_residual_layers"]) != (n_filters, k):
        cfg["decoder_seanet"] = {**dec, "n_filters": n_filters, "n_residual_layers": k}
        model.mimi = build_mimi(MimiConfig(**cfg)).eval()
        stamp_state_names(model)
    mimi = model.mimi
    steps = round(mimi.encoder_frame_rate / mimi.frame_rate)
    latents = torch.randn(1, frames, model.config.mimi.quantizer.dimension) * model.flow_lm.emb_std + model.flow_lm.emb_mean
    with torch.no_grad():
        for repeat in range(2):  # the first pass warms up
            state = init_states(mimi, batch_size=1, sequence_length=(frames + 50) * steps)
            start = time.monotonic()
            for t in range(frames):
                mimi.decode_from_latent(latents[:, t : t + 1], state)
                increment_steps(mimi, state, increment=steps)
            seconds = time.monotonic() - start
    return seconds / (frames / model.config.mimi.frame_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="package yaml")
    parser.add_argument("eval_dir", type=Path, help="evaluation set (items.jsonl): texts and the prompt")
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--steps", type=int, nargs="*", default=[1, 8])
    parser.add_argument("--decoders", nargs="*", default=[], help="NFxK sizes of the decoder to time alone")
    parser.add_argument("--frames", type=int, default=250, help="frames of the decoder-only run (25 Hz)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    items = [json.loads(line) for line in open(args.eval_dir / "items.jsonl", encoding="utf-8")][: args.items]
    print(f"{args.threads} threads; generation on {len(items)} texts of {args.eval_dir.name}")
    for steps in args.steps:
        rtf, audio = generation_rtf(args.config, items, steps, args.seed)
        print(f"  generation, {steps} LSD steps: {rtf:.3f} s per s of audio ({audio:.0f} s of audio)", flush=True)
    for size in args.decoders:
        print(f"  decoder alone, n_filters x blocks {size}: {decoder_rtf(args.config, size, args.frames):.3f} s per s", flush=True)


if __name__ == "__main__":
    main()
