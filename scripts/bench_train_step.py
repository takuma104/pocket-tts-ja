"""Measure FlowLM training throughput and peak VRAM on this GPU with synthetic batches.

Builds the trainable model exactly like training/train.py (24-layer English teacher,
LSD, bf16 autocast, per-layer torch.compile, fused AdamW) and times optimizer steps on
random latents of a fixed shape, so the numbers bound what a real run can reach once the
loader keeps up.

    uv run python scripts/bench_train_step.py --batch-size 16 --target-sec 10
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from training.args import TrainArgs
from training.modules.builders import build_models
from training.train_utils import _compile_models
from pocket_tts_ja.mimi_utils import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-sec", type=float, default=10.0)
    parser.add_argument("--prompt-sec", type=float, default=5.0)
    parser.add_argument("--text-tokens", type=int, default=60)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--flow-batch-multiplier", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device("cuda")
    targs = TrainArgs(
        model_config=str(DEFAULT_CONFIG),
        model_overrides={"flow_lm.transformer.num_layers": args.num_layers},
        start_from_pretrained=args.num_layers == 24,
        reset_text_embedding=True,
        flow_batch_multiplier=args.flow_batch_multiplier,
        compile=not args.no_compile,
    )
    model, mimi, _ = build_models(targs)
    model.to(device).train()
    if targs.compile:
        _compile_models(model, mimi)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    B, frame_rate = args.batch_size, 12.5
    T = int(args.target_sec * frame_rate)
    Tv = int(args.prompt_sec * frame_rate)
    latents = torch.randn(B, T, 32, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    voice = torch.randn(B, Tv, 32, device=device)
    n_voice = torch.full((B,), Tv, dtype=torch.long)
    tokens = [torch.randint(0, 4000, (args.text_tokens,)) for _ in range(B)]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        with autocast:
            loss, _ = model(latents, mask, tokens, voice, num_voice_prompt_frames=n_voice)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        if step < 3:
            print(f"step {step}: {times[-1]:.2f}s (warmup/compile)", flush=True)
    steady = times[len(times) // 3 :]
    per_step = sum(steady) / len(steady)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"layers={args.num_layers} params={n_params / 1e6:.1f}M B={B} target={args.target_sec}s "
        f"prompt={args.prompt_sec}s text={args.text_tokens} fbm={args.flow_batch_multiplier}: "
        f"{per_step * 1000:.0f} ms/micro-step, {1 / per_step:.2f} micro-steps/s, "
        f"peak {peak:.1f} GiB"
    )


if __name__ == "__main__":
    main()
