"""W1: a stock-wheel Mimi encoder (ELU SEANet, causal) that regresses DACVAE's prompt latents.

The wheel clones a voice by running its Mimi encoder on the prompt (24 kHz, any loudness) and
projecting the latents with the FlowLM's speaker_proj. The students were trained on DACVAE's
latents of the reference at -16 dB, so this encoder learns that map from the pairs of
scripts/wheel/encode_targets.py: the stock encoder (SEANet + encoder transformer, from the stock
weights) with a fresh stride-8 downsampling to 25 Hz x 32 dims (pocket_tts_ja.wheel_codec). The
input gain is drawn per clip (--gain-dbfs, RMS), the target stays the -16 dB latent, so the
encoder normalises loudness itself.

--delay K: the encoder is causal, DACVAE's encoder is not, so output frame t + K is trained on
DACVAE's frame t (the clip gets K frames of zeros after it, and the first K outputs are trained
on DACVAE's latent of silence). In the wheel the prompt then starts with K frames of silence and
loses its last K frames.

Whole clips, right-padded within a batch (the encoder is causal, so padding does not reach the
frames before it); MSE on the raw latents (the FlowLM projects them linearly). One clip in 100
(by name) is held out, plus the ten release voices (--val-wav).

    uv run python scripts/wheel/train_encoder.py runs/irodori_distill/W1e_d2 \
        --targets data/irodori_distill/w1_enc_targets --delay 2

Writes run_dir/checkpoint.pt (resumed automatically), run_dir/encoder_step_<step>.safetensors
(EMA, `mimi.*` names) and run_dir/progress.jsonl.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from pocket_tts_ja.dacvae import DACVAE, REF_LOUDNESS_DB, load_wav, normalize_loudness
from pocket_tts_ja.dacvae import SAMPLE_RATE as DV_SR
from pocket_tts_ja.dv_loader import SILENCE_LATENT
from pocket_tts_ja.wheel_codec import build_dv_mimi

ENCODER_MODULES = ("encoder", "encoder_transformer", "downsample")
FRESH = ("downsample",)
HOP = 960
SR = 24000

Clip = tuple[np.ndarray, Path]  # DACVAE latent [frames, 32] float16, 24 kHz int16 input (read per batch)


def encoder_named_parameters(mimi: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(f"{m}.{n}", p) for m in ENCODER_MODULES for n, p in getattr(mimi, m).named_parameters()]


def encoder_state(mimi: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {f"mimi.{m}.{n}": t.detach().float().cpu().contiguous()
            for m in ENCODER_MODULES for n, t in getattr(mimi, m).state_dict().items()}


def held_out(key: str) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 == 0


def batch_loss(
    mimi: torch.nn.Module, clips: list[Clip], delay: int, silence: torch.Tensor, gains_db: list[float], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(squared error summed over dims and frames, the frame count)."""
    frames = max(z.shape[0] for z, _ in clips) + delay
    x = torch.zeros(len(clips), 1, frames * HOP)
    target = silence.expand(len(clips), frames, -1).clone()
    mask = torch.zeros(len(clips), frames, dtype=torch.bool)
    for i, ((z, wav), g) in enumerate(zip(clips, gains_db, strict=True)):
        y = np.load(wav).astype(np.float32) / 32767
        rms = float(np.sqrt(np.mean(y**2))) + 1e-6
        x[i, 0, : len(y)] = torch.from_numpy(np.clip(y * 10 ** (g / 20) / rms, -1, 1))
        target[i, delay : delay + z.shape[0]] = torch.from_numpy(z.astype(np.float32))
        mask[i, : delay + z.shape[0]] = True
    out = mimi.encode_to_latent(x.to(device))[:, :frames]
    target, mask = target.to(device), mask.to(device)
    err = (out - target).square().sum(-1)
    return (err * mask).sum(), mask.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--targets", type=Path, action="append", required=True, help="encode_targets.py output")
    parser.add_argument("--val-wav", type=Path, default=Path("exp/release_cc/voices"), help="held-out prompts (<name>.wav)")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--batch-seconds", type=float, default=160.0, help="audio per batch (whole clips)")
    parser.add_argument("--gain-dbfs", type=float, nargs=2, default=(-40.0, -12.0), help="input RMS range")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-4, help="stock-initialised modules")
    parser.add_argument("--lr-fresh", type=float, default=1e-3, help="the downsampling")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--final-lr-scale", type=float, default=0.1)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    clips, val = [], []
    for d in args.targets:
        for wav in sorted((d / "wav").glob("*.npy")):
            (val if held_out(wav.stem) else clips).append((np.load(d / "lat" / wav.name), wav))
    clips.sort(key=lambda c: c[0].shape[0])
    print(f"{len(clips)} training clips ({sum(c[0].shape[0] for c in clips) / 25 / 3600:.1f} h), {len(val)} held out", flush=True)
    # the release voices: DACVAE's prompt latents, made here once
    codec = DACVAE.load(str(device))
    voices = []
    vdir = args.run_dir / "val_voices"
    vdir.mkdir(exist_ok=True)
    for p in sorted(args.val_wav.glob("*.wav")):
        x48 = load_wav(p)
        z = codec.encode(normalize_loudness(x48, DV_SR, REF_LOUDNESS_DB))[0].cpu().numpy().astype(np.float16)
        x24 = load_wav(p, SR)
        np.save(vdir / f"{p.stem}.npy", (np.clip(x24, -1, 1) * 32767).round().astype(np.int16))
        voices.append((z, vdir / f"{p.stem}.npy"))
    del codec
    silence = torch.from_numpy(np.load(SILENCE_LATENT)[30].astype(np.float32))  # steady digital silence
    var = float(np.concatenate([z.astype(np.float32) for z, _ in val]).var(axis=0).sum())

    mimi = build_dv_mimi().to(device).train()
    named = encoder_named_parameters(mimi)
    for p in mimi.parameters():
        p.requires_grad_(False)
    for _, p in named:
        p.requires_grad_(True)
    fresh = [p for n, p in named if n.split(".")[0] in FRESH]
    stock = [p for n, p in named if n.split(".")[0] not in FRESH]
    opt = torch.optim.AdamW([{"params": stock, "lr": args.lr}, {"params": fresh, "lr": args.lr_fresh}], weight_decay=0.0)
    ema = {n: p.detach().clone() for n, p in named}
    step = 0
    ckpt = args.run_dir / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        mimi.load_state_dict(state["mimi"])
        opt.load_state_dict(state["opt"])
        ema, step = state["ema"], state["step"]
        rng.setstate(state["rng"])
        print(f"resumed at step {step}", flush=True)
    base_lrs = [args.lr, args.lr_fresh]
    log = open(args.run_dir / "progress.jsonl", "a")

    def make_batch() -> list[Clip]:
        """Neighbouring clips in length order, up to --batch-seconds of padded audio."""
        i = rng.randrange(len(clips))
        n = max(1, int(args.batch_seconds * 25 / clips[i][0].shape[0]))
        j = min(len(clips), i + n)
        return clips[max(0, j - n) : j]

    def evaluate(items: list[Clip]) -> float:
        sq = cnt = 0.0
        for k in range(0, len(items), 8):
            part = items[k : k + 8]
            s, c = batch_loss(mimi, part, args.delay, silence, [-20.0] * len(part), device)
            sq, cnt = sq + float(s), cnt + float(c)
        return sq / cnt

    t0 = time.time()
    acc: list[float] = []
    while step < args.max_steps:
        progress = max(0.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
        scale = min(1.0, (step + 1) / args.warmup) * (
            args.final_lr_scale + (1 - args.final_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress)))
        for g, lr in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr * scale
        batch = make_batch()
        gains = [rng.uniform(*args.gain_dbfs) for _ in batch]
        sq, cnt = batch_loss(mimi, batch, args.delay, silence, gains, device)
        loss = sq / cnt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in named], 10.0)
        opt.step()
        with torch.no_grad():
            for n, p in named:
                ema[n].lerp_(p.detach(), 1 - args.ema)
        step += 1
        acc.append(float(loss))
        if step % args.log_every == 0:
            row = {"step": step, "mse": round(float(np.mean(acc)), 4), "lr_scale": round(scale, 3),
                   "steps_per_s": round(args.log_every / (time.time() - t0), 2),
                   "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            acc, t0 = [], time.time()
        if step % args.valid_every == 0 or step == args.max_steps:
            mimi.eval()
            live = {n: p.detach().clone() for n, p in named}
            with torch.no_grad():
                for n, p in named:
                    p.copy_(ema[n])
                mse_val, mse_voices = evaluate(val), evaluate(voices)
                safetensors.torch.save_file(encoder_state(mimi), str(args.run_dir / f"encoder_step_{step:08d}.safetensors"))
                for n, p in named:
                    p.copy_(live[n])
            mimi.train()
            # R2: share of the latent variance (summed over dims, held-out clips) explained
            row = {"valid": step, "mse": round(mse_val, 4), "r2": round(1 - mse_val / var, 4),
                   "voices_mse": round(mse_voices, 4), "voices_r2": round(1 - mse_voices / var, 4)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            torch.save({"mimi": mimi.state_dict(), "opt": opt.state_dict(), "ema": ema, "step": step,
                        "rng": rng.getstate()}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            t0 = time.time()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
