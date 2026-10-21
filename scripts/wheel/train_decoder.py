"""P1-side: can a stock-wheel Mimi decoder (ELU SEANet, causal) speak DACVAE latents with harmonics?

Decoder distillation: the target of every window is what the DACVAE decoder (non-causal, Snake)
makes of the same latents, resampled to 24 kHz. The latents are the teacher's (P1,
data/irodori_distill/teacher/<set>/latents/<id>.npy). The student decoder is the stock Mimi
decode side rebuilt for 25 Hz input (pocket_tts_ja.wheel_codec): decoder transformer and SEANet
from the stock weights, the latent projection and the upsampling fresh. Each window of --frames
frames is decoded with --context frames before it as streaming context, as inference does.

Losses: multi-scale log-mel L1 alone for the first --gan-start steps (the fresh projection has to
find the latent space), then least-squares GAN + feature matching against multi-period,
multi-scale STFT and multi-band STFT discriminators (the discriminators train alone from
--gan-start minus --d-warmup).

Validation (EMA weights) on teacher latents of voices never trained on (the ten release voices x
JSUT, exp/irodori_distill/eval/voices/teacher_latents): mel L1 to the DACVAE decode and, on the
voiced frames of the female voices, the 4-6 / 6-8 kHz comb and level re the DACVAE decode
(scripts/eval/hf_structure.py). The decoded validation clips are kept for the full scoring.

The latent directories are scanned again at every validation, so the probe can start while P1 is
still writing them. Each training clip is decoded with DACVAE once, whole, and its 24 kHz target
kept in memory (float16, --limit clips at most): decoding the windows at every step took most of
the GPU time the probe shares with the P1 generation.

    uv run python scripts/wheel/train_decoder.py runs/irodori_distill/W1_probe \
        --latents data/irodori_distill/teacher/s3/latents --max-steps 30000

Writes run_dir/checkpoint.pt (resumed automatically), run_dir/decoder_step_<step>.safetensors
(EMA, `mimi.*` names; pocket_tts_ja.wheel_codec.load_dv_decoder), run_dir/progress.jsonl and
run_dir/val/<step>/{target,student}/<key>.wav.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from pocket_tts_ja.codec_losses import (
    Discriminators,
    MultiScaleMelLoss,
    decode_window,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from pocket_tts_ja.dacvae import DACVAE, HOP as DV_HOP, SAMPLE_RATE as DV_SR
from pocket_tts_ja.wheel_codec import FRESH_MODULES, build_dv_mimi, decoder_named_parameters, decoder_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from hf_structure import N_FFT, band_stats, refine_f0, spectrum, voiced_f0  # noqa: E402

SR = 24000


Clip = tuple[np.ndarray, np.ndarray]  # latents [T, 32] float16, DACVAE decode at 24 kHz [T * 960] float16
_TARGETS: dict[Path, Clip] = {}


@torch.no_grad()
def load_clips(codec: DACVAE, dirs: list[Path], min_frames: int, limit: int | None) -> list[Clip]:
    """The clips of the latent directories (the first `limit` in the order of a hash of their
    names, which new files disturb little), each with its whole DACVAE decode; decoded once per
    process, dropped from memory when a rescan no longer selects them."""
    paths = sorted((p for d in dirs for p in d.glob("*.npy")), key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
    chosen = set(paths[:limit])
    for p in [p for p in _TARGETS if p not in chosen]:
        del _TARGETS[p]
    hop = DV_HOP * SR // DV_SR
    out = []
    for p in paths[:limit]:
        if p not in _TARGETS:
            z = np.load(p)
            if z.shape[0] < min_frames:
                continue
            y = AF.resample(codec.decode(torch.from_numpy(z.astype(np.float32)).to(codec.device)[None])[0], DV_SR, SR)
            y = y[: z.shape[0] * hop]
            _TARGETS[p] = (z, F.pad(y, (0, z.shape[0] * hop - len(y))).half().cpu().numpy())
        out.append(_TARGETS[p])
    return out


class EMA:
    def __init__(self, named: list[tuple[str, torch.nn.Parameter]], decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in named}

    @torch.no_grad()
    def update(self, named: list[tuple[str, torch.nn.Parameter]]) -> None:
        for n, p in named:
            self.shadow[n].lerp_(p.detach(), 1 - self.decay)

    @torch.no_grad()
    def swap(self, named: list[tuple[str, torch.nn.Parameter]]) -> None:
        for n, p in named:
            tmp = p.detach().clone()
            p.copy_(self.shadow[n])
            self.shadow[n].copy_(tmp)


@torch.no_grad()
def validate(
    mimi: torch.nn.Module, codec: DACVAE, val: list[tuple[str, str, np.ndarray]], out: Path, mel: MultiScaleMelLoss,
    device: torch.device,
) -> dict[str, float]:
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "student").mkdir(parents=True, exist_ok=True)
    mels, combs, levels = [], [], []
    hop = DV_HOP * SR // DV_SR
    for key, gender, z_np in val:
        z = torch.from_numpy(z_np.astype(np.float32)).to(device)
        target = AF.resample(codec.decode(z[None])[0], DV_SR, SR)
        student = decode_window(mimi, z[None], 0)[0, 0][: z.shape[0] * hop]
        n = min(len(target), len(student))
        target, student = target[:n], student[:n]
        mels.append(float(mel(student[None], target[None])))
        t_np, s_np = target.cpu().numpy(), student.cpu().numpy()
        sf.write(str(out / "target" / f"{key}.wav"), np.clip(t_np, -1, 1), SR, subtype="PCM_16")
        sf.write(str(out / "student" / f"{key}.wav"), np.clip(s_np, -1, 1), SR, subtype="PCM_16")
        if gender != "F":
            continue
        step = SR / (N_FFT * 4)
        for start, f0 in voiced_f0(t_np, SR):
            f0 = refine_f0(np.log(spectrum(t_np[start : start + N_FFT]) + 1e-9), f0, step)
            ts = band_stats(spectrum(t_np[start : start + N_FFT]), f0, step)
            ss = band_stats(spectrum(s_np[start : start + N_FFT]), f0, step)
            combs.append([ts[2][0], ts[3][0], ss[2][0], ss[3][0]])
            levels.append([ss[2][1] / (ts[2][1] + 1e-12), ss[3][1] / (ts[3][1] + 1e-12)])
    c = np.mean(combs, axis=0)
    lv = 10 * np.log10(np.mean(levels, axis=0))
    return {"mel": float(np.mean(mels)), "comb46_target": float(c[0]), "comb68_target": float(c[1]),
            "comb46": float(c[2]), "comb68": float(c[3]), "level46": float(lv[0]), "level68": float(lv[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--latents", type=Path, action="append", required=True, help="directory of <id>.npy")
    parser.add_argument("--limit", type=int, default=20000, help="at most this many training clips (RAM: ~0.2 MB/s)")
    parser.add_argument("--val-latents", type=Path, default=Path("exp/irodori_distill/eval/voices/teacher_latents"))
    parser.add_argument("--val-items", type=Path, default=Path("exp/irodori_distill/eval/voices/items.jsonl"))
    parser.add_argument("--val-per-voice", type=int, default=4)
    parser.add_argument("--frames", type=int, default=64, help="trained window (25 Hz frames)")
    parser.add_argument("--context", type=int, default=50, help="max streaming context before the window")
    parser.add_argument("--prefix-prob", type=float, default=0.3, help="share of batches at the clip start")
    parser.add_argument("--batch-size", type=int, default=4, help="4 fits beside the P1 generation (~11 GB)")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-4, help="stock-initialised modules")
    parser.add_argument("--lr-fresh", type=float, default=3e-4, help="latent projection and upsampling")
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gan-start", type=int, default=4000)
    parser.add_argument("--d-warmup", type=int, default=1000)
    parser.add_argument("--w-mel", type=float, default=15.0)
    parser.add_argument("--w-adv", type=float, default=1.0)
    parser.add_argument("--w-fm", type=float, default=10.0)
    parser.add_argument("--band-ffts", type=int, nargs="*", default=[2048, 1024, 512])
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    codec = DACVAE.load(str(device))
    for p in codec.model.parameters():
        p.requires_grad_(False)
    def scan() -> tuple[list[Clip], int]:
        clips = load_clips(codec, args.latents, args.frames + 1, args.limit)
        print(f"{len(clips)} training clips, {sum(c[0].shape[0] for c in clips) / 25 / 3600:.1f} h", flush=True)
        # the context of a batch is drawn so that at least a tenth of the clips are long enough
        return clips, max(0, min(args.context, int(np.percentile([c[0].shape[0] for c in clips], 90)) - args.frames))

    clips, max_context = scan()
    items = {it["key"]: it for it in map(json.loads, open(args.val_items, encoding="utf-8"))}
    per_voice: dict[str, int] = {}
    val: list[tuple[str, str, np.ndarray]] = []
    for p in sorted(args.val_latents.glob("*.npy")):
        it = items[p.stem]
        if per_voice.get(it["speaker"], 0) < args.val_per_voice:
            per_voice[it["speaker"]] = per_voice.get(it["speaker"], 0) + 1
            val.append((p.stem, it["gender"], np.load(p)))

    mimi = build_dv_mimi().to(device).train()
    named = decoder_named_parameters(mimi)
    for p in mimi.parameters():
        p.requires_grad_(False)
    for _, p in named:
        p.requires_grad_(True)
    fresh = [p for n, p in named if n.split(".")[0] in FRESH_MODULES]
    stock = [p for n, p in named if n.split(".")[0] not in FRESH_MODULES]
    opt_g = torch.optim.AdamW([{"params": stock, "lr": args.lr}, {"params": fresh, "lr": args.lr_fresh}],
                              betas=(0.8, 0.99), weight_decay=0.0)
    disc = Discriminators(band_ffts=tuple(args.band_ffts)).to(device)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.0)
    mel = MultiScaleMelLoss(SR).to(device)
    ema = EMA(named, args.ema)
    step = 0
    ckpt = args.run_dir / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        mimi.load_state_dict(state["mimi"])
        disc.load_state_dict(state["disc"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        ema.shadow = state["ema"]
        step = state["step"]
        rng.setstate(state["rng"])
        print(f"resumed at step {step}", flush=True)
    base_lrs = [g["lr"] for g in opt_g.param_groups]
    log = open(args.run_dir / "progress.jsonl", "a")
    hop = DV_HOP * SR // DV_SR
    t0 = time.time()
    acc: dict[str, list[float]] = {}
    while step < args.max_steps:
        for g, lr in zip(opt_g.param_groups, base_lrs, strict=True):
            g["lr"] = lr * min(1.0, (step + 1) / args.warmup)
        ctx = 0 if rng.random() < args.prefix_prob or max_context < 8 else rng.randint(8, max_context)
        length = ctx + args.frames
        inputs, targets = [], []
        with torch.no_grad():
            while len(inputs) < args.batch_size:
                z_np, y_np = clips[rng.randrange(len(clips))]
                if z_np.shape[0] < length:
                    continue
                a = 0 if ctx == 0 else rng.randint(0, z_np.shape[0] - length)
                inputs.append(torch.from_numpy(z_np[a : a + length].astype(np.float32)).to(device))
                y = y_np[(a + ctx) * hop : (a + ctx + args.frames) * hop]
                targets.append(torch.from_numpy(y.astype(np.float32)).to(device))
        x = torch.stack(inputs)
        real = torch.stack(targets)
        fake = decode_window(mimi, x, ctx)[:, 0, : args.frames * hop]
        gan = step >= args.gan_start
        d_on = step >= args.gan_start - args.d_warmup
        logs: dict[str, float] = {}
        if d_on:
            loss_d = discriminator_loss(disc(real), disc(fake.detach()))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            logs["d"] = float(loss_d)
        loss_mel = mel(fake, real)
        loss = args.w_mel * loss_mel
        logs["mel"] = float(loss_mel)
        if gan:
            d_real, d_fake = disc(real), disc(fake)
            loss_adv = generator_loss(d_fake)
            loss_fm = feature_loss(d_real, d_fake)
            loss = loss + args.w_adv * loss_adv + args.w_fm * loss_fm
            logs["adv"], logs["fm"] = float(loss_adv), float(loss_fm)
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in named], 10.0)
        opt_g.step()
        ema.update(named)
        step += 1
        for k, v in logs.items():
            acc.setdefault(k, []).append(v)
        if step % args.log_every == 0:
            row = {"step": step, **{k: round(float(np.mean(v)), 4) for k, v in acc.items()},
                   "steps_per_s": round(args.log_every / (time.time() - t0), 2),
                   "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            acc, t0 = {}, time.time()
        if step % args.valid_every == 0 or step == args.max_steps:
            mimi.eval()
            ema.swap(named)
            res = validate(mimi, codec, val, args.run_dir / "val" / f"{step:08d}", mel, device)
            if step % args.save_every == 0 or step == args.max_steps:
                safetensors.torch.save_file(decoder_state(mimi), str(args.run_dir / f"decoder_step_{step:08d}.safetensors"))
            ema.swap(named)
            mimi.train()
            row = {"valid": step, **{k: round(v, 3) for k, v in res.items()}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            torch.save({"mimi": mimi.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(), "ema": ema.shadow, "step": step, "rng": rng.getstate()},
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            clips, max_context = scan()
            t0 = time.time()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
