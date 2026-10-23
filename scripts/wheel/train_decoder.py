"""W1: a stock-wheel Mimi decoder (ELU SEANet, causal) for DACVAE latents.

Decoder distillation: the target of every window is what the DACVAE decoder (non-causal, Snake)
makes of the same latents, resampled to 24 kHz, decoded once beforehand
(scripts/wheel/decode_targets.py: <targets>/lat/<id>.npy, <targets>/wav/<id>.npy; each clip ends
in a few frames of DACVAE's silence latent, as a student's output does after EOS). The student
decoder is the stock Mimi decode side rebuilt for 25 Hz input (pocket_tts_ja.wheel_codec): decoder
transformer and SEANet from the stock weights, the latent projection and the upsampling fresh.
Each window of --frames frames is decoded with up to --context frames before it as streaming
context, as inference does.

--delay K trains the decoder to put out at frame t the audio of frame t - K. DACVAE's encoder is
non-causal, so the end of frame t is partly carried by the latents after it; a causal decoder
without delay has to guess it (the P1-side probe at K = 0 wobbled in pitch and lost voicing). The
wheel decodes the frames generated after EOS too (model_recommended_frames_after_eos), so the
delay costs K frames of silence at the start and K frames of latency, no change to the wheel.

Losses: multi-scale log-mel L1 alone for the first --gan-start steps (the fresh projection has to
find the latent space), then least-squares GAN + feature matching against multi-period,
multi-scale STFT and multi-band STFT discriminators on a --gan-seconds slice of each window (the
discriminators train alone from --gan-start minus --d-warmup). --gan-start beyond --max-steps
trains on the mel loss alone. The learning rates warm up, then follow a cosine to
--final-lr-scale of their value at --max-steps.

Validation (EMA weights) on voices never trained on (the ten release voices x JSUT,
exp/irodori_distill/eval/voices): the teacher's latents and a student's free generation
(--val-student), each against DACVAE's decode of the same latents: mel L1; on the voiced frames,
the F0 error in cents (median, p90) and the share of the target's voiced frames still voiced
(scripts/eval/hf_structure.py voiced_f0); for the female voices, the 4-6 kHz comb and level; and
the decoder's frame-grid buzz, the 200 Hz and 25 Hz lines of the long-term spectrum
(scripts/eval/spectral_lines.py: ~0.6 dB for the DACVAE decode). Early in training the 200 Hz buzz
is strong enough that voiced_f0 locks onto it, so the F0 numbers only mean something once the
line is down. The decoded clips are kept for the full scoring (runs/irodori_distill/p1side_eval.sh).

The target directories are scanned again at every validation, so training can start while
decode_targets.py is still writing them.

    uv run python scripts/wheel/train_decoder.py runs/irodori_distill/W1_main \
        --targets data/irodori_distill/w1_targets --delay 2 --batch-size 16 --max-steps 150000

--init CKPT starts a fine-tune from another run's checkpoint.pt (decoder, EMA and discriminators;
fresh optimizers, step 0), for the soup runs of W1s (several short runs from W1 150k with different
recipes, averaged by scripts/wheel/soup_decoders.py, as main's CC decoder soup4). Pass
--gan-start 0 --d-warmup 0: the discriminators are trained already.

Writes run_dir/checkpoint.pt (resumed automatically), run_dir/decoder_step_<step>.safetensors
(EMA, `mimi.*` names; pocket_tts_ja.wheel_codec.load_dv_decoder), run_dir/progress.jsonl and
run_dir/val/<step>/<set>/{target,student}/<key>.wav (student aligned to the target: delay removed).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import torch
import torchaudio.functional as AF

from pocket_tts_ja.codec_losses import (
    Discriminators,
    MultiScaleMelLoss,
    decode_window,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from pocket_tts_ja.dacvae import DACVAE
from pocket_tts_ja.dacvae import HOP as DV_HOP
from pocket_tts_ja.dacvae import SAMPLE_RATE as DV_SR
from pocket_tts_ja.dv_loader import SILENCE_LATENT
from pocket_tts_ja.wheel_codec import (
    FRESH_MODULES,
    build_dv_mimi,
    decoder_named_parameters,
    decoder_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from hf_structure import N_FFT, band_stats, refine_f0, spectrum, voiced_f0  # noqa: E402
from spectral_lines import contrast, ltas  # noqa: E402

SR = 24000
HOP = DV_HOP * SR // DV_SR  # 960 samples per 25 Hz frame


Clip = tuple[np.ndarray, Path]  # latents [frames, 32] float16, the 24 kHz int16 target (read per window)


def scan(dirs: list[Path], clips: dict[Path, Clip], min_frames: int) -> list[Clip]:
    """The complete clips of the target directories; the latents of new ones are read once."""
    for d in dirs:
        for wav in (d / "wav").glob("*.npy"):
            if wav not in clips:
                z = np.load(d / "lat" / wav.name)
                if z.shape[0] >= min_frames:
                    clips[wav] = (z, wav)
    return list(clips.values())


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


def load_val(latents: Path, items_path: Path, per_voice: int, silence: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    items = {it["key"]: it for it in map(json.loads, open(items_path, encoding="utf-8"))}
    count: dict[str, int] = {}
    val = []
    for p in sorted(latents.glob("*.npy")):
        it = items[p.stem]
        z = np.load(p)
        if count.get(it["speaker"], 0) < per_voice and z.shape[0] >= 25:
            count[it["speaker"]] = count.get(it["speaker"], 0) + 1
            val.append((p.stem, it["gender"], np.concatenate([z.astype(np.float16), silence])))
    return val


@torch.no_grad()
def validate(
    mimi: torch.nn.Module, codec: DACVAE, val: list[tuple[str, str, np.ndarray]], out: Path, mel: MultiScaleMelLoss,
    delay: int, device: torch.device,
) -> dict[str, float]:
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "student").mkdir(parents=True, exist_ok=True)
    mels, combs, levels, cents = [], [], [], []
    voiced = both = 0
    for key, gender, z_np in val:
        z = torch.from_numpy(z_np.astype(np.float32)).to(device)
        n = (z.shape[0] - delay) * HOP
        target = AF.resample(codec.decode(z[None])[0], DV_SR, SR)[:n]
        student = decode_window(mimi, z[None], 0)[0, 0][delay * HOP : delay * HOP + n]
        target = torch.nn.functional.pad(target, (0, n - len(target)))
        mels.append(float(mel(student[None], target[None])))
        t_np, s_np = target.cpu().numpy(), student.cpu().numpy()
        sf.write(str(out / "target" / f"{key}.wav"), np.clip(t_np, -1, 1), SR, subtype="PCM_16")
        sf.write(str(out / "student" / f"{key}.wav"), np.clip(s_np, -1, 1), SR, subtype="PCM_16")
        t_f0, s_f0 = voiced_f0(t_np, SR), dict(voiced_f0(s_np, SR))
        voiced += len(t_f0)
        for start, f0 in t_f0:
            if start in s_f0:
                both += 1
                cents.append(abs(1200 * float(np.log2(s_f0[start] / f0))))
        if gender != "F":
            continue
        step = SR / (N_FFT * 4)
        for start, f0 in t_f0:
            f0 = refine_f0(np.log(spectrum(t_np[start : start + N_FFT]) + 1e-9), f0, step)
            ts = band_stats(spectrum(t_np[start : start + N_FFT]), f0, step)
            ss = band_stats(spectrum(s_np[start : start + N_FFT]), f0, step)
            combs.append([ts[2][0], ss[2][0]])
            levels.append(ss[2][1] / (ts[2][1] + 1e-12))
    db, _ = ltas(sorted((out / "student").glob("*.wav")))
    c = np.mean(combs, axis=0) if combs else np.full(2, np.nan)
    off = np.array(cents) if cents else np.full(1, np.nan)  # an early decoder may voice nothing
    return {"mel": float(np.mean(mels)), "f0_cents_med": float(np.median(off)), "f0_cents_p90": float(np.percentile(off, 90)),
            "voiced": both / max(voiced, 1), "comb46_target": float(c[0]), "comb46": float(c[1]),
            "level46": float(10 * np.log10(np.mean(levels))) if levels else float("nan"),
            "line200": contrast(db, SR, 200.0), "line25": contrast(db, SR, 25.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--targets", type=Path, action="append", required=True, help="decode_targets.py output")
    parser.add_argument("--val-items", type=Path, default=Path("exp/irodori_distill/eval/voices/items.jsonl"))
    parser.add_argument("--val-teacher", type=Path, default=Path("exp/irodori_distill/eval/voices/teacher_latents"))
    parser.add_argument("--val-student", type=Path, default=Path("exp/irodori_distill/eval/voices/P2_60k_cfg1.5_latents"))
    parser.add_argument("--val-per-voice", type=int, default=4)
    parser.add_argument("--silence-frames", type=int, default=12, help="silence latent after each validation clip")
    parser.add_argument("--delay", type=int, default=0, help="output frames behind the input (latency, lookahead)")
    parser.add_argument("--frames", type=int, default=64, help="trained window (25 Hz frames)")
    parser.add_argument("--context", type=int, default=50, help="max streaming context before the window")
    parser.add_argument("--prefix-prob", type=float, default=0.3, help="share of batches at the clip start")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=1e-4, help="stock-initialised modules")
    parser.add_argument("--lr-fresh", type=float, default=3e-4, help="latent projection and upsampling")
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--final-lr-scale", type=float, default=0.1, help="cosine decay ends at this share of the LR")
    parser.add_argument("--gan-start", type=int, default=4000)
    parser.add_argument("--d-warmup", type=int, default=1000)
    parser.add_argument("--gan-seconds", type=float, default=1.0, help="slice of the window the discriminators see")
    parser.add_argument("--w-mel", type=float, default=15.0)
    parser.add_argument("--w-adv", type=float, default=1.0)
    parser.add_argument("--w-fm", type=float, default=10.0)
    parser.add_argument("--band-ffts", type=int, nargs="*", default=[2048, 1024, 512])
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=None, help="checkpoint.pt of a run to fine-tune from")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    codec = DACVAE.load(str(device))
    for p in codec.model.parameters():
        p.requires_grad_(False)
    silence = np.load(SILENCE_LATENT)[: args.silence_frames].astype(np.float16)
    val_sets = {"teacher": load_val(args.val_teacher, args.val_items, args.val_per_voice, silence)}
    if args.val_student.exists():
        val_sets["student"] = load_val(args.val_student, args.val_items, args.val_per_voice, silence)
    known: dict[Path, Clip] = {}

    def rescan() -> tuple[list[Clip], int]:
        clips = scan(args.targets, known, args.frames)
        print(f"{len(clips)} training clips, {sum(c[0].shape[0] for c in clips) / 25 / 3600:.1f} h", flush=True)
        # the context of a batch is drawn so that at least a tenth of the clips are long enough
        return clips, max(0, min(args.context, int(np.percentile([c[0].shape[0] for c in clips], 90)) - args.frames))

    clips, max_context = rescan()

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
    elif args.init is not None:
        state = torch.load(args.init, map_location=device, weights_only=False)
        mimi.load_state_dict(state["mimi"])
        disc.load_state_dict(state["disc"])
        ema.shadow = state["ema"]
        print(f"fine-tuning from {args.init} (its step {state['step']})", flush=True)
    base_lrs = [args.lr, args.lr_fresh]
    log = open(args.run_dir / "progress.jsonl", "a")
    gan_len = min(args.frames * HOP, int(args.gan_seconds * SR))
    t0 = time.time()
    acc: dict[str, list[float]] = {}
    while step < args.max_steps:
        progress = max(0.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
        scale = min(1.0, (step + 1) / args.warmup) * (
            args.final_lr_scale + (1 - args.final_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress)))
        for g, lr in zip(opt_g.param_groups, base_lrs, strict=True):
            g["lr"] = lr * scale
        ctx = 0 if rng.random() < args.prefix_prob or max_context < 8 else rng.randint(8, max_context)
        length = ctx + args.frames
        inputs, targets = [], []
        while len(inputs) < args.batch_size:
            z_np, wav = clips[rng.randrange(len(clips))]
            if z_np.shape[0] < length:
                continue
            a = 0 if ctx == 0 else rng.randint(0, z_np.shape[0] - length)
            inputs.append(torch.from_numpy(z_np[a : a + length].astype(np.float32)))
            s0 = (a + ctx - args.delay) * HOP
            y = np.load(wav, mmap_mode="r")[max(0, s0) : s0 + args.frames * HOP].astype(np.float32) / 32767
            targets.append(torch.from_numpy(np.pad(y, (args.frames * HOP - len(y), 0))))
        x = torch.stack(inputs).to(device)
        real = torch.stack(targets).to(device)
        fake = decode_window(mimi, x, ctx)[:, 0, : args.frames * HOP]
        gan = step >= args.gan_start
        d_on = step >= args.gan_start - args.d_warmup
        logs: dict[str, float] = {}
        if d_on:
            g0 = rng.randint(0, args.frames * HOP - gan_len)
            real_g, fake_g = real[:, g0 : g0 + gan_len], fake[:, g0 : g0 + gan_len]
            loss_d = discriminator_loss(disc(real_g), disc(fake_g.detach()))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            logs["d"] = float(loss_d)
        loss_mel = mel(fake, real)
        loss = args.w_mel * loss_mel
        logs["mel"] = float(loss_mel)
        if gan:
            d_real, d_fake = disc(real_g), disc(fake_g)
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
                   "lr_scale": round(scale, 3), "steps_per_s": round(args.log_every / (time.time() - t0), 2),
                   "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            acc, t0 = {}, time.time()
        if step % args.valid_every == 0 or step == args.max_steps:
            mimi.eval()
            ema.swap(named)
            vrow: dict[str, float] = {"valid": step}
            for name, val in val_sets.items():
                res = validate(mimi, codec, val, args.run_dir / "val" / f"{step:08d}" / name, mel, args.delay, device)
                vrow.update({f"{name}_{k}": round(v, 3) for k, v in res.items()})
            if step % args.save_every == 0 or step == args.max_steps:
                safetensors.torch.save_file(decoder_state(mimi), str(args.run_dir / f"decoder_step_{step:08d}.safetensors"))
            ema.swap(named)
            mimi.train()
            print(json.dumps(vrow), flush=True)
            log.write(json.dumps(vrow) + "\n")
            log.flush()
            torch.save({"mimi": mimi.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(), "ema": ema.shadow, "step": step, "rng": rng.getstate()},
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            clips, max_context = rescan()
            t0 = time.time()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
