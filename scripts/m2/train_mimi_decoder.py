"""M2: fine-tune the decode side of Mimi on Japanese speech (encoder and latent space frozen).

Every step takes clips from their start, encodes them with the frozen encoder (the latents a
FlowLM is trained on) and decodes a window of --frames frames with everything before it as
streaming context, as inference does. With probability --prefix-prob the window is the start
of the clip, where the first consonant has nothing in front of it.

    uv run python scripts/m2/train_mimi_decoder.py runs/M2_pilot \
        --data irodori=data/synth/s3_full/manifest.jsonl:data/synth/s3_full/audio \
        --valid-glob "jvs=data/jvs_ver1/jvs00*/parallel100/wav24kHz16bit/*.wav"

--data NAME=MANIFEST:AUDIO_DIR[:WEIGHT]: rows with "id" and "duration"; the audio is
AUDIO_DIR/<id>.wav. A batch comes from one source, picked with probability WEIGHT (default 1).
One id in 200 (by hash) is held out for validation. --valid-glob adds validation-only audio
(JVS is never trained on). --w-adv 0 --w-fm 0 trains on the mel loss alone (no discriminators).

SIGTERM to the pid in run_dir/pid writes a checkpoint and exits.

Validation also records, for the voiced frames of high voices (F0 >= 165 Hz) of each validation
set, the comb of the bands 4-6 and 6-8 kHz at the recording's F0 and their level re the recording
(scripts/m2/hf_structure.py): whether the decoder puts harmonics or noise above 4 kHz.

Writes run_dir/checkpoint.pt (at every validation; resumed automatically),
run_dir/decoder_step_<step>.safetensors (EMA weights under their `mimi.*` names, for
pocket_tts_ja.codec_ft.load_decoder) and run_dir/progress.jsonl. wandb gets scalars only.

GPU memory: batch 16 with the discriminators peaks at about 5 GB while only they train and
more once the decoder trains too (gpu_gb in the log is the peak of each interval).
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import safetensors.torch
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from hf_structure import N_FFT, PAD, band_stats, refine_f0, spectrum, voiced_f0
from pocket_tts_ja.codec_ft import (
    Discriminators,
    FineSpectrumLoss,
    HighBandWaveLoss,
    MultiScaleMelLoss,
    SharedSpectrumLoss,
    VoicingBlendMelLoss,
    decode_window,
    decoder_named_parameters,
    decoder_state,
    discriminator_loss,
    expand_residual_layers,
    feature_loss,
    generator_loss,
    load_decoder,
)
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config, to_mono_float

VALID_MOD = 200
VALID_CLIPS = 128
VALID_FRAMES = 64
HIGH_BANDS = ((4000, 6000), (6000, 8000))
HIGH_VOICE_F0 = 165.0


def is_valid_id(utt_id: str) -> bool:
    return int(hashlib.md5(utt_id.encode()).hexdigest(), 16) % VALID_MOD == 0


def read_clip(path: str, sample_rate: int, frame: int, max_frames: int) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = to_mono_float(wav, sr, sample_rate)
    frames = min(len(wav) // frame, max_frames)
    return torch.from_numpy(wav[: frames * frame].copy())


class ClipDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: list[str], sample_rate: int, frame: int, max_frames: int) -> None:
        self.paths = paths
        self.sample_rate = sample_rate
        self.frame = frame
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return read_clip(self.paths[index], self.sample_rate, self.frame, self.max_frames)


class BucketBatches:
    """Endless batches of clips of similar length, so a window can start late in all of them."""

    def __init__(
        self, durations: list[float], sources: list[tuple[range, float]], batch_size: int, seed: int
    ) -> None:
        self.sources: list[list[list[int]]] = []
        self.source_weights = [weight for _, weight in sources]
        for indices, _ in sources:
            buckets: dict[int, list[int]] = defaultdict(list)
            for index in indices:
                buckets[int(durations[index])].append(index)
            self.sources.append([b for b in buckets.values() if len(b) >= batch_size])
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            buckets = self.rng.choices(self.sources, weights=self.source_weights)[0]
            bucket = self.rng.choices(buckets, weights=[len(b) for b in buckets])[0]
            yield self.rng.sample(bucket, self.batch_size)


class Collate:
    def __init__(self, frame: int, frames: int, prefix_prob: float, gain_db: tuple[float, float]) -> None:
        self.frame = frame
        self.frames = frames
        self.prefix_prob = prefix_prob
        self.gain_db = gain_db

    def __call__(self, wavs: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        need = self.frames * self.frame
        wavs = [w if len(w) >= need else torch.nn.functional.pad(w, (0, need - len(w))) for w in wavs]
        latest = min(len(w) for w in wavs) // self.frame - self.frames
        start = 0 if latest <= 0 or random.random() < self.prefix_prob else random.randint(1, latest)
        length = (start + self.frames) * self.frame
        out = []
        for wav in wavs:
            gain = 10 ** (random.uniform(*self.gain_db) / 20)
            gain = min(gain, 0.99 / (float(wav.abs().max()) + 1e-9))
            out.append(wav[:length] * gain)
        return torch.stack(out)[:, None], start


def load_rows(
    specs: list[str], min_seconds: float, max_seconds: float
) -> tuple[list[str], list[float], list[tuple[range, float]], list[str]]:
    """Training paths with durations, the index range and weight of each source, the held-out paths."""
    paths: list[str] = []
    durations: list[float] = []
    sources: list[tuple[range, float]] = []
    held_out: list[str] = []
    for spec in specs:
        name, rest = spec.split("=", 1)
        manifest, audio_dir, *weight = rest.split(":")
        kept = 0
        first = len(paths)
        for line in open(manifest):
            row = json.loads(line)
            if not (min_seconds <= row["duration"] <= max_seconds):
                continue
            path = str(Path(audio_dir) / f"{row['id']}.wav")
            if is_valid_id(row["id"]):
                held_out.append(path)
            else:
                paths.append(path)
                durations.append(float(row["duration"]))
                kept += 1
        sources.append((range(first, len(paths)), float(weight[0]) if weight else 1.0))
        print(f"data {name}: {kept} training clips, {sum(durations[first:]) / 3600:.1f} h, weight {sources[-1][1]}", flush=True)
    return paths, durations, sources, held_out


def spread(items: list[str], count: int) -> list[str]:
    items = sorted(items)
    if len(items) <= count:
        return items
    return [items[int(i * len(items) / count)] for i in range(count)]


@contextmanager
def ema_weights(mimi: torch.nn.Module, ema: dict[str, torch.Tensor]) -> Iterator[None]:
    """Run the block with the EMA weights in the model."""
    backup = {name: param.detach().clone() for name, param in decoder_named_parameters(mimi)}
    with torch.no_grad():
        for name, param in decoder_named_parameters(mimi):
            param.copy_(ema[name])
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in decoder_named_parameters(mimi):
                param.copy_(backup[name])


def high_band_structure(
    refs: list[npt.NDArray[np.float32]], fakes: list[npt.NDArray[np.float32]], sample_rate: int
) -> dict[str, float]:
    """Comb at the recording's F0 and level re the recording of HIGH_BANDS, over the voiced frames
    of high voices."""
    import hf_structure

    saved = hf_structure.BANDS
    hf_structure.BANDS = HIGH_BANDS
    step = sample_rate / (N_FFT * PAD)
    combs, p_ref, p_fake = [], [], []
    try:
        for ref, fake in zip(refs, fakes, strict=True):
            for start, f0 in voiced_f0(ref, sample_rate):
                spec_ref = spectrum(ref[start : start + N_FFT])
                f0 = refine_f0(np.log(spec_ref + 1e-9), f0, step)
                if f0 < HIGH_VOICE_F0:
                    continue
                stats_ref, stats_fake = band_stats(spec_ref, f0, step), band_stats(spectrum(fake[start : start + N_FFT]), f0, step)
                combs.append([s[0] for s in stats_fake])
                p_ref.append([s[1] for s in stats_ref])
                p_fake.append([s[1] for s in stats_fake])
    finally:
        hf_structure.BANDS = saved
    if not combs:
        return {}
    comb = np.mean(combs, axis=0)
    level = 10 * np.log10(np.mean(p_fake, axis=0) / np.mean(p_ref, axis=0))
    out = {}
    for (lo, hi), c, lv in zip(HIGH_BANDS, comb, level, strict=True):
        band = f"{lo // 1000}_{hi // 1000}k"
        out[f"comb{band}"], out[f"level{band}"] = float(c), float(lv)
    return out


@torch.no_grad()
def validate(
    mimi: torch.nn.Module, mel_loss: MultiScaleMelLoss, sets: dict[str, list[torch.Tensor]], device: torch.device
) -> dict[str, float]:
    out = {}
    for name, clips in sets.items():
        total = 0.0
        refs, fakes = [], []
        for clip in clips:
            wav = clip.to(device)[None, None]
            fake = decode_window(mimi, mimi.encode_to_latent(wav), 0)[..., : wav.shape[-1]]
            total += float(mel_loss(fake, wav))
            refs.append(clip.numpy())
            fakes.append(fake[0, 0].float().cpu().numpy())
        out[f"mel_{name}"] = total / max(len(clips), 1)
        out |= {f"{key}_{name}": value for key, value in high_band_structure(refs, fakes, mimi.sample_rate).items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", action="append", required=True, help="NAME=MANIFEST:AUDIO_DIR[:WEIGHT]")
    parser.add_argument("--valid-glob", action="append", default=[], help="NAME=GLOB of validation-only wavs")
    parser.add_argument("--frames", type=int, default=32, help="frames of the trained window (80 ms each)")
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--prefix-prob", type=float, default=0.5, help="share of windows at the start of the clip")
    parser.add_argument("--gain-db", type=float, nargs=2, default=(-6.0, 3.0), metavar=("LO", "HI"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500, help="decoder LR warmup steps")
    parser.add_argument("--d-warmup", type=int, default=2000, help="steps of discriminator-only training")
    parser.add_argument("--final-lr-scale", type=float, default=0.1, help="cosine decay ends at this share of the LR")
    parser.add_argument("--w-mel", type=float, default=15.0)
    parser.add_argument("--w-adv", type=float, default=1.0)
    parser.add_argument("--w-fm", type=float, default=10.0)
    parser.add_argument("--w-spec", type=float, default=0.0,
                        help="weight of the shared long-term spectrum loss (against tones at the stride frequencies)")
    parser.add_argument("--spec-band", type=float, nargs=2, default=(300.0, 11000.0), metavar=("LO", "HI"),
                        help="band of that loss in Hz; a tone moves to just outside it if it can")
    parser.add_argument("--high-cross", type=float, default=0.0,
                        help="Hz; above it the mel loss against the recording is replaced by --high-mode "
                             "(0: the mel loss covers the whole band)")
    parser.add_argument("--high-mode", choices=("stock", "wave", "fine", "voicing"), default="stock",
                        help="stock: mel loss against the stock decoder's output of the same latents (the level of "
                             "its high band is kept); wave: its high-passed waveform is kept (HighBandWaveLoss); "
                             "fine: codec_ft.FineSpectrumLoss against the recording; voicing: the recording in "
                             "unvoiced frames, the stock decoder in voiced ones (VoicingBlendMelLoss)")
    parser.add_argument("--w-high", type=float, default=15.0)
    parser.add_argument("--init", type=Path, default=None, help="decoder weights to start from (default: stock)")
    parser.add_argument("--init-disc", type=Path, default=None,
                        help="checkpoint.pt of another run whose discriminators to start from (new run only)")
    parser.add_argument("--residual-layers", type=int, default=1,
                        help="residual blocks per upsampling stage of the SEANet decoder (added ones start as identities)")
    parser.add_argument("--gan-lowpass", type=float, default=0.0,
                        help="Hz; the discriminators see both signals low-passed (with --high-cross: they would "
                             "ask for the recording's high band again)")
    parser.add_argument("--gan-seconds", type=float, default=1.0, help="slice of the window the discriminators see")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--clip", type=float, default=100.0, help="gradient norm clip of the decoder")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--valid-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu-memory-gb", type=float, default=0.0,
                        help="cap this process's GPU memory (the GPU is shared with evaluations)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    if args.gpu_memory_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_gb * 2**30 / total), 0)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "pid").write_text(f"{os.getpid()}\n")  # SIGTERM this one, not the loader workers
    use_gan = args.w_adv > 0 or args.w_fm > 0
    g_start = args.d_warmup if use_gan else 0

    mimi = load_mimi(load_pocket_config(), device)
    sample_rate, frame = mimi.sample_rate, mimi.frame_size
    stock = load_mimi(load_pocket_config(), device) if args.high_cross > 0 and args.high_mode != "fine" else None
    if args.init is not None:
        load_decoder(mimi, args.init)
    expand_residual_layers(mimi, args.residual_layers)
    params = [param for _, param in decoder_named_parameters(mimi)]
    for param in params:
        param.requires_grad_(True)
    ema = {name: param.detach().clone() for name, param in decoder_named_parameters(mimi)}
    mel_loss = MultiScaleMelLoss(sample_rate).to(device)  # validation: always the whole band
    train_mel, high_loss = mel_loss, None
    if args.high_cross > 0:
        # the mel bands of the whole-band loss, shared out between the two parts by their width in mel
        share = math.log10(1 + args.high_cross / 700) / math.log10(1 + sample_rate / 2 / 700)
        full = (5, 10, 20, 40, 80, 160, 320)
        low_mels = tuple(max(2, round(n * share)) for n in full)
        high_mels = tuple(max(2, n - m) for n, m in zip(full, low_mels))
        train_mel = MultiScaleMelLoss(sample_rate, n_mels=low_mels, f_max=args.high_cross).to(device)
        if args.high_mode == "stock":
            high_loss = MultiScaleMelLoss(sample_rate, n_mels=high_mels, f_min=args.high_cross)
        elif args.high_mode == "wave":
            high_loss = HighBandWaveLoss(sample_rate, args.high_cross)
        elif args.high_mode == "voicing":
            high_loss = VoicingBlendMelLoss(sample_rate, args.high_cross, n_mels=high_mels)
        else:
            high_loss = FineSpectrumLoss(args.high_cross, sample_rate)
        high_loss = high_loss.to(device)
    spec_loss = SharedSpectrumLoss(sample_rate, f_lo=args.spec_band[0], f_hi=args.spec_band[1]).to(device)
    opt_g = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    disc = Discriminators().to(device) if use_gan else None
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.0) if disc else None
    print(f"decoder parameters {sum(p.numel() for p in params) / 1e6:.2f}M"
          + (f", discriminators {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M" if disc else ""), flush=True)

    step = 0
    checkpoint = args.run_dir / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        step = int(saved["step"])
        with torch.no_grad():
            for name, param in decoder_named_parameters(mimi):
                param.copy_(saved["decoder"][name])
        ema = saved["ema"]
        opt_g.load_state_dict(saved["opt_g"])
        if disc is not None and opt_d is not None:
            disc.load_state_dict(saved["disc"])
            opt_d.load_state_dict(saved["opt_d"])
        print(f"resumed from step {step}", flush=True)
    elif args.init_disc is not None and disc is not None:
        disc.load_state_dict(torch.load(args.init_disc, map_location=device, weights_only=True)["disc"])
        print(f"discriminators from {args.init_disc}", flush=True)

    min_seconds = (args.frames + 2) * frame / sample_rate
    paths, durations, sources, held_out = load_rows(args.data, min_seconds, args.max_seconds)
    max_frames = int(args.max_seconds * sample_rate) // frame
    valid_paths = {"heldout": spread(held_out, VALID_CLIPS)}
    for spec in args.valid_glob:
        name, pattern = spec.split("=", 1)
        valid_paths[name] = spread(glob.glob(pattern), VALID_CLIPS)
    valid_sets = {
        name: [read_clip(p, sample_rate, frame, VALID_FRAMES) for p in items] for name, items in valid_paths.items()
    }
    print("validation clips: " + ", ".join(f"{k} {len(v)}" for k, v in valid_sets.items()), flush=True)
    loader = DataLoader(
        ClipDataset(paths, sample_rate, frame, max_frames),
        batch_sampler=BucketBatches(durations, sources, args.batch_size, args.seed + step),
        collate_fn=Collate(frame, args.frames, args.prefix_prob, tuple(args.gain_db)),
        num_workers=args.workers,
        # forkserver (the default from Python 3.14) leaves the workers running after this process:
        # they watch the fork server, and it stays up for as long as they hold its pipe
        multiprocessing_context="fork",
        prefetch_factor=4,
        persistent_workers=True,
    )

    if not args.no_wandb:
        from pocket_tts_ja.wandb_log import init_run

        init_run(args.run_dir, vars(args) | {"run_dir": str(args.run_dir)}, job_type="m2")

    def log(event: str, at: int, values: dict[str, float]) -> None:
        with open(args.run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps({"event": event, "step": at, **values}) + "\n")
        if not args.no_wandb:
            from pocket_tts_ja.wandb_log import log_scalars

            log_scalars(event, at, values)

    def save(at: int, export: bool) -> None:
        state: dict[str, Any] = {
            "step": at,
            "decoder": {name: param.detach() for name, param in decoder_named_parameters(mimi)},
            "ema": ema,
            "opt_g": opt_g.state_dict(),
        }
        if disc is not None and opt_d is not None:
            state |= {"disc": disc.state_dict(), "opt_d": opt_d.state_dict()}
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(checkpoint)
        if export:
            with ema_weights(mimi, ema):
                safetensors.torch.save_file(decoder_state(mimi), str(args.run_dir / f"decoder_step_{at:08d}.safetensors"))

    def lr_scale(at: int) -> float:
        progress = min(1.0, max(0.0, (at - g_start) / max(1, args.max_steps - g_start)))
        return args.final_lr_scale + (1 - args.final_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))

    stop = False

    def on_term(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    if step == 0:
        log("valid", 0, validate(mimi, mel_loss, valid_sets, device))
        print(f"valid @ 0: {json.loads(open(args.run_dir / 'progress.jsonl').readlines()[-1])}", flush=True)
    sums: dict[str, float] = defaultdict(float)
    count = 0
    began = time.time()
    segment = int(args.gan_seconds * sample_rate)
    lowpass: torch.Tensor | None = None
    if args.gan_lowpass > 0:
        from scipy.signal import firwin

        lowpass = torch.tensor(firwin(255, args.gan_lowpass, fs=sample_rate), dtype=torch.float32, device=device)[None, None]

    def for_disc(x: torch.Tensor) -> torch.Tensor:
        if lowpass is None:
            return x
        return torch.nn.functional.conv1d(x[:, None], lowpass, padding=lowpass.shape[-1] // 2)[:, 0]

    for wav, start in loader:
        if step >= args.max_steps or stop:
            break
        wav = wav.to(device, non_blocking=True)
        train_g = step >= g_start
        scale = lr_scale(step)
        for group in opt_g.param_groups:
            group["lr"] = args.lr * scale * min(1.0, max(0.0, (step - g_start + 1) / max(1, args.warmup)))
        with torch.no_grad():
            latents = mimi.encode_to_latent(wav)
        real = wav[..., start * frame :]
        with torch.set_grad_enabled(train_g):
            fake = decode_window(mimi, latents, start)[..., : real.shape[-1]]
        stats: dict[str, float] = {}
        if disc is not None and opt_d is not None:
            for group in opt_d.param_groups:
                group["lr"] = args.lr_d * scale
            offsets = torch.randint(0, real.shape[-1] - segment + 1, (wav.shape[0], 1), device=device)
            index = offsets + torch.arange(segment, device=device)[None]
            real_seg = for_disc(real[:, 0].gather(1, index))
            fake_seg = for_disc(fake[:, 0].gather(1, index))
            d_real, d_fake = disc(real_seg), disc(fake_seg.detach())
            loss_d = discriminator_loss(d_real, d_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            stats["grad_d"] = float(torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000.0))
            opt_d.step()
            stats["loss_d"] = float(loss_d.detach())
            stats["d_real"] = float(torch.stack([r.detach().mean() for r, _ in d_real]).mean())
            stats["d_fake"] = float(torch.stack([f.detach().mean() for f, _ in d_fake]).mean())
            del d_real, d_fake, loss_d  # their feature maps would stay allocated through the decoder step
        if train_g:
            mel = train_mel(fake, real)
            loss_g = args.w_mel * mel
            stats["mel"] = float(mel.detach())
            if high_loss is not None:
                if stock is None:
                    high = high_loss(fake, real)
                else:
                    with torch.no_grad():
                        kept = decode_window(stock, latents, start)[..., : real.shape[-1]]
                    high = high_loss(fake, real, kept) if args.high_mode == "voicing" else high_loss(fake, kept)
                loss_g = loss_g + args.w_high * high
                stats["high"] = float(high.detach())
            if args.w_spec > 0:
                shared = spec_loss(fake, real)
                loss_g = loss_g + args.w_spec * shared
                stats["spec"] = float(shared.detach())
            if disc is not None:
                with torch.no_grad():
                    d_real = disc(real_seg)
                d_fake = disc(fake_seg)
                adv, fm = generator_loss(d_fake), feature_loss(d_real, d_fake)
                loss_g = loss_g + args.w_adv * adv + args.w_fm * fm
                stats |= {"adv": float(adv.detach()), "fm": float(fm.detach())}
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            stats["grad_g"] = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            opt_g.step()
            with torch.no_grad():
                for name, param in decoder_named_parameters(mimi):
                    ema[name].lerp_(param, 1 - args.ema)
        else:
            with torch.no_grad():
                stats["mel"] = float(mel_loss(fake, real))
        step += 1
        count += 1
        for key, value in stats.items():
            sums[key] += value
        if step % args.log_every == 0:
            mean = {key: value / count for key, value in sums.items()}
            mean["lr"] = opt_g.param_groups[0]["lr"]
            mean["it_per_s"] = count / (time.time() - began)
            mean["gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            log("train", step, mean)
            print(f"step {step} | " + " ".join(f"{k} {v:.4g}" for k, v in mean.items()), flush=True)
            sums.clear()
            count = 0
            began = time.time()
        if step % args.valid_every == 0 or step == args.max_steps:
            with ema_weights(mimi, ema):
                values = validate(mimi, mel_loss, valid_sets, device)
            log("valid", step, values)
            print(f"valid @ {step}: " + " ".join(f"{k} {v:.4f}" for k, v in values.items()), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save(step, export=True)
        elif step % args.valid_every == 0:
            save(step, export=False)  # a restart resumes from here
    if stop:
        save(step, export=False)
        print(f"stopped at step {step} (checkpoint written)", flush=True)
    else:
        print("M2_TRAIN_DONE", flush=True)
    if not args.no_wandb:
        import wandb

        wandb.finish()
    os._exit(0)  # loader workers can keep the interpreter alive


if __name__ == "__main__":
    main()
