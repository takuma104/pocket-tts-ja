"""M2 high band: can the decoder put harmonics above 4 kHz at all, when it only has to fit a few clips?

Every decoder so far (the stock one and every fine-tune) stops making harmonics at about 4.2 kHz
(scripts/m2/hf_structure.py --own-f0: the comb at the output's own F0 matches the recording up to
4 kHz and falls to about 1 dB in 4.5-5 kHz). This trains the decode side on a handful of clips,
over and over, with a loss that sees the harmonics of the band above --cross (codec_ft.FineSpectrumLoss
against the recording; below it the mel loss against the recording), and follows the comb of the
band on those clips and on held-out clips of the same speaker. Memorizing a few clips takes away
the uncertainty of the pitch and of the content: if even then no harmonics appear above 4 kHz, the
structure (or the optimization) is what keeps them out; if they appear, the structure can make
them and the question is what the full training lacks. --residual-layers 3 repeats it with the
zero-initialized blocks of codec_ft.expand_residual_layers.

    uv run python scripts/m2/overfit_probe.py exp/m2/overfit/res1 --train data/m2/probe_hfcf_train \
        --heldout data/m2/probe_hfcf_dev --init runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import safetensors.torch
import soundfile as sf
import torch

from pocket_tts_ja.codec_ft import (
    FineSpectrumLoss,
    MultiScaleMelLoss,
    decode_window,
    decoder_named_parameters,
    decoder_state,
    expand_residual_layers,
    load_decoder,
)
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_structure import N_FFT, PAD, band_stats, refine_f0, spectrum, voiced_f0  # noqa: E402

BANDS = ((2500, 4000), (4000, 5000), (5000, 6000), (6000, 8000))


def read_set(directory: Path, frame: int, max_frames: int) -> list[torch.Tensor]:
    clips = []
    for line in open(directory / "manifest.jsonl", encoding="utf-8"):
        row = json.loads(line)
        wav, sr = sf.read(str(directory / "audio" / f"{row['id']}.wav"), dtype="float32")
        assert sr == 24000, sr
        frames = min(len(wav) // frame, max_frames)
        clips.append(torch.from_numpy(wav[: frames * frame].copy()))
    return clips


def structure(refs: list[npt.NDArray[np.float32]], outs: list[npt.NDArray[np.float32]]) -> dict[str, list[float]]:
    """Per band: comb of the recording, of the output at the recording's F0 and at its own F0, and
    the output's level re the recording (dB), over the recording's voiced frames."""
    import hf_structure

    saved = hf_structure.BANDS
    hf_structure.BANDS = BANDS
    step = 24000 / (N_FFT * PAD)
    rows: dict[str, list[list[float]]] = {"comb_rec": [], "comb_out": [], "comb_own": [], "p_rec": [], "p_out": []}
    try:
        for ref, out in zip(refs, outs, strict=True):
            for start, f0_raw in voiced_f0(ref, 24000):
                spec_ref, spec_out = spectrum(ref[start : start + N_FFT]), spectrum(out[start : start + N_FFT])
                f0 = refine_f0(np.log(spec_ref + 1e-9), f0_raw, step)
                # as hf_structure.py --own-f0: refined on 1-2.5 kHz, so the bands above are not selected for
                f0_own = refine_f0(np.log(spec_out + 1e-9), f0_raw, step, span=0.04, lo=1000, hi=2500)
                rec, at_rec, at_own = band_stats(spec_ref, f0, step), band_stats(spec_out, f0, step), band_stats(spec_out, f0_own, step)
                rows["comb_rec"].append([s[0] for s in rec])
                rows["comb_out"].append([s[0] for s in at_rec])
                rows["comb_own"].append([s[0] for s in at_own])
                rows["p_rec"].append([s[1] for s in rec])
                rows["p_out"].append([s[1] for s in at_rec])
    finally:
        hf_structure.BANDS = saved
    level = 10 * np.log10(np.mean(rows["p_out"], axis=0) / np.mean(rows["p_rec"], axis=0))
    return {
        "frames": [float(len(rows["comb_rec"]))],
        **{key: np.mean(rows[key], axis=0).tolist() for key in ("comb_rec", "comb_out", "comb_own")},
        "level": level.tolist(),
    }


@torch.no_grad()
def evaluate(mimi: torch.nn.Module, clips: list[torch.Tensor], device: torch.device) -> tuple[dict[str, list[float]], list[npt.NDArray[np.float32]]]:
    refs, outs = [], []
    for clip in clips:
        wav = clip.to(device)[None, None]
        out = decode_window(mimi, mimi.encode_to_latent(wav), 0)[0, 0, : wav.shape[-1]]
        refs.append(clip.numpy())
        outs.append(out.float().cpu().numpy())
    return structure(refs, outs), outs


def table_row(label: str, s: dict[str, list[float]]) -> str:
    cells = [f"{a:.1f} / {b:.1f} / {c:.1f} / {d:+.1f}" for a, b, c, d in zip(s["comb_rec"], s["comb_out"], s["comb_own"], s["level"])]
    return f"| {label} | " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train", type=Path, required=True, help="directory from extract_corpus.py")
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=None, help="decoder to start from (default: stock)")
    parser.add_argument("--residual-layers", type=int, default=1)
    parser.add_argument("--cross", type=float, default=4000.0, help="Hz: mel loss below, fine STFT loss above")
    parser.add_argument("--high", choices=("fine", "mel"), default="fine",
                        help="loss above --cross: fine STFT (sees harmonics) or the mel loss (does not)")
    parser.add_argument("--w-mel", type=float, default=15.0)
    parser.add_argument("--w-high", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    mimi = load_mimi(load_pocket_config(), device)
    if args.init is not None:
        load_decoder(mimi, args.init)
    expand_residual_layers(mimi, args.residual_layers)
    params = [p for _, p in decoder_named_parameters(mimi)]
    for p in params:
        p.requires_grad_(True)
    frame = mimi.frame_size
    train = read_set(args.train, frame, args.max_frames)
    heldout = read_set(args.heldout, frame, args.max_frames)
    sr = mimi.sample_rate
    share = math.log10(1 + args.cross / 700) / math.log10(1 + sr / 2 / 700)
    full = (5, 10, 20, 40, 80, 160, 320)
    low_mels = tuple(max(2, round(n * share)) for n in full)
    high_mels = tuple(max(2, n - m) for n, m in zip(full, low_mels))
    low_loss = MultiScaleMelLoss(sr, n_mels=low_mels, f_max=args.cross).to(device)
    high_loss = (FineSpectrumLoss(args.cross, sr) if args.high == "fine"
                 else MultiScaleMelLoss(sr, n_mels=high_mels, f_min=args.cross)).to(device)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    head = " | ".join(f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS)
    print(f"{len(train)} training clips, {len(heldout)} held out, decoder {sum(p.numel() for p in params) / 1e6:.2f}M parameters")
    print(f"comb of the recording / output at the recording's F0 / output at its own F0 / level re the recording\n| step / set | {head} |")
    print("|---|" + "---|" * len(BANDS), flush=True)
    log = open(args.run_dir / "progress.jsonl", "a")

    def report(step: int) -> None:
        for name, clips in (("train", train), ("heldout", heldout)):
            stats, _ = evaluate(mimi, clips, device)
            print(table_row(f"{step} {name}", stats), flush=True)
            log.write(json.dumps({"step": step, "set": name, **stats}) + "\n")
            log.flush()

    report(0)
    usable = min(len(c) // frame for c in train)
    for step in range(1, args.steps + 1):
        batch = torch.stack([c[: usable * frame] for c in random.sample(train, min(args.batch_size, len(train)))])
        wav = batch.to(device)[:, None]
        with torch.no_grad():
            latents = mimi.encode_to_latent(wav)
        fake = decode_window(mimi, latents, 0)[..., : wav.shape[-1]]
        mel, high = low_loss(fake, wav), high_loss(fake, wav)
        loss = args.w_mel * mel + args.w_high * high
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 100.0)
        opt.step()
        if step % 200 == 0:
            log.write(json.dumps({"step": step, "mel": float(mel.detach()), "high": float(high.detach())}) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            report(step)
    safetensors.torch.save_file(decoder_state(mimi), str(args.run_dir / f"decoder_step_{args.steps:08d}.safetensors"))
    print("OVERFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
