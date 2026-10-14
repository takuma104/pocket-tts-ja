"""M2 high band, Phase 3: can the decoder's own features give the F0 that drives the harmonic source?

The harmonic source (pocket_tts_ja.harmonic) works with the recording's F0, which a TTS does not
have. The decoder already knows the pitch (its output is a median 4 cents off the recording's), so
a small causal head on the decoder transformer's output (512 channels, 200 Hz, the frames the source
is made at) should be able to say it. This trains such a head on the training sources (target: the
recording's F0 from harmonic.f0_track, refined on the spectrum, and its voicing) with the decoder
frozen, and measures on JVS (never trained on):

    cents        |predicted - target| in the frames voiced in both (median, p90, share > 1%)
    voicing      agreement of the voiced / unvoiced decision
    comb 4-6k    the recording's comb at 4-6 kHz in a 43 ms window at each 5 ms frame, at the
                 predicted F0 and at the target F0: how well harmonics made from it would sit on
                 the recording's (the number that matters for the source)

--head classify predicts a distribution over log-F0 bins 10 cents wide (50-800 Hz) instead of a
value, trained on a target blurred over 25 cents (as CREPE does), and reads the F0 off as the
weighted mean of the bins around the most likely one: a regression on F0 is pulled between octave
candidates and lands between them.

    uv run python scripts/m2/f0_probe.py exp/m2/f0_probe --decoder runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.modules.stateful_module import init_states
from pocket_tts_ja.codec_ft import load_decoder
from pocket_tts_ja.harmonic import V3, f0_track, refine_f0_track
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config, to_mono_float

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_structure import contrast  # noqa: E402

SOURCES = {
    "emilia": "data/m2/emilia_hq",
    "hfc_f": "data/m2/hfc_f",
    "hfc_m": "data/m2/hfc_m",
    "jvnv": "data/m2/jvnv",
}


BIN_CENTS = 10.0
F0_LO, F0_HI = 50.0, 800.0
N_BINS = int(1200 * np.log2(F0_HI / F0_LO) / BIN_CENTS) + 1


class F0Head(nn.Module):
    """Causal convolutions on the decoder transformer's output -> per frame either (log2 F0, voicing
    logit) or (bin logits, voicing logit)."""

    def __init__(self, dim: int = 512, hidden: int = 256, kernel: int = 3, layers: int = 3, classify: bool = False) -> None:
        super().__init__()
        self.kernel = kernel
        self.classify = classify
        self.convs = nn.ModuleList([nn.Conv1d(dim if i == 0 else hidden, hidden, kernel) for i in range(layers)])
        self.out = nn.Conv1d(hidden, (N_BINS if classify else 1) + 1, 1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = emb
        for conv in self.convs:
            x = F.gelu(conv(F.pad(x, (self.kernel - 1, 0))))
        y = self.out(x)
        if self.classify:
            return y[:, :-1], y[:, -1]  # [B, bins, frames]
        return y[:, 0] + 8.0, y[:, 1]  # log2 F0 around 256 Hz


def bin_cents() -> torch.Tensor:
    return torch.arange(N_BINS, dtype=torch.float32) * BIN_CENTS  # cents above F0_LO


def decode_bins(logits: torch.Tensor, width: int = 4) -> torch.Tensor:
    """[B, bins, frames] -> F0 in Hz: the probability-weighted mean of the bins within `width` of the best."""
    prob = torch.sigmoid(logits)
    best = prob.argmax(1, keepdim=True)
    idx = (best + torch.arange(-width, width + 1, device=logits.device)[None, :, None]).clamp(0, N_BINS - 1)
    p = torch.gather(prob, 1, idx)
    cents = (idx.float() * BIN_CENTS * p).sum(1) / p.sum(1).clamp(min=1e-6)
    return F0_LO * 2 ** (cents / 1200)


@torch.no_grad()
def features(mimi: torch.nn.Module, wav: torch.Tensor) -> torch.Tensor:
    """[B, T] waveform -> [B, 512, frames] decoder transformer output at 200 Hz."""
    latents = mimi.encode_to_latent(wav[:, None])
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, wav.shape[0], (latents.shape[1] + 8) * ratio)
    latent = mimi.quantizer(latents.transpose(-1, -2))
    emb = mimi._to_encoder_framerate(latent, state)
    (emb,) = mimi.decoder_transformer(emb, state)
    return emb


@torch.no_grad()
def targets(wav: torch.Tensor, frames: int, hop: int, sr: int) -> torch.Tensor:
    f0 = torch.stack([
        torch.from_numpy(f0_track(w.cpu().numpy(), sr, frames, V3.allowed_range, V3.fill_frames)) for w in wav
    ]).to(wav.device)
    return refine_f0_track(wav, f0, hop, sr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--decoder", type=Path, default=None, help="decoder file (default: stock)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--jvs-glob", default="data/jvs_ver1/jvs0[0-9][0-9]/parallel100/wav24kHz16bit/VOICEACTRESS100_00[1-2].wav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=("regress", "classify"), default="regress")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    mimi = load_mimi(load_pocket_config(), device)
    if args.decoder is not None:
        load_decoder(mimi, args.decoder)
    sr = mimi.sample_rate
    hop = round(sr / mimi.encoder_frame_rate)
    frame = mimi.frame_size
    clip_len = int(args.seconds * sr) // frame * frame
    paths = []
    for name, root in SOURCES.items():
        rows = [json.loads(line) for line in open(f"{root}/manifest.jsonl", encoding="utf-8")]
        paths += [(name, f"{root}/audio/{r['id']}.wav") for r in rows if r["duration"] >= args.seconds]
    print(f"{len(paths)} training clips", flush=True)

    def batch() -> torch.Tensor:
        out = []
        for _, path in random.sample(paths, args.batch_size):
            wav, _ = sf.read(path, dtype="float32")
            if len(wav) < clip_len:  # the manifest's duration is rounded
                wav = np.pad(wav, (0, clip_len - len(wav)))
            start = random.randint(0, len(wav) - clip_len)
            out.append(torch.from_numpy(wav[start : start + clip_len].copy()))
        return torch.stack(out).to(device)

    head = F0Head(mimi.decoder.dimension, args.hidden, layers=args.layers, classify=args.head == "classify").to(device)
    centers = bin_cents().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(1, args.steps + 1):
        wav = batch()
        emb = features(mimi, wav)
        f0 = targets(wav, emb.shape[-1], hop, sr)
        out, logit = head(emb)
        voiced = f0 > 0
        if head.classify:
            target_c = 1200 * torch.log2(f0.clamp(min=F0_LO) / F0_LO)  # [B, frames]
            soft = torch.exp(-0.5 * ((centers[None, :, None] - target_c[:, None]) / 25.0) ** 2)
            bce = F.binary_cross_entropy_with_logits(out, soft, reduction="none").mean(1)
            loss_f0 = (bce * voiced).sum() / voiced.sum().clamp(min=1) * 10
            with torch.no_grad():
                cents = 1200 * torch.log2(decode_bins(out) / f0.clamp(min=1.0))
        else:
            cents = 1200 * (out - torch.log2(f0.clamp(min=1.0)))
            loss_f0 = (cents.abs() * voiced).sum() / voiced.sum().clamp(min=1) / 100
        loss_v = F.binary_cross_entropy_with_logits(logit, voiced.float())
        loss = loss_f0 + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            med = float(cents.detach().abs()[voiced].median()) if voiced.any() else float("nan")
            print(f"step {step} | loss {float(loss_f0):.3f} median cents {med:.1f} voicing {float(loss_v):.3f}", flush=True)
    torch.save(head.state_dict(), args.out / "f0_head.pt")

    # JVS: never trained on
    n_fft, pad = 1024, 8
    step_hz = sr / (n_fft * pad)
    errs, agree, comb_pred, comb_tgt = [], [], [], []
    for path in sorted(glob.glob(args.jvs_glob)):
        wav, file_sr = sf.read(path, dtype="float32")
        wav = to_mono_float(wav, file_sr, sr)
        wav = wav[: len(wav) // frame * frame]
        w = torch.from_numpy(wav)[None].to(device)
        with torch.no_grad():
            emb = features(mimi, w)
            f0 = targets(w, emb.shape[-1], hop, sr)[0].cpu().numpy()
            out, logit = head(emb)
        pred = (decode_bins(out) if head.classify else 2 ** out)[0].cpu().numpy()
        pred_v = (logit[0] > 0).cpu().numpy()
        agree.append(float(np.mean(pred_v == (f0 > 0))))
        both = (f0 > 0) & pred_v
        errs += list(1200 * np.log2(pred[both] / f0[both]))
        for c in np.flatnonzero(both)[::4]:
            s0 = c * hop - n_fft // 2
            if s0 < 0 or s0 + n_fft > len(wav) or np.sqrt(np.mean(wav[s0 : s0 + n_fft] ** 2)) < 0.01:
                continue
            log = np.log(np.abs(np.fft.rfft(wav[s0 : s0 + n_fft] * np.hanning(n_fft), n_fft * pad)) + 1e-9)
            comb_pred.append(contrast(log, float(pred[c]), 4000, 6000, step_hz))
            comb_tgt.append(contrast(log, float(f0[c]), 4000, 6000, step_hz))
    e = np.abs(errs)
    result = {
        "cents_median": float(np.median(e)), "cents_p90": float(np.percentile(e, 90)), "over_1pct": float(np.mean(e > 17.2)),
        "voicing_agreement": float(np.mean(agree)), "comb4_6k_at_pred": float(np.mean(comb_pred)),
        "comb4_6k_at_target": float(np.mean(comb_tgt)), "frames": len(e),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
