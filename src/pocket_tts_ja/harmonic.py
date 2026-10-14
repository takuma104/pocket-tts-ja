"""A harmonic source for the Mimi decoder's band above 4 kHz (plan M2 high band, Phase 3 B1).

Every decoder so far fills the band above 4 kHz of a voice with noise bursts that come with the
glottal pulses but do not repeat their waveform from one period to the next (scripts/m2/
hf_coherence.py): harmonics need a waveform that stays in phase over many periods, which the
decoder does not learn to make in general (it does for a few memorized clips). This gives it one:
sines at the harmonics of F0, with the phase accumulated sample by sample, summed per band with
smooth band edges (no harmonic pops in or out as F0 moves), a sine and a cosine sum per band. A
gain per channel and decoder frame (200 Hz), computed from the decoder transformer's output by a
1x1 convolution that starts at zero, scales them; their sum is added to the decoder's waveform.
So the decoder starts out exactly as it was and learns how much of each band is harmonic.

F0 comes from WORLD (dio + stonemask) at 5 ms, the decoder's frame period. In this diagnostic
version it is the recording's F0, which a TTS does not have: the question is whether a coherent
harmonic source lets the decoder put natural harmonics into unseen voices at all.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import increment_steps, init_states

BANDS: tuple[tuple[float, float], ...] = ((3000.0, 4500.0), (4500.0, 6000.0), (6000.0, 8000.0), (8000.0, 11000.0))
EDGE = 500.0  # Hz, raised-cosine band edges
TOP = 11500.0  # no harmonic above this (Nyquist 12 kHz)
F0_FLOOR, F0_CEIL = 60.0, 700.0
FRAME_MS = 5.0


def f0_track(wav: npt.NDArray[np.float32], sample_rate: int, frames: int) -> npt.NDArray[np.float32]:
    """F0 in Hz at the decoder's frames (one per 5 ms from the start), 0 where unvoiced."""
    import pyworld

    x = wav.astype(np.float64)
    f0, t = pyworld.dio(x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS)
    f0 = pyworld.stonemask(x, f0, t, sample_rate)
    out = np.zeros(frames, dtype=np.float32)
    n = min(frames, len(f0))
    out[:n] = f0[:n]
    return out


def band_weight(freq: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """1 inside [lo, hi], smooth (smoothstep) edges EDGE wide outside it, 0 above TOP."""
    rise = ((freq - (lo - EDGE)) / EDGE).clamp(0, 1)
    fall = (((hi + EDGE) - freq) / EDGE).clamp(0, 1)
    w = rise * rise * (3 - 2 * rise) * fall * fall * (3 - 2 * fall)
    return w * (freq < TOP)


@torch.no_grad()
def harmonic_source(
    f0: torch.Tensor, hop: int, sample_rate: int, bands: Sequence[tuple[float, float]] = BANDS, phase0: torch.Tensor | None = None
) -> torch.Tensor:
    """[B, frames] F0 (Hz, 0 = unvoiced) -> [B, 2 * len(bands), frames * hop] band-limited harmonic sums.

    Channels: sin and cos sums of each band, each scaled to unit RMS whatever the number of
    harmonics in it. Through unvoiced stretches the phase keeps running at the nearest voiced F0
    and the source fades out and in over one frame, so a voicing change does not click.
    """
    batch, frames = f0.shape
    voiced = (f0 > 0).float()
    # F0 through the unvoiced stretches: the previous voiced value (or the next one at the start)
    filled = f0.clone()
    for b in range(batch):
        idx = torch.nonzero(f0[b] > 0).flatten()
        if len(idx) == 0:
            filled[b] = 100.0
            continue
        pos = torch.arange(frames, device=f0.device)
        nearest = torch.searchsorted(idx, pos, right=True) - 1
        nearest = nearest.clamp(min=0)
        filled[b] = f0[b, idx[nearest]]
    length = frames * hop
    # sample-rate F0 and voicing by linear interpolation between frame values
    f0_s = F.interpolate(filled[:, None], size=length, mode="linear", align_corners=False)[:, 0]
    v_s = F.interpolate(voiced[:, None], size=length, mode="linear", align_corners=False)[:, 0]
    phase = 2 * torch.pi * torch.cumsum(f0_s / sample_rate, dim=-1)
    if phase0 is not None:
        phase = phase + phase0[:, None]
    sums = torch.zeros(batch, 2 * len(bands), length, device=f0.device)
    norm = torch.zeros(batch, len(bands), length, device=f0.device)
    f_min, f_max = float(f0_s.min().clamp(min=F0_FLOOR)), float(f0_s.max())
    unit = torch.polar(torch.ones_like(phase), phase)  # e^{i phase}; e^{i k phase} by repeated products
    z = torch.ones_like(unit)
    for k in range(1, int(TOP / f_min) + 1):
        z = z * unit
        if k % 32 == 0:  # keep the magnitude at 1 against rounding
            z = z / z.abs()
        for i, (lo, hi) in enumerate(bands):
            if k * f_max < lo - EDGE or k * f_min > hi + EDGE:
                continue
            w = band_weight(k * f0_s, lo, hi)
            sums[:, 2 * i] += w * z.imag
            sums[:, 2 * i + 1] += w * z.real
            norm[:, i] += w * w
    scale = torch.rsqrt(norm / 2 + 1e-3).repeat_interleave(2, dim=1)
    return sums * scale * v_s[:, None]


class HarmonicHead(nn.Module):
    """Per-frame gains of the source channels from the decoder transformer's output."""

    def __init__(self, dim: int, channels: int, hop: int) -> None:
        super().__init__()
        self.hop = hop
        self.gain = nn.Conv1d(dim, channels, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, emb: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """emb [B, dim, frames], source [B, channels, frames * hop] -> [B, 1, frames * hop].

        The gain moves linearly from the previous frame's value to this frame's over the frame
        (causal: a frame's samples depend on that frame and the one before)."""
        g = self.gain(emb)
        prev = torch.cat([g[..., :1], g[..., :-1]], dim=-1)
        w = (torch.arange(self.hop, device=g.device, dtype=g.dtype) + 1) / self.hop
        g_s = (prev[..., None] * (1 - w) + g[..., None] * w).flatten(-2)
        return (g_s * source).sum(dim=1, keepdim=True)


def decode_harmonic(
    mimi: MimiModel, head: HarmonicHead, latents: torch.Tensor, start: int, source: torch.Tensor, context_frames: int = 8
) -> torch.Tensor:
    """codec_ft.decode_window plus the harmonic part: [B, T, 32] latents from the start of a clip,
    source for the decoded window ([B, C, (T - start) * frame]) -> [B, 1, (T - start) * frame]."""
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, latents.shape[0], (latents.shape[1] + context_frames) * ratio)
    if start > 0:
        with torch.no_grad():
            mimi.decode_from_latent(latents[:, :start], state)
        increment_steps(mimi, state, increment=start * ratio)
    latent = mimi.quantizer(latents[:, start:].transpose(-1, -2))
    emb = mimi._to_encoder_framerate(latent, state)
    (emb,) = mimi.decoder_transformer(emb, state)
    out = mimi.decoder(emb, state)
    return out + head(emb, source[..., : out.shape[-1]])


PREFIX = "mimi.harmonic."


def head_state(head: HarmonicHead) -> dict[str, torch.Tensor]:
    """The head's tensors under the names they get next to codec_ft.decoder_state's."""
    return {PREFIX + name: t.detach().float().cpu().contiguous() for name, t in head.state_dict().items()}


def load_harmonic_head(path: Path, dim: int, hop: int, device: torch.device) -> HarmonicHead | None:
    """The head stored in a decoder file, or None for a file without one."""
    import safetensors.torch

    state = {k.removeprefix(PREFIX): v for k, v in safetensors.torch.load_file(str(path)).items() if k.startswith(PREFIX)}
    if not state:
        return None
    head = HarmonicHead(dim, state["gain.weight"].shape[0], hop)
    head.load_state_dict(state)
    return head.to(device).eval()
