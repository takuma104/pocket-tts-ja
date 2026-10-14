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

The bands, their edges and the F0 settings are a HarmonicConfig, stored in the metadata of the
decoder file (a file without it is V1, the first run's). V2 starts at 4 kHz (the decoder already
makes the harmonics below), has bands 500-1500 Hz wide whose edges sum to one across a boundary,
lets dio search a wider range and bridges unvoiced gaps of up to 20 ms inside a voiced stretch.

WORLD's F0 is not precise enough for harmonics around 5 kHz: on JVS (female) it is a median 21-25
cents (0.8% low on the median) off the F0 at which the recording's harmonics stand out most, and at
WORLD's F0 the recording's own 4-6 kHz comb drops from 5.1 to 2.5 dB (around the 20th harmonic a
1.5% error moves a harmonic by a third of the spacing). On synthetic tones both agree within a few
cents; on speech the spectrum is the arbiter. V3 therefore refines every 5 ms frame on the
spectrum (refine_f0_track): the F0 within 4% of WORLD's whose harmonics in 1-4 kHz stand out most
in a 43 ms window.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import increment_steps, init_states

TOP = 11500.0  # no harmonic above this (Nyquist 12 kHz)
F0_FLOOR, F0_CEIL = 60.0, 700.0
FRAME_MS = 5.0


@dataclass(frozen=True)
class HarmonicConfig:
    bands: tuple[tuple[float, float], ...]
    edge: float  # Hz, width of the smooth band edges
    centered: bool  # edges centred on the band limits (adjacent bands sum to one) or outside them
    allowed_range: float  # pyworld.dio's
    fill_frames: int  # unvoiced gaps of up to this many 5 ms frames inside voicing are bridged
    refine: bool = False  # refine_f0_track on the waveform before the source is made

    def metadata(self) -> dict[str, str]:
        return {"harmonic_config": json.dumps(asdict(self))}

    @staticmethod
    def from_metadata(metadata: dict[str, str] | None) -> "HarmonicConfig":
        if not metadata or "harmonic_config" not in metadata:
            return V1
        d = json.loads(metadata["harmonic_config"])
        return HarmonicConfig(
            tuple(tuple(b) for b in d["bands"]), d["edge"], d["centered"], d["allowed_range"], d["fill_frames"], d.get("refine", False)
        )


V1 = HarmonicConfig(((3000.0, 4500.0), (4500.0, 6000.0), (6000.0, 8000.0), (8000.0, 11000.0)), 500.0, False, 0.1, 0)
V2 = HarmonicConfig(
    ((4000.0, 4500.0), (4500.0, 5000.0), (5000.0, 5500.0), (5500.0, 6000.0), (6000.0, 6750.0), (6750.0, 7500.0),
     (7500.0, 8500.0), (8500.0, 10000.0), (10000.0, 11000.0)),
    250.0, True, 0.2, 4,
)
V3 = HarmonicConfig(V2.bands, V2.edge, V2.centered, V2.allowed_range, V2.fill_frames, refine=True)
# V3 up to 8 kHz only: with the source above it the decoder's 8-11 kHz became twice as harmonic as speech
V4 = HarmonicConfig(
    tuple(b for b in V2.bands if b[1] <= 7500.0) + ((7500.0, 8000.0),), V2.edge, V2.centered, V2.allowed_range, V2.fill_frames,
    refine=True,
)
CONFIGS = {"v1": V1, "v2": V2, "v3": V3, "v4": V4}
BANDS = V1.bands


def f0_track(
    wav: npt.NDArray[np.float32], sample_rate: int, frames: int, allowed_range: float = 0.1, fill_frames: int = 0
) -> npt.NDArray[np.float32]:
    """F0 in Hz at the decoder's frames (one per 5 ms from the start), 0 where unvoiced."""
    import pyworld

    x = wav.astype(np.float64)
    f0, t = pyworld.dio(x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS, allowed_range=allowed_range)
    f0 = pyworld.stonemask(x, f0, t, sample_rate)
    if fill_frames:
        voiced = np.flatnonzero(f0 > 0)
        for a, b in zip(voiced[:-1], voiced[1:], strict=True):
            if 1 < b - a <= fill_frames + 1:
                f0[a + 1 : b] = np.interp(np.arange(a + 1, b), [a, b], [f0[a], f0[b]])
    out = np.zeros(frames, dtype=np.float32)
    n = min(frames, len(f0))
    out[:n] = f0[:n]
    return out


@torch.no_grad()
def refine_f0_track(
    wav: torch.Tensor, f0: torch.Tensor, hop: int, sample_rate: int, span: float = 0.04, n_fft: int = 1024, pad: int = 4,
    lo: float = 1000.0, hi: float = 4000.0,
) -> torch.Tensor:
    """[B, T] waveform, [B, frames] F0 (0 = unvoiced) -> the F0 of each voiced frame refined on the spectrum.

    Frame i is the 43 ms window centred on sample i * hop. Two passes: 81 candidates within `span`
    of the given F0, then 21 within a step of the best; the score is the mean log magnitude at the
    harmonics in [lo, hi] minus that halfway between them.
    """
    batch, frames = f0.shape
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft * pad, hop, win_length=n_fft, window=window, center=True, return_complex=True)
    log = spec.abs().clamp_min(1e-7).log().transpose(1, 2)[:, :frames]  # [B, frames, bins]
    if log.shape[1] < frames:
        log = F.pad(log, (0, 0, 0, frames - log.shape[1]))
    bin_hz = sample_rate / (n_fft * pad)
    k = torch.arange(1, int(hi / F0_FLOOR) + 1, device=wav.device, dtype=torch.float32)

    def best(center: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        cand = center[..., None] * (1 + rel)  # [B, frames, C]
        harm = cand[..., None] * k  # [B, frames, C, K]
        mask = ((harm >= lo) & (harm <= hi)).float()
        peaks = torch.gather(log, 2, (harm / bin_hz).round().long().clamp(max=log.shape[-1] - 1).flatten(2)).view_as(harm)
        valleys = torch.gather(log, 2, ((harm + 0.5 * cand[..., None]) / bin_hz).round().long().clamp(max=log.shape[-1] - 1).flatten(2))
        score = ((peaks - valleys.view_as(harm)) * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return torch.gather(cand, 2, score.argmax(-1, keepdim=True))[..., 0]

    voiced = f0 > 0
    base = torch.where(voiced, f0, torch.full_like(f0, 200.0))
    first = best(base, torch.linspace(-span, span, 81, device=wav.device))
    step = 2 * span / 80
    second = best(first, torch.linspace(-step, step, 21, device=wav.device))
    return torch.where(voiced, second, torch.zeros_like(f0))


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def band_weight(freq: torch.Tensor, lo: float, hi: float, edge: float = 500.0, centered: bool = False) -> torch.Tensor:
    """Weight of a harmonic at `freq` in the band [lo, hi]: smooth edges `edge` wide, outside the band
    or centred on its limits; 0 above TOP."""
    if centered:
        w = smoothstep((freq - lo) / edge + 0.5) - smoothstep((freq - hi) / edge + 0.5)
    else:
        w = smoothstep((freq - (lo - edge)) / edge) * smoothstep(((hi + edge) - freq) / edge)
    return w * (freq < TOP)


@torch.no_grad()
def harmonic_source(
    f0: torch.Tensor, hop: int, sample_rate: int, config: HarmonicConfig = V1, phase0: torch.Tensor | None = None
) -> torch.Tensor:
    """[B, frames] F0 (Hz, 0 = unvoiced) -> [B, 2 * len(bands), frames * hop] band-limited harmonic sums.

    Channels: sin and cos sums of each band of `config`, each scaled to unit RMS whatever the number of
    harmonics in it. Through unvoiced stretches the phase keeps running at the nearest voiced F0
    and the source fades out and in over one frame, so a voicing change does not click.
    """
    bands, edge = config.bands, config.edge
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
            if k * f_max < lo - edge or k * f_min > hi + edge:
                continue
            w = band_weight(k * f0_s, lo, hi, edge, config.centered)
            sums[:, 2 * i] += w * z.imag
            sums[:, 2 * i + 1] += w * z.real
            norm[:, i] += w * w
    scale = torch.rsqrt(norm / 2 + 1e-3).repeat_interleave(2, dim=1)
    return sums * scale * v_s[:, None]


BIN_CENTS = 10.0
F0_LO, F0_HI = 50.0, 800.0
N_BINS = int(1200 * np.log2(F0_HI / F0_LO) / BIN_CENTS) + 1


class F0Head(nn.Module):
    """Causal convolutions on the decoder transformer's output -> per frame either (log2 F0, voicing
    logit) or, with `classify`, (logits over log-F0 bins BIN_CENTS wide from F0_LO, voicing logit).
    Classifying (as CREPE does) avoids the regression's landing between octave candidates."""

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

    def f0(self, emb: torch.Tensor) -> torch.Tensor:
        """[B, dim, frames] -> F0 in Hz, 0 where the head says unvoiced."""
        out, voicing = self(emb)
        hz = decode_bins(out) if self.classify else 2**out
        return hz * (voicing > 0)

    @staticmethod
    def from_state(state: dict[str, torch.Tensor]) -> "F0Head":
        convs = sorted({int(k.split(".")[1]) for k in state if k.startswith("convs.")})
        hidden, dim, kernel = state["convs.0.weight"].shape
        head = F0Head(dim, hidden, kernel, len(convs), classify=state["out.weight"].shape[0] == N_BINS + 1)
        head.load_state_dict(state)
        return head


def bin_cents(device: torch.device | None = None) -> torch.Tensor:
    return torch.arange(N_BINS, dtype=torch.float32, device=device) * BIN_CENTS  # cents above F0_LO


def decode_bins(logits: torch.Tensor, width: int = 4) -> torch.Tensor:
    """[B, bins, frames] -> F0 in Hz: the probability-weighted mean of the bins within `width` of the best."""
    prob = torch.sigmoid(logits)
    best = prob.argmax(1, keepdim=True)
    idx = (best + torch.arange(-width, width + 1, device=logits.device)[None, :, None]).clamp(0, N_BINS - 1)
    p = torch.gather(prob, 1, idx)
    cents = (idx.float() * BIN_CENTS * p).sum(1) / p.sum(1).clamp(min=1e-6)
    return F0_LO * 2 ** (cents / 1200)


def f0_loss(out: torch.Tensor, voicing: torch.Tensor, target: torch.Tensor, blur_cents: float = 25.0) -> torch.Tensor:
    """F0Head (classify) outputs against a target F0 track [B, frames] (0 = unvoiced): binary
    cross-entropy against bins blurred over `blur_cents` in the voiced frames, plus the voicing's."""
    voiced = target > 0
    target_c = 1200 * torch.log2(target.clamp(min=F0_LO) / F0_LO)
    soft = torch.exp(-0.5 * ((bin_cents(out.device)[None, :, None] - target_c[:, None]) / blur_cents) ** 2)
    bce = F.binary_cross_entropy_with_logits(out, soft, reduction="none").mean(1)
    return (bce * voiced).sum() / voiced.sum().clamp(min=1) * 10 + F.binary_cross_entropy_with_logits(voicing, voiced.float())


class HarmonicHead(nn.Module):
    """Per-frame gains of the source channels from the decoder transformer's output, and optionally
    the F0 head that drives the source (without it the F0 is given, e.g. the recording's)."""

    def __init__(self, dim: int, channels: int, hop: int, f0: F0Head | None = None) -> None:
        super().__init__()
        self.hop = hop
        self.gain = nn.Conv1d(dim, channels, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        self.f0 = f0

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
    mimi: MimiModel, head: HarmonicHead, latents: torch.Tensor, start: int, source: torch.Tensor | None,
    context_frames: int = 8, parts: bool = False, config: HarmonicConfig | None = None, phase0: torch.Tensor | None = None,
    f0_outputs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """codec_ft.decode_window plus the harmonic part: [B, T, 32] latents from the start of a clip,
    source for the decoded window ([B, C, (T - start) * frame]) -> [B, 1, (T - start) * frame]
    (with `parts`: the decoder's own output and the harmonic part, separately).

    With `source` None the head's F0 head predicts the F0 from the decoder transformer's output and
    the source is made from it with `config` (no recording needed: what a TTS can run). Its raw
    outputs are appended to `f0_outputs` for a loss."""
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
    if source is None:
        assert head.f0 is not None and config is not None
        raw, voicing = head.f0(emb)
        if f0_outputs is not None:
            f0_outputs.append((raw, voicing))
        with torch.no_grad():
            hz = decode_bins(raw) if head.f0.classify else 2**raw
            source = harmonic_source(hz * (voicing > 0), head.hop, mimi.sample_rate, config, phase0)
    harm = head(emb, source[..., : out.shape[-1]])
    return (out, harm) if parts else out + harm


PREFIX = "mimi.harmonic."


def head_state(head: HarmonicHead) -> dict[str, torch.Tensor]:
    """The head's tensors under the names they get next to codec_ft.decoder_state's."""
    return {PREFIX + name: t.detach().float().cpu().contiguous() for name, t in head.state_dict().items()}


def load_harmonic_head(path: Path, dim: int, hop: int, device: torch.device) -> tuple[HarmonicHead, HarmonicConfig] | None:
    """The head stored in a decoder file and its config, or None for a file without one."""
    import safetensors
    import safetensors.torch

    state = {k.removeprefix(PREFIX): v for k, v in safetensors.torch.load_file(str(path)).items() if k.startswith(PREFIX)}
    if not state:
        return None
    with safetensors.safe_open(str(path), framework="pt") as f:
        config = HarmonicConfig.from_metadata(f.metadata())
    f0_state = {k.removeprefix("f0."): v for k, v in state.items() if k.startswith("f0.")}
    head = HarmonicHead(dim, state["gain.weight"].shape[0], hop, F0Head.from_state(f0_state) if f0_state else None)
    head.load_state_dict(state)
    assert head.gain.out_channels == 2 * len(config.bands), (head.gain.out_channels, config)
    return head.to(device).eval(), config
