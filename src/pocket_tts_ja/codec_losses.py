"""Losses and discriminators for training Mimi-architecture decoders (ported from mimi_ft codec_ft.py).

The usual codec recipe: multi-scale log-mel L1, least-squares adversarial losses and feature
matching against multi-period (HiFi-GAN), multi-scale STFT (EnCodec) and optionally DAC's
multi-band STFT discriminators. ``decode_window`` decodes a window of a stock-wheel Mimi decoder
with everything before it as streaming context, exactly as inference does.
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import increment_steps, init_states

DECODE_CONTEXT_FRAMES = 8
LEAK = 0.1

DiscOutput = tuple[torch.Tensor, list[torch.Tensor]]  # logits, feature maps


def decode_window(mimi: MimiModel, latents: torch.Tensor, start: int) -> torch.Tensor:
    """[B, T, D] latents from the start of a clip -> [B, 1, (T - start) * frame] waveform.

    The frames before `start` only build the streaming state (no gradient).
    """
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, latents.shape[0], (latents.shape[1] + DECODE_CONTEXT_FRAMES) * ratio)
    if start > 0:
        with torch.no_grad():
            mimi.decode_from_latent(latents[:, :start], state)
        increment_steps(mimi, state, increment=start * ratio)  # the stream position is advanced by hand
    return mimi.decode_from_latent(latents[:, start:], state)


class MultiScaleMelLoss(nn.Module):
    """Mean over STFT sizes of the L1 distance between log-mel spectrograms."""

    def __init__(
        self,
        sample_rate: int,
        n_ffts: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        n_mels: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320),
    ) -> None:
        super().__init__()
        self.specs = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=mels, power=1.0, norm=None, mel_scale="htk"
            )
            for n_fft, mels in zip(n_ffts, n_mels, strict=True)
        )

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        total = fake.new_zeros(())
        for spec in self.specs:
            total = total + F.l1_loss(spec(fake).clamp_min(1e-5).log10(), spec(real).clamp_min(1e-5).log10())
        return total / len(self.specs)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: tuple[int, ...] = (32, 128, 256, 512, 512)) -> None:
        super().__init__()
        self.period = period
        dims = (1, *channels)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(dims[i], dims[i + 1], (5, 1), (3 if i < len(channels) - 1 else 1, 1), padding=(2, 0)))
            for i in range(len(channels))
        )
        self.post = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> DiscOutput:
        pad = (self.period - x.shape[-1] % self.period) % self.period
        z = F.pad(x[:, None], (0, pad), mode="reflect")
        z = z.view(z.shape[0], 1, -1, self.period)
        fmaps = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), LEAK)
            fmaps.append(z)
        return self.post(z), fmaps


class STFTDiscriminator(nn.Module):
    """One scale of EnCodec's MS-STFT discriminator: 2D convolutions over the complex STFT."""

    def __init__(self, n_fft: int, filters: int = 32) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(2, filters, (3, 9), padding=(1, 4)))]
            + [
                weight_norm(nn.Conv2d(filters, filters, (3, 9), stride=(1, 2), dilation=(d, 1), padding=(d, 4)))
                for d in (1, 2, 4)
            ]
            + [weight_norm(nn.Conv2d(filters, filters, (3, 3), padding=(1, 1)))]
        )
        self.post = weight_norm(nn.Conv2d(filters, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> DiscOutput:
        spec = torch.stft(x, self.n_fft, self.n_fft // 4, window=self.window, normalized=True, return_complex=True)
        z = torch.view_as_real(spec).permute(0, 3, 2, 1)  # [B, 2, frames, bins]
        fmaps = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), LEAK)
            fmaps.append(z)
        return self.post(z), fmaps


class BandSTFTDiscriminator(nn.Module):
    """One scale of DAC's multi-band STFT discriminator: each frequency band through its own convolutions."""

    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

    def __init__(self, n_fft: int, filters: int = 32) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        bins = n_fft // 2 + 1
        self.bands = [(int(lo * bins), int(hi * bins)) for lo, hi in self.BANDS]

        def stack() -> nn.ModuleList:
            return nn.ModuleList(
                [weight_norm(nn.Conv2d(2, filters, (3, 9), padding=(1, 4)))]
                + [weight_norm(nn.Conv2d(filters, filters, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
                + [weight_norm(nn.Conv2d(filters, filters, (3, 3), padding=(1, 1)))]
            )

        self.band_convs = nn.ModuleList(stack() for _ in self.bands)
        self.post = weight_norm(nn.Conv2d(filters, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> DiscOutput:
        spec = torch.stft(x, self.n_fft, self.n_fft // 4, window=self.window, normalized=True, return_complex=True)
        z = torch.view_as_real(spec).permute(0, 3, 2, 1)
        fmaps, outs = [], []
        for (lo, hi), convs in zip(self.bands, self.band_convs, strict=True):
            b = z[..., lo:hi]
            for conv in convs:
                b = F.leaky_relu(conv(b), LEAK)
                fmaps.append(b)
            outs.append(b)
        return self.post(torch.cat(outs, dim=-1)), fmaps


class Discriminators(nn.Module):
    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        n_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128),
        band_ffts: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.nets = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
            + [STFTDiscriminator(n) for n in n_ffts]
            + [BandSTFTDiscriminator(n) for n in band_ffts]
        )

    def forward(self, x: torch.Tensor) -> list[DiscOutput]:
        """x: [B, T] waveform."""
        return [net(x) for net in self.nets]


def discriminator_loss(real: list[DiscOutput], fake: list[DiscOutput]) -> torch.Tensor:
    losses = [((1 - r) ** 2).mean() + (f**2).mean() for (r, _), (f, _) in zip(real, fake, strict=True)]
    return torch.stack(losses).mean()


def generator_loss(fake: list[DiscOutput]) -> torch.Tensor:
    return torch.stack([((1 - f) ** 2).mean() for f, _ in fake]).mean()


def feature_loss(real: list[DiscOutput], fake: list[DiscOutput]) -> torch.Tensor:
    losses = [
        F.l1_loss(f, r.detach())
        for (_, real_maps), (_, fake_maps) in zip(real, fake, strict=True)
        for r, f in zip(real_maps, fake_maps, strict=True)
    ]
    return torch.stack(losses).mean()
