"""Fine-tuning the decode side of Mimi (plan M2): parameters, windowed decoding, losses.

Only the path from the 32-dim latent to the waveform is trained (the latent projection, the
upsampling, the decoder transformer and the SEANet decoder). The encoder and the latent space
stay as they are, so a trained FlowLM keeps working and the result drops into a released
safetensors by replacing those `mimi.*` tensors.

Losses follow the usual codec recipe: multi-scale log-mel L1, least-squares adversarial losses
and feature matching against multi-period (HiFi-GAN) and multi-scale STFT (EnCodec)
discriminators.
"""

from collections.abc import Iterator
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import increment_steps, init_states

DECODER_MODULES = ("quantizer", "upsample", "decoder_transformer", "decoder")
DECODE_CONTEXT_FRAMES = 8
LEAK = 0.1

DiscOutput = tuple[torch.Tensor, list[torch.Tensor]]  # logits, feature maps


def decoder_named_parameters(mimi: MimiModel) -> Iterator[tuple[str, nn.Parameter]]:
    for module in DECODER_MODULES:
        for name, param in getattr(mimi, module).named_parameters():
            yield f"{module}.{name}", param


def decoder_state(mimi: MimiModel) -> dict[str, torch.Tensor]:
    """The decode-side tensors under the names they have in a released file (`mimi.*`)."""
    return {
        f"mimi.{module}.{name}": tensor.detach().float().cpu().contiguous()
        for module in DECODER_MODULES
        for name, tensor in getattr(mimi, module).state_dict().items()
    }


def load_decoder(mimi: MimiModel, path: Path) -> None:
    """Replace the decode side of `mimi` with a fine-tuned one (file from `decoder_state`)."""
    state = safetensors.torch.load_file(str(path))
    own = mimi.state_dict()
    expected = {f"mimi.{module}.{name}" for module in DECODER_MODULES for name in getattr(mimi, module).state_dict()}
    assert set(state) == expected, f"{path}: not a decoder file ({len(set(state) ^ expected)} names differ)"
    for name, tensor in state.items():
        target = own[name.removeprefix("mimi.")]
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def decode_window(mimi: MimiModel, latents: torch.Tensor, start: int) -> torch.Tensor:
    """[B, T, 32] latents from the start of a clip -> [B, 1, (T - start) * frame] waveform.

    The frames before `start` only build the streaming state (no gradient), so the trained
    window is decoded exactly as in inference, with everything before it as context.
    """
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, latents.shape[0], (latents.shape[1] + DECODE_CONTEXT_FRAMES) * ratio)
    if start > 0:
        with torch.no_grad():
            mimi.decode_from_latent(latents[:, :start], state)
        increment_steps(mimi, state, increment=start * ratio)  # the stream position is advanced by hand
    return mimi.decode_from_latent(latents[:, start:], state)


class MultiScaleMelLoss(nn.Module):
    """Mean over STFT sizes of the L1 distance between log-mel spectrograms (short windows
    weigh the bursts and transitions that tell consonants apart)."""

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


class SharedSpectrumLoss(nn.Module):
    """Deviation of the long-term spectrum that the clips of a batch share, at 12 Hz resolution.

    A decoder fine-tuned on the mel loss puts tones at multiples of its strides' frequencies
    (200 Hz, 1.2 kHz) into fricatives: a metallic noise for a listener, invisible to a mel band,
    which is hundreds of Hz wide up there. Such a line sits in the same bin for every clip, so
    it survives the mean over the batch of the log ratio of the time-averaged power spectra,
    where the differences of the single clips cancel. Squared, so that narrow lines dominate.

    Only the band from `f_lo` to `f_hi` counts, and a bin counts down to 40 dB under the mean
    power of the band: where the recordings have next to no energy the log ratio is large and
    arbitrary, and its gradient swamps the rest (without both limits the loss removed the lines
    but cost half of the mel loss's gain in Kana-CER). The upper limit must not leave room,
    though: with 11 kHz the adversarial run kept one line, at 11.2 kHz (a multiple of 200 Hz,
    6 dB over its surroundings), where the training audio already rolls off. Up to Nyquist
    the floor does the limiting.
    """

    def __init__(self, sample_rate: int, n_fft: int = 2048, f_lo: float = 300.0, f_hi: float = 11000.0) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.lo = round(f_lo * n_fft / sample_rate)
        self.hi = round(f_hi * n_fft / sample_rate)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def spectrum(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav[:, 0], self.n_fft, self.n_fft // 4, window=self.window, return_complex=True)
        power = spec.abs().pow(2).mean(dim=-1)[:, self.lo : self.hi]
        return (power + 1e-4 * power.mean(dim=-1, keepdim=True) + 1e-12).log10()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return (self.spectrum(fake) - self.spectrum(real)).mean(dim=0).pow(2).mean()


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


class Discriminators(nn.Module):
    def __init__(
        self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), n_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    ) -> None:
        super().__init__()
        self.nets = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods] + [STFTDiscriminator(n) for n in n_ffts]
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
