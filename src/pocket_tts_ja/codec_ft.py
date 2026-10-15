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
from pocket_tts.modules.seanet import SEANetDecoder, SEANetResnetBlock
from pocket_tts.modules.stateful_module import increment_steps, init_states
from training.modules.utils import stamp_state_names

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


def residual_layers(decoder: SEANetDecoder) -> int:
    """Residual blocks per upsampling stage of a SEANet decoder."""
    return sum(isinstance(layer, SEANetResnetBlock) for layer in decoder.model) // len(decoder.ratios)


def expand_residual_layers(mimi: MimiModel, n_residual_layers: int) -> None:
    """Give every upsampling stage of the SEANet decoder `n_residual_layers` residual blocks.

    The blocks SEANet would build for that setting (dilation 2**j for the j-th) are inserted after
    the existing ones, with the last convolution of each new block zeroed: the new blocks are
    identities, so the decoder's output is unchanged until they train. The trained tensors keep
    their values; only the indices of `decoder.model` after an insertion move (a file from
    `decoder_state` of an expanded decoder carries the new names, and `load_decoder` expands to
    match it).
    """
    decoder: SEANetDecoder = mimi.decoder
    have = residual_layers(decoder)
    assert n_residual_layers >= have, (n_residual_layers, have)
    if n_residual_layers == have:
        return
    first = next(layer for layer in decoder.model if isinstance(layer, SEANetResnetBlock))
    kernel = first.block[1].conv.kernel_size[0]
    pad_mode = first.block[1].pad_mode
    layers: list[nn.Module] = []
    for i, layer in enumerate(decoder.model):
        layers.append(layer)
        is_last_of_stage = isinstance(layer, SEANetResnetBlock) and not isinstance(
            decoder.model[i + 1] if i + 1 < len(decoder.model) else None, SEANetResnetBlock
        )
        if not is_last_of_stage:
            continue
        dim = layer.block[1].conv.in_channels
        compress = dim // layer.block[1].conv.out_channels
        for j in range(have, n_residual_layers):
            block = SEANetResnetBlock(dim, kernel_sizes=[kernel, 1], dilations=[2**j, 1], pad_mode=pad_mode, compress=compress)
            nn.init.zeros_(block.block[-1].conv.weight)
            nn.init.zeros_(block.block[-1].conv.bias)
            device, dtype = layer.block[1].conv.weight.device, layer.block[1].conv.weight.dtype
            layers.append(block.to(device=device, dtype=dtype))
    decoder.model = nn.ModuleList(layers)
    decoder.n_residual_layers = n_residual_layers
    stamp_state_names(mimi)


def load_decoder(mimi: MimiModel, path: Path) -> None:
    """Replace the decode side of `mimi` with a fine-tuned one (file from `decoder_state`).

    A file from a decoder with more residual blocks per stage expands `mimi` to match first. The
    tensors of a harmonic head (pocket_tts_ja.harmonic) are left to harmonic.load_harmonic_head."""
    state = {k: v for k, v in safetensors.torch.load_file(str(path)).items() if not k.startswith("mimi.harmonic.")}
    last = max(int(name.split(".")[3]) for name in state if name.startswith("mimi.decoder.model."))
    expand_residual_layers(mimi, (last + 1 - 3) // len(mimi.decoder.ratios) - 2)  # entries: 1 + stages * (2 + n) + 2
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
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.specs = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=mels, power=1.0, norm=None, mel_scale="htk",
                f_min=f_min, f_max=f_max,
            )
            for n_fft, mels in zip(n_ffts, n_mels, strict=True)
        )

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        total = fake.new_zeros(())
        for spec in self.specs:
            total = total + F.l1_loss(spec(fake).clamp_min(1e-5).log10(), spec(real).clamp_min(1e-5).log10())
        return total / len(self.specs)


class FineSpectrumLoss(nn.Module):
    """L1 distance of log STFT magnitudes above `f_lo`, at fine frequency resolution only.

    Above about 4 kHz the decoder cannot place the harmonics of a voice (the recording has them
    up to 8 kHz). A loss that integrates over several harmonics - every mel band up there, and
    every short window - then only asks for the energy of the band, and the decoder delivers it
    as noise: a hiss that follows the voicing, worst for high voices. Bins a few tens of Hz wide
    see the valleys between the harmonics; noise that fills them costs as much as a missing
    peak, so the cheapest answer to harmonics that cannot be placed is a quiet band, while a
    fricative, which is noise in the recording too, keeps its level.

    A bin counts down to `floor` times the mean magnitude of the recording's band (per clip):
    below that the log ratio is large and arbitrary.
    """

    def __init__(self, f_lo: float, sample_rate: int, n_ffts: tuple[int, ...] = (1024, 2048), floor: float = 3e-3) -> None:
        super().__init__()
        self.n_ffts = n_ffts
        self.lows = [round(f_lo * n_fft / sample_rate) for n_fft in n_ffts]
        self.floor = floor
        for n_fft in n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        total = fake.new_zeros(())
        for n_fft, low in zip(self.n_ffts, self.lows, strict=True):
            window = getattr(self, f"window_{n_fft}")
            mag_fake = torch.stft(fake[:, 0], n_fft, n_fft // 4, window=window, return_complex=True).abs()[:, low:]
            mag_real = torch.stft(real[:, 0], n_fft, n_fft // 4, window=window, return_complex=True).abs()[:, low:]
            floor = self.floor * mag_real.mean(dim=(1, 2), keepdim=True) + 1e-7
            total = total + F.l1_loss((mag_fake + floor).log10(), (mag_real + floor).log10())
        return total / len(self.n_ffts)


class HighBandWaveLoss(nn.Module):
    """L1 distance between the high-passed waveforms, relative to the level of the target's.

    Keeps the band above `f_lo` of the stock decoder sample by sample: a mel loss against its
    output only keeps the level of the band, and noise at the right level is not what the stock
    decoder has there. 0 for the stock decoder itself, about 1.4 for unrelated noise of the
    same level.
    """

    def __init__(self, sample_rate: int, f_lo: float, taps: int = 255) -> None:
        super().__init__()
        from scipy.signal import firwin

        kernel = torch.tensor(firwin(taps, f_lo, fs=sample_rate, pass_zero=False), dtype=torch.float32)
        self.register_buffer("kernel", kernel[None, None], persistent=False)

    def forward(self, fake: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.kernel.shape[-1] // 2
        high_fake, high_target = F.conv1d(fake, self.kernel, padding=pad), F.conv1d(target, self.kernel, padding=pad)
        return (high_fake - high_target).abs().mean() / (high_target.abs().mean() + 1e-5)


class VoicingBlendMelLoss(nn.Module):
    """Mel loss of the band above `f_min` against the recording in unvoiced frames and against the
    stock decoder's output in voiced ones.

    The stock decoder is 6 to 7 dB quieter than the recording above 4 kHz everywhere. Raised to
    the recording's level, the band is right in a fricative (noise in the recording too) and
    wrong in a vowel, where the decoder has only noise for the recording's harmonics (see
    FineSpectrumLoss). So the target of a frame is the recording where its high band carries
    the frame (energy above `f_min` against energy below 1 kHz, a soft switch around -3 dB) and
    the stock decoder's output elsewhere, silence included.
    """

    def __init__(
        self,
        sample_rate: int,
        f_min: float,
        n_ffts: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        n_mels: tuple[int, ...] = (2, 3, 7, 14, 27, 55, 110),
    ) -> None:
        super().__init__()
        self.mel = MultiScaleMelLoss(sample_rate, n_ffts, n_mels, f_min=f_min)
        self.n_fft, self.hop = 1024, 256
        self.low = round(1000 * self.n_fft / sample_rate)
        self.high = round(f_min * self.n_fft / sample_rate)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def unvoiced(self, real: torch.Tensor) -> torch.Tensor:
        """[B, frames] in 0..1: how far the recording's frame is noise in the high band."""
        power = torch.stft(real[:, 0], self.n_fft, self.hop, window=self.window, return_complex=True).abs().pow(2)
        low, high = power[:, : self.low].sum(dim=1), power[:, self.high :].sum(dim=1)
        ratio = 10 * torch.log10((high + 1e-10) / (low + 1e-10))
        loud = (low + high) > 1e-3 * (low + high).amax(dim=1, keepdim=True)
        return torch.sigmoid((ratio + 3.0) / 2.0) * loud

    def forward(self, fake: torch.Tensor, real: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        weight = self.unvoiced(real)
        total = fake.new_zeros(())
        for spec in self.mel.specs:
            mel_real, mel_stock = spec(real), spec(stock)
            w = F.interpolate(weight[:, None], size=mel_real.shape[-1], mode="linear", align_corners=True)
            target = w[:, :, None] * mel_real + (1 - w[:, :, None]) * mel_stock
            total = total + F.l1_loss(spec(fake).clamp_min(1e-5).log10(), target.clamp_min(1e-5).log10())
        return total / len(self.mel.specs)


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


class GridModulationLoss(nn.Module):
    """Amplitude modulation of the high bands locked to the decoder's 5 ms frame grid.

    The decoder makes 200 Hz frames and upsamples them by 6, 5 and 4 into 24 kHz. Its high bands
    come out louder at some positions of every 120-sample frame than at others, the same positions
    in every clip: folded on the grid, the relative envelope of 4-11 kHz deviates by 4-15% from
    flat in stock Mimi and the fine-tuned decoders (the recordings: 1%, the noise floor), with
    clip profiles that correlate 0.7-0.97 with their mean. Heard as a buzz at 200 Hz and its
    multiples (1.2, 2.4 kHz) under the voice - mechanical, and louder once the high band is.

    Per band: the Hilbert envelope of the band-passed output over its own 10 ms mean, minus 1,
    weighted by the recording's band level (where the band is silent, the relative envelope is
    arbitrary), folded on `period` samples, averaged over the batch (what differs between clips
    cancels, the grid's pattern stays), the mean over positions removed; the mean square of what
    is left. The training windows start on frame boundaries, so the grid is the same in a batch.

    With `latent_frames` > 0 the same is done one level up: the envelope's means over the 5 ms
    frames (which removes the pattern above), over their 160 ms mean, folded on the 16 frames of
    an 80 ms latent frame. The upsampling from the 12.5 Hz latents leaves a pattern there too
    (4-8 kHz: 4-7% against the recordings' 0.6%, also 2.5-3% at 1-4 kHz): a 12.5 Hz flutter.
    """

    def __init__(
        self,
        sample_rate: int,
        period: int = 120,
        bands: tuple[tuple[float, float], ...] = ((2000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0), (8000.0, 11000.0)),
        latent_frames: int = 0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.period = period
        self.bands = bands
        self.latent_frames = latent_frames

    @staticmethod
    def folded(rel: torch.Tensor, weight: torch.Tensor, period: int) -> torch.Tensor:
        """Mean square of the batch's weighted mean profile on `period`, its mean removed."""
        batch = rel.shape[0]
        fold = (rel * weight).view(batch, -1, period).sum(1) / weight.view(batch, -1, period).sum(1).clamp(min=1e-6)
        profile = fold.mean(0)
        return (profile - profile.mean()).pow(2).mean()

    @staticmethod
    def relative(env: torch.Tensor, ref: torch.Tensor, span: int) -> tuple[torch.Tensor, torch.Tensor]:
        """env over its moving mean of `span` - 1, and the recording's moving mean as the weight
        (zero within `span` of the ends, where the FFT band-pass wraps around)."""
        n = env.shape[-1]
        mean = F.avg_pool1d(env[:, None], span + 1, 1, span // 2, count_include_pad=False)[:, 0, :n]
        weight = F.avg_pool1d(ref[:, None], span + 1, 1, span // 2, count_include_pad=False)[:, 0, :n].clone()
        weight[:, :span] = 0
        weight[:, -span:] = 0
        return env / (mean + 1e-6) - 1, weight / (weight.mean(dim=-1, keepdim=True) + 1e-9)

    def envelopes(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """[B, T] -> per band [B, T] Hilbert envelope (FFT band-pass, one-sided spectrum doubled)."""
        n = wav.shape[-1]
        spec = torch.fft.fft(wav.float())
        freq = torch.fft.fftfreq(n, 1 / self.sample_rate, device=wav.device)
        return [torch.fft.ifft(spec * ((freq >= lo) & (freq <= hi)) * 2).abs() for lo, hi in self.bands]

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """fake, real: [B, 1, T] waveforms whose sample 0 is on the decoder's frame grid."""
        unit = self.period * max(self.latent_frames, 1)
        n = fake.shape[-1] // unit * unit
        fake, real = fake[:, 0, :n], real[:, 0, :n]
        total = fake.new_zeros(())
        with torch.no_grad():
            real_env = self.envelopes(real)
        for env, ref in zip(self.envelopes(fake), real_env, strict=True):
            rel, weight = self.relative(env, ref, 2 * self.period)
            total = total + self.folded(rel, weight, self.period)
            if self.latent_frames > 0:
                frames = env.view(env.shape[0], -1, self.period).mean(-1)
                ref_frames = ref.view(ref.shape[0], -1, self.period).mean(-1)
                rel, weight = self.relative(frames, ref_frames, 2 * self.latent_frames)
                total = total + self.folded(rel, weight, self.latent_frames)
        return total


def zero_phase_highpass(wav: torch.Tensor, sample_rate: int, lo: float, hi: float) -> torch.Tensor:
    """[..., T] waveform without what lies below `lo` Hz, a raised-cosine slope up to `hi` Hz,
    by an FFT over the whole clip (no phase shift; the clip wraps around at its ends)."""
    n = wav.shape[-1]
    freq = torch.fft.rfftfreq(n, 1 / sample_rate, device=wav.device)
    gain = (0.5 - 0.5 * torch.cos(torch.pi * ((freq - lo) / (hi - lo)).clamp(0, 1))).to(wav.dtype)
    return torch.fft.irfft(torch.fft.rfft(wav.float()) * gain, n).to(wav.dtype)


class SubsonicLoss(nn.Module):
    """Content below `cutoff` that the recording does not have.

    The decoders put a slow wander under the waveform, at the latent rate and its multiples
    (12.5 / 25 Hz; mostly below 40 Hz), 41-48 dB under the signal against the recordings' 55 dB.
    The mel loss hardly sees it (a few bins at the bottom of the largest STFT). Inaudible as such,
    but UTMOSv2 marks the decoders down by about 0.3 for it, and it costs headroom. Per clip: the
    power of the output below `cutoff` over the recording's, in dB, with a floor at -60 dB of the
    recording's power; only an excess counts (squared, in units of 10 dB).
    """

    def __init__(self, sample_rate: int, cutoff: float = 40.0, floor_db: float = -60.0) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.floor = 10 ** (floor_db / 10)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """fake, real: [B, 1, T]."""
        n = fake.shape[-1]
        freq = torch.fft.rfftfreq(n, 1 / self.sample_rate, device=fake.device)
        low = freq < self.cutoff
        f_spec = torch.fft.rfft(fake[:, 0].float()).abs().pow(2)
        with torch.no_grad():
            r_spec = torch.fft.rfft(real[:, 0].float()).abs().pow(2)
            floor = self.floor * r_spec.sum(-1)
            r_low = r_spec[:, low].sum(-1) + floor
        excess = 10 * torch.log10((f_spec[:, low].sum(-1) + floor) / r_low)
        return (F.relu(excess) / 10).pow(2).mean()


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
