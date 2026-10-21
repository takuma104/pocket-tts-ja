"""Irodori-TTS's codec (Semantic-DACVAE-Japanese-32dim) outside the Irodori environment.

Mirrors ``irodori_tts/codec.py`` (third-party/Irodori-TTS @ 89f9d8f) so that latents encoded or
decoded here match the teacher's:

- encode takes the posterior mean (``quantizer.in_proj`` -> mean), not a sample;
- the decoder's watermark branch is bypassed (``alpha = 0`` and a pass-through), as Irodori's
  checkpoints were trained without it;
- the optional loudness normalisation before encoding is audiotools' ``normalize(-16)`` +
  ``ensure_max_of_audio`` (Irodori applies it to speaker references).

Latents are ``(B, T, 32)`` at 25 Hz (hop 1920 at 48 kHz), unnormalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch

REPO_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
SAMPLE_RATE = 48000
HOP = 1920
FRAME_RATE = SAMPLE_RATE / HOP  # 25.0
LATENT_DIM = 32
REF_LOUDNESS_DB = -16.0


def load_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at ``sample_rate``.

    Resampled with ``torchaudio.functional.resample`` as Irodori does: the encoder is sensitive
    to the resampler (soxr HQ instead moves 24 kHz JVS latents by 0.056 mean abs, 0.8 max).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(torch.from_numpy(wav)[None], sr, sample_rate)[0]
        wav = wav.numpy()
    return wav


def normalize_loudness(wav: np.ndarray, sample_rate: int, target_db: float) -> np.ndarray:
    """audiotools' loudness normalisation, as in ``DACVAECodec._normalize_loudness``."""
    from audiotools import AudioSignal

    signal = AudioSignal(torch.from_numpy(wav).view(1, 1, -1), sample_rate)
    signal.normalize(target_db)
    signal.ensure_max_of_audio()
    return signal.audio_data.view(-1).float().numpy()


@dataclass
class DACVAE:
    model: torch.nn.Module
    device: torch.device

    @classmethod
    def load(cls, device: str = "cuda", repo_id: str = REPO_ID) -> DACVAE:
        from dacvae import DACVAE as _DACVAE
        from huggingface_hub import hf_hub_download

        model = _DACVAE.load(hf_hub_download(repo_id=repo_id, filename="weights.pth"))
        model = model.eval().to(device)
        decoder = model.decoder
        decoder.alpha = 0.0

        def _watermark_passthrough(
            x: torch.Tensor, message: torch.Tensor | None = None, _decoder=decoder
        ) -> torch.Tensor:
            del message
            return _decoder.wm_model.encoder_block.forward_no_conv(x)

        decoder.watermark = _watermark_passthrough
        assert int(model.sample_rate) == SAMPLE_RATE
        return cls(model=model, device=torch.device(device))

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor | np.ndarray) -> torch.Tensor:
        """``(B, samples)`` or ``(samples,)`` 48 kHz mono -> ``(B, T, 32)`` float32 mean latents."""
        x = torch.as_tensor(wav, dtype=torch.float32)
        if x.ndim == 1:
            x = x[None]
        x = x[:, None].to(self.device)
        z = self.model.encoder(self.model._pad(x))
        mean, _scale = self.model.quantizer.in_proj(z).chunk(2, dim=1)
        return mean.transpose(1, 2).float()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, T, 32)`` -> ``(B, samples)`` float32 at 48 kHz."""
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=torch.float32)
        return self.model.decode(z)[:, 0].float()

    def encode_file(self, path: str | Path, loudness_db: float | None = None) -> torch.Tensor:
        """One file -> ``(T, 32)`` on CPU; ``loudness_db`` as Irodori's speaker references."""
        wav = load_wav(path)
        if loudness_db is not None:
            wav = normalize_loudness(wav, SAMPLE_RATE, loudness_db)
        return self.encode(wav)[0].cpu()
