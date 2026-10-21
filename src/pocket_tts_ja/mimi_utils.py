"""The frozen Pocket TTS Mimi codec (and main's / mimi_ft's fine-tuned decoders) for comparisons.

Ported from main / mimi_ft (``src/pocket_tts_ja/mimi_utils.py``, ``codec_ft.load_decoder``); only
stock-size decoder files are supported (keep4k, M2CH2_soup4, M2C_base2 are all stock size).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import safetensors.torch
import soxr
import torch

from pocket_tts.models.mimi import MimiModel, build_mimi
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.utils import download_if_necessary
from training.modules.utils import stamp_state_names

POCKET_TTS_ROOT = Path(__file__).resolve().parents[2] / "third-party" / "pocket-tts"
# Every released language ships the same Mimi weights (verified on main by hashing the
# mimi.* tensors of english, english_2026-04_24l, english_2026-09_24l, french_24l).
DEFAULT_CONFIG = POCKET_TTS_ROOT / "pocket_tts" / "config" / "english_2026-04_24l.yaml"
DECODE_CONTEXT_FRAMES = 8
# Modules a decoder file (mimi_ft's `decoder_state`) replaces: the 44 decode-side tensors.
DECODER_MODULES = ("quantizer", "upsample", "decoder_transformer", "decoder")


def load_pocket_config(path: Path = DEFAULT_CONFIG) -> Config:
    return load_config(path)


def load_mimi(config: Config, device: torch.device, decoder: Path | None = None) -> MimiModel:
    """The frozen Mimi from the config's bundled weights; `decoder` swaps in a fine-tuned decode side."""
    mimi = build_mimi(config.mimi)
    assert config.weights_path is not None
    state = safetensors.torch.load_file(download_if_necessary(str(config.weights_path)))
    mimi.load_state_dict(
        {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")},
        strict=True,
    )
    if decoder is not None:
        load_decoder(mimi, decoder)
    mimi = mimi.eval().to(device)
    stamp_state_names(mimi)
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi


def load_decoder(mimi: MimiModel, path: Path) -> None:
    """Replace the decode side of `mimi` with a stock-size fine-tuned one."""
    state = safetensors.torch.load_file(str(path))
    own = mimi.state_dict()
    expected = {f"mimi.{module}.{name}" for module in DECODER_MODULES for name in getattr(mimi, module).state_dict()}
    assert set(state) == expected, f"{path}: not a stock-size decoder file ({len(set(state) ^ expected)} names differ)"
    for name, tensor in state.items():
        target = own[name.removeprefix("mimi.")]
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def to_mono_float(wav: npt.NDArray[np.float32], sr: int, target_sr: int) -> npt.NDArray[np.float32]:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr, quality="HQ").astype(np.float32)
    return wav


@torch.no_grad()
def encode(mimi: MimiModel, wav: npt.NDArray[np.float32], device: torch.device) -> torch.Tensor:
    """[T] waveform at mimi.sample_rate -> [frames, 32] continuous latents."""
    x = torch.from_numpy(wav)[None, None].to(device)
    return mimi.encode_to_latent(x)[0]


@torch.no_grad()
def decode(mimi: MimiModel, latents: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
    """[frames, 32] latents -> [T] waveform (trimmed to num_samples when given)."""
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, 1, (latents.shape[0] + DECODE_CONTEXT_FRAMES) * ratio)
    out = mimi.decode_from_latent(latents[None], state)[0, 0]
    return out[:num_samples] if num_samples is not None else out
