"""Stock-wheel Mimi decoders for DACVAE latents (plan P1-side / W1).

The stock wheel can only decode with the Mimi module tree (ELU SEANet, causal convolutions, a
2-layer transformer), but its YAML sets the frame rate and the latent dimension. Here that tree is
built for DACVAE latents at 24 kHz: 32 dims at 25 Hz (DACVAE's own rate), or two frames stacked
into 64 dims at 12.5 Hz (the stock rate). The SEANet runs at 200 Hz as in the stock Mimi, so the
decoder transformer and SEANet decoder start from the stock weights; the latent projection
(``quantizer``) and the transposed-conv upsampling (stride 8 at 25 Hz) start fresh.
"""

from pathlib import Path

import safetensors.torch
import torch

from pocket_tts.models.mimi import MimiModel, build_mimi
from pocket_tts.utils.config import MimiConfig
from pocket_tts.utils.utils import download_if_necessary
from pocket_tts_ja.mimi_utils import load_pocket_config
from training.modules.utils import stamp_state_names

DECODER_MODULES = ("quantizer", "upsample", "decoder_transformer", "decoder")
FRESH_MODULES = ("quantizer", "upsample")


def dv_mimi_config(frame_rate: float = 25.0, latent_dim: int = 32) -> MimiConfig:
    config = load_pocket_config().mimi.model_dump()
    config["frame_rate"] = frame_rate
    config["quantizer"]["dimension"] = latent_dim
    config["inner_dim"] = latent_dim
    return MimiConfig(**config)


def build_dv_mimi(frame_rate: float = 25.0, latent_dim: int = 32, init_stock: bool = True) -> MimiModel:
    """The decode side for DACVAE latents; decoder transformer and SEANet from the stock Mimi."""
    mimi = build_mimi(dv_mimi_config(frame_rate, latent_dim))
    if init_stock:
        stock = load_pocket_config()
        assert stock.weights_path is not None
        state = safetensors.torch.load_file(download_if_necessary(str(stock.weights_path)))
        own = mimi.state_dict()
        for name, tensor in state.items():
            key = name.removeprefix("mimi.")
            if not name.startswith("mimi.") or key.split(".")[0] in FRESH_MODULES:
                continue
            if key in own and own[key].shape == tensor.shape:
                own[key].copy_(tensor)
    stamp_state_names(mimi)
    return mimi


def decoder_named_parameters(mimi: MimiModel) -> list[tuple[str, torch.nn.Parameter]]:
    return [(f"{m}.{n}", p) for m in DECODER_MODULES if hasattr(mimi, m) for n, p in getattr(mimi, m).named_parameters()]


def decoder_state(mimi: MimiModel) -> dict[str, torch.Tensor]:
    """The decode-side tensors under their names in a released file (`mimi.*`)."""
    return {
        f"mimi.{m}.{n}": t.detach().float().cpu().contiguous()
        for m in DECODER_MODULES
        if hasattr(mimi, m)
        for n, t in getattr(mimi, m).state_dict().items()
    }


def load_dv_decoder(path: Path, device: torch.device, frame_rate: float = 25.0, latent_dim: int = 32) -> MimiModel:
    mimi = build_dv_mimi(frame_rate, latent_dim, init_stock=False)
    own = mimi.state_dict()
    state = safetensors.torch.load_file(str(path))
    for name, tensor in state.items():
        own[name.removeprefix("mimi.")].copy_(tensor)
    mimi = mimi.eval().to(device)
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi
