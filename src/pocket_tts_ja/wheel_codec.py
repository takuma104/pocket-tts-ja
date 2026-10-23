"""Stock-wheel Mimi decoders for DACVAE latents (plan P1-side / W1).

The stock wheel can only decode with the Mimi module tree (ELU SEANet, causal convolutions, a
2-layer transformer), but its YAML sets the frame rate and the latent dimension. Here that tree is
built for DACVAE latents at 24 kHz: 32 dims at 25 Hz (DACVAE's own rate), or two frames stacked
into 64 dims at 12.5 Hz (the stock rate). The SEANet runs at 200 Hz as in the stock Mimi, so the
decoder transformer and SEANet decoder start from the stock weights; the latent projection
(``quantizer``) and the transposed-conv upsampling (stride 8 at 25 Hz) start fresh.

The SEANet decoder can have more residual blocks per upsampling stage than the stock one
(``decoder_res_layers``; the stock wheel reads them from the config's ``mimi.decoder_seanet``). A
trained decoder is deepened with ``deepen_decoder``: every new block ends in a zeroed 1x1 conv, so
the deeper decoder starts out computing exactly what the shallower one did.
"""

import re
from collections.abc import Iterable
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
# "<prefix>decoder.model.<index>.<rest>": a tensor of the SEANet decoder's layer list
_DECODER_LAYER = re.compile(r"^(.*?(?<![a-z_])decoder\.model\.)(\d+)(\..*)$")


def dv_mimi_config(frame_rate: float = 25.0, latent_dim: int = 32, decoder_res_layers: int = 1) -> MimiConfig:
    config = load_pocket_config().mimi.model_dump()
    config["frame_rate"] = frame_rate
    config["quantizer"]["dimension"] = latent_dim
    config["inner_dim"] = latent_dim
    if decoder_res_layers != config["seanet"]["n_residual_layers"]:
        config["decoder_seanet"] = {**config["seanet"], "n_residual_layers": decoder_res_layers}
    return MimiConfig(**config)


def decoder_res_layers(names: Iterable[str]) -> int:
    """Residual blocks per upsampling stage of the SEANet decoder whose tensors are named `names`."""
    blocks = {int(m.group(2)) for n in names if (m := _DECODER_LAYER.match(n)) and m.group(3).startswith(".block.")}
    stages = len(load_pocket_config().mimi.seanet.ratios)
    assert blocks and len(blocks) % stages == 0, sorted(blocks)
    return len(blocks) // stages


def added_layers(k: int) -> list[int]:
    """Indices, in the SEANet decoder's layer list, of the residual blocks beyond the first of each stage."""
    stages = len(load_pocket_config().mimi.seanet.ratios)
    return [1 + (2 + k) * s + 2 + b for s in range(stages) for b in range(1, k)]


def is_added(name: str, k: int) -> bool:
    m = _DECODER_LAYER.match(name)
    return m is not None and int(m.group(2)) in added_layers(k)


def deepen_decoder(state: dict[str, torch.Tensor], fresh: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """`state` (a decoder with one residual block per stage, any naming prefix) laid out as `fresh`,
    the same tensors of a deeper decoder: the trained layers move to their new places, and each
    added block comes from `fresh` with its last (1x1) conv zeroed, so it starts as the identity."""
    stages = len(load_pocket_config().mimi.seanet.ratios)
    assert decoder_res_layers(state) == 1, "deepens a stock-depth decoder"
    k = decoder_res_layers(fresh)
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        m = _DECODER_LAYER.match(name)
        if m is None:
            out[name] = tensor
            continue
        i = int(m.group(2))
        # stock layout: 0 conv, then per stage [ELU, convtr, block], then ELU and the last conv
        if i == 0:
            j = 0
        elif i <= 3 * stages:
            j = 1 + (2 + k) * ((i - 1) // 3) + (i - 1) % 3
        else:
            j = i + stages * (k - 1)
        out[f"{m.group(1)}{j}{m.group(3)}"] = tensor
    for name, tensor in fresh.items():
        if is_added(name, k):
            last_conv = _DECODER_LAYER.match(name).group(3).startswith(".block.3.")  # type: ignore[union-attr]
            out[name] = torch.zeros_like(tensor) if last_conv else tensor.detach().clone()
    assert set(out) == set(fresh) and all(out[n].shape == fresh[n].shape for n in out), sorted(set(out) ^ set(fresh))[:5]
    return out


def build_dv_mimi(
    frame_rate: float = 25.0, latent_dim: int = 32, init_stock: bool = True, decoder_res_layers: int = 1
) -> MimiModel:
    """The decode side for DACVAE latents; decoder transformer and SEANet from the stock Mimi."""
    mimi = build_mimi(dv_mimi_config(frame_rate, latent_dim, decoder_res_layers))
    if init_stock:
        assert decoder_res_layers == 1, "a deeper decoder starts from a trained one (deepen_decoder)"
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
    state = safetensors.torch.load_file(str(path))
    mimi = build_dv_mimi(frame_rate, latent_dim, init_stock=False, decoder_res_layers=decoder_res_layers(state))
    own = mimi.state_dict()
    for name, tensor in state.items():
        own[name.removeprefix("mimi.")].copy_(tensor)
    mimi = mimi.eval().to(device)
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi
