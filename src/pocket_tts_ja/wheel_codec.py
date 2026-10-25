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
the deeper decoder starts out computing exactly what the shallower one did. It can also be wider
(``decoder_filters``, the SEANet's n_filters; the stock wheel reads it from the same section): a
trained decoder is widened with ``widen_decoder`` (net2wider). The residual blocks' kernel, dilation
base and compression (``decoder_knobs``, also read by the stock wheel from that section) differ from
the stock ones only in a decoder trained from scratch (``stock_seanet=False``: the decoder transformer
still starts from the stock weights). Such a decoder's file carries them in its safetensors metadata
(``save_decoder``), since the weights alone do not tell the dilation.

Not for the stock wheel (plan W2): ``snake=True`` gives every ELU of the SEANet decoder Snake's
periodic term (``SnakeELU``), the activation of DACVAE's own decoder. The term starts at zero, so a
trained ELU decoder fitted with it starts out unchanged.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from pocket_tts.models.mimi import MimiModel, build_mimi
from pocket_tts.modules.conv import StreamingConv1d
from pocket_tts.modules.seanet import SEANetResnetBlock
from pocket_tts.utils.config import MimiConfig
from pocket_tts.utils.utils import download_if_necessary
from pocket_tts_ja.mimi_utils import load_pocket_config
from training.modules.utils import stamp_state_names

DECODER_MODULES = ("quantizer", "upsample", "decoder_transformer", "decoder")
FRESH_MODULES = ("quantizer", "upsample")
DECODER_KNOBS = ("residual_kernel_size", "dilation_base", "compress")
# "<prefix>decoder.model.<index>.<rest>": a tensor of the SEANet decoder's layer list
_DECODER_LAYER = re.compile(r"^(.*?(?<![a-z_])decoder\.model\.)(\d+)(\..*)$")


def dv_mimi_config(
    frame_rate: float = 25.0, latent_dim: int = 32, decoder_res_layers: int = 1, decoder_filters: int | None = None,
    decoder_knobs: dict[str, int] | None = None,
) -> MimiConfig:
    config = load_pocket_config().mimi.model_dump()
    config["frame_rate"] = frame_rate
    config["quantizer"]["dimension"] = latent_dim
    config["inner_dim"] = latent_dim
    seanet = config["seanet"]
    assert set(decoder_knobs or {}) <= set(DECODER_KNOBS), decoder_knobs
    dec = {**seanet, "n_residual_layers": decoder_res_layers, "n_filters": decoder_filters or seanet["n_filters"],
           **(decoder_knobs or {})}
    if dec != seanet:
        config["decoder_seanet"] = dec
    return MimiConfig(**config)


def decoder_knobs(mimi: MimiModel) -> dict[str, int]:
    """The residual-block settings of ``mimi``'s SEANet decoder that differ from the stock ones."""
    stock = load_pocket_config().mimi.seanet
    blocks = [m for m in mimi.decoder.model if isinstance(m, SEANetResnetBlock)]
    first = blocks[0].block[1].conv  # each block: ELU, dilated conv (dim -> dim / compress), ELU, 1x1 conv
    per_stage = len(blocks) // len(stock.ratios)
    knobs = {
        "residual_kernel_size": first.kernel_size[0],
        "dilation_base": blocks[1].block[1].conv.dilation[0] if per_stage > 1 else stock.dilation_base,
        "compress": first.in_channels // first.out_channels,
    }
    return {k: v for k, v in knobs.items() if v != getattr(stock, k)}


def file_decoder_knobs(path: Path) -> dict[str, int]:
    """``decoder_knobs`` as saved in a decoder file's metadata (none: the stock ones)."""
    with safetensors.safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return json.loads(meta.get("decoder_knobs", "{}"))


def save_decoder(mimi: MimiModel, path: Path) -> None:
    knobs = decoder_knobs(mimi)
    safetensors.torch.save_file(decoder_state(mimi), str(path), metadata={"decoder_knobs": json.dumps(knobs)} if knobs else None)


def decoder_res_layers(names: Iterable[str]) -> int:
    """Residual blocks per upsampling stage of the SEANet decoder whose tensors are named `names`."""
    blocks = {int(m.group(2)) for n in names if (m := _DECODER_LAYER.match(n)) and m.group(3).startswith(".block.")}
    stages = len(load_pocket_config().mimi.seanet.ratios)
    assert blocks and len(blocks) % stages == 0, sorted(blocks)
    return len(blocks) // stages


def decoder_filters(state: dict[str, torch.Tensor]) -> int:
    """n_filters of the SEANet decoder of `state`: the input channels of its last conv."""
    last = max(int(m.group(2)) for n in state if (m := _DECODER_LAYER.match(n)))
    (weight,) = [t for n, t in state.items() if (m := _DECODER_LAYER.match(n)) and int(m.group(2)) == last
                 and m.group(3) == ".conv.weight"]
    return int(weight.shape[1])


def widen_decoder(
    state: dict[str, torch.Tensor], fresh: dict[str, torch.Tensor], seed: int = 0, noise: float = 0.02
) -> dict[str, torch.Tensor]:
    """`state` (a decoder, any naming prefix) laid out as `fresh`, the same decoder with more filters:
    net2wider. Every channel of the wider decoder copies one of the narrower's (the first ones are
    themselves, the rest drawn at random), and the layer that reads a channel's copies divides its
    weights among them, so the wider decoder computes what the narrower one did. Adam would give the
    copies the same updates for ever, so the weights into each added copy get Gaussian noise of
    `noise` times their layer's weight std (the start is then close to, not exactly, the narrower
    decoder). The same seed widens a run's weights and its EMA alike."""
    gen = torch.Generator().manual_seed(seed)
    k = decoder_res_layers(state)
    assert decoder_res_layers(fresh) == k, "widen_decoder keeps the depth (deepen first)"
    stages = len(load_pocket_config().mimi.seanet.ratios)
    (prefix,) = {m.group(1) for n in state if (m := _DECODER_LAYER.match(n))}
    out = dict(state)

    def produce(layer: str, g: torch.Tensor, dim: int, old: int) -> None:
        """the output channels of `layer` (weight dim `dim`) become copies g of the old ones"""
        w, b = out[f"{layer}.weight"], out[f"{layer}.bias"]
        w2 = w.index_select(dim, g.to(w.device)).clone()
        added = w2.narrow(dim, old, len(g) - old)
        added += noise * float(w.std()) * torch.randn(added.shape, generator=gen).to(w)
        out[f"{layer}.weight"], out[f"{layer}.bias"] = w2, b[g.to(b.device)].clone()

    def consume(layer: str, g: torch.Tensor, dim: int, old: int) -> None:
        """the input channels of `layer` read copies g: each copy takes 1 / (its source's copies)"""
        count = torch.zeros(old).index_add_(0, g, torch.ones(len(g)))
        share = (1 / count[g]).to(out[f"{layer}.weight"])
        shape = [1] * out[f"{layer}.weight"].dim()
        shape[dim] = len(g)
        w = out[f"{layer}.weight"]
        out[f"{layer}.weight"] = w.index_select(dim, g.to(w.device)) * share.view(shape)

    def mapping(old: int, new: int) -> torch.Tensor:
        assert new >= old, (old, new)
        return torch.cat([torch.arange(old), torch.randint(old, (new - old,), generator=gen)])

    def layer(i: int, rest: str = "") -> str:
        return f"{prefix}{i}{rest}"

    def size(name: str, which: dict[str, torch.Tensor], dim: int) -> int:
        return int(which[f"{name}.weight"].shape[dim])

    first = layer(0, ".conv")
    old, new = size(first, state, 0), size(first, fresh, 0)
    g = mapping(old, new)
    produce(first, g, 0, old)
    for s in range(stages):
        up = layer(2 + (2 + k) * s, ".convtr")  # weight [in, out, kernel]
        consume(up, g, 0, old)
        old, new = size(up, state, 1), size(up, fresh, 1)
        g = mapping(old, new)
        produce(up, g, 1, old)
        for b in range(k):
            block = layer(3 + (2 + k) * s + b, ".block")
            consume(f"{block}.1.conv", g, 1, old)
            h_old, h_new = size(f"{block}.1.conv", state, 0), size(f"{block}.1.conv", fresh, 0)
            h = mapping(h_old, h_new)
            produce(f"{block}.1.conv", h, 0, h_old)
            consume(f"{block}.3.conv", h, 1, h_old)
            produce(f"{block}.3.conv", g, 0, old)
    consume(layer(2 + (2 + k) * stages, ".conv"), g, 1, old)
    assert set(out) == set(fresh) and all(out[n].shape == fresh[n].shape for n in out), \
        [(n, tuple(out[n].shape), tuple(fresh[n].shape)) for n in out if n in fresh and out[n].shape != fresh[n].shape][:5]
    return out


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


class _SnakeELUFn(torch.autograd.Function):
    """elu(x) + beta sin^2(alpha x) / alpha keeping only x for the backward (the plain expression keeps
    four activation-sized tensors per layer: W1w2 + Snake ran out of 32 GB at batch 16)."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, alpha: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, alpha, beta)
        return F.elu(x) + beta * torch.sin(alpha * x).pow(2) / (alpha + 1e-9)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, g: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x, alpha, beta = ctx.saved_tensors  # type: ignore[attr-defined]
        a = alpha + 1e-9
        sin2 = torch.sin(2 * alpha * x)
        sq = torch.sin(alpha * x).pow(2)
        gx = g * (torch.where(x > 0, 1.0, torch.exp(x)) + beta * sin2 * alpha / a)
        g_alpha = (g * beta * (x * sin2 / a - sq / a**2)).sum(dim=(0, 2), keepdim=True)
        g_beta = (g * sq / a).sum(dim=(0, 2), keepdim=True)
        return gx, g_alpha, g_beta


class SnakeELU(nn.Module):
    """ELU plus Snake's periodic term (BigVGAN, DAC), per channel: elu(x) + beta sin^2(alpha x) / alpha.
    beta starts at zero: an ELU decoder fitted with these starts out computing what it did."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SnakeELUFn.apply(x, self.alpha, self.beta)  # type: ignore[no-any-return]


def is_snake(name: str) -> bool:
    return name.endswith((".alpha", ".beta"))


def snake_decoder(mimi: MimiModel) -> None:
    """Swaps every ELU of the SEANet decoder for a SnakeELU as wide as the conv after it."""

    def swap(layers: nn.ModuleList) -> None:
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.ELU):
                after = layers[i + 1]
                conv = after.conv if isinstance(after, StreamingConv1d) else after.convtr
                layers[i] = SnakeELU(conv.in_channels)
            elif isinstance(layer, SEANetResnetBlock):
                swap(layer.block)

    swap(mimi.decoder.model)


def build_dv_mimi(
    frame_rate: float = 25.0, latent_dim: int = 32, init_stock: bool = True, decoder_res_layers: int = 1,
    decoder_filters: int | None = None, snake: bool = False, decoder_knobs: dict[str, int] | None = None,
    stock_seanet: bool = True,
) -> MimiModel:
    """The decode side for DACVAE latents; decoder transformer and SEANet from the stock Mimi.

    ``stock_seanet=False`` leaves the SEANet decoder at its random initialisation (any size).
    """
    mimi = build_mimi(dv_mimi_config(frame_rate, latent_dim, decoder_res_layers, decoder_filters, decoder_knobs))
    if init_stock:
        if stock_seanet:
            assert decoder_res_layers == 1, "a deeper decoder starts from a trained one (deepen_decoder)"
            assert decoder_filters in (None, load_pocket_config().mimi.seanet.n_filters), "a wider one too (widen_decoder)"
            assert not decoder_knobs, "other residual blocks start from scratch (stock_seanet=False)"
        stock = load_pocket_config()
        assert stock.weights_path is not None
        state = safetensors.torch.load_file(download_if_necessary(str(stock.weights_path)))
        own = mimi.state_dict()
        for name, tensor in state.items():
            key = name.removeprefix("mimi.")
            if not name.startswith("mimi.") or key.split(".")[0] in FRESH_MODULES:
                continue
            if not stock_seanet and key.split(".")[0] == "decoder":
                continue
            if key in own and own[key].shape == tensor.shape:
                own[key].copy_(tensor)
    if snake:
        snake_decoder(mimi)
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
    mimi = build_dv_mimi(frame_rate, latent_dim, init_stock=False, decoder_res_layers=decoder_res_layers(state),
                         decoder_filters=decoder_filters(state), snake=any(map(is_snake, state)),
                         decoder_knobs=file_decoder_knobs(path))
    own = mimi.state_dict()
    for name, tensor in state.items():
        own[name.removeprefix("mimi.")].copy_(tensor)
    mimi = mimi.eval().to(device)
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi
