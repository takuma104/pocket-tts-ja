"""Average the weights of decoders fine-tuned from the same start (a "model soup", Wortsman et al. 2022).

The runs of plan M2 high band all start from M2_gs2gan_keep4k and differ in one option each; their UTMOSv2 is
within the run-to-run spread. Averaging their exported (EMA) decoders tests whether they sit in one basin, where
the mean is often better than any of them, at no training cost. Floating point tensors are averaged (in float32,
saved in the first file's dtype); any other tensor must be the same in every file. The first file's metadata is kept.

    uv run python scripts/m2/average_decoders.py runs/M2H_soup5.safetensors \
        runs/M2H_H2ghSG4/decoder_step_00010000.safetensors runs/M2H_H2ghSG4W15/decoder_step_00010000.safetensors
"""

import argparse
from pathlib import Path

import safetensors
import safetensors.torch
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("decoders", type=Path, nargs="+")
    parser.add_argument("--weights", type=float, nargs="*", default=None, help="one per decoder (default: equal)")
    args = parser.parse_args()
    weights = args.weights or [1.0] * len(args.decoders)
    assert len(weights) == len(args.decoders), "one weight per decoder"
    total = sum(weights)

    with safetensors.safe_open(str(args.decoders[0]), "pt") as f:
        metadata = f.metadata()
    states = [safetensors.torch.load_file(str(p)) for p in args.decoders]
    keys = set(states[0])
    for p, s in zip(args.decoders[1:], states[1:], strict=True):
        assert set(s) == keys, f"{p} has other tensors than {args.decoders[0]}"
    out: dict[str, torch.Tensor] = {}
    for k, first in states[0].items():
        if first.is_floating_point():
            mean = sum(w / total * s[k].float() for w, s in zip(weights, states, strict=True))
            out[k] = mean.to(first.dtype).contiguous()  # type: ignore[union-attr]
        else:
            assert all(torch.equal(first, s[k]) for s in states[1:]), f"{k} differs and is not averageable"
            out[k] = first
    args.out.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(out, str(args.out), metadata=metadata)
    print(f"{args.out}: mean of {len(states)} decoders, {len(out)} tensors")


if __name__ == "__main__":
    main()
