"""W1s: average W1 decoders fine-tuned from one checkpoint (a model soup, as main's CC decoder soup4).

The inputs are decoder_step_*.safetensors of scripts/wheel/train_decoder.py runs started with
--init from the same checkpoint, so their weights stay close enough to average. Floating tensors
are averaged; any other tensor must be the same in every input.

    uv run python scripts/wheel/soup_decoders.py runs/irodori_distill/W1s_soup4.safetensors \
        runs/irodori_distill/W1s_{same,hi,gan15,mel}/decoder_step_00010000.safetensors
"""

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch

from pocket_tts_ja.wheel_codec import file_decoder_knobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    states = [safetensors.torch.load_file(str(p)) for p in args.inputs]
    keys = set(states[0])
    assert all(set(s) == keys for s in states), "the decoders have different tensors"
    soup: dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        ts = [s[k] for s in states]
        if ts[0].is_floating_point():
            soup[k] = torch.stack([t.float() for t in ts]).mean(0).to(ts[0].dtype).contiguous()
        else:
            assert all(torch.equal(t, ts[0]) for t in ts), k
            soup[k] = ts[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    knobs = {json.dumps(file_decoder_knobs(p), sort_keys=True) for p in args.inputs}
    assert len(knobs) == 1, "the decoders have different residual blocks"
    meta = knobs.pop()
    safetensors.torch.save_file(soup, str(args.out), metadata={"decoder_knobs": meta} if meta != "{}" else None)
    spread = max(float((s[k].float() - soup[k].float()).norm() / soup[k].float().norm().clamp_min(1e-12))
                 for s in states for k in keys if soup[k].is_floating_point())
    print(f"{args.out}: {len(states)} decoders, {len(soup)} tensors, largest relative distance to the soup {spread:.4f}")


if __name__ == "__main__":
    main()
