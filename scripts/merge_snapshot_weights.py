"""Full model.safetensors from a training snapshot: the base weights with flow_lm.* replaced.

Snapshots (scripts/train_ja.py) hold only the EMA FlowLM. A model config's weights_path must
also carry Mimi, so this merges the snapshot into the base file of the config. Used to start a
new run from a snapshot (e.g. a cosine fine-tune) and for packaging.

    uv run python scripts/merge_snapshot_weights.py configs/model/ja_24l_4k.yaml \
        runs/B1_4k/snapshots/step_00300000.safetensors runs/B1_4k/weights_300000.safetensors
"""

import argparse
from pathlib import Path

import safetensors.torch
import yaml
from pocket_tts.utils.utils import download_if_necessary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.model_config.read_text())
    base = safetensors.torch.load_file(str(download_if_necessary(cfg["weights_path"])))
    snap = safetensors.torch.load_file(str(args.snapshot))
    assert all(k.startswith("flow_lm.") for k in snap), "snapshot should hold flow_lm.* only"
    merged = {k: v for k, v in base.items() if not k.startswith("flow_lm.")}
    merged.update({k: v.contiguous() for k, v in snap.items()})
    dropped = sorted(set(k for k in base if k.startswith("flow_lm.")) - set(snap))
    safetensors.torch.save_file(merged, str(args.out))
    print(f"{len(merged)} tensors -> {args.out} ({len(snap)} from the snapshot; base flow_lm keys not in it: {dropped[:5]})")


if __name__ == "__main__":
    main()
