"""Train Pocket TTS on Japanese with the upstream training loop, unmodified.

Swaps four names inside `training.train` before running its `main`:
  - SubprocessDataLoader / DataLoader -> latent-only loaders whose voice prompts are other
    utterances of the same speaker (pocket_tts_ja.loader)
  - encode_batch -> passes the precomputed latents through (no audio, no stitch)
  - ProgressLog -> also mirrors scalars to wandb (pocket_tts_ja.wandb_log)
and wraps save_checkpoint to keep an EMA snapshot (flow_lm weights only, safetensors) of
every checkpoint in run_dir/snapshots/, so evaluation can pick the best step even after the
resumable checkpoints have rotated out.

    uv run python scripts/train_ja.py configs/train/A5_8k.yaml
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

import safetensors.torch
import yaml
from torch import nn

import training.train as tt
from pocket_tts_ja.loader import JaLatentDataLoader, JaSubprocessDataLoader, encode_batch_latents
from pocket_tts_ja.wandb_log import make_progress_log_cls
from training.args import dump_args, load_args

_upstream_save_checkpoint = tt.save_checkpoint


def save_checkpoint_with_snapshot(
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: object,
    ema: object,
    num_keep: int,
    mimi: nn.Module | None = None,
):
    _upstream_save_checkpoint(run_dir, step, model, optimizer, ema, num_keep, mimi)
    export = Path(run_dir) / "model.safetensors"
    if mimi is None or not export.exists():
        return
    state = safetensors.torch.load_file(str(export))
    flow = {k: v for k, v in state.items() if k.startswith("flow_lm.")}
    snap_dir = Path(run_dir) / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    tmp = snap_dir / f"step_{step:08d}.tmp"
    safetensors.torch.save_file(flow, str(tmp))
    tmp.rename(snap_dir / f"step_{step:08d}.safetensors")


def main(config_path: str):
    args = load_args(config_path)
    tt.SubprocessDataLoader = JaSubprocessDataLoader
    tt.DataLoader = JaLatentDataLoader
    tt.encode_batch = encode_batch_latents
    tt.ProgressLog = make_progress_log_cls(args.run_dir, yaml.safe_load(dump_args(args)))
    tt.save_checkpoint = save_checkpoint_with_snapshot
    tt.main(config_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: train_ja.py <config.yaml>"
    main(sys.argv[1])
