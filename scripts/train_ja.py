"""Train Pocket TTS on Japanese with the upstream training loop, unmodified.

Swaps four names inside `training.train` before running its `main`:
  - SubprocessDataLoader / DataLoader -> latent-only loaders whose voice prompts are other
    utterances of the same speaker (pocket_tts_ja.loader)
  - encode_batch -> passes the precomputed latents through (no audio, no stitch)
  - ProgressLog -> also mirrors scalars to wandb (pocket_tts_ja.wandb_log)
and wraps save_checkpoint to keep an EMA snapshot (flow_lm weights only, safetensors) of
every checkpoint in run_dir/snapshots/, so evaluation can pick the best step even after the
resumable checkpoints have rotated out.

Top-level config keys starting with ja_ are ours; the upstream trainer gets the config without
them (run_dir/upstream_config.yaml):
  ja_post_eos_frames: N   append N frames of silence after every training target and supervise
                          them with the flow loss, the EOS target unchanged (pocket_tts_ja.loader,
                          pocket_tts_ja.post_eos); validation stays as before, so its losses
                          compare with runs without it

A file run_dir/next_config (the path of another config with the same run_dir) ends the run
right after the next checkpoint with exit code 0; scripts/run_train.sh then goes on with that
config, so a new mix or LR stage takes over without losing steps.

    uv run python scripts/train_ja.py configs/train/A5_8k.yaml
"""

import multiprocessing
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import resource
import sys
import traceback
from pathlib import Path

from functools import partial

import safetensors.torch
import yaml
from torch import nn

import training.train as tt
from pocket_tts_ja.loader import JaLatentDataLoader, JaSubprocessDataLoader, encode_batch_latents
from pocket_tts_ja.post_eos import forward_with_post_eos
from pocket_tts_ja.wandb_log import make_progress_log_cls
from training.args import dump_args, load_args
from training.modules.model import TrainableTTS

_upstream_save_checkpoint = tt.save_checkpoint


class Handover(Exception):
    """run_dir/next_config was found after a checkpoint."""


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
    if (Path(run_dir) / "next_config").exists():
        raise Handover(f"step {step}: handing over to {(Path(run_dir) / 'next_config').read_text().strip()}")


def split_config(config_path: str) -> tuple[str, dict[str, object]]:
    """(path of the config for the upstream trainer, our ja_ keys)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    ja = {k: raw.pop(k) for k in list(raw) if k.startswith("ja_")}
    if not ja:
        return config_path, ja
    out = Path(raw["run_dir"]) / "upstream_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(out), ja


def main(config_path: str):
    config_path, ja = split_config(config_path)
    args = load_args(config_path)
    post_eos = int(ja.pop("ja_post_eos_frames", 0))  # type: ignore[call-overload]
    assert not ja, f"unknown ja_ config keys: {sorted(ja)}"
    tt.SubprocessDataLoader = JaSubprocessDataLoader
    tt.DataLoader = JaLatentDataLoader
    if post_eos > 0:
        tt.SubprocessDataLoader = partial(JaSubprocessDataLoader, post_eos_frames=post_eos)
        TrainableTTS.forward = forward_with_post_eos  # type: ignore[method-assign]
    tt.encode_batch = encode_batch_latents
    tt.ProgressLog = make_progress_log_cls(
        args.run_dir, {**yaml.safe_load(dump_args(args)), "ja_post_eos_frames": post_eos}
    )
    tt.save_checkpoint = save_checkpoint_with_snapshot
    tt.main(config_path)
    import wandb

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: train_ja.py <config.yaml>"
    # an ssh shell starts with 1024 file descriptors; the latent files alone are several hundred
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    code = 0
    try:
        main(sys.argv[1])
    except Handover as stop:
        print(stop, flush=True)
    except Exception:
        # the loader threads would keep a failed run alive (and its GPU memory) for ever
        traceback.print_exc()
        sys.stderr.flush()
        code = 1
    # Loader subprocesses and wandb's service can keep the interpreter alive after training
    # finished (holding GPU memory); everything is saved by now. os._exit skips the cleanup
    # of multiprocessing, so the loader subprocesses must be killed here or they stay behind.
    for child in multiprocessing.active_children():
        child.kill()
    os._exit(code)
