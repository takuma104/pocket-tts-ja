"""P2: train a PocketTTS FlowLM on the teacher's DACVAE latents with the upstream training loop.

Like main's scripts/train_ja.py, swaps names inside `training.train` before running its `main`:
  - SubprocessDataLoader / DataLoader -> pocket_tts_ja.dv_loader: the teacher's latents, the
    teacher's reference latent as the voice prompt, speech + tail under a three-valued mask
  - encode_batch -> passes the latents through
  - TrainableTTS.forward -> pocket_tts_ja.post_eos (the flow loss covers the tail too)
  - build_models -> the FlowLM of model_config, warm-started from ja_init_from (main's CC 6L
    student) except what is tied to the latent space (input_linear, bos_emb, speaker_proj_weight,
    flow_net, out_eos: upstream's fresh init), with the latent statistics fixed to
    ja_latent_stats (plan §2; stats_ema_steps must stay 0). The codec is the 25 Hz stock-wheel
    decode side (pocket_tts_ja.wheel_codec), which only gives the trainer its rates.
    With distill_cfg_coef > 0 (P3d, CFG baking, main's D0 recipe), the student starts as an exact
    copy of distill_teacher_weights (a student of this latent space, with distill_teacher_config
    = model_config; ja_init_from unset) and a frozen copy is the teacher; only the backbone and
    the conditioning train (upstream's latent CFG distillation, flow and EOS heads frozen).
    With distill_teacher_config another (deeper) model of this latent space (S2, main's D1 recipe),
    the teacher is that model with distill_teacher_weights and the student starts from its bottom
    and top layers (upstream's shrink_checkpoint.shrink; same d_model, so every other tensor, the
    heads too, is the teacher's); the CFG target is the teacher's, the rest as above.
  - write_samples -> decodes with DACVAE (48 kHz)
  - ProgressLog -> also mirrors scalars to wandb (pocket_tts_ja.wandb_log)
and wraps save_checkpoint to keep an EMA snapshot (flow_lm weights only, safetensors) of every
checkpoint in run_dir/snapshots/.

Top-level config keys starting with ja_ are ours; the upstream trainer gets the config without
them (run_dir/upstream_config.yaml):
  ja_init_from: flow_lm.* safetensors to warm-start from ("" for upstream's random init)
  ja_latent_stats: npz with per-dimension mean and std (scripts/train/make_split.py)
  ja_tail_frames: frames supervised after the speech end (default 12)
  ja_prompt_aug: P3g, how the training prompts vary (pocket_tts_ja.dv_loader.PromptAug fields;
    the validation rows keep the teacher's reference)
  ja_init_full: P3h, ja_init_from is a student of this latent space: load all of it (strict),
    the latent-tied parts too
  ja_context_noise: P3c, noise on the past latents in training (pocket_tts_ja.post_eos.ContextNoise
    fields)
  ja_text_aug: P3e, the stock wheel's chunk forms in training: several sentences in one row,
    ", " after commas (pocket_tts_ja.dv_loader.TextAug fields; the validation rows stay single)
  ja_head_only: H, a new flow head on a trained backbone: ja_init_from gives everything but
    flow_net (strict), flow_net starts fresh at model_config's flow size, and only it trains

    uv run python scripts/train/train_student.py configs/train/P2_6l.yaml
"""

import multiprocessing
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import copy
import logging
import math
import resource
import sys
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile
import torch
import yaml
from torch import nn

import training.train as tt
from pocket_tts.models.mimi import MimiModel
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import Config
from pocket_tts_ja.dacvae import DACVAE, SAMPLE_RATE as DV_SR
from pocket_tts_ja.dv_loader import (
    TAIL_FRAMES,
    DvLatentDataLoader,
    DvSubprocessDataLoader,
    PromptAug,
    TextAug,
    encode_batch_latents,
)
from pocket_tts_ja.post_eos import ContextNoise, forward_with_post_eos
from pocket_tts_ja.wandb_log import make_progress_log_cls
from pocket_tts_ja.wheel_codec import build_dv_mimi
from training.args import TrainArgs, dump_args, load_args
from training.modules.builders import _load_flow_lm_state, load_model_config
from training.modules.model import TrainableTTS
from training.modules.samplers import build_flow
from training.modules.utils import disable_grad, dit_init, gaussian_init, stamp_state_names
from training.scripts.shrink_checkpoint import shrink

logger = logging.getLogger(__name__)

# FlowLM parts whose meaning depends on the codec's latent space: started fresh
LATENT_TIED = ("input_linear.", "bos_emb", "speaker_proj_weight", "flow_net.", "out_eos.", "emb_mean", "emb_std")

_upstream_save_checkpoint = tt.save_checkpoint


def build_teacher(args: TrainArgs, latent_dim: int) -> nn.Module:
    """Depth distillation's teacher: the FlowLM of distill_teacher_config, distill_teacher_weights loaded strictly."""
    config = load_model_config(args.distill_teacher_config, args.distill_teacher_overrides)
    flow_lm = TTSModel._from_pydantic_config(
        config, temp=0.7, sampler_decode_steps=1, noise_clamp=None, eos_threshold=0.0, origin=None
    ).flow_lm
    d_model = config.flow_lm.transformer.d_model
    flow_lm.speaker_proj_weight = nn.Parameter(torch.zeros((d_model, latent_dim), dtype=torch.float32))
    flow_lm.load_state_dict(_load_flow_lm_state(args.distill_teacher_weights, args.distill_teacher_use_ema), strict=True)
    return flow_lm.eval()


def make_build_models(
    init_from: str, stats: Path, init_full: bool = False, head_only: bool = False
) -> Callable[[TrainArgs], tuple[TrainableTTS, MimiModel, Config]]:
    def build_models(args: TrainArgs) -> tuple[TrainableTTS, MimiModel, Config]:
        assert not args.start_from_pretrained and not args.head_distill
        distill = args.distill_cfg_coef > 0
        depth = distill and args.distill_teacher_config != args.model_config
        if distill:
            assert args.distill_teacher_config and not init_from, "the student starts from its teacher"
        assert args.stats_ema_steps == 0, "the latent statistics are fixed (ja_latent_stats)"
        config = load_model_config(args.model_config, args.model_overrides)
        tts_model = TTSModel._from_pydantic_config(
            config, temp=0.7, sampler_decode_steps=1, noise_clamp=None, eos_threshold=0.0, origin=None
        )
        flow_lm = tts_model.flow_lm
        latent_dim = config.mimi.quantizer.dimension
        d_model = config.flow_lm.transformer.d_model
        flow_lm.speaker_proj_weight = nn.Parameter(torch.zeros((d_model, latent_dim), dtype=torch.float32))
        flow = build_flow(args.flow.type, **args.flow.kwargs)
        assert flow.num_time_conds == 2, "the stock flow head (LSD)"

        gaussian_init(flow_lm.transformer)
        gaussian_init(flow_lm.input_linear)
        gaussian_init(flow_lm.conditioner)
        nn.init.trunc_normal_(flow_lm.speaker_proj_weight, std=1 / math.sqrt(latent_dim))
        dit_init(flow_lm.flow_net)
        if depth:
            teacher = build_teacher(args, latent_dim)
            state, kept = shrink(teacher.state_dict(), len(flow_lm.transformer.layers))
            flow_lm.load_state_dict(state, strict=True)
            logger.info(f"depth distillation (CFG {args.distill_cfg_coef}) of {args.distill_teacher_weights}, "
                        f"student from its layers {kept}")
        elif distill:
            state = _load_flow_lm_state(args.distill_teacher_weights, args.distill_teacher_use_ema)
            flow_lm.load_state_dict(state, strict=True)
            logger.info(f"CFG distillation ({args.distill_cfg_coef}) of {args.distill_teacher_weights}")
        elif init_from and head_only:
            state = _load_flow_lm_state(init_from, args.init_from_use_ema)
            missing, unexpected = flow_lm.load_state_dict({k: v for k, v in state.items() if not k.startswith("flow_net.")}, strict=False)
            assert not unexpected and missing and all(k.startswith("flow_net.") for k in missing), (missing, unexpected)
            for name, p in flow_lm.named_parameters():
                p.requires_grad_(name.startswith("flow_net."))
            n = sum(p.numel() for p in flow_lm.flow_net.parameters())
            logger.info(f"backbone of {init_from} (frozen), a fresh flow head of {n / 1e6:.1f}M params")
        elif init_from and init_full:
            flow_lm.load_state_dict(_load_flow_lm_state(init_from, args.init_from_use_ema), strict=True)
            logger.info(f"continued from {init_from} (all of it)")
        elif init_from:
            state = _load_flow_lm_state(init_from, args.init_from_use_ema)
            kept = {k: v for k, v in state.items() if not k.startswith(LATENT_TIED)}
            missing, unexpected = flow_lm.load_state_dict(kept, strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith(LATENT_TIED) for k in missing), missing
            logger.info(f"warm-started from {init_from}; fresh: {sorted({k.split('.')[0] for k in missing})}")
        s = np.load(stats)
        if distill:
            assert np.allclose(flow_lm.emb_mean.numpy(), s["mean"], atol=1e-5), "the teacher's statistics differ"
        flow_lm.emb_mean.copy_(torch.from_numpy(s["mean"]))
        flow_lm.emb_std.copy_(torch.from_numpy(s["std"]))
        logger.info(f"latent statistics from {stats} ({int(s['frames'])} frames)")

        mimi = build_dv_mimi(config.mimi.frame_rate, latent_dim, init_stock=False).eval()
        for p in mimi.parameters():
            p.requires_grad_(False)
        model = TrainableTTS(flow_lm, flow, args)
        if distill:
            if not depth:
                teacher = copy.deepcopy(flow_lm).eval()
            disable_grad(teacher)
            stamp_state_names(teacher)
            model.__dict__["distill_teacher"] = teacher  # outside the module tree, as upstream
            disable_grad(flow_lm.flow_net)
            disable_grad(flow_lm.out_eos)
        return model, mimi, config

    return build_models


_CODEC: list[DACVAE] = []


def write_samples_dacvae(
    model: TrainableTTS,
    mimi: MimiModel,
    tokenize: Callable[[str], list[int]],
    args: TrainArgs,
    run_dir: Path,
    step: int,
    voice_latents: torch.Tensor,
    device: torch.device,
) -> None:
    """upstream write_samples, decoded with DACVAE: run_dir/samples/step<step>_<i>.wav (48 kHz)."""
    if not _CODEC:
        _CODEC.append(DACVAE.load(str(device)))
    out_dir = run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    model.eval()
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in args.sample_sentences]
    with torch.no_grad():
        outs = model.generate(
            tokens, [voice_latents] * len(tokens), max_frames=750, temp=args.sample_temp,
            cfg_coef=args.sample_cfg_coef, eos_threshold=-4.0, eos_countdown=TAIL_FRAMES,
        )
        for i, latents in enumerate(outs):
            if latents.shape[0] < 2:
                logger.warning(f"sample {i} at step {step}: empty generation, skipped")
                continue
            audio = _CODEC[0].decode(latents[None].float())[0]
            soundfile.write(str(out_dir / f"step{step:08d}_{i}.wav"), audio.cpu().numpy(), DV_SR)
    model.train()
    logger.info(f"wrote {len(tokens)} samples at step {step}")


def save_checkpoint_with_snapshot(
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: object,
    ema: object,
    num_keep: int,
    mimi: nn.Module | None = None,
) -> None:
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


def split_config(config_path: str) -> tuple[str, dict[str, object]]:
    """(path of the config for the upstream trainer, our ja_ keys)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    ja = {k: raw.pop(k) for k in list(raw) if k.startswith("ja_")}
    out = Path(raw["run_dir"]) / "upstream_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(out), ja


def main(config_path: str) -> None:
    config_path, ja = split_config(config_path)
    args = load_args(config_path)
    init_from = str(ja.pop("ja_init_from", ""))
    stats = Path(str(ja.pop("ja_latent_stats")))
    tail = int(ja.pop("ja_tail_frames", TAIL_FRAMES))  # type: ignore[call-overload]
    aug_spec = ja.pop("ja_prompt_aug", None)
    aug = PromptAug(**aug_spec) if isinstance(aug_spec, dict) else None
    init_full = bool(ja.pop("ja_init_full", False))
    head_only = bool(ja.pop("ja_head_only", False))
    assert not (head_only and init_full), "ja_head_only loads ja_init_from but its flow head"
    noise_spec = ja.pop("ja_context_noise", None)
    TrainableTTS.ja_context_noise = ContextNoise(**noise_spec) if isinstance(noise_spec, dict) else None  # type: ignore[attr-defined]
    text_spec = ja.pop("ja_text_aug", None)
    text_aug = TextAug(**text_spec) if isinstance(text_spec, dict) else None
    assert not ja, f"unknown ja_ config keys: {sorted(ja)}"
    tt.SubprocessDataLoader = partial(DvSubprocessDataLoader, tail_frames=tail, prompt_aug=aug, text_aug=text_aug)
    tt.DataLoader = partial(DvLatentDataLoader, tail_frames=tail)
    tt.encode_batch = encode_batch_latents
    TrainableTTS.forward = forward_with_post_eos  # type: ignore[method-assign]
    tt.build_models = make_build_models(init_from, stats, init_full, head_only)
    tt.write_samples = write_samples_dacvae
    tt.ProgressLog = make_progress_log_cls(
        args.run_dir,
        {**yaml.safe_load(dump_args(args)), "ja_init_from": init_from, "ja_latent_stats": str(stats), "ja_tail_frames": tail,
         "ja_prompt_aug": aug_spec, "ja_init_full": init_full, "ja_head_only": head_only, "ja_context_noise": noise_spec,
         "ja_text_aug": text_spec},
    )
    tt.save_checkpoint = save_checkpoint_with_snapshot
    tt.main(config_path)
    import wandb

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: train_student.py <config.yaml>"
    # an ssh shell starts with 1024 file descriptors; the latent files alone are several hundred
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    code = 0
    try:
        main(sys.argv[1])
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
