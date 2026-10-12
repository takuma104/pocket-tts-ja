#!/usr/bin/env python3
"""Batched synthesis with Irodori-TTS-v4.1-Small-Yomi (plan §4.4, S0/S3).

Input is a jsonl with one row per utterance; each row needs at least::

    {"id": str, "text": str, "ref_wav": str | null, "seed": int}

and may give ``"seconds"``: the clip length to generate, instead of the duration predictor's
(for very short text, which the predictor stretches to 2-3 s).

``ref_wav`` is a speaker reference (any format/rate torchaudio or soundfile can
read; relative paths are resolved against ``--ref-root``). ``null`` synthesizes
without a speaker reference (Irodori's ``--no-ref``: the reference is empty and
the speaker CFG branch is disabled, exactly as in the reference pipeline).

Each row becomes ``<output-dir>/<id>.wav`` (24 kHz mono 16-bit; Irodori's 48 kHz
output is resampled with soxr HQ). Existing outputs are skipped, so interrupted
runs can be resumed. Files are written atomically (tmp + rename).

Logs go to ``--log-dir`` (default ``<output-dir>_logs``):

- ``failures_shard{i}.jsonl``: rows that could not be synthesized (bad ref,
  empty text, CUDA errors after a per-item retry, ...).
- ``meta_shard{i}.jsonl``: one line per written file (predicted frames,
  whether the 30 s duration cap or the 256-token text cap was hit, clipping).
- ``generation_config_shard{i}.json``: the effective settings.

The sampler is adapted from Irodori-TTS-Experiments ``scripts/batch_synth.py``
(the pipeline behind the release JKYB numbers) and keeps its defaults: RF with
40 steps, independent CFG (text 3.0 / speaker 5.0, applied for t in [0.5, 1.0]),
DiT in bf16 with the text encoder and the DACVAE codec in fp32, SDPA
"efficient" backend, tail trimming and the silentcipher watermark. The only
extension is per-item references: rows are sorted by predicted length and
batched across different references (padded and masked, as in training); rows
with and without a reference are never mixed in one batch.

Run inside the Irodori-TTS environment, e.g. on galleria2::

    PYTHONPATH=~/co/pocket-tts-ja-synth-env/Irodori-TTS \\
      ~/.local/bin/uv run --project ~/co/pocket-tts-ja-synth-env/Irodori-TTS --no-sync \\
      python scripts/synth/irodori_synth.py --input rows.jsonl --output-dir out/audio \\
      --checkpoint ~/co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch
from irodori_tts.attention import _SDPA_PRIORITY
from irodori_tts.codec import unpatchify_latent
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    find_flattening_point,
)
from irodori_tts.text_normalization import normalize_text
from torch.nn.attention import SDPBackend

DEFAULT_CHECKPOINT = str(
    Path.home() / "co/Irodori-TTS-v4.1-Small-Yomi/model.safetensors"
)
TEXT_MODULES = (
    "pretrained_text_backbone",
    "text_encoder",
    "text_norm",
    "caption_encoder",
    "caption_norm",
)


# --------------------------------------------------------------------------- runtime


def set_sdpa_backend(name: str) -> None:
    """``cudnn`` (Irodori's default) re-plans per sequence length in bf16; ``efficient`` doesn't."""
    priorities = {
        "cudnn": [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    }
    _SDPA_PRIORITY[:] = priorities[name]


def load_runtime(
    checkpoint: str,
    *,
    device: str,
    precision: str,
    text_precision: str,
    codec_precision: str,
) -> InferenceRuntime:
    """Load the runtime; the text/caption encoder is cast back to fp32 if requested."""
    path = (
        checkpoint if Path(checkpoint).is_file() else download_hf_checkpoint(checkpoint)
    )
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(path),
            model_device=device,
            model_precision=precision,
            codec_device=device,
            codec_precision=codec_precision,
        )
    )
    if text_precision == "fp32":
        for name in TEXT_MODULES:
            module = getattr(runtime.model, name, None)
            if module is not None:
                module.float()
    return runtime


# --------------------------------------------------------------------------- sampler


@dataclass(frozen=True)
class SynthItem:
    key: str
    text: str
    seed: int
    ref_wav: str | None
    seconds: float | None = None  # clip length to use instead of the duration predictor's


@dataclass(frozen=True)
class SynthOutput:
    item: SynthItem
    audio: torch.Tensor  # (channels, samples) at the codec rate, trimmed (+ watermark)
    sample_rate: int
    pred_frames: int
    clamped: bool


@dataclass(frozen=True)
class SamplingSettings:
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    trim_tail: bool = True
    watermark: bool = True


class RefLoadError(RuntimeError):
    """A reference waveform could not be loaded/encoded."""


class MultiRefSynthesizer:
    """``batch_synth.BatchSynthesizer`` with a per-item speaker reference."""

    def __init__(self, runtime: InferenceRuntime, settings: SamplingSettings) -> None:
        if runtime.model_cfg.flow_parameterization != "rf_velocity":
            raise ValueError("Only RF checkpoints are supported.")
        if not runtime.model_cfg.use_speaker_condition_resolved:
            raise ValueError("Expected a speaker-conditioned checkpoint.")
        self.runtime = runtime
        self.model = runtime.model
        self.settings = settings
        self.device = runtime.model_device
        self.dtype = self.model.in_proj.weight.dtype  # DiT dtype
        self.hop_length = int(runtime.codec.model.hop_length)
        self.sample_rate = int(runtime.codec.sample_rate)
        self.text_max_len = int(runtime.default_text_max_len)
        self.caption_max_len = int(runtime.default_caption_max_len)
        self._ref_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}
        self._ref_cache_limit = 8192

    # ---- references

    def reference(self, ref_wav: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Patched reference latent (1, S, D) and mask (1, S); cached per path."""
        cached = self._ref_cache.get(ref_wav)
        if cached is not None:
            return cached
        try:
            with torch.inference_mode():
                latent, mask = self.runtime._load_reference_latent(
                    req=SamplingRequest(
                        text="", ref_wav=ref_wav, no_ref=ref_wav is None
                    ),
                    batch_size=1,
                    messages=[],
                )
        except Exception as exc:
            raise RefLoadError(f"{type(exc).__name__}: {exc}") from exc
        if latent is None or mask is None:
            raise RefLoadError("runtime returned no reference latent")
        entry = (latent.to(self.dtype), mask)
        if len(self._ref_cache) >= self._ref_cache_limit:
            self._ref_cache.clear()
        self._ref_cache[ref_wav] = entry
        return entry

    def _ref_batch(
        self, refs: Sequence[str | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = [self.reference(ref) for ref in refs]
        if all(ref == refs[0] for ref in refs):  # same layout as BatchSynthesizer
            latent, mask = pieces[0]
            return latent.expand(len(refs), -1, -1), mask.expand(len(refs), -1)
        max_len = max(int(p[0].shape[1]) for p in pieces)
        dim = int(pieces[0][0].shape[2])
        latent = torch.zeros(
            (len(refs), max_len, dim), device=self.device, dtype=self.dtype
        )
        mask = torch.zeros((len(refs), max_len), device=self.device, dtype=torch.bool)
        for i, (lat, msk) in enumerate(pieces):
            n = int(lat.shape[1])
            latent[i, :n] = lat[0]
            mask[i, :n] = msk[0]
        return latent, mask

    # ---- conditions

    def _encode(
        self, texts: Sequence[str], refs: Sequence[str | None]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = len(texts)
        text_ids, text_mask = self.runtime.tokenizer.batch_encode(
            list(texts), max_length=self.text_max_len
        )
        text_ids = text_ids.to(self.device)
        text_mask = text_mask.to(self.device)
        caption_ids = None
        caption_mask = None
        if self.runtime.model_cfg.use_caption_condition:
            caption_ids, caption_mask = self.runtime.caption_tokenizer.batch_encode(
                [""] * batch, max_length=self.caption_max_len
            )
            caption_mask.zero_()
            caption_ids = caption_ids.to(self.device)
            caption_mask = caption_mask.to(self.device)
        ref_latent, ref_mask = self._ref_batch(refs)
        has_speaker = ref_mask.any(dim=1)
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = self.model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )
        text_state = text_state.to(self.dtype)
        if caption_state is not None:
            caption_state = caption_state.to(self.dtype)
        duration_features = build_duration_features(
            list(texts),
            token_counts=text_mask.sum(dim=1),
            max_text_len=self.text_max_len,
            has_speaker=has_speaker,
        ).to(self.device)
        return (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            duration_features,
            has_speaker,
        )

    # ---- duration

    def _frame_bounds(self) -> tuple[int, int]:
        s = self.settings
        min_frames = max(
            1, math.ceil(s.min_seconds * self.sample_rate / self.hop_length)
        )
        max_frames = max(
            1, math.floor(s.max_seconds * self.sample_rate / self.hop_length)
        )
        return min_frames, max_frames

    @torch.inference_mode()
    def predict_latent_steps(self, items: Sequence[SynthItem]) -> list[int]:
        """Predicted latent frame counts, clamped like ``InferenceRuntime.synthesize``."""
        normalized = [normalize_text(item.text).strip() for item in items]
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            duration_features,
            has_speaker,
        ) = self._encode(normalized, [item.ref_wav for item in items])
        batch = len(normalized)
        has_caption = None
        if self.runtime.model_cfg.use_caption_condition:
            has_caption = torch.zeros((batch,), dtype=torch.bool, device=self.device)
        pred_log_frames = self.model.predict_duration_log_frames(
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            duration_features=duration_features,
            has_speaker=has_speaker,
            has_caption=has_caption,
        )
        frames = torch.expm1(pred_log_frames).float().reshape(batch).tolist()
        min_frames, max_frames = self._frame_bounds()
        scale = self.settings.duration_scale
        # The predictor never saw clips under 3 s: it gives very short text 2-3 s, and the
        # sampler fills the spare time with babble. Rows may bring their own length.
        frames = [
            f * scale if item.seconds is None else item.seconds * self.sample_rate / self.hop_length
            for f, item in zip(frames, items, strict=True)
        ]
        return [max(min_frames, min(max_frames, round(f))) for f in frames]

    # ---- sampling

    @torch.inference_mode()
    def sample_latents(
        self,
        items: Sequence[SynthItem],
        latent_steps: Sequence[int],
        *,
        pad_multiple: int = 1,
    ) -> list[torch.Tensor]:
        """Sample latents (frames, latent_dim); all items must agree on having a reference."""
        cfg = self.runtime.model_cfg
        normalized = [normalize_text(item.text).strip() for item in items]
        refs = [item.ref_wav for item in items]
        no_ref = refs[0] is None
        if any((ref is None) != no_ref for ref in refs):
            raise ValueError("Rows with and without a reference cannot share a batch.")
        batch = len(normalized)
        patch = int(cfg.latent_patch_size)
        patched_steps = [math.ceil(steps / patch) for steps in latent_steps]
        max_len = math.ceil(max(patched_steps) / pad_multiple) * pad_multiple
        latent_dim = int(cfg.patched_latent_dim)

        x_t = torch.zeros(
            (batch, max_len, latent_dim), device=self.device, dtype=self.dtype
        )
        latent_mask = torch.zeros(
            (batch, max_len), device=self.device, dtype=torch.bool
        )
        for i, (item, length) in enumerate(zip(items, patched_steps, strict=True)):
            generator = torch.Generator(device=self.device).manual_seed(int(item.seed))
            x_t[i, :length] = torch.randn(
                (1, length, latent_dim),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )[0]
            latent_mask[i, :length] = True

        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            _,
            _,
        ) = self._encode(normalized, refs)
        branches: list[tuple[str, float]] = []
        if self.settings.cfg_scale_text > 0:
            branches.append(("text", float(self.settings.cfg_scale_text)))
        cfg_scale_speaker = 0.0 if no_ref else float(self.settings.cfg_scale_speaker)
        if cfg_scale_speaker > 0:
            branches.append(("speaker", cfg_scale_speaker))
        mult = 1 + len(branches)

        def _uncond(
            name: str, which: str
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            state, mask = {
                "text": (text_state, text_mask),
                "speaker": (speaker_state, speaker_mask),
                "caption": (caption_state, caption_mask),
            }[which]
            if name != which or state is None or mask is None:
                return state, mask
            return torch.zeros_like(state), torch.zeros_like(mask)

        cfg_states: dict[str, list[torch.Tensor]] = {}
        for which in ("text", "speaker", "caption"):
            parts = [_uncond("cond", which)] + [
                _uncond(name, which) for name, _ in branches
            ]
            if parts[0][0] is None:
                continue
            cfg_states[which + "_state"] = [p[0] for p in parts]  # type: ignore[misc]
            cfg_states[which + "_mask"] = [p[1] for p in parts]  # type: ignore[misc]

        def _cat(name: str) -> torch.Tensor | None:
            values = cfg_states.get(name)
            return None if values is None else torch.cat(values, dim=0)

        cfg_text_state = _cat("text_state")
        cfg_speaker_state = _cat("speaker_state")
        cfg_caption_state = _cat("caption_state")
        kv_cond = self.model.build_context_kv_cache(
            text_state=text_state,
            speaker_state=speaker_state,
            caption_state=caption_state,
        )
        kv_cfg = self.model.build_context_kv_cache(
            text_state=cfg_text_state,
            speaker_state=cfg_speaker_state,
            caption_state=cfg_caption_state,
        )
        cfg_masks = {
            "text_mask": _cat("text_mask"),
            "speaker_mask": _cat("speaker_mask"),
            "caption_mask": _cat("caption_mask"),
        }
        cfg_latent_mask = latent_mask.repeat(mult, 1)

        num_steps = int(self.settings.num_steps)
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        t_schedule = (1.0 - u) * 0.999
        for step in range(num_steps):
            t = t_schedule[step]
            t_next = t_schedule[step + 1]
            tt = torch.full((batch,), t, device=self.device, dtype=self.dtype)
            use_cfg = bool(branches) and (
                self.settings.cfg_min_t <= t.item() <= self.settings.cfg_max_t
            )
            if use_cfg:
                v_out = self.model.forward_with_encoded_conditions(
                    x_t=x_t.repeat(mult, 1, 1),
                    t=tt.repeat(mult),
                    text_state=cfg_text_state,
                    text_mask=cfg_masks["text_mask"],
                    speaker_state=cfg_speaker_state,
                    speaker_mask=cfg_masks["speaker_mask"],
                    caption_state=cfg_caption_state,
                    caption_mask=cfg_masks["caption_mask"],
                    latent_mask=cfg_latent_mask,
                    context_kv_cache=kv_cfg,
                )
                chunks = v_out.chunk(mult, dim=0)
                v = chunks[0]
                for (_, scale), chunk in zip(branches, chunks[1:], strict=True):
                    v = v + scale * (chunks[0] - chunk)
            else:
                v = self.model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=tt,
                    text_state=text_state,
                    text_mask=text_mask,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask,
                    caption_state=caption_state,
                    caption_mask=caption_mask,
                    latent_mask=latent_mask,
                    context_kv_cache=kv_cond,
                )
            x_t = x_t + v * (t_next - t)

        latents: list[torch.Tensor] = []
        for i, steps in enumerate(latent_steps):
            z = unpatchify_latent(
                x_t[i : i + 1, : patched_steps[i]],
                patch_size=patch,
                latent_dim=cfg.latent_dim,
            )
            latents.append(z[0, :steps])
        return latents

    @torch.inference_mode()
    def decode(self, latents: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Decode, trim the flat tail, and watermark like ``InferenceRuntime.synthesize``."""
        audios: list[torch.Tensor] = []
        for z in latents:
            audio = self.runtime.codec.decode_latent(z[None]).cpu()[0]
            max_samples = int(z.shape[0]) * self.hop_length
            if self.settings.trim_tail:
                flattening = find_flattening_point(
                    z, window_size=20, std_threshold=0.05, mean_threshold=0.1
                )
                if flattening > 0:
                    max_samples = min(max_samples, int(flattening) * self.hop_length)
            audios.append(audio[:, :max_samples])
        if self.settings.watermark and self.runtime.watermarker.ready:
            audios = self.runtime.watermarker.encode_batch(
                audios, sample_rate=self.sample_rate
            )
        return audios

    def synthesize_batch(
        self, items: Sequence[SynthItem], steps: Sequence[int], *, pad_multiple: int = 1
    ) -> list[SynthOutput]:
        latents = self.sample_latents(items, steps, pad_multiple=pad_multiple)
        audios = self.decode(latents)
        _, max_frames = self._frame_bounds()
        return [
            SynthOutput(
                item=item,
                audio=audio,
                sample_rate=self.sample_rate,
                pred_frames=int(n),
                clamped=int(n) >= max_frames,
            )
            for item, audio, n in zip(items, audios, steps, strict=True)
        ]


def length_batches(
    order: Sequence[int],
    steps: Sequence[int],
    max_batch_size: int,
    max_batch_frames: int,
) -> Iterator[list[int]]:
    """Group length-sorted indices so that batch_size * max_length stays within budget."""
    batch: list[int] = []
    for index in order:
        candidate = batch + [index]
        if batch and (
            len(candidate) > max_batch_size
            or len(candidate) * max(steps[i] for i in candidate) > max_batch_frames
        ):
            yield batch
            candidate = [index]
        batch = candidate
    if batch:
        yield batch


# --------------------------------------------------------------------------- IO


def to_pcm16(
    audio: torch.Tensor, sample_rate: int, out_rate: int
) -> tuple[np.ndarray, int]:
    """(channels, samples) float -> mono int16 at ``out_rate``; returns (pcm, n_clipped)."""
    x = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
    x = x.mean(axis=0) if x.ndim == 2 and x.shape[0] > 1 else x.reshape(-1)
    if sample_rate != out_rate:
        x = soxr.resample(x, sample_rate, out_rate, quality="HQ")
    n_clipped = int(np.count_nonzero(np.abs(x) > 1.0))
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm, n_clipped


def write_wav_atomic(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    sf.write(str(tmp), pcm, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)


class JsonlLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def load_items(path: Path, ref_root: Path) -> list[SynthItem]:
    items: list[SynthItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("id", "text", "seed"):
                if field not in row:
                    raise ValueError(f"{path}:{lineno}: missing field {field!r}")
            if "ref_wav" not in row:
                raise ValueError(
                    f"{path}:{lineno}: missing field 'ref_wav' (use null for no ref)"
                )
            key = str(row["id"])
            if not key or "/" in key or key.startswith("."):
                raise ValueError(f"{path}:{lineno}: invalid id {key!r}")
            if key in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {key!r}")
            seen.add(key)
            ref = row["ref_wav"]
            if ref is not None:
                ref_path = Path(str(ref)).expanduser()
                ref = str(ref_path if ref_path.is_absolute() else ref_root / ref_path)
            seconds = row.get("seconds")
            items.append(
                SynthItem(
                    key=key,
                    text=str(row["text"]),
                    seed=int(row["seed"]),
                    ref_wav=ref,
                    seconds=None if seconds is None else float(seconds),
                )
            )
    return items


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="jsonl: id, text, ref_wav, seed"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="default: <output-dir>_logs"
    )
    parser.add_argument(
        "--ref-root", type=Path, default=Path.cwd(), help="base for relative ref_wav"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="model.safetensors or HF repo"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=["fp32", "bf16"], default="bf16", help="DiT precision"
    )
    parser.add_argument(
        "--text-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Text/caption encoder precision (bf16 costs ~0.5pt JKYB).",
    )
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--sdpa-backend", choices=["efficient", "cudnn"], default="efficient"
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument(
        "--no-watermark", action="store_true", help="skip the silentcipher watermark"
    )
    parser.add_argument("--no-trim-tail", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-frames", type=int, default=8192)
    parser.add_argument("--pad-multiple", type=int, default=1)
    parser.add_argument("--duration-batch-size", type=int, default=256)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Rows are length-sorted and batched within chunks of this size.",
    )
    parser.add_argument("--out-sample-rate", type=int, default=24000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir: Path = args.log_dir or out_dir.with_name(out_dir.name + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.input, args.ref_root)[args.shard :: args.num_shards]
    if args.limit is not None:
        items = items[: args.limit]
    todo = [item for item in items if not (out_dir / f"{item.key}.wav").is_file()]

    settings = SamplingSettings(
        num_steps=args.num_steps,
        cfg_scale_text=args.cfg_scale_text,
        cfg_scale_speaker=args.cfg_scale_speaker,
        cfg_min_t=args.cfg_min_t,
        cfg_max_t=args.cfg_max_t,
        max_seconds=args.max_seconds,
        min_seconds=args.min_seconds,
        trim_tail=not args.no_trim_tail,
        watermark=not args.no_watermark,
    )
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["settings"] = asdict(settings)
    (log_dir / f"generation_config_shard{args.shard}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"rows={len(items)} todo={len(todo)} out={out_dir}", flush=True)
    if not todo:
        print("done", flush=True)
        return 0

    set_sdpa_backend(args.sdpa_backend)
    runtime = load_runtime(
        args.checkpoint,
        device=args.device,
        precision=args.precision,
        text_precision=args.text_precision,
        codec_precision=args.codec_precision,
    )
    synth = MultiRefSynthesizer(runtime, settings)
    tokenizer = runtime.tokenizer

    failures = JsonlLog(log_dir / f"failures_shard{args.shard}.jsonl")
    meta = JsonlLog(log_dir / f"meta_shard{args.shard}.jsonl")

    def fail(item: SynthItem, stage: str, error: str) -> None:
        failures.write(
            {
                "id": item.key,
                "ref_wav": item.ref_wav,
                "seed": item.seed,
                "stage": stage,
                "error": error,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    start = time.perf_counter()
    stats = {"done": 0, "failed": 0, "audio_sec": 0.0}
    last_report = [0.0]

    def report(force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - last_report[0] < 30.0:
            return
        last_report[0] = now
        elapsed = now - start
        done = stats["done"]
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(todo) - done - stats["failed"]) / rate if rate > 0 else float("nan")
        print(
            f"progress {done + stats['failed']}/{len(todo)} ok={done} failed={stats['failed']} "
            f"elapsed={elapsed:.0f}s rate={rate:.2f}/s audio_rate={stats['audio_sec'] / elapsed:.1f}s/s "
            f"eta={eta:.0f}s",
            flush=True,
        )

    def save(output: SynthOutput) -> None:
        item = output.item
        pcm, n_clipped = to_pcm16(
            output.audio, output.sample_rate, args.out_sample_rate
        )
        seconds = len(pcm) / args.out_sample_rate
        if len(pcm) < int(0.1 * args.out_sample_rate):
            fail(item, "output", f"audio too short ({seconds:.3f}s)")
            stats["failed"] += 1
            return
        write_wav_atomic(out_dir / f"{item.key}.wav", pcm, args.out_sample_rate)
        n_tokens = int(
            tokenizer.encode(normalize_text(item.text).strip()).numel()
        )  # incl. BOS
        meta.write(
            {
                "id": item.key,
                "ref_wav": item.ref_wav,
                "seed": item.seed,
                "seconds": round(seconds, 3),
                "pred_frames": output.pred_frames,
                "duration_clamped": output.clamped,
                "text_tokens": n_tokens,
                "text_truncated": n_tokens > synth.text_max_len,
                "n_clipped": n_clipped,
            }
        )
        stats["done"] += 1
        stats["audio_sec"] += seconds

    def run_batch(batch: list[SynthItem], steps: list[int]) -> None:
        try:
            outputs = synth.synthesize_batch(
                batch, steps, pad_multiple=args.pad_multiple
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            if len(batch) == 1:
                fail(batch[0], "sample", f"{type(exc).__name__}: {exc}")
                stats["failed"] += 1
                return
            print(
                f"batch of {len(batch)} failed ({type(exc).__name__}: {exc}); retrying singly",
                flush=True,
            )
            for item, n in zip(batch, steps, strict=True):
                run_batch([item], [n])
            return
        for output in outputs:
            try:
                save(output)
            except Exception as exc:  # noqa: BLE001
                fail(output.item, "write", f"{type(exc).__name__}: {exc}")
                stats["failed"] += 1

    for chunk_start in range(0, len(todo), args.chunk_size):
        chunk = todo[chunk_start : chunk_start + args.chunk_size]
        # Validate rows and references up front.
        valid: list[SynthItem] = []
        for item in chunk:
            if not normalize_text(item.text).strip():
                fail(item, "input", "empty text after normalization")
                stats["failed"] += 1
                continue
            try:
                synth.reference(item.ref_wav)
            except RefLoadError as exc:
                fail(item, "reference", str(exc))
                stats["failed"] += 1
                continue
            valid.append(item)
        # Duration prediction, then length-sorted batches per reference mode.
        steps: list[int] = []
        for i in range(0, len(valid), args.duration_batch_size):
            sub = valid[i : i + args.duration_batch_size]
            try:
                steps.extend(synth.predict_latent_steps(sub))
            except Exception:  # noqa: BLE001
                for item in sub:
                    try:
                        steps.extend(synth.predict_latent_steps([item]))
                    except Exception as exc:  # noqa: BLE001
                        fail(item, "duration", f"{type(exc).__name__}: {exc}")
                        stats["failed"] += 1
                        steps.append(-1)
        for no_ref in (False, True):
            idx = [
                i
                for i, item in enumerate(valid)
                if (item.ref_wav is None) == no_ref and steps[i] > 0
            ]
            order = sorted(idx, key=lambda i: steps[i])
            for batch_idx in length_batches(
                order, steps, args.batch_size, args.max_batch_frames
            ):
                run_batch([valid[i] for i in batch_idx], [steps[i] for i in batch_idx])
                report()
    report(force=True)
    failures.close()
    meta.close()
    elapsed = time.perf_counter() - start
    summary = {
        "shard": args.shard,
        "rows": len(items),
        "todo": len(todo),
        "ok": stats["done"],
        "failed": stats["failed"],
        "elapsed_sec": round(elapsed, 1),
        "sentences_per_sec": round(stats["done"] / elapsed, 3),
        "audio_sec": round(stats["audio_sec"], 1),
        "audio_sec_per_sec": round(stats["audio_sec"] / elapsed, 2),
    }
    (log_dir / f"summary_shard{args.shard}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("summary " + json.dumps(summary), flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
