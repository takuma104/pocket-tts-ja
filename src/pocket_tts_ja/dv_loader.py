"""Latent-only data loading for the DACVAE students (plan §2, P2).

Drop-in replacement for pocket-tts' `training.dataloader.DataLoader` / `SubprocessDataLoader`
(same constructor arguments, same `Batch` output), like main's `pocket_tts_ja.loader`, so the
upstream training loop runs unchanged (scripts/train/train_student.py swaps them in).

One sample = the teacher's latent sequence of one sentence (25 Hz, 32 dims, unnormalised) + its
transcript + the voice prompt the teacher was given: the DACVAE latent of the reference wav at
-16 dB (scripts/teacher/encode_refs.py), whole, as the teacher saw it.

The target is the latent up to the speech end (scripts/teacher/pack.py) plus tail_frames after
it: what the teacher made after the speech where it made anything, then DACVAE's latent of
digital silence (dv_silence_latent.npy). In practice it is almost all silence latent: Irodori's
duration predictor fits the window to the speech, whose last mora decays in the final 2-3 stored
frames (s3: median 0 frames after the speech end). The mask says which is which
(uint8: 1 speech, 2 tail, 0 padding; pocket_tts_ja.post_eos): the EOS target is the first tail
frame, and the flow loss covers the tail too, so the frames the wheel keeps generating after EOS
are supervised.

Manifest lines (scripts/train/make_split.py): {"id", "transcript", "speaker", "ref_file",
"lat_file", "lat_start", "lat_len", "speech_end", ...}; lat_file and ref_file are relative to the
manifest's directory, lat_file holds float16 [frames, 32] arrays back to back.
"""

import json
import logging
import multiprocessing.queues
import queue
import random
import signal
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as torch_mp

from pocket_tts.modules.text_conditioner import Tokenizer, encoder_from_serialized
from training.dataloader.loader import DataLoader
from training.dataloader.types import Batch

logger = logging.getLogger(__name__)

TAIL_FRAMES = 12  # 0.48 s after the speech end
MIN_PROMPT_FRAMES = 25  # 1 s, the least prompt_trim_max_sec leaves
SILENCE_LATENT = Path(__file__).with_name("dv_silence_latent.npy")


@dataclass
class DvBatch(Batch):
    # [B] frames of each target before its tail
    num_speech_frames: torch.Tensor | None = None


# One memory map per latent file for the whole process: the trainer builds a new validation
# loader at every validation (see main's loader).
_MMAPS: dict[str, npt.NDArray[np.float16]] = {}
_REFS: dict[str, npt.NDArray[np.float16]] = {}


class _Corpus:
    """All manifest rows as compact arrays."""

    def __init__(self, jsonl: str):
        root = Path(jsonl).parent
        files: dict[str, int] = {}
        refs: dict[str, int] = {}
        file_idx, ref_idx, start, length, end = [], [], [], [], []
        texts: list[str] = []
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                file_idx.append(files.setdefault(d["lat_file"], len(files)))
                ref_idx.append(refs.setdefault(d["ref_file"], len(refs)))
                start.append(d["lat_start"])
                length.append(d["lat_len"])
                end.append(d["speech_end"])
                texts.append(d["transcript"])
        self.paths = [str(root / p) for p in files]
        self.ref_paths = [str(root / p) for p in refs]
        self.file_idx = np.asarray(file_idx, dtype=np.int32)
        self.ref_idx = np.asarray(ref_idx, dtype=np.int32)
        self.start = np.asarray(start, dtype=np.int64)
        self.length = np.asarray(length, dtype=np.int32)
        self.speech_end = np.asarray(end, dtype=np.int32)
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def latents(self, i: int, frames: int) -> npt.NDArray[np.float16]:
        """The first `frames` stored frames of row i (fewer if the row is shorter)."""
        path = self.paths[int(self.file_idx[i])]
        if path not in _MMAPS:
            _MMAPS[path] = np.load(path, mmap_mode="r")
        s = int(self.start[i])
        return _MMAPS[path][s : s + min(frames, int(self.length[i]))]

    def prompt(self, i: int) -> npt.NDArray[np.float16]:
        path = self.ref_paths[int(self.ref_idx[i])]
        if path not in _REFS:
            _REFS[path] = np.load(path)
        return _REFS[path]


class DvLatentDataLoader(DataLoader):
    def __init__(
        self,
        jsonl: str,
        tokenize: Callable[[str], list[int]],
        batch_size: int,
        sample_rate: int,
        frame_rate: float,
        max_duration_sec: float,
        max_voice_prompt_sec: float,
        rank: int,
        world_size: int,
        seed: int = 0,
        shuffle: bool = True,
        io_workers: int = 16,
        num_bucket_batches: int = 1,
        prompt_trim_max_sec: float = 0.0,
        final_punct_dropout: float = 0.0,
        tail_frames: int = TAIL_FRAMES,
    ):
        # Deliberately not calling DataLoader.__init__: it indexes audio manifests.
        self.jsonl = jsonl
        self.corpus = _Corpus(jsonl)
        self.entries = list(range(rank, len(self.corpus), world_size))
        assert self.entries, f"no entries for rank {rank} in {jsonl}"
        self.tokenize = tokenize
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.max_duration_sec = max_duration_sec
        self.max_voice_prompt_sec = max_voice_prompt_sec
        self.prompt_trim_max_sec = prompt_trim_max_sec
        self.final_punct_dropout = final_punct_dropout
        self.shuffle = shuffle
        self.io_workers = io_workers
        self.num_bucket_batches = num_bucket_batches
        self._failures = 0
        self.rng = random.Random(seed)
        self.stitch_frames = 0
        self.max_target_frames = int(max_duration_sec * frame_rate)
        cap = max_voice_prompt_sec if max_voice_prompt_sec > 0 else 30.0
        self.max_prompt_frames = int(cap * frame_rate)
        self.tail_frames = tail_frames
        self.silence = np.load(SILENCE_LATENT)
        assert len(self.silence) >= tail_frames, f"{SILENCE_LATENT} has {len(self.silence)} frames"

    def get_entry(self, index: int) -> int:  # type: ignore[override]
        return self.entries[index]

    def _prompt(self, i: int) -> npt.NDArray[np.float16]:
        prompt = self.corpus.prompt(i)[: self.max_prompt_frames]
        if self.prompt_trim_max_sec > 0:
            drop = self.rng.randint(0, int(round(self.prompt_trim_max_sec * self.frame_rate)))
            prompt = prompt[: max(prompt.shape[0] - drop, min(prompt.shape[0], MIN_PROMPT_FRAMES))]
        return prompt

    def _sample(self, entry: int) -> tuple[Any, ...]:  # type: ignore[override]
        i = int(entry)
        speech = int(self.corpus.speech_end[i])
        want = speech + self.tail_frames
        target = np.asarray(self.corpus.latents(i, want))
        if len(target) < want:
            target = np.concatenate([target, self.silence[: want - len(target)]])
        target = target[: self.max_target_frames]
        tokens = self._tokens(self.corpus.texts[i])
        return (
            tokens,
            torch.from_numpy(np.asarray(self._prompt(i), dtype=np.float32)),
            torch.from_numpy(np.asarray(target, dtype=np.float32)),
            min(speech, len(target)),
        )

    def _row_len(self, sample: tuple[Any, ...]) -> float:
        tokens, prompt, target, _ = sample
        return prompt.shape[0] + len(tokens) + target.shape[0]

    def _collate(self, batch: list[tuple[Any, ...]]) -> Batch:
        tokens, prompts, targets, speech = zip(*batch, strict=True)
        return DvBatch(
            audio=torch.zeros(len(batch), 1, 0),
            num_audio_frames=torch.tensor([t.shape[0] for t in targets], dtype=torch.long),
            text_tokens=list(tokens),
            voice_audio=torch.zeros(len(batch), 1, 0),
            num_voice_prompt_frames=torch.tensor([max(1, p.shape[0]) for p in prompts], dtype=torch.long),
            tail_latents=self._pad_latents(targets, 0),
            prompt_latents=self._pad_latents(prompts, 1),
            num_speech_frames=torch.tensor(speech, dtype=torch.long),
        )


@torch.no_grad()
def encode_batch_latents(
    mimi: torch.nn.Module, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`training.dataloader.encode_batch` for latent-only batches: the mask is uint8, 1 on the
    speech, 2 on the tail after it, 0 on padding (pocket_tts_ja.post_eos)."""
    assert batch.tail_latents is not None and batch.prompt_latents is not None
    latents = batch.tail_latents.to(device).float()
    T = latents.shape[1]
    n = batch.num_audio_frames.to(device).clamp(max=T)
    t = torch.arange(T, device=device)[None, :]
    mask = t < n[:, None]
    speech = getattr(batch, "num_speech_frames", None)
    if speech is not None:
        speech = speech.to(device).clamp(max=T)
        mask = mask.to(torch.uint8) + (t >= speech[:, None]).to(torch.uint8) * mask
    voice = batch.prompt_latents.to(device).float()
    n_voice = batch.num_voice_prompt_frames.to(device).clamp(max=voice.shape[1])
    return latents, mask, voice, n_voice


def _ignore_stop_signals() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def _feed_queue(
    q: "multiprocessing.queues.Queue[Batch]",
    serialized_tokenizer: tuple[str, bytes],
    loader_kwargs: dict[str, Any],
) -> None:
    _ignore_stop_signals()
    torch_mp.set_sharing_strategy("file_system")
    loader = DvLatentDataLoader(tokenize=encoder_from_serialized(*serialized_tokenizer), **loader_kwargs)
    for batch in loader:
        q.put(batch)


class DvSubprocessDataLoader:
    """Same interface as training.dataloader.SubprocessDataLoader, with DvLatentDataLoader."""

    def __init__(
        self,
        jsonl: str,
        sentence_piece: Tokenizer,
        batch_size: int,
        sample_rate: int,
        frame_rate: float,
        max_duration_sec: float,
        max_voice_prompt_sec: float,
        rank: int,
        world_size: int,
        seed: int = 0,
        shuffle: bool = True,
        io_workers: int = 16,
        num_procs: int = 3,
        depth: int = 8,
        num_bucket_batches: int = 1,
        prompt_trim_max_sec: float = 0.0,
        final_punct_dropout: float = 0.0,
        tail_frames: int = TAIL_FRAMES,
    ):
        ctx = torch_mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=depth)
        self._procs = []
        for i in range(num_procs):
            loader_kwargs = {
                "jsonl": jsonl,
                "batch_size": batch_size,
                "sample_rate": sample_rate,
                "frame_rate": frame_rate,
                "max_duration_sec": max_duration_sec,
                "max_voice_prompt_sec": max_voice_prompt_sec,
                "rank": rank * num_procs + i,
                "world_size": world_size * num_procs,
                "seed": seed + rank * num_procs + i,
                "shuffle": shuffle,
                "io_workers": io_workers,
                "num_bucket_batches": num_bucket_batches,
                "prompt_trim_max_sec": prompt_trim_max_sec,
                "final_punct_dropout": final_punct_dropout,
                "tail_frames": tail_frames,
            }
            proc = ctx.Process(
                target=_feed_queue,
                args=(self._queue, sentence_piece.serialize(), loader_kwargs),
                daemon=True,
                name=f"dv-dataloader-{i}",
            )
            proc.start()
            self._procs.append(proc)

    def close(self) -> None:
        for p in self._procs:
            if p.is_alive():
                p.kill()
        for p in self._procs:
            p.join(timeout=10)

    def __iter__(self) -> Iterator[Batch]:
        while True:
            try:
                batch = self._queue.get(timeout=60)
            except queue.Empty:
                dead = [p for p in self._procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"dataloader subprocess(es) died (exit codes {[p.exitcode for p in dead]})"
                    ) from None
                continue
            yield batch
