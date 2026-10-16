"""Latent-only data loading with voice prompts from other utterances of the same speaker.

Drop-in replacement for pocket-tts' `training.dataloader.DataLoader` /
`SubprocessDataLoader` (same constructor arguments, same `Batch` output), so the upstream
training loop runs unchanged (see scripts/train_ja.py, which swaps them in).

One sample = target utterance i (its full precomputed Mimi latents, trailing silence
trimmed, capped at max_duration_sec) + its whole transcript + a voice prompt built from
*other* utterances j != i of the same speaker label (one diarized speaker in one video):

  50%  the first 3–10 s of one other utterance
  35%  one other utterance in full (capped at max_voice_prompt_sec, 30 s if <= 0)
  15%  two other utterances back to back, capped likewise (long prompts: upstream #344
       found students trained only on 5 s prompts stop early on prompts > 12 s)

then the upstream prompt-trim augmentation (drop 0..prompt_trim_max_sec off the end).
Nothing is read from audio files: the Mimi encoder is causal, so a prefix of the stored
latents equals a cold encode of that prefix, and no stitch re-encoding is needed.

post_eos_frames > 0 appends that many frames of digital silence (silence_latent.npy,
scripts/make_silence_latent.py) to every target that is not cut at max_duration_sec, and the
batch says where the speech ends (JaBatch.num_speech_frames). The EOS target stays on the first
frame after the speech; the flow loss also covers the silence (pocket_tts_ja.post_eos). The
stock wheel keeps generating a few frames after EOS (5 for Japanese text), and without this
nothing ever showed the model what they should be.

Manifest lines (scripts/finalize_emilia_ja.py): {"id", "transcript", "speaker",
"lat_file", "lat_start", "lat_len", "speech_end", ...}; lat_file is relative to the
manifest's directory and holds float16 [frames, 32] arrays.
"""

import json
import logging
import multiprocessing.queues
import queue
import random
import signal
from collections import defaultdict
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

TRAIL_FRAMES = 3  # keep ~0.24 s after the detected speech end, so EOS has a consistent target
PROMPT_MODES = (("prefix", 0.50), ("full", 0.35), ("concat", 0.15))
PREFIX_SEC = (3.0, 10.0)
MIN_PROMPT_FRAMES = 13  # ~1 s
# Utterances under 2 s are targets only: the very short synthetic clips (S9) share their speaker
# labels with the long ones and would otherwise become a fifth of those speakers' voice prompts.
MIN_PROMPT_SOURCE_FRAMES = 25

SILENCE_LATENT = Path(__file__).with_name("silence_latent.npy")


@dataclass
class JaBatch(Batch):
    # [B] frames of each target before the appended silence (post_eos_frames); None without it
    num_speech_frames: torch.Tensor | None = None


# One memory map per latent file for the whole process: the trainer builds a new validation
# loader at every validation and never drops the old one (its prefetch thread stays blocked),
# so per-loader maps leak one file descriptor per file and validation.
_MMAPS: dict[str, npt.NDArray[np.float16]] = {}


class _Corpus:
    """All manifest rows as compact arrays, plus speaker -> row indices."""

    def __init__(self, jsonl: str):
        root = Path(jsonl).parent
        files: dict[str, int] = {}
        file_idx, start, length, end, spk = [], [], [], [], []
        texts: list[str] = []
        speakers: dict[str, int] = {}
        with open(jsonl) as f:
            for line in f:
                d = json.loads(line)
                file_idx.append(files.setdefault(d["lat_file"], len(files)))
                start.append(d["lat_start"])
                length.append(d["lat_len"])
                end.append(d.get("speech_end", d["lat_len"]))
                spk.append(speakers.setdefault(d["speaker"], len(speakers)))
                texts.append(d["transcript"])
        self.paths = [str(root / p) for p in files]
        self.file_idx = np.asarray(file_idx, dtype=np.int32)
        self.start = np.asarray(start, dtype=np.int64)
        self.length = np.asarray(length, dtype=np.int32)
        self.speech_end = np.asarray(end, dtype=np.int32)
        self.speaker = np.asarray(spk, dtype=np.int32)
        self.texts = texts
        by_spk: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(self.speaker.tolist()):
            by_spk[s].append(i)
        self.by_speaker = {s: np.asarray(v, dtype=np.int64) for s, v in by_spk.items()}
        usable = np.minimum(self.length, self.speech_end + TRAIL_FRAMES) >= MIN_PROMPT_SOURCE_FRAMES
        self.prompt_pool = {s: v[usable[v]] for s, v in self.by_speaker.items()}

    def __len__(self) -> int:
        return len(self.texts)

    def latents(self, i: int, trim_silence: bool = True) -> npt.NDArray[np.float16]:
        path = self.paths[int(self.file_idx[i])]
        if path not in _MMAPS:
            _MMAPS[path] = np.load(path, mmap_mode="r")
        n = int(self.length[i])
        if trim_silence:
            n = min(n, int(self.speech_end[i]) + TRAIL_FRAMES)
        s = int(self.start[i])
        return _MMAPS[path][s : s + max(1, n)]

    def others(self, i: int) -> npt.NDArray[np.int64]:
        """Rows that can prompt row i: the speaker's other utterances of 2 s or more (any other
        one if there is none)."""
        spk = int(self.speaker[i])
        cands = self.prompt_pool[spk]
        cands = cands[cands != i]
        if len(cands) == 0:
            cands = self.by_speaker[spk]
            cands = cands[cands != i]
        return cands


class JaLatentDataLoader(DataLoader):
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
        post_eos_frames: int = 0,
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
        self.post_eos_frames = post_eos_frames
        self.silence: npt.NDArray[np.float16] | None = None
        if post_eos_frames > 0:
            self.silence = np.load(SILENCE_LATENT)
            assert len(self.silence) >= post_eos_frames, f"{SILENCE_LATENT} has {len(self.silence)} frames"
            self.silence = self.silence[:post_eos_frames]

    def get_entry(self, index: int) -> int:  # type: ignore[override]
        return self.entries[index]

    def _choose_mode(self) -> str:
        r = self.rng.random()
        for name, p in PROMPT_MODES:
            if r < p:
                return name
            r -= p
        return PROMPT_MODES[-1][0]

    def _prompt(self, i: int) -> npt.NDArray[np.float16]:
        others = self.corpus.others(i)
        if len(others) == 0:
            raise ValueError(f"row {i}: speaker has no other utterance")
        mode = self._choose_mode()
        if mode == "concat" and len(others) >= 2:
            j, k = self.rng.sample(others.tolist(), 2)
            prompt = np.concatenate([self.corpus.latents(j), self.corpus.latents(k)])
        else:
            prompt = self.corpus.latents(int(self.rng.choice(others.tolist())))
            if mode == "prefix":
                n = int(self.rng.uniform(*PREFIX_SEC) * self.frame_rate)
                prompt = prompt[: max(MIN_PROMPT_FRAMES, n)]
        prompt = prompt[: self.max_prompt_frames]
        if self.prompt_trim_max_sec > 0:
            drop = self.rng.randint(0, int(round(self.prompt_trim_max_sec * self.frame_rate)))
            keep = max(prompt.shape[0] - drop, min(prompt.shape[0], MIN_PROMPT_FRAMES))
            prompt = prompt[:keep]
        return prompt

    def _sample(self, entry: int) -> tuple[Any, ...]:  # type: ignore[override]
        i = int(entry)
        full = self.corpus.latents(i)
        target = full[: self.max_target_frames]
        speech = len(target)
        if self.silence is not None and len(target) == len(full):
            target = np.concatenate([target, self.silence])
        tokens = self._tokens(self.corpus.texts[i])
        prompt = self._prompt(i)
        return (
            tokens,
            torch.from_numpy(np.asarray(prompt, dtype=np.float32)),
            torch.from_numpy(np.asarray(target, dtype=np.float32)),
            speech,
        )

    def _row_len(self, sample: tuple[Any, ...]) -> float:
        tokens, prompt, target, _ = sample
        return prompt.shape[0] + len(tokens) + target.shape[0]

    def _collate(self, batch: list[tuple[Any, ...]]) -> Batch:
        tokens, prompts, targets, speech = zip(*batch, strict=True)
        return JaBatch(
            audio=torch.zeros(len(batch), 1, 0),
            num_audio_frames=torch.tensor([t.shape[0] for t in targets], dtype=torch.long),
            text_tokens=list(tokens),
            voice_audio=torch.zeros(len(batch), 1, 0),
            num_voice_prompt_frames=torch.tensor(
                [max(1, p.shape[0]) for p in prompts], dtype=torch.long
            ),
            tail_latents=self._pad_latents(targets, 0),
            prompt_latents=self._pad_latents(prompts, 1),
            num_speech_frames=(
                torch.tensor(speech, dtype=torch.long) if self.silence is not None else None
            ),
        )


@torch.no_grad()
def encode_batch_latents(
    mimi: torch.nn.Module, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`training.dataloader.encode_batch` for latent-only batches (no stitch audio).

    With appended silence (JaBatch.num_speech_frames) the mask is uint8 instead of bool:
    1 on the speech, 2 on the silence after it, 0 on padding (pocket_tts_ja.post_eos)."""
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


def _ignore_stop_signals():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def _feed_queue(
    q: "multiprocessing.queues.Queue[Batch]",
    serialized_tokenizer: tuple[str, bytes],
    loader_kwargs: dict[str, Any],
):
    _ignore_stop_signals()
    torch_mp.set_sharing_strategy("file_system")
    loader = JaLatentDataLoader(
        tokenize=encoder_from_serialized(*serialized_tokenizer), **loader_kwargs
    )
    for batch in loader:
        q.put(batch)


class JaSubprocessDataLoader:
    """Same interface as training.dataloader.SubprocessDataLoader, with JaLatentDataLoader."""

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
        post_eos_frames: int = 0,
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
                "post_eos_frames": post_eos_frames,
            }
            proc = ctx.Process(
                target=_feed_queue,
                args=(self._queue, sentence_piece.serialize(), loader_kwargs),
                daemon=True,
                name=f"ja-dataloader-{i}",
            )
            proc.start()
            self._procs.append(proc)

    def close(self):
        for p in self._procs:
            if p.is_alive():
                p.kill()
        for p in self._procs:
            p.join(timeout=10)

    def __iter__(self) -> Iterator[Batch]:
        batches = 0
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
            batches += 1
            yield batch
