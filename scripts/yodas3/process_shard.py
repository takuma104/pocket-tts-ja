"""Y1: turn one YODAS3 ja shard (whole YouTube videos + captions) into TTS utterances.

Per video (CPU worker pool):
  1. decode the webm/opus member (PyAV), downmix to mono, resample to 24 kHz and 16 kHz
  2. silero VAD (16 kHz) and greedy merging of speech regions into 3–20 s utterances,
     cutting at pauses >= 0.5 s (shorter gaps are merged while the utterance stays <= 20 s)
  3. attach the caption text: YouTube ASR captions carry per-word start times, a word goes
     to the utterance its start falls in; uploader subtitles are attached by >= 50% overlap
  4. the speech end (last 80 ms frame within 35 dB of peak)
Per batch (GPU, main process):
  5. whisper-large-v3-turbo transcription (the label candidate, and the second opinion for
     the caption-agreement filter applied later by finalize_yodas3.py) — skipped for
     DNSMOS OVRL < 2.9 and for voices close to --tts-blocklist (known TTS channels)
  6. ECAPA speaker embedding (per-video speaker clustering happens in finalize)
  7. Mimi latents (fp32/TF32, right-padded batches are exact since the encoder is causal)
  8. DNSMOS P.835 (SIG/BAK/OVRL), converted to torch (pocket_tts_ja.dnsmos.TorchDNSMOS)

Utterances without caption words are dropped before the GPU stages. Outputs (resumable,
one set per shard) go to --out:
  latents/<shard>.npy  float16 [frames, 32]     ecapa/<shard>.npy float16 [utts, 192]
  utts/<shard>.jsonl   one row per utterance (video, start, end, caption, whisper, dnsmos,
                       lat_start, lat_len, speech_end, caption_kind, ...)

    uv run python scripts/yodas3/process_shard.py --shard 0000 \
        --root /mnt/artifacts/yodas3/data/ja --out data/yodas3
"""

import argparse
import io
import json
import multiprocessing as mp
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import soxr
import torch

SR24, SR16 = 24000, 16000
MAX_VIDEO_SEC = 4 * 3600.0  # multi-hour streams (mostly music/chat) are skipped
WINDOW_SEC = 600.0  # videos are decoded and segmented in 10-minute windows
FRAME = 1920
MIN_UTT, MAX_UTT, CUT_GAP = 3.0, 20.0, 0.5
PAD = 0.1

_vad: Any = None


def _init_worker():
    global _vad
    torch.set_num_threads(1)
    import silero_vad

    _vad = silero_vad.load_silero_vad()


def decode_windows(data: bytes, win_sec: float = WINDOW_SEC) -> Iterator[tuple[float, npt.NDArray[np.float32], int]]:
    """Decode a webm/opus member to mono float32 in consecutive windows of win_sec seconds:
    (window start in seconds, samples, sample rate). Bounds a worker's memory to one window
    whatever the video length."""
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        chunks: list[npt.NDArray[np.float32]] = []
        n, start = 0, 0.0
        for frame in container.decode(stream):
            a = frame.to_ndarray()
            if a.ndim == 2 and a.shape[0] > 1 and frame.format.is_planar:
                a = a.mean(axis=0)
            elif a.ndim == 2 and not frame.format.is_planar:
                a = a.reshape(-1, len(frame.layout.channels)).mean(axis=1)
            else:
                a = a.reshape(-1)
            chunks.append(a.astype(np.float32))
            n += len(chunks[-1])
            if n >= win_sec * sr:
                yield start, np.concatenate(chunks), sr
                start += n / sr
                chunks, n = [], 0
        if chunks:
            yield start, np.concatenate(chunks), sr


def speech_end_frames(wav: npt.NDArray[np.float32]) -> int:
    n = len(wav) // FRAME
    if n == 0:
        return 0
    rms = np.sqrt((wav[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    above = np.nonzero(db >= max(db.max() - 35.0, -55.0))[0]
    return int(above[-1]) + 1 if len(above) else n


def merge_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    utts: list[tuple[float, float]] = []
    cur: list[float] | None = None
    for s, e in regions:
        if cur is not None and s - cur[1] < CUT_GAP and e - cur[0] <= MAX_UTT:
            cur[1] = e
            continue
        if cur is not None:
            utts.append((cur[0], cur[1]))
        cur = [s, e]
    if cur is not None:
        utts.append((cur[0], cur[1]))
    out = []
    for s, e in utts:
        if e - s > MAX_UTT:  # one long region with no pause: split evenly
            k = int(np.ceil((e - s) / MAX_UTT))
            step = (e - s) / k
            out += [(s + i * step, s + (i + 1) * step) for i in range(k)]
        elif e - s >= MIN_UTT:
            out.append((s, e))
    return out


def join_words(words: list[str]) -> str:
    text = ""
    for w in words:
        if text and text[-1].isascii() and text[-1].isalnum() and w[:1].isascii() and w[:1].isalnum():
            text += " "
        text += w
    return text


def captions(transcript: str | None) -> tuple[str, list[tuple[float, float, str]]]:
    """(kind, [(start_s, end_s, text)]) — one entry per word for ASR tracks."""
    if not transcript:
        return "none", []
    segs = json.loads(transcript)
    if any(s.get("words") for s in segs):
        words = []
        for s in segs:
            for w in s.get("words") or []:
                t = (s["start"] + w["t"]) / 1000
                words.append((t, t, w["w"]))
        words.sort(key=lambda x: x[0])
        return "asr", words
    return "manual", [
        (s["start"] / 1000, (s["start"] + s["duration"]) / 1000, s["text"]) for s in segs
    ]


def caption_text(kind: str, caps: list[tuple[float, float, str]], s: float, e: float) -> str:
    if kind == "asr":
        return join_words([w for t, _, w in caps if s - 0.25 <= t < e - 0.1])
    picked = []
    for cs, ce, text in caps:
        ov = min(ce, e) - max(cs, s)
        if ce > cs and ov / (ce - cs) >= 0.5:
            picked.append(text.replace("\n", " "))
    return " ".join(picked)


def process_video(job: tuple[str, bytes, dict[str, Any]]) -> list[dict[str, Any]]:
    import silero_vad

    vid, data, meta = job
    kind, caps = captions(meta.get("transcript"))
    out: list[dict[str, Any]] = []
    k = 0
    try:
        for w_start, wav, sr in decode_windows(data):
            if len(wav) < sr * MIN_UTT:
                continue
            w24 = soxr.resample(wav, sr, SR24).astype(np.float32)
            w16 = soxr.resample(wav, sr, SR16).astype(np.float32)
            del wav
            ts = silero_vad.get_speech_timestamps(
                torch.from_numpy(w16), _vad, sampling_rate=SR16, min_silence_duration_ms=300,
                min_speech_duration_ms=250, speech_pad_ms=30,
            )
            regions = [(t["start"] / SR16, t["end"] / SR16) for t in ts]
            total = len(w24) / SR24
            window: list[dict[str, Any]] = []
            for s, e in merge_regions(regions):
                s, e = max(0.0, s - PAD), min(total, e + PAD)
                text = caption_text(kind, caps, w_start + s, w_start + e)
                if not text.strip():
                    continue
                a24 = w24[int(s * SR24) : int(e * SR24)]
                if not len(a24) or float(np.abs(a24).max()) < 1e-3:
                    continue
                window.append(
                    {
                        "id": f"{vid}_{k:04d}", "video": vid, "start": round(w_start + s, 3),
                        "end": round(w_start + e, 3), "duration": round(e - s, 3), "caption": text,
                        "caption_kind": kind, "speech_end": speech_end_frames(a24),
                        "a24": a24.astype(np.float16),
                    }
                )
                k += 1
            out += window
    except Exception:  # noqa: BLE001 - corrupt or truncated member: keep what was decoded
        pass
    return out


def iter_members(tar_path: Path, meta: dict[str, dict[str, Any]]) -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    with open(tar_path, "rb", buffering=16 * 1024 * 1024) as f, tarfile.open(fileobj=f, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".webm"):
                continue
            vid = Path(m.name).stem
            fobj = tf.extractfile(m)
            if fobj is None or vid not in meta:
                continue
            yield vid, fobj.read(), meta[vid]


class GPUStages:
    def __init__(
        self, device: torch.device, asr_model: str, blocklist: Path | None = None,
        block_cos: float = 0.9, min_dnsmos: float = 2.9, max_padded_sec: float = 120.0,
    ):
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config

        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.mimi = load_mimi(load_pocket_config(), device)
        self.ecapa = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": str(device)},
            savedir="data/models/spkrec-ecapa-voxceleb",
        )
        from pocket_tts_ja.dnsmos import TorchDNSMOS

        self.dnsmos = TorchDNSMOS(device)
        self.blocklist = np.load(blocklist).astype(np.float32) if blocklist and blocklist.exists() else None
        self.block_cos = block_cos
        self.min_dnsmos = min_dnsmos
        self.max_padded_sec = max_padded_sec
        self.proc = WhisperProcessor.from_pretrained(asr_model)
        self.asr = WhisperForConditionalGeneration.from_pretrained(
            asr_model, dtype=torch.float16, attn_implementation="sdpa"
        ).to(device).eval()

    @torch.no_grad()
    def __call__(self, utts: list[dict[str, Any]]):
        for u in utts:
            u["a24"] = u["a24"].astype(np.float32)
            u["a16"] = soxr.resample(u["a24"], SR24, SR16).astype(np.float32)
        lats: list[npt.NDArray[np.float16]] = []
        embs: list[npt.NDArray[np.float16]] = []
        for sub in _sub_batches(utts, max_padded_sec=self.max_padded_sec):
            lens24 = [len(u["a24"]) for u in sub]
            T = max(lens24)
            T += (-T) % FRAME
            x = torch.zeros(len(sub), T)
            for i, u in enumerate(sub):
                x[i, : len(u["a24"])] = torch.from_numpy(u["a24"])
            lat = self.mimi.encode_to_latent(x.to(self.device)[:, None])
            lats += [lat[i, : int(np.ceil(n / FRAME))].half().cpu().numpy() for i, n in enumerate(lens24)]
            lens16 = [len(u["a16"]) for u in sub]
            T16 = max(lens16)
            x16 = torch.zeros(len(sub), T16)
            for i, u in enumerate(sub):
                x16[i, : len(u["a16"])] = torch.from_numpy(u["a16"])
            emb = self.ecapa.encode_batch(
                x16.to(self.device), wav_lens=torch.tensor([n / T16 for n in lens16], device=self.device)
            )[:, 0]
            embs += list(torch.nn.functional.normalize(emb, dim=-1).half().cpu().numpy())
        mos = self.dnsmos([u["a16"] for u in utts])
        # Whisper is the expensive stage: skip utterances the later filters drop anyway
        # (low DNSMOS, or a known mass-produced TTS voice from earlier shards).
        need = []
        for i, u in enumerate(utts):
            u["latents"], u["ecapa"] = lats[i], embs[i]
            u["dnsmos_sig"], u["dnsmos_bak"], u["dnsmos_ovrl"] = (round(float(x), 3) for x in mos[i])
            u["whisper"] = ""
            tts = (
                self.blocklist is not None
                and float((self.blocklist @ embs[i].astype(np.float32)).max()) >= self.block_cos
            )
            u["skipped"] = "dnsmos" if u["dnsmos_ovrl"] < self.min_dnsmos else ("tts_voice" if tts else "")
            if not u["skipped"]:
                need.append(u)
        if need:
            feats = self.proc.feature_extractor(
                [u["a16"] for u in need], sampling_rate=SR16, return_tensors="pt", device=str(self.device)
            ).input_features.to(self.device, torch.float16)
            ids = self.asr.generate(feats, language="ja", task="transcribe", max_new_tokens=220)
            for u, text in zip(need, self.proc.batch_decode(ids, skip_special_tokens=True), strict=True):
                u["whisper"] = text.strip()


def _sub_batches(utts: list[dict[str, Any]], max_padded_sec: float) -> Iterator[list[dict[str, Any]]]:
    """Consecutive runs of `utts` (already length-sorted) whose padded length fits the cap."""
    cur: list[dict[str, Any]] = []
    for u in utts:
        longest = max([len(u["a24"])] + [len(c["a24"]) for c in cur]) / SR24
        if cur and longest * (len(cur) + 1) > max_padded_sec:
            yield cur
            cur = []
        cur.append(u)
    if cur:
        yield cur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--root", type=Path, default=Path("/mnt/artifacts/yodas3/data/ja"))
    parser.add_argument("--out", type=Path, default=Path("data/yodas3"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--max-padded-sec", type=float, default=120.0,
                        help="Mimi/ECAPA sub-batch size in padded seconds (lower = less GPU memory)")
    parser.add_argument("--tts-blocklist", type=Path, default=None,
                        help="mass-voice centroids from finalize_yodas3.py (skip Whisper near them)")
    args = parser.parse_args()

    for d in ("latents", "ecapa", "utts"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    done_flag = args.out / "utts" / f"{args.shard}.jsonl"
    if done_flag.exists():
        print(f"{args.shard}: already done")
        return
    table = pq.read_table(args.root / "metadata" / f"{args.shard}.parquet",
                          columns=["id", "transcript", "length"]).to_pydict()
    # Long videos are skipped before decoding: a multi-hour stream decodes to >10 GB per worker.
    meta = {
        table["id"][i]: {"transcript": table["transcript"][i]}
        for i in range(len(table["id"]))
        if (table["length"][i] or 0) <= MAX_VIDEO_SEC
    }
    print(f"{args.shard}: {len(meta)}/{len(table['id'])} videos <= {MAX_VIDEO_SEC:.0f}s", flush=True)
    gpu = GPUStages(torch.device("cuda"), args.asr_model, blocklist=args.tts_blocklist,
                    max_padded_sec=args.max_padded_sec)

    rows: list[dict[str, Any]] = []
    lat_chunks: list[npt.NDArray[np.float16]] = []
    emb_rows: list[npt.NDArray[np.float16]] = []
    n_frames = 0
    pending: list[dict[str, Any]] = []
    t0 = time.time()
    audio_sec = 0.0
    videos = 0

    def flush(force: bool):
        nonlocal pending, n_frames, audio_sec
        if not force and len(pending) < args.batch * 4:
            return
        pending.sort(key=lambda u: len(u["a24"]))
        n_full = len(pending) if force else len(pending) - len(pending) % args.batch
        todo, pending = pending[:n_full], pending[n_full:]
        for b in range(0, len(todo), args.batch):
            batch = todo[b : b + args.batch]
            gpu(batch)
            for u in batch:
                lat = u.pop("latents")
                emb_rows.append(u.pop("ecapa"))
                u.pop("a24")
                u.pop("a16", None)
                u["lat_start"], u["lat_len"] = n_frames, int(lat.shape[0])
                n_frames += int(lat.shape[0])
                lat_chunks.append(lat)
                audio_sec += u["duration"]
                rows.append(u)

    ctx = mp.get_context("spawn")
    jobs = iter_members(args.root / "audio" / f"{args.shard}.tar", meta)
    if args.limit_videos:
        import itertools

        jobs = itertools.islice(jobs, args.limit_videos)

    def handle(utts: list[dict[str, Any]]):
        nonlocal videos
        videos += 1
        pending.extend(utts)
        flush(force=False)
        if videos % 200 == 0:
            el = time.time() - t0
            print(f"{args.shard}: {videos} videos, {len(rows)} utts, {audio_sec / 3600:.1f} h kept "
                  f"in {el / 60:.1f} min ({audio_sec / max(el, 1):.0f}x realtime)", flush=True)

    # Bounded in-flight dispatch: Pool.imap would read the whole tar ahead and queue every
    # decoded video in memory if the GPU falls behind.
    from collections import deque

    with ctx.Pool(args.workers, initializer=_init_worker, maxtasksperchild=50) as pool:
        inflight: deque[Any] = deque()
        for job in jobs:
            inflight.append(pool.apply_async(process_video, (job,)))
            while len(inflight) >= args.workers + args.workers // 2:
                handle(inflight.popleft().get())
        while inflight:
            handle(inflight.popleft().get())
    flush(force=True)

    key = args.shard
    np.save(args.out / "latents" / f"{key}.npy",
            np.concatenate(lat_chunks) if lat_chunks else np.zeros((0, 32), np.float16))
    np.save(args.out / "ecapa" / f"{key}.npy",
            np.stack(emb_rows) if emb_rows else np.zeros((0, 192), np.float16))
    tmp = done_flag.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for i, r in enumerate(rows):
            r["shard"] = key
            r["ecapa_row"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(done_flag)
    el = time.time() - t0
    print(f"{key}: done, {videos} videos, {len(rows)} utts, {audio_sec / 3600:.1f} h in {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
