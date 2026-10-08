"""A0 (2/2): encode every manifest utterance once — Mimi latents, ECAPA embedding, speech end.

Reader processes stream each tar shard sequentially in large chunks (random seeks over the
SMB mount run at ~0.5 MB/s, sequential reads at ~130 MB/s), decode the mp3 members listed in
the manifest and hand 24 kHz waveforms to the main process, which batches them on the GPU:

  - Mimi encoder (fp32/TF32; the encoder is causal, so right-padding a batch is exact)
  - ECAPA speaker embedding (speechbrain/spkrec-ecapa-voxceleb, 16 kHz) for the
    speaker-consistency filter
  - speech end: last 80 ms frame within 35 dB of the utterance peak, so the loader can trim
    trailing silence (training on long trailing silence teaches the model to skip EOS)

Per shard it writes data/emilia_ja/latents/<subset>__<shard>.npy (float16 [frames, 32], all
utterances concatenated), .../ecapa/<same>.npy (float16 [utts, 192]) and an index
.../latents/<same>.json. Shards already written are skipped, so the job is resumable.

    uv run python scripts/precompute_emilia_ja.py --readers 4
"""

import argparse
import io
import json
import multiprocessing as mp
import queue
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr
import torch
import torchaudio.functional as AF

SR = 24000
FRAME = 1920  # samples per 12.5 Hz frame
CHUNK = 64 * 1024 * 1024
ROOTS = {
    "Emilia": Path("/mnt/ext11/Emilia-Dataset/Emilia/JA"),
    "Emilia-YODAS": Path("/mnt/ext11/Emilia-Dataset/Emilia-YODAS/JA"),
}


def shard_key(subset: str, shard: str) -> str:
    return f"{subset}__{Path(shard).stem}"


def speech_end_frames(wav: npt.NDArray[np.float32]) -> int:
    n = len(wav) // FRAME
    if n == 0:
        return 0
    rms = np.sqrt((wav[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(db.max() - 35.0, -55.0)
    above = np.nonzero(db >= thr)[0]
    return int(above[-1]) + 1 if len(above) else n


def read_shard(subset: str, shard: str, entries: list[dict[str, Any]], out: "mp.Queue[Any]"):
    """Stream one tar sequentially and emit (key, id, wav, speech_end) for listed members."""
    key = shard_key(subset, shard)
    entries = sorted(entries, key=lambda e: e["offset"])
    buf = b""
    buf_start = 0
    with open(ROOTS[subset] / shard, "rb", buffering=0) as f:
        for e in entries:
            end = e["offset"] + e["size"]
            while buf_start + len(buf) < end:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                # drop what lies before this member; the file position stays buf_start + len(buf)
                keep_from = min(max(0, e["offset"] - buf_start), len(buf))
                buf = buf[keep_from:] + chunk
                buf_start += keep_from
            lo = e["offset"] - buf_start
            data = buf[lo : lo + e["size"]]
            if lo < 0 or len(data) != e["size"]:
                out.put(("error", key, e["id"], "short read", 0))
                continue
            try:
                wav, sr = sf.read(io.BytesIO(data), dtype="float32")
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                if sr != SR:
                    wav = soxr.resample(wav, sr, SR, quality="HQ").astype(np.float32)
                wav = wav[: int(round(e["duration"] * SR))]  # mp3 padding
                out.put(("item", key, e["id"], wav, speech_end_frames(wav)))
            except Exception as exc:  # noqa: BLE001 - record and continue
                out.put(("error", key, e["id"], repr(exc), 0))
    out.put(("done", key, len(entries), None, 0))


def reader_main(jobs: list[tuple[str, str, list[dict[str, Any]]]], out: "mp.Queue[Any]"):
    torch.set_num_threads(1)
    for subset, shard, entries in jobs:
        read_shard(subset, shard, entries, out)


class Encoder:
    def __init__(self, device: torch.device):
        from speechbrain.inference.speaker import EncoderClassifier

        from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config

        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.mimi = load_mimi(load_pocket_config(), device)
        self.ecapa = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
            savedir="data/models/spkrec-ecapa-voxceleb",
        )

    @torch.no_grad()
    def __call__(
        self, wavs: list[npt.NDArray[np.float32]]
    ) -> tuple[list[npt.NDArray[np.float16]], npt.NDArray[np.float16]]:
        lens = [len(w) for w in wavs]
        T = max(lens)
        T += (-T) % FRAME
        x = torch.zeros(len(wavs), T)
        for i, w in enumerate(wavs):
            x[i, : len(w)] = torch.from_numpy(w)
        x = x.to(self.device)
        lat = self.mimi.encode_to_latent(x[:, None])  # [B, frames, 32]
        lats = [
            lat[i, : int(np.ceil(n / FRAME))].half().cpu().numpy() for i, n in enumerate(lens)
        ]
        x16 = AF.resample(x, SR, 16000)
        rel = torch.tensor([n / T for n in lens], device=self.device)
        emb = self.ecapa.encode_batch(x16, wav_lens=rel)[:, 0]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return lats, emb.half().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/emilia_ja/manifest_all.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/emilia_ja"))
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--batch-seconds", type=float, default=480.0)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()

    lat_dir, ecapa_dir = args.out / "latents", args.out / "ecapa"
    lat_dir.mkdir(parents=True, exist_ok=True)
    ecapa_dir.mkdir(parents=True, exist_ok=True)

    by_shard: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with open(args.manifest) as f:
        for line in f:
            d = json.loads(line)
            by_shard[(d["subset"], d["shard"])].append(
                {"id": d["id"], "offset": d["offset"], "size": d["size"], "duration": d["duration"]}
            )
    todo = [
        (s, sh, ents)
        for (s, sh), ents in sorted(by_shard.items())
        if not (lat_dir / f"{shard_key(s, sh)}.json").exists()
    ]
    if args.limit_shards:
        todo = todo[: args.limit_shards]
    # Largest shards first so the readers finish together.
    todo.sort(key=lambda j: -sum(e["size"] for e in j[2]))
    expected = {shard_key(s, sh): len(ents) for s, sh, ents in todo}
    print(f"{len(todo)} shards to encode, {sum(expected.values())} utterances", flush=True)
    if not todo:
        return

    ctx = mp.get_context("spawn")
    q: "mp.Queue[Any]" = ctx.Queue(maxsize=2048)
    assignments: list[list[tuple[str, str, list[dict[str, Any]]]]] = [[] for _ in range(args.readers)]
    for i, job in enumerate(todo):
        assignments[i % args.readers].append(job)
    procs = [ctx.Process(target=reader_main, args=(a, q), daemon=True) for a in assignments if a]
    for p in procs:
        p.start()

    enc = Encoder(torch.device("cuda"))
    results: dict[str, dict[str, Any]] = defaultdict(dict)  # key -> id -> (lat, emb, end)
    announced: dict[str, int] = {}
    errors: dict[str, list[str]] = defaultdict(list)
    pending: list[tuple[str, str, npt.NDArray[np.float32], int]] = []
    written = 0
    t0 = time.time()
    audio_sec = 0.0

    def flush_batches(force: bool):
        nonlocal pending, audio_sec
        total = sum(len(p[2]) for p in pending) / SR
        if not pending or (not force and total < args.batch_seconds):
            return
        pending.sort(key=lambda p: len(p[2]))
        i = 0
        while i < len(pending):
            # batches of similar length, capped by count and by padded seconds
            j = i
            while j < len(pending) and j - i < args.max_batch:
                if (j - i + 1) * len(pending[j][2]) / SR > args.batch_seconds / 4:
                    break
                j += 1
            j = max(j, i + 1)
            chunk = pending[i:j]
            lats, embs = enc([c[2] for c in chunk])
            for (key, uid, wav, end), lat, emb in zip(chunk, lats, embs, strict=True):
                results[key][uid] = (lat, emb, min(end, lat.shape[0]))
                audio_sec += len(wav) / SR
            i = j
        pending = []

    def write_ready():
        nonlocal written
        for key in list(announced):
            got = len(results[key]) + len(errors[key])
            if got < announced[key]:
                continue
            ids = sorted(results[key])
            lat_list = [results[key][u][0] for u in ids]
            starts = np.cumsum([0] + [lt.shape[0] for lt in lat_list[:-1]]).tolist()
            np.save(lat_dir / f"{key}.npy", np.concatenate(lat_list) if lat_list else np.zeros((0, 32), np.float16))
            np.save(ecapa_dir / f"{key}.npy", np.stack([results[key][u][1] for u in ids]))
            index = {
                "ids": ids,
                "lat_start": starts,
                "lat_len": [lt.shape[0] for lt in lat_list],
                "speech_end": [results[key][u][2] for u in ids],
                "errors": errors[key],
            }
            (lat_dir / f"{key}.json").write_text(json.dumps(index))
            del results[key], announced[key]
            written += 1
            el = time.time() - t0
            print(
                f"[{el / 60:6.1f} min] wrote {key}: {len(ids)} utts ({len(errors[key])} errors); "
                f"{written}/{len(todo)} shards, {audio_sec / 3600:.1f} h encoded, "
                f"{audio_sec / max(el, 1):.0f}x realtime",
                flush=True,
            )

    alive = len(procs)
    done_shards = 0
    while done_shards < len(todo):
        try:
            kind, key, uid, payload, end = q.get(timeout=5)
        except queue.Empty:
            flush_batches(force=True)
            write_ready()
            if not any(p.is_alive() for p in procs) and q.empty():
                raise RuntimeError("readers exited before all shards were done")
            continue
        if kind == "item":
            pending.append((key, uid, payload, end))
            flush_batches(force=False)
        elif kind == "error":
            errors[key].append(f"{uid}: {payload}")
        elif kind == "done":
            announced[key] = uid  # uid carries the entry count here
            done_shards += 1
            flush_batches(force=True)
        write_ready()
    flush_batches(force=True)
    write_ready()
    for p in procs:
        p.join(timeout=10)
    print(f"done: {written} shards in {(time.time() - t0) / 60:.1f} min", flush=True)
    del alive


if __name__ == "__main__":
    main()
