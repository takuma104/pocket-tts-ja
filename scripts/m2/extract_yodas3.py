"""M2 CC data: full-band, clean YODAS3 ja clips as 24 kHz wavs for the decoder fine-tune (CC BY 3.0).

The CC line of the decoder may learn only from CC BY audio, and YODAS3 (espnet/yodas3: YouTube videos, opus in webm,
mostly 96-135 kbps) is the full-band natural speech among it. The clips are the utterances of the TTS training set
(data/yodas3_kept/train.jsonl: speech whose captions agree with Whisper, DNSMOS OVRL >= 2.9, the voices of known TTS
channels removed), cut again from the videos with the times scripts/yodas3/process_shard.py found (data/yodas3/utts).
--select keeps those of videos whose audio reaches --min-max-freq (the metadata's max_freq_hz, set by the encoder's
low-pass) with DNSMOS P.835 at least --min-ovrl / --min-sig / --min-bak.

    uv run python scripts/m2/extract_yodas3.py --select   # writes <out>/manifest.jsonl
    uv run python scripts/m2/extract_yodas3.py            # reads it, writes <out>/audio/<id>.wav

Each worker takes one tar shard, reads its index and only the videos it needs, decodes them (PyAV, 48 kHz, channels
averaged as process_shard.py does) up to its last clip, holding only the audio its clips still need, and writes the
clips resampled to 24 kHz (soxr). Clips already written are skipped.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import soundfile as sf
import soxr

SR = 24000
ROOT = Path("/mnt/artifacts/yodas3/data/ja")
UTTS = Path("/home/takuma/co/pocket-tts-ja/data/yodas3")
KEPT = Path("/home/takuma/co/pocket-tts-ja/data/yodas3_kept/train.jsonl")


def select(args: argparse.Namespace) -> None:
    video_meta: dict[str, tuple[float, float]] = {}
    for path in sorted((args.root / "metadata").glob("*.parquet")):
        for row in pq.read_table(path, columns=["id", "audio_bitrate", "max_freq_hz"]).to_pylist():
            video_meta[row["id"]] = (float(row["audio_bitrate"] or 0.0), float(row["max_freq_hz"] or 0.0))
    speakers = {}
    for line in open(args.kept):
        row = json.loads(line)
        speakers[row["id"]] = row["speaker"]
    rows = []
    for path in sorted((args.utts / "utts").glob("*.jsonl")):
        for line in open(path):
            u = json.loads(line)
            if u["id"] not in speakers:
                continue
            bitrate, max_freq = video_meta.get(u["video"], (0.0, 0.0))
            if not (max_freq >= args.min_max_freq and u["dnsmos_ovrl"] >= args.min_ovrl
                    and u["dnsmos_sig"] >= args.min_sig and u["dnsmos_bak"] >= args.min_bak):
                continue
            rows.append({
                "id": f"y3_{u['id']}", "duration": u["duration"], "video": u["video"], "shard": u["shard"],
                "start": u["start"], "end": u["end"], "speaker": speakers[u["id"]], "dnsmos_ovrl": u["dnsmos_ovrl"],
                "dnsmos_sig": u["dnsmos_sig"], "dnsmos_bak": u["dnsmos_bak"], "max_freq_hz": max_freq,
                "bitrate": bitrate,
            })
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{len(rows)} clips, {hours:.1f} h, {len({r['speaker'] for r in rows})} speakers, "
          f"{len({r['video'] for r in rows})} videos, {len({r['shard'] for r in rows})} shards", flush=True)


def mono(frame: Any) -> npt.NDArray[np.float32]:
    """One decoded audio frame with its channels averaged (as process_shard.py does)."""
    a = frame.to_ndarray()
    if a.ndim == 2 and a.shape[0] > 1 and frame.format.is_planar:
        a = a.mean(axis=0)
    elif a.ndim == 2 and not frame.format.is_planar:
        a = a.reshape(-1, len(frame.layout.channels)).mean(axis=1)
    else:
        a = a.reshape(-1)
    return a.astype(np.float32)


def cut_clips(data: bytes, rows: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], npt.NDArray[np.float32], int]]:
    """(row, clip at the stream's rate, rate) for the rows of one webm/opus member, decoding it only up to its last
    clip and keeping only the audio from the earliest clip still to cut (a few seconds, whatever the video's length)."""
    import av

    rows = sorted(rows, key=lambda r: r["start"])
    k = 0
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        buf: list[npt.NDArray[np.float32]] = []
        buf_start = buf_len = 0  # samples of the video before buf, in buf

        def take(row: dict[str, Any]) -> npt.NDArray[np.float32]:
            audio = np.concatenate(buf) if buf else np.zeros(0, np.float32)
            buf[:] = [audio]
            return audio[max(round(row["start"] * sr) - buf_start, 0) : max(round(row["end"] * sr) - buf_start, 0)]

        for frame in container.decode(stream):
            buf.append(mono(frame))
            buf_len += len(buf[-1])
            while k < len(rows) and buf_start + buf_len >= round(rows[k]["end"] * sr):
                yield rows[k], take(rows[k]), sr
                k += 1
            if k == len(rows):
                return
            drop = min(round(rows[k]["start"] * sr) - buf_start, buf_len)
            if drop > 10 * sr:  # now and then, forget what no clip left needs
                audio = np.concatenate(buf)[drop:]
                buf[:] = [audio]
                buf_start, buf_len = buf_start + drop, len(audio)
        for row in rows[k:]:  # past the end of the stream: what there is
            yield row, take(row), sr


def extract_shard(job: tuple[Path, Path, list[dict[str, Any]]]) -> tuple[str, int, int]:
    tar_path, audio_dir, rows = job
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not (audio_dir / f"{row['id']}.wav").exists():
            by_video[row["video"]].append(row)
    written = errors = 0
    if not by_video:
        return tar_path.name, 0, 0
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            video = member.name.rsplit("/", 1)[-1].removesuffix(".webm")
            if not member.name.endswith(".webm") or video not in by_video:
                continue
            todo = len(by_video[video])
            try:
                for row, clip, sr in cut_clips(tf.extractfile(member).read(), by_video[video]):
                    todo -= 1
                    if len(clip) < sr:
                        errors += 1
                        continue
                    clip = soxr.resample(clip, sr, SR) if sr != SR else clip
                    tmp = audio_dir / f".{row['id']}.wav"  # renamed when complete: a stopped run leaves no partial clip
                    sf.write(tmp, np.clip(clip, -1.0, 1.0), SR, subtype="PCM_16", format="WAV")
                    os.replace(tmp, audio_dir / f"{row['id']}.wav")
                    written += 1
            except Exception:  # noqa: BLE001 - a corrupt member: skip the clips left
                errors += todo
    return tar_path.name, written, errors


def extract(args: argparse.Namespace) -> None:
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in open(args.out / "manifest.jsonl"):
        row = json.loads(line)
        by_shard[row["shard"]].append(row)
    jobs = [(args.root / "audio" / f"{shard}.tar", audio_dir, rows) for shard, rows in sorted(by_shard.items())]
    total = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, written, errors in pool.imap_unordered(extract_shard, jobs):
            total += written
            print(f"{name}: {written} written, {errors} errors (total {total})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help="YODAS3 ja: audio/<shard>.tar and metadata/*.parquet")
    parser.add_argument("--utts", type=Path, default=UTTS, help="process_shard.py's output (utts/<shard>.jsonl)")
    parser.add_argument("--kept", type=Path, default=KEPT, help="the TTS training rows (finalize_yodas3.py)")
    parser.add_argument("--out", type=Path, default=Path("data/m2/yodas3"))
    parser.add_argument("--min-max-freq", type=float, default=15000.0)
    parser.add_argument("--min-ovrl", type=float, default=3.2)
    parser.add_argument("--min-sig", type=float, default=3.5)
    parser.add_argument("--min-bak", type=float, default=3.9)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    select(args) if args.select else extract(args)


if __name__ == "__main__":
    main()
