"""P1: pack the QC-passed teacher latents of a set into shards with a training manifest.

The teacher's latent ends at the length the duration predictor chose, often with a short stretch
of digital silence that Irodori's flat-tail test (20 frames) does not catch. The speech end is
taken from the 16 kHz QC audio instead: the frame (40 ms = one latent frame) after the last one
whose RMS is within --floor-db of the clip's loudest frame. The latent itself is stored whole.

    uv run python scripts/teacher/pack.py data/irodori_distill/teacher s3
    -> <root>/<set>/part_<k>.npy  float16 [frames, 32], clips back to back
       <root>/<set>/manifest.jsonl id, set, transcript, text (TTS input), speaker = syn:<ref speaker>,
                                   ref_speaker, ref_id, lat_file, lat_start, lat_len, speech_start,
                                   speech_end, duration (s, speech_end / 25)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_RATE = 25.0
QC_SR = 16000
QC_HOP = int(QC_SR / FRAME_RATE)  # 640 samples = one latent frame
PART_FRAMES = 1 << 22  # ~4.2M frames (268 MB) per shard


def speech_bounds(wav: np.ndarray, frames: int, floor_db: float) -> tuple[int, int]:
    n = min(frames, len(wav) // QC_HOP)
    if n == 0:
        return 0, 0
    rms = np.sqrt(np.mean(wav[: n * QC_HOP].reshape(n, QC_HOP).astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.nonzero(db > db.max() - floor_db)[0]
    return int(loud[0]), int(loud[-1]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("set")
    parser.add_argument("--floor-db", type=float, default=45.0)
    args = parser.parse_args()

    d = args.root / args.set
    rows = {r["id"]: r for r in map(json.loads, open(args.root / f"rows_{args.set}.jsonl", encoding="utf-8"))}
    passed = [q["id"] for q in map(json.loads, open(d / "qc.jsonl", encoding="utf-8")) if q["pass"]]
    for old in d.glob("part_*.npy"):
        old.unlink()
    part: list[np.ndarray] = []
    part_frames = 0
    k = 0
    tails: list[int] = []
    with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
        for key in passed:
            z = np.load(d / "latents" / f"{key}.npy")
            wav, sr = sf.read(str(d / "audio16k" / f"{key}.wav"), dtype="float32")
            assert sr == QC_SR
            start, end = speech_bounds(wav, z.shape[0], args.floor_db)
            if end <= start:
                continue
            if part_frames + z.shape[0] > PART_FRAMES and part:
                np.save(d / f"part_{k:03d}.npy", np.concatenate(part))
                part, part_frames, k = [], 0, k + 1
            r = rows[key]
            f.write(json.dumps({
                "id": key, "set": args.set, "transcript": r["transcript"], "text": r["text"],
                "speaker": f"syn:{r['ref_speaker']}", "ref_speaker": r["ref_speaker"], "ref_id": r["ref_id"],
                "lat_file": f"part_{k:03d}.npy", "lat_start": part_frames, "lat_len": int(z.shape[0]),
                "speech_start": start, "speech_end": end, "duration": round(end / FRAME_RATE, 3),
            }, ensure_ascii=False) + "\n")
            part.append(z.astype(np.float16))
            part_frames += z.shape[0]
            tails.append(int(z.shape[0]) - end)
    if part:
        np.save(d / f"part_{k:03d}.npy", np.concatenate(part))
    t = np.array(tails)
    print(f"{args.set}: {len(t)} clips packed into {k + 1} shards; frames after the speech end: "
          f"median {np.median(t):.0f}, p10 {np.percentile(t, 10):.0f}, p90 {np.percentile(t, 90):.0f}, 0: {np.mean(t == 0):.1%}")


if __name__ == "__main__":
    main()
