"""Y2: filter YODAS3 utterances, pick their labels and assign speaker labels.

Inputs are the per-shard outputs of process_shard.py under --root (utts/*.jsonl, ecapa/*.npy,
latents/*.npy). For every utterance:

  - caption agreement: CER between the pyopenjtalk readings of the YouTube caption and of
    the Whisper transcript (pocket_tts_ja.kana); the reading level ignores orthographic
    variants (時/とき) but catches misrecognitions and boundary words
  - label: the Whisper transcript (punctuated, same style as the Emilia transcripts),
    normalized with pocket_tts_ja.text.normalize
Filters: agreement CER <= --max-cer, DNSMOS OVRL >= --min-dnsmos, 1.5 <= chars/sec <= 15,
non-empty label, not a JKYB-Parakeet sentence.
Speakers: per video, average-linkage clustering of the kept utterances' ECAPA embeddings
(cosine distance cut at --cluster-dist); clusters with >= 2 utterances become speaker labels
`<video>#<k>`, the rest is dropped (a voice prompt needs another utterance of the speaker).
Mass-produced voices: speaker clusters whose ECAPA centroid has cos >= --tts-cos with clusters
of at least --tts-min-videos other videos are dropped. Text-to-speech channels (e.g. Wikipedia
read aloud by a TTS engine) reuse one synthetic voice across hundreds of videos with
near-identical embeddings, whereas one real person varies across recordings.
Split: valid = md5(video) % 200 == 0. Writes <root>/{train,valid}.jsonl for the loader.

    uv run python scripts/yodas3/finalize_yodas3.py --root data/yodas3
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.cluster.hierarchy import fcluster, linkage

from pocket_tts_ja.kana import cer, reading
from pocket_tts_ja.text import normalize


def score_rows(rows: list[dict[str, Any]]) -> list[tuple[float, str]]:
    out = []
    for r in rows:
        try:
            out.append((cer(reading(r["caption"]), reading(r["whisper"])), normalize(r["whisper"])))
        except Exception:  # noqa: BLE001 - pyopenjtalk rejects some inputs
            out.append((1.0, ""))
    return out


def cluster_video(emb: np.ndarray, dist: float) -> np.ndarray:
    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    z = linkage(emb.astype(np.float64), method="average", metric="cosine")
    return fcluster(z, t=dist, criterion="distance") - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/yodas3"))
    parser.add_argument("--max-cer", type=float, default=0.10)
    parser.add_argument("--min-dnsmos", type=float, default=3.0)
    parser.add_argument("--cluster-dist", type=float, default=0.55)
    parser.add_argument("--tts-cos", type=float, default=0.92)
    parser.add_argument("--tts-min-videos", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    jkyb_path = hf_hub_download(
        "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet", "data/common_kanji_source.jsonl",
        repo_type="dataset",
    )
    jkyb = {normalize(json.loads(line)["text"]) for line in open(jkyb_path)}

    rows: list[dict[str, Any]] = []
    for p in sorted((args.root / "utts").glob("*.jsonl")):
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    chunks = [rows[i : i + 2000] for i in range(0, len(rows), 2000)]
    with ProcessPoolExecutor(args.workers) as ex:
        scored = [s for part in ex.map(score_rows, chunks) for s in part]

    stats: Counter = Counter()
    kept_by_video: dict[str, list[int]] = defaultdict(list)
    cers = []
    for i, (r, (c, label)) in enumerate(zip(rows, scored, strict=True)):
        r["agree_cer"], r["transcript"] = round(c, 4), label
        cers.append(c)
        cps = len(re.sub(r"\s", "", label)) / max(r["duration"], 1e-3)
        if not label:
            stats["drop_empty"] += 1
        elif c > args.max_cer:
            stats["drop_agreement"] += 1
        elif r["dnsmos_ovrl"] < args.min_dnsmos:
            stats["drop_dnsmos"] += 1
        elif not 1.5 <= cps <= 15:
            stats["drop_chars_per_sec"] += 1
        elif label in jkyb:
            stats["drop_jkyb"] += 1
        else:
            kept_by_video[r["video"]].append(i)

    emb_cache: dict[str, np.ndarray] = {}

    def emb_of(r: dict[str, Any]) -> np.ndarray:
        if r["shard"] not in emb_cache:
            emb_cache[r["shard"]] = np.load(args.root / "ecapa" / f"{r['shard']}.npy").astype(np.float32)
        return emb_cache[r["shard"]][r["ecapa_row"]]

    def emb_of_mean(rs: list[dict[str, Any]]) -> np.ndarray:
        v = np.mean([emb_of(r) for r in rs], axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    # per-video speaker clusters
    clusters: list[tuple[str, int, list[int]]] = []  # (video, k, row indices)
    for video, idx in kept_by_video.items():
        labels = cluster_video(np.stack([emb_of(rows[i]) for i in idx]), args.cluster_dist)
        members: dict[int, list[int]] = defaultdict(list)
        for i, k in zip(idx, labels.tolist(), strict=True):
            members[k].append(i)
        for k, m in members.items():
            if len(m) < 2:
                stats["drop_singleton_speaker"] += len(m)
                continue
            clusters.append((video, k, m))

    # mass-produced (TTS) voices: near-identical centroids across many videos
    flagged: set[int] = set()
    if clusters:
        cent = np.stack([emb_of_mean([rows[i] for i in m]) for _, _, m in clusters])
        vids = np.array([hash(v) for v, _, _ in clusters])
        for a in range(0, len(cent), 4096):
            sim = cent[a : a + 4096] @ cent.T
            for r in range(sim.shape[0]):
                close = np.nonzero(sim[r] >= args.tts_cos)[0]
                if len(np.unique(vids[close])) - 1 >= args.tts_min_videos:
                    flagged.add(a + r)
    stats["drop_mass_voice_utts"] = sum(len(clusters[c][2]) for c in flagged)
    stats["mass_voice_clusters"] = len(flagged)

    out: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    spk_sizes = []
    for c, (video, k, m) in enumerate(clusters):
        if c in flagged:
            continue
        split = "valid" if int(hashlib.md5(video.encode()).hexdigest(), 16) % 200 == 0 else "train"
        spk_sizes.append(len(m))
        for i in m:
            r = rows[i]
            out[split].append(
                {
                    "id": r["id"], "transcript": r["transcript"], "speaker": f"{video}#{k}",
                    "source_type": "yodas3", "group": video, "duration": r["duration"],
                    "dnsmos": r["dnsmos_ovrl"], "agree_cer": r["agree_cer"],
                    "caption_kind": r["caption_kind"], "lat_file": f"latents/{r['shard']}.npy",
                    "lat_start": r["lat_start"], "lat_len": r["lat_len"], "speech_end": r["speech_end"],
                }
            )

    summary: dict[str, Any] = {
        "utts_in": len(rows),
        "hours_in": round(sum(r["duration"] for r in rows) / 3600, 1),
        "agree_cer_percentiles": {p: round(float(np.percentile(cers, p)), 3) for p in (10, 25, 50, 75, 90)},
        **dict(stats),
        "speakers": len(spk_sizes),
        "utts_per_speaker_median": float(np.median(spk_sizes)) if spk_sizes else 0,
    }
    for split, rs in out.items():
        with open(args.root / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {"utts": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 1)}
    (args.root / "finalize_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
