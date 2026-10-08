"""A0 (3/3): join the encoded latents with the manifest and apply the speaker-consistency filter.

For every speaker label (within its split), each utterance's ECAPA embedding is compared with
the leave-one-out centroid of the label's other utterances (pairwise cosine for 2-utterance
labels). Utterances below --min-cos are dropped: they are the second voice of a multi-voice
label (JLPT listening drills, dramas) or a mis-diarized segment, and would teach the model to
ignore its voice prompt. Labels left with a single utterance are dropped too.

Writes data/emilia_ja/{train,valid}.jsonl for pocket_tts_ja.loader.

    uv run python scripts/finalize_emilia_ja.py
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

KEEP_FIELDS = ("id", "transcript", "speaker", "source_type", "duration", "kbps", "dnsmos", "group")


def consistency(emb: np.ndarray) -> np.ndarray:
    n = emb.shape[0]
    if n == 1:
        return np.ones(1)
    if n == 2:
        c = float(emb[0] @ emb[1])
        return np.array([c, c])
    total = emb.sum(axis=0, keepdims=True)
    loo = total - emb
    loo /= np.linalg.norm(loo, axis=1, keepdims=True) + 1e-8
    return (emb * loo).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/emilia_ja"))
    parser.add_argument("--min-cos", type=float, default=0.5)
    args = parser.parse_args()

    rows: dict[str, dict[str, Any]] = {}
    with open(args.root / "manifest_all.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rows[d["id"]] = d
    emb_of: dict[str, np.ndarray] = {}
    missing = 0
    for idx_path in sorted((args.root / "latents").glob("*.json")):
        key = idx_path.stem
        index = json.loads(idx_path.read_text())
        emb = np.load(args.root / "ecapa" / f"{key}.npy").astype(np.float32)
        for k, uid in enumerate(index["ids"]):
            r = rows.get(uid)
            if r is None:
                continue
            r["lat_file"] = f"latents/{key}.npy"
            r["lat_start"] = int(index["lat_start"][k])
            r["lat_len"] = int(index["lat_len"][k])
            r["speech_end"] = int(index["speech_end"][k])
            emb_of[uid] = emb[k]
    for r in rows.values():
        if "lat_file" not in r:
            missing += 1

    stats: Counter = Counter()
    by_spk: dict[tuple[str, str], list[str]] = defaultdict(list)
    for uid, r in rows.items():
        if "lat_file" in r:
            by_spk[(r["split"], r["speaker"])].append(uid)
    cos_all: list[float] = []
    keep: set[str] = set()
    for _, uids in by_spk.items():
        cos = consistency(np.stack([emb_of[u] for u in uids]))
        cos_all.extend(cos.tolist())
        ok = [u for u, c in zip(uids, cos, strict=True) if c >= args.min_cos]
        stats["dropped_inconsistent"] += len(uids) - len(ok)
        if len(ok) < 2:
            stats["dropped_single_after_filter"] += len(ok)
            continue
        keep.update(ok)

    out: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    for uid, r in rows.items():
        if uid in keep:
            row = {k: r[k] for k in KEEP_FIELDS}
            row.update({k: r[k] for k in ("lat_file", "lat_start", "lat_len", "speech_end")})
            out[r["split"]].append(row)
    summary: dict[str, Any] = {
        "missing_latents": missing,
        "min_cos": args.min_cos,
        "cos_percentiles": {
            p: round(float(np.percentile(cos_all, p)), 3) for p in (1, 5, 10, 25, 50, 75)
        },
        **dict(stats),
    }
    for split, rs in out.items():
        with open(args.root / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {
            "utts": len(rs),
            "hours": round(sum(r["duration"] for r in rs) / 3600, 1),
            "speakers": len({r["speaker"] for r in rs}),
        }
    (args.root / "finalize_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
