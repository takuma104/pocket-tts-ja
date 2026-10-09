"""Training rows for the level-augmented copy written by precompute_emilia_ja.py --gain-db.

For every row of --train (e.g. the rechecked Emilia train manifest) whose shard was re-encoded,
emit the same row pointing at the gain-shifted latents, with id "<id>#g" and speaker
"gain:<speaker>" (its own voice-prompt pool: prompt and target then share the speaker's gain).

    uv run python scripts/make_gain_manifest.py --train data/emilia_ja/train_rechecked.jsonl \
        --gain-dir data/emilia_ja_gain --out data/emilia_ja_gain/train.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gain-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    index: dict[str, dict[str, tuple[int, int, int]]] = {}
    for p in sorted((args.gain_dir / "latents").glob("*.json")):
        d = json.loads(p.read_text())
        index[p.stem] = {u: (s, n, e) for u, s, n, e in zip(d["ids"], d["lat_start"], d["lat_len"], d["speech_end"], strict=True)}
    rows, per_spk = [], Counter()
    for line in open(args.train):
        r = json.loads(line)
        key = Path(r["lat_file"]).stem
        hit = index.get(key, {}).get(r["id"])
        if hit is None:
            continue
        per_spk[r["speaker"]] += 1
        start, length, end = hit
        rows.append({**r, "id": f"{r['id']}#g", "speaker": f"gain:{r['speaker']}", "source_type": "emilia_gain",
                     "lat_file": f"latents/{key}.npy", "lat_start": start, "lat_len": length, "speech_end": end})
    # a speaker needs another utterance for its voice prompt
    rows = [r for r in rows if per_spk[r["speaker"].removeprefix("gain:")] >= 2]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["lat_len"] for r in rows) * 0.08 / 3600
    print(f"{len(rows)} rows ({hours:.1f}h, {len({r['speaker'] for r in rows})} speakers) -> {args.out}")


if __name__ == "__main__":
    main()
