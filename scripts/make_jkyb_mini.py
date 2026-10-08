"""JKYB-mini: a fixed, stratified subset of JKYB-Parakeet for frequent in-training evaluation.

One sentence per (kanji, reading) pair at most, so the subset spans as many readings as
possible; sampled per reading_category in proportion to the full set, with the appendix
(jukujikun) category kept whole up to its share. The output is a JKYB-schema jsonl, so it
can be scored with `jkyb-eval tts <audio> --dataset <file>`.

    uv run python scripts/make_jkyb_mini.py --size 1500 --out data/eval/jkyb_mini.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILE = "data/common_kanji_source.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/eval/jkyb_mini.jsonl"))
    args = parser.parse_args()

    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    rng = random.Random(args.seed)

    # key = "<kanji>_<reading>_<k>": keep one random sentence per (kanji, reading)
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[r["key"].rsplit("_", 1)[0]].append(r)
    singles = [rng.choice(v) for _, v in sorted(by_pair.items())]

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in singles:
        by_cat[r["reading_category"]].append(r)
    total = len(rows)
    picked: list[dict] = []
    for cat, cands in sorted(by_cat.items()):
        share = round(args.size * sum(r["reading_category"] == cat for r in rows) / total)
        rng.shuffle(cands)
        picked.extend(cands[: max(1, share)])
    picked.sort(key=lambda r: r["key"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {c: sum(r["reading_category"] == c for r in picked) for c in by_cat}
    print(f"wrote {len(picked)} rows to {args.out}: {counts}")


if __name__ == "__main__":
    main()
