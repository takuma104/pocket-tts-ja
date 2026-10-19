"""Check that a training mix of the CC BY line holds CC BY material only (release check).

Every row must be one of:
- `yodas3`: YODAS3 (CC BY 3.0),
- `yodas` / `emilia_gain`: an Emilia-YODAS clip (CC BY 4.0) or its gain copy, looked up by id in
  the Emilia manifest (the NC part of Emilia has subset "Emilia"),
- `irodori`: synthetic speech whose reference voice (`ref_id`) is an Emilia-YODAS clip.
Anything else, or an id that the manifest lists under another subset, is reported and makes the
exit status 1. The manifest is the filtered one, so a few reference voices of the synthetic sets
are not in it (picked from all of Emilia-YODAS): those whose video has only Emilia-YODAS clips
in the manifest are listed separately, to be looked up in the raw tars (`tar tf <shard> | grep
<id>`; the six of mix v8 were, 2026-10-20). Prints rows and hours per source.

    uv run python scripts/release/audit_cc_mix.py data/mix/BCC_train_v8.jsonl data/mix/BCC_valid.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST = Path("data/emilia_ja/manifest_all.jsonl")
CC_SUBSET = "Emilia-YODAS"


def video(clip_id: str) -> str:
    return clip_id.rsplit("_W", 1)[0]


def load_subsets(path: Path) -> tuple[dict[str, str], dict[str, set[tuple[str, str]]]]:
    """Subset per clip id, and (subset, shard) pairs per video."""
    subsets: dict[str, str] = {}
    videos: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            subsets[r["id"]] = r["subset"]
            videos[video(r["id"])].add((r["subset"], r.get("shard", "")))
    return subsets, videos


def audit(mix: Path, subsets: dict[str, str], videos: dict[str, set[tuple[str, str]]]) -> int:
    hours: dict[str, float] = defaultdict(float)
    rows: Counter[str] = Counter()
    bad: Counter[str] = Counter()
    examples: dict[str, str] = {}
    by_video: dict[str, str] = {}
    with open(mix) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("source_type", "?")
            rows[st] += 1
            hours[st] += float(r.get("duration", 0.0))
            if st == "yodas3":
                continue
            if st in ("yodas", "emilia_gain"):
                key = str(r["id"]).split("#", 1)[0]
            elif st == "irodori":
                key = str(r.get("ref_id") or "")
            else:
                bad[f"source_type {st}"] += 1
                examples.setdefault(f"source_type {st}", str(r.get("id")))
                continue
            subset = subsets.get(key)
            seen = {sub for sub, _ in videos.get(video(key), set())}
            if subset is None and seen == {CC_SUBSET}:
                by_video[key] = " ".join(sorted(shard for _, shard in videos[video(key)]))
                continue
            if subset != CC_SUBSET:
                reason = f"{st}: {'not in the manifest' if subset is None else 'subset ' + subset}"
                bad[reason] += 1
                examples.setdefault(reason, key)
    print(f"## {mix}")
    for st in sorted(rows):
        print(f"  {st:12s} {rows[st]:8d} rows {hours[st] / 3600:8.1f} h")
    for reason, n in sorted(bad.items()):
        print(f"  NOT CC BY: {reason}: {n} rows (e.g. {examples[reason]})")
    for key, shards in sorted(by_video.items()):
        print(f"  not in the manifest, video only in {CC_SUBSET}: {key} ({shards})")
    if not bad:
        print("  all rows CC BY" + (" (check the ids above in the raw tars)" if by_video else ""))
    return sum(bad.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mixes", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    subsets, videos = load_subsets(args.manifest)
    print(f"manifest: {len(subsets)} ids, {dict(Counter(subsets.values()))}")
    total = sum(audit(m, subsets, videos) for m in args.mixes)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
