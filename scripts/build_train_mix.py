"""Combine per-source latent manifests into one training manifest for pocket_tts_ja.loader.

Each --source is NAME=PATH[:HOURS_FRACTION]. Rows keep their fields; `lat_file` is rewritten
relative to the output manifest's directory (the loader resolves it from there). A fraction
subsamples (or repeats) that source so it makes up that share of the output hours; sources
without one are taken whole.

Synthetic rows (speaker "syn:<ref speaker>", with "ref_id") also get the reference speaker's
real utterance added under the same speaker label, so voice prompts for synthetic targets
can be the real recording the synthesis was conditioned on.

    uv run python scripts/build_train_mix.py --out data/mix/B1_train.jsonl \
        --source emilia=data/emilia_ja/train.jsonl --source irodori=data/synth/s3/manifest.jsonl:0.15
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out_dir = args.out.parent.resolve()

    sources: list[tuple[str, Path, float | None, list[dict[str, Any]]]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for spec in args.source:
        name, rest = spec.split("=", 1)
        path_s, _, frac = rest.partition(":")
        path = Path(path_s)
        rows = read(path)
        rel = os.path.relpath(path.parent.resolve(), out_dir)
        for r in rows:
            r["lat_file"] = os.path.normpath(os.path.join(rel, r["lat_file"]))
            r.setdefault("source_type", name)
            by_id[r["id"]] = r
        sources.append((name, path, float(frac) if frac else None, rows))

    fixed_hours = sum(sum(r["duration"] for r in rows) for _, _, f, rows in sources if f is None) / 3600
    frac_total = sum(f for _, _, f, _ in sources if f is not None)
    assert frac_total < 1.0, "fractions must sum to < 1"
    total_hours = fixed_hours / (1 - frac_total)
    out: list[dict[str, Any]] = []
    report = {}
    for name, _, frac, rows in sources:
        if frac is None:
            picked = rows
        else:
            want = frac * total_hours * 3600
            pool = rows[:]
            rng.shuffle(pool)
            picked, acc, i = [], 0.0, 0
            while acc < want and pool:  # wraps around (repeats) if the source is too small
                r = pool[i % len(pool)]
                picked.append(r)
                acc += r["duration"]
                i += 1
        out += picked
        report[name] = {"rows": len(picked), "hours": round(sum(r["duration"] for r in picked) / 3600, 1)}

    # real reference utterances for synthetic speakers
    refs_added = 0
    seen: set[str] = set()
    for r in list(out):
        ref_id = r.get("ref_id")
        if not ref_id or r["speaker"] in seen:
            continue
        seen.add(r["speaker"])
        real = by_id.get(ref_id)
        if real is not None:
            out.append({**real, "id": f"{ref_id}#ref", "speaker": r["speaker"]})
            refs_added += 1
    rng.shuffle(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["reference_rows_added"] = refs_added
    report["total_hours"] = round(sum(r["duration"] for r in out) / 3600, 1)
    Path(str(args.out) + ".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
