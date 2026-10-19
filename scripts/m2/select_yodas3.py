"""M2 CC data: training sets of YODAS3 clips for the decoder, from score_yodas3.py's measures (plan mimi_ft_cc_plan.md).

The pool: clips with --min-voiced voiced frames, not band-limited (the long-term spectrum at 8-11 kHz at least
--min-top dB under 1-4 kHz, and 11-12 kHz no more than --max-drop dB under 8-11 kHz) and not clipped (at most
--max-clipped of the samples at full scale). Clips are split by their median F0 (--f0-split Hz: F above, M below),
and per gender the --per-gender best by --rank are kept:

    all   the whole pool (no ranking; --per-gender is ignored)
    hf    the high-band harmonics: comb at 4-6 kHz plus 6-8 kHz
    mos   UTMOSv2 after the 40 Hz high-pass (mos_eval.py's caches under --mos-root, every name in it)

Writes <out>/<name>_f.jsonl and <name>_m.jsonl (id, duration capped at --max-seconds: the trainer drops longer rows and
reads a clip from its start) for train_mimi_decoder.py --data, and prints what the sets hold.

    uv run python scripts/m2/select_yodas3.py data/m2/yodas3 --name hf --rank hf --per-gender 10000
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="extract_yodas3.py's output with score_yodas3.py's scores.jsonl")
    parser.add_argument("--name", required=True)
    parser.add_argument("--rank", choices=("all", "hf", "mos"), default="all")
    parser.add_argument("--per-gender", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=None, help="default: <dir>/sets")
    parser.add_argument("--mos-root", type=Path, default=Path("exp/m2/y3mos"))
    parser.add_argument("--min-voiced", type=int, default=20)
    parser.add_argument("--min-top", type=float, default=-30.0)
    parser.add_argument("--max-drop", type=float, default=12.0)
    parser.add_argument("--max-clipped", type=float, default=1e-4)
    parser.add_argument("--f0-split", type=float, default=165.0)
    parser.add_argument("--max-seconds", type=float, default=7.99)
    args = parser.parse_args()
    manifest = {r["id"]: r for r in map(json.loads, open(args.dir / "manifest.jsonl"))}
    scores = {r["id"]: r for r in map(json.loads, open(args.dir / "scores.jsonl"))}
    mos: dict[str, float] = {}
    if args.rank == "mos":
        for cache in sorted(args.mos_root.glob("mos_hp40_*.json")):
            mos.update({k: v["utmosv2"] for k, v in json.loads(cache.read_text()).items()})

    def keep(s: dict[str, Any]) -> bool:
        return (s.get("voiced", 0) >= args.min_voiced and s["top"][0] >= args.min_top
                and s["top"][1] - s["top"][0] >= -args.max_drop and s["clipped"] <= args.max_clipped)

    pool = [s for s in scores.values() if keep(s) and (args.rank != "mos" or s["id"] in mos)]
    print(f"{len(scores)} scored, pool {len(pool)}" + (f" (with UTMOSv2: {len(mos)})" if mos else ""))
    out = args.out or args.dir / "sets"
    out.mkdir(parents=True, exist_ok=True)
    for gender, side in (("f", 1), ("m", -1)):
        group = [s for s in pool if side * (s["f0"] - args.f0_split) >= 0]
        if args.rank == "hf":
            group.sort(key=lambda s: -(s["comb"][2] + s["comb"][3]))
        elif args.rank == "mos":
            group.sort(key=lambda s: -mos[s["id"]])
        chosen = group if args.rank == "all" else group[: args.per_gender]
        with open(out / f"{args.name}_{gender}.jsonl", "w") as f:
            for s in chosen:
                f.write(json.dumps({"id": s["id"], "duration": min(manifest[s["id"]]["duration"], args.max_seconds)}) + "\n")
        hours = sum(min(manifest[s["id"]]["duration"], args.max_seconds) for s in chosen) / 3600
        comb = np.array([s["comb"] for s in chosen])
        speakers = len({manifest[s["id"]]["speaker"] for s in chosen})
        line = (f"{args.name}_{gender}: {len(chosen)} of {len(group)} clips, {hours:.1f} h (first 8 s), {speakers} speakers, "
                f"comb 4-6k {np.median(comb[:, 2]):.2f} 6-8k {np.median(comb[:, 3]):.2f} dB, "
                f"DNSMOS OVRL {np.median([manifest[s['id']]['dnsmos_ovrl'] for s in chosen]):.2f}")
        if mos:
            line += f", UTMOSv2 {np.median([mos[s['id']] for s in chosen]):.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
