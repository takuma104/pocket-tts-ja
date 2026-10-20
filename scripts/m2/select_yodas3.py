"""M2 CC data: training sets of YODAS3 clips for the decoder, from score_yodas3.py's measures (plan mimi_ft_cc_plan.md).

The pool: clips with --min-voiced voiced frames, not band-limited (the long-term spectrum at 8-11 kHz at least
--min-top dB under 1-4 kHz, and 11-12 kHz no more than --max-drop dB under 8-11 kHz) and not clipped (at most
--max-clipped of the samples at full scale). Clips are split by their median F0 (--f0-split Hz: F above, M below),
and per gender the --per-gender best by --rank are kept:

    all   the whole pool (no ranking; --per-gender is ignored)
    hf    the high-band harmonics: comb at 4-6 kHz plus 6-8 kHz
    mos   UTMOSv2 after the 40 Hz high-pass (mos_eval.py's caches under --mos-root, every name in it)
    mosz  UTMOSv2 and UTMOS22 (same caches), each as a z-score within the gender, summed

--max-per-speaker caps the clips of one speaker (0: no cap), the best ones kept. The videos of --exclude (by default
yodas3_exclude.txt next to this script: synthetic voices heard on the listening pages) are left out. Writes <out>/<name>_f.jsonl and <name>_m.jsonl (id, duration capped at --max-seconds: the trainer drops longer rows and
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
    parser.add_argument("--rank", choices=("all", "hf", "mos", "mosz"), default="all")
    parser.add_argument("--per-gender", type=int, default=10000)
    parser.add_argument("--max-per-speaker", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: <dir>/sets")
    parser.add_argument("--mos-root", type=Path, default=Path("exp/m2/y3mos"))
    parser.add_argument("--min-voiced", type=int, default=20)
    parser.add_argument("--min-top", type=float, default=-30.0)
    parser.add_argument("--max-drop", type=float, default=12.0)
    parser.add_argument("--max-clipped", type=float, default=1e-4)
    parser.add_argument("--f0-split", type=float, default=165.0)
    parser.add_argument("--max-seconds", type=float, default=7.99)
    parser.add_argument("--exclude", type=Path, default=Path(__file__).with_name("yodas3_exclude.txt"),
                        help="video ids to leave out, one per line ('#' starts a comment)")
    args = parser.parse_args()
    manifest = {r["id"]: r for r in map(json.loads, open(args.dir / "manifest.jsonl"))}
    scores = {r["id"]: r for r in map(json.loads, open(args.dir / "scores.jsonl"))}
    mos: dict[str, float] = {}
    mos22: dict[str, float] = {}
    if args.rank in ("mos", "mosz"):
        for cache in sorted(args.mos_root.glob("mos_hp40_*.json")):
            for k, v in json.loads(cache.read_text()).items():
                mos[k], mos22[k] = v["utmosv2"], v["utmos22"]

    def keep(s: dict[str, Any]) -> bool:
        return (s.get("voiced", 0) >= args.min_voiced and s["top"][0] >= args.min_top
                and s["top"][1] - s["top"][0] >= -args.max_drop and s["clipped"] <= args.max_clipped)

    excluded = {line.split("#")[0].strip() for line in open(args.exclude)} - {""} if args.exclude.exists() else set()
    pool = [s for s in scores.values() if keep(s) and (args.rank not in ("mos", "mosz") or s["id"] in mos)]
    dropped = sum(manifest[s["id"]]["video"] in excluded for s in pool)
    pool = [s for s in pool if manifest[s["id"]]["video"] not in excluded]
    print(f"{len(scores)} scored, pool {len(pool)}" + (f" (with UTMOSv2: {len(mos)})" if mos else "")
          + f", {dropped} clips of {len(excluded)} excluded videos left out")
    out = args.out or args.dir / "sets"
    out.mkdir(parents=True, exist_ok=True)
    for gender, female in (("f", True), ("m", False)):
        group = [s for s in pool if (s["f0"] >= args.f0_split) == female]
        if args.rank == "hf":
            group.sort(key=lambda s: -(s["comb"][2] + s["comb"][3]))
        elif args.rank == "mos":
            group.sort(key=lambda s: -mos[s["id"]])
        elif args.rank == "mosz":
            v2, v22 = (np.array([m[s["id"]] for s in group]) for m in (mos, mos22))
            z = (v2 - v2.mean()) / v2.std() + (v22 - v22.mean()) / v22.std()
            group = [group[i] for i in np.argsort(-z, kind="stable")]
        if args.max_per_speaker:
            count: dict[str, int] = {}
            capped = []
            for s in group:
                spk = manifest[s["id"]]["speaker"]
                count[spk] = count.get(spk, 0) + 1
                if count[spk] <= args.max_per_speaker:
                    capped.append(s)
            group = capped
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
            line += (f", UTMOSv2 {np.mean([mos[s['id']] for s in chosen]):.3f}"
                     f" UTMOS22 {np.mean([mos22[s['id']] for s in chosen]):.3f} (means)")
        print(line, flush=True)


if __name__ == "__main__":
    main()
