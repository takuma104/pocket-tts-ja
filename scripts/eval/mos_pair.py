"""Paired differences of cached neural MOS between decoders (same utterances), from mos_eval.py's cache.

    uv run python scripts/eval/mos_pair.py exp/irodori_distill/p0/recon_jvs M2H_soup12hp M2H_H2ghSG4p100E M2H_H2ghSG4p100Er3

prints, for each name after the reference (the first one), the mean of (name - reference) over the files both have, ±
its standard error, for all files and for the female speakers (items.jsonl's gender), for UTMOSv2 and UTMOS22.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("ref")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--prefix", default="mos_hp40_", help="cache file prefix (mos_hp40_ = scores after the 40 Hz high-pass)")
    args = parser.parse_args()
    gender = {}
    items = args.root / "items.jsonl"
    if items.exists():
        for line in items.read_text().splitlines():
            item = json.loads(line)
            gender[item["key"]] = item.get("gender", "")

    def load(name: str) -> dict[str, dict[str, float]]:
        return json.loads((args.root / f"{args.prefix}{name}.json").read_text())

    ref = load(args.ref)
    for name in args.names:
        scores = load(name)
        keys = [k for k in ref if k in scores]
        for metric in ("utmosv2", "utmos22"):
            cells = []
            for subset in (keys, [k for k in keys if gender.get(k) == "F"]):
                d = np.array([scores[k][metric] - ref[k][metric] for k in subset])
                cells.append(f"{d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f}" if len(d) > 1 else "-")
            print(f"{name} - {args.ref} {metric}: {cells[0]}  F {cells[1]}  (n {len(keys)})")


if __name__ == "__main__":
    main()
