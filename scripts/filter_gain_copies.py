"""Drop the gain copies of clips that a filter dropped from their original manifest.

A gain copy's id is the original's id plus "#g" (data/emilia_ja_gain). A copy is kept when its
original is in the filtered manifest or was never in the unfiltered one.

    uv run python scripts/filter_gain_copies.py data/emilia_ja_gain/train_yodas.jsonl \
        --before data/emilia_ja/train_rechecked_yodas.jsonl --after data/emilia_ja/train_rechecked_yodas_tail.jsonl \
        --out data/emilia_ja_gain/train_yodas_tail.jsonl
"""

import argparse
import json
from pathlib import Path


def ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gain", type=Path)
    parser.add_argument("--before", type=Path, required=True, help="original manifest before the filter")
    parser.add_argument("--after", type=Path, required=True, help="original manifest after the filter")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dropped = ids(args.before) - ids(args.after)
    kept = n = 0
    with open(args.gain, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if json.loads(line)["id"].split("#", 1)[0] not in dropped:
                out.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    print(f"{len(dropped)} originals dropped; kept {kept} / {n} gain copies -> {args.out}")


if __name__ == "__main__":
    main()
