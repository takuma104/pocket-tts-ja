"""JKYB-mini accuracy by how often the training text shows the target's (kanji, reading) unit.

Diagnosis only (C0-3): the benchmark keys are never used to choose synthetic text. The units
and their exposures before C2 come from `build_s6_texts.py exposure` (Sudachi counts over the
mix transcripts plus the tagged targets of the accepted S3 rows); `--added` adds the accepted
rows of the C2 sets, so the same benchmark sentences can be followed from bucket to bucket as
the hard-mining sets come in. Rows are bucketed by the exposure *before* C2 and the table also
gives each bucket's median exposure after the listed sets.

    uv run python scripts/jkyb_exposure_curve.py --remote galleria2-direct \
        --added data/synth/s7_texts.jsonl=data/synth/s7/qc.jsonl \
        --added data/synth/s6_texts.jsonl=data/synth/s6a/qc.jsonl \
        --runs before=runs/B1_4k/eval/step00370000,runs/B1_4k/eval/step00375000 \
               s7=runs/B1_4k/eval/step00395000,runs/B1_4k/eval/step00400000
"""

import argparse
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

REMOTE_REPO = "~/co/pocket-tts-ja"
TAG = "jkyb_mini_rotation_cfg2.0"
BINS = [(0, 0), (1, 9), (10, 29), (30, 99), (100, 299), (300, 999), (1000, 10**9)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 96) if "ァ" <= c <= "ヶ" else c for c in text)


def to_katakana(text: str) -> str:
    return "".join(chr(ord(c) + 96) if "ぁ" <= c <= "ゖ" else c for c in text)


def unit_of(row: dict[str, Any], table: dict[str, Any]) -> str | None:
    """Exposure unit of a benchmark row: the tagged kanji with its reading, else the key's."""
    match = re.search(r"<(.+?)>", row["tagged_text"])
    assert match is not None
    word = match.group(1)
    category = row["reading_category"]
    cands: list[str] = []
    for reading in row["readings"]["natural"]:
        if category == "joyo_appendix_reading":
            cands.append(f"F:{word}_{reading}")
        else:
            cands.append(f"{word}_{to_hiragana(reading) if category == 'kun_yomi' else reading}")
    # a sound-changed target (ペイ for 併_ヘイ, ぴき for 匹_ひき): the benchmark key has the plain reading
    key_word, key_reading, _ = row["key"].rsplit("_", 2)
    if category == "joyo_appendix_reading":
        cands.append(f"F:{key_word}_{to_katakana(key_reading)}")
    else:
        cands += [f"{key_word}_{key_reading[:n]}" for n in range(len(key_reading), 0, -1)]
    return next((c for c in cands if c in table), None)


def details(step_dir: Path, remote: str | None) -> dict[str, bool]:
    path = step_dir / TAG / "results" / "details" / "all.jsonl"
    if not path.exists() and remote:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{remote}:{REMOTE_REPO}/{path.as_posix()}", str(path)], check=True)
    return {r["key"]: bool(r["target_exact"]) for r in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/jkyb_mini.jsonl"))
    parser.add_argument("--exposure", type=Path, default=Path("data/synth/work/s6_exposure.json"))
    parser.add_argument("--added", nargs="*", default=[], metavar="TEXTS=QC",
                        help="a C2 set: its texts (unit / units / extra_targets) and the QC rows that accepted them")
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=STEP_DIR[,STEP_DIR...]",
                        help="eval step directories pooled under one column")
    parser.add_argument("--remote", default=None, help="fetch missing details/all.jsonl from this host")
    args = parser.parse_args()

    table: dict[str, dict[str, int]] = json.loads(args.exposure.read_text())["units"]
    before: Counter[str] = Counter({u: v["train"] + v["target"] for u, v in table.items()})
    added: Counter[str] = Counter()
    for spec in args.added:
        texts, qc = spec.split("=")
        accepted = {r["id"] for r in read_jsonl(Path(qc)) if r["accepted"]}
        for r in read_jsonl(Path(texts)):
            if r["id"] in accepted:
                added.update(r["units"] if "units" in r else [r["unit"]] + [t["unit"] for t in r["extra_targets"]])

    rows = read_jsonl(args.dataset)
    units = {r["key"]: unit_of(r, table) for r in rows}
    columns: dict[str, list[dict[str, bool]]] = {}
    for spec in args.runs:
        label, dirs = spec.split("=")
        columns[label] = [details(Path(d), args.remote) for d in dirs.split(",")]

    def cell(keys: list[str], evals: list[dict[str, bool]]) -> str:
        hits = [e[k] for e in evals for k in keys if k in e]
        return f"{100 * sum(hits) / len(hits):.1f}%" if hits else "–"

    print(f"| 露出（C2 前） | 例文 | C2 後の露出の中央値 | {' | '.join(columns)} |")
    print("|---|---:|---:|" + "---:|" * len(columns))
    groups: list[tuple[str, list[str]]] = []
    for lo, hi in BINS:
        label = "0" if hi == 0 else f"{lo}〜{hi}" if hi < 10**8 else f"{lo}〜"
        groups.append((label, [k for k, u in units.items() if u is not None and lo <= before[u] <= hi]))
    groups.append(("単位に対応なし", [k for k, u in units.items() if u is None]))
    groups.append(("全体", list(units)))
    for label, keys in groups:
        after = [before[units[k]] + added[units[k]] for k in keys if units[k] is not None]
        median = f"{statistics.median(after):.0f}" if after else "–"
        print(f"| {label} | {len(keys)} | {median} | " + " | ".join(cell(keys, e) for e in columns.values()) + " |")


if __name__ == "__main__":
    main()
