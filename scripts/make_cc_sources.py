"""CC-BY line (plan B-CC): the parts of the existing manifests that carry no NC-licensed speech.

Emilia has two parts: Emilia proper (CC BY-NC 4.0) and Emilia-YODAS (CC BY 4.0, source_type
"yodas"). The synthetic speech was cloned from reference utterances of both. Next to each
input this writes the rows that only involve Emilia-YODAS:

  data/emilia_ja/train_rechecked_yodas.jsonl, valid_yodas.jsonl   Emilia-YODAS rows
  data/emilia_ja_gain/train_yodas.jsonl                            their gain-changed copies
  data/synth/<set>/manifest_ccref.jsonl                            synthetic rows whose reference
                                                                   voice is an Emilia-YODAS utterance

The files stay in the directories of their sources, so `lat_file` needs no rewriting;
scripts/build_train_mix.py combines them (with data/yodas3_kept) into the training manifest.

    uv run python scripts/make_cc_sources.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

SYNTH_SETS = ("s3_full", "s4", "s5", "s6a", "s6b", "s7", "s9")


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{path}: {len(rows)} rows, {hours:.1f} h, {len({r['speaker'] for r in rows})} speakers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    emilia = args.data / "emilia_ja"

    # every Emilia utterance that was ever a candidate, so that reference ids resolve
    source: dict[str, str] = {r["id"]: r["source_type"] for name in ("train", "valid") for r in read(emilia / f"{name}.jsonl")}
    write(emilia / "train_rechecked_yodas.jsonl", [r for r in read(emilia / "train_rechecked_full.jsonl") if r["source_type"] == "yodas"])
    write(emilia / "valid_yodas.jsonl", [r for r in read(emilia / "valid.jsonl") if r["source_type"] == "yodas"])
    gain = args.data / "emilia_ja_gain"
    write(gain / "train_yodas.jsonl", [r for r in read(gain / "train_full.jsonl") if source.get(r["id"].removesuffix("#g")) == "yodas"])
    for name in SYNTH_SETS:
        rows = read(args.data / "synth" / name / "manifest.jsonl")
        write(args.data / "synth" / name / "manifest_ccref.jsonl", [r for r in rows if source.get(r.get("ref_id", "")) == "yodas"])


if __name__ == "__main__":
    main()
