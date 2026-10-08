"""Tabulate eval_tts.py results (runs/<run>/eval/step*/<set>/metrics.json) as markdown.

    uv run python scripts/summarize_evals.py runs/A5_8k runs/A5_4k
"""

import argparse
import json
import re
from pathlib import Path

COLUMNS = {
    "jkyb": [("accuracy", "Acc"), ("accuracy_on_yomi", "on"), ("accuracy_kun_yomi", "kun"),
             ("accuracy_joyo_appendix_reading", "appx"), ("target_kana_cer@1", "tgtCER"),
             ("sentence_kana_cer@1", "sentCER"), ("text_cer@1", "textCER"), ("no_eos_count", "noEOS")],
    "jvs": [("kana_cer@1", "kanaCER"), ("kana_cer@1_M", "M"), ("kana_cer@1_F", "F"),
            ("text_cer@1", "textCER"), ("sim_ecapa", "SIM"), ("no_eos_count", "noEOS"),
            ("silent_count", "silent")],
}


def fmt(key: str, v: float | None) -> str:
    if v is None:
        return "–"
    if key.endswith("_count"):
        return f"{int(v)}"
    if key == "sim_ecapa":
        return f"{v:.3f}"
    return f"{100 * v:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for kind, cols in COLUMNS.items():
        rows = []
        for run in args.runs:
            for m in sorted(run.glob("eval/step*/*/metrics.json")):
                tag = m.parent.name
                if not tag.startswith(kind) or "limit" in tag:
                    continue
                d = json.loads(m.read_text())
                step = int(re.search(r"step(\d+)", str(m)).group(1))  # type: ignore[union-attr]
                rows.append((run.name, tag, step, d))
        if not rows:
            continue
        print(f"\n### {kind}\n")
        print("| run | set | step | " + " | ".join(c for _, c in cols) + " |")
        print("|---|---|---:|" + "---:|" * len(cols))
        for run, tag, step, d in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            print(f"| {run} | {tag} | {step} | " + " | ".join(fmt(k, d.get(k)) for k, _ in cols) + " |")


if __name__ == "__main__":
    main()
