"""Row-by-row comparison of JKYB-schema evals against a baseline (kana oracle, seeds, settings).

For each other run: how many of the baseline's errors it gets right, and how many of the
baseline's correct rows it breaks, overall and by the baseline's error type
(analyze_jkyb_errors.classify) and reading category.

    uv run python scripts/compare_jkyb_runs.py \
        runs/B1_4k/eval/step00300000/jkyb_mini_rotation_cfg2.0/results \
        kata=runs/B1_4k/eval/step00300000/jkyb_mini_kata_rotation_cfg2.0/results \
        hira=runs/B1_4k/eval/step00300000/jkyb_mini_hira_rotation_cfg2.0/results
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_jkyb_errors import (  # noqa: E402
    CATEGORIES,
    ERROR_TYPES,
    JKYB_FULL,
    classify,
    rate,
    read_jsonl,
    reading_tables,
    table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="jkyb-eval results directory")
    parser.add_argument("others", nargs="+", help="LABEL=results directory")
    parser.add_argument("--out", type=Path, default=None, help="markdown report (default: stdout only)")
    args = parser.parse_args()

    base = {r["key"]: r for r in read_jsonl(args.baseline / "details" / "all.jsonl")}
    readings, _ = reading_tables(list(base.values()) + (read_jsonl(JKYB_FULL) if JKYB_FULL else []))
    etype = {k: classify(r, readings) for k, r in base.items() if not r["target_exact"]}
    runs = {}
    for spec in args.others:
        label, path = spec.split("=", 1)
        runs[label] = {r["key"]: bool(r["target_exact"]) for r in read_jsonl(Path(path) / "details" / "all.jsonl")}
    labels = list(runs)
    wrong = [k for k, r in base.items() if not r["target_exact"]]
    right = [k for k, r in base.items() if r["target_exact"]]

    def fixed(keys: list[str], label: str) -> str:
        return f"{rate(sum(runs[label].get(k, False) for k in keys), len(keys))}"

    out = [f"# JKYB 行ごとの比較（基準: {args.baseline.parent}）", ""]
    out.append(table(
        ["", "基準", *labels],
        [["Accuracy", rate(len(right), len(base)), *[rate(sum(runs[lb].values()), len(runs[lb])) for lb in labels]],
         [f"基準の誤り（{len(wrong)}）が正解に", "–", *[fixed(wrong, lb) for lb in labels]],
         [f"基準の正解（{len(right)}）が誤りに", "–",
          *[rate(sum(not runs[lb].get(k, False) for k in right), len(right)) for lb in labels]]],
    ))
    out += ["", "## 基準の誤りタイプ別: 正解になった割合", ""]
    by_type = Counter(etype.values())
    out.append(table(["誤りタイプ", "件数", *labels],
                     [[name, str(by_type[t]), *[fixed([k for k in wrong if etype[k] == t], lb) for lb in labels]]
                      for t, name in ERROR_TYPES.items() if by_type[t]]))
    out += ["", "## 読み分類別: 正解率", ""]
    out.append(table(
        ["分類", "例文数", "基準", *labels],
        [[name, str(sum(r["reading_category"] == c for r in base.values())),
          rate(sum(r["target_exact"] for r in base.values() if r["reading_category"] == c),
               sum(r["reading_category"] == c for r in base.values())),
          *[fixed([k for k, r in base.items() if r["reading_category"] == c], lb) for lb in labels]]
         for c, name in CATEGORIES.items()],
    ))
    still = [k for k in wrong if not any(runs[lb].get(k, False) for lb in labels)]
    out += ["", f"## どの条件でも誤りのまま（{len(still)} 件）", ""]
    out.append(table(["例文（<> が対象）", "正解", "基準の読み", "誤りタイプ"],
                     [[base[k]["tagged_text"].replace("<", "&lt;"), base[k]["reference_target"], base[k]["mapped_target"] or "（なし）",
                       ERROR_TYPES[etype[k]]] for k in still[:40]]))
    report = "\n".join(out) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
