"""Kana-CER of a scored eval set per voice prompt (e.g. the voices/ of a release, one per item).

Runs in the benchmark environment, like scripts/score_jvs_cer.py, on an eval directory that
eval_tts.py --from-audio has scored (asr_audio/kana.jsonl). The prompt of each item comes from
--items (relative to the directory): eval_tts.py --from-audio writes items.json again with the
set's own prompts, so runs/run_voices_cc.sh keeps its voices assignment in items_voices.json:

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/release/per_prompt_cer.py runs/D1_cc_6l/eval/step00160000/jsut_rotation_cfg1.0_wheelvoices \
        --items items_voices.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jkyb_eval.metrics import character_error_rate
from jkyb_eval.normalization import canonicalize_yomi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--items", default="items.json", help="file in each directory with key / yomi / prompt")
    args = parser.parse_args()
    for d in args.dirs:
        items = json.loads((d / args.items).read_text())
        kana = {json.loads(line)["key"]: json.loads(line)["yomi"] for line in open(d / "asr_audio" / "kana.jsonl")}
        cer: dict[str, list[float]] = defaultdict(list)
        for it in items:
            ref, hyp = canonicalize_yomi(it["yomi"]), canonicalize_yomi(kana[it["key"]])
            cer[Path(it["prompt"]).stem].append(min(1.0, character_error_rate(ref, hyp)))
        print(f"## {d} ({args.items})")
        print("| prompt | items | Kana-CER | items with CER > 20% |")
        print("|---|---:|---:|---:|")
        for name, v in sorted(cer.items()):
            print(f"| {name} | {len(v)} | {100 * sum(v) / len(v):.2f}% | {sum(x > 0.2 for x in v)} |")


if __name__ == "__main__":
    main()
