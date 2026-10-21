"""Lay out an evaluation set (pocket_tts_ja.eval_sets) for generation and scoring.

    <out>/items.jsonl        key, text, speaker, gender, yomi, prompt (the scripts/eval layout)
    <out>/orig/<key>.wav     link to the reference recording (similarity.py compares with it)
    <out>/teacher_rows.jsonl Irodori input rows (id, text, ref_wav = prompt, seed from the key)

    uv run python scripts/eval/make_eval_dir.py voices exp/irodori_distill/eval/voices
"""

import argparse
import json
from pathlib import Path

from pocket_tts_ja.eval_sets import SETS, build, item_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", choices=SETS)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    items = build(args.set)
    (args.out / "orig").mkdir(parents=True, exist_ok=True)
    with open(args.out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            row = {"key": it.key, "text": it.text, "speaker": it.speaker, "gender": it.gender,
                   "yomi": it.yomi, "prompt": it.prompt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out / "teacher_rows.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"id": it.key, "text": it.text, "ref_wav": it.prompt, "seed": item_seed(it.key)},
                               ensure_ascii=False) + "\n")
    for it in items:
        link = args.out / "orig" / f"{it.key}.wav"
        if not link.exists():
            link.symlink_to(Path(it.reference).resolve())
    print(f"{args.out}: {len(items)} items")


if __name__ == "__main__":
    main()
