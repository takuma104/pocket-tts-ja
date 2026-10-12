"""How often is the first kana of an utterance wrong (the onset consonant)?

kana-whisper gets the first kana of generated speech wrong far more often than the others
("図書館。" -> ボショカン, "ヒマラヤの山は" -> キマラヤ): nothing before it helps to tell the
consonant apart, so it shows how clear the consonants are on their own. Per result:

  exact        clips whose reading is exactly the reference
  Kana-CER     edits over reference kana, all clips together
  1st only     clips whose only error is the first kana, as a share of the clips whose other
               kana are all right (exact + these)
  1st / 2nd / 3rd-5th / later
               share of reference kana at that position that are substituted or deleted
               (references of 8 kana or more)

Reference points (10/12): JVS recordings 0.6% / 4.5% (1st only / 1st), the same recordings through
Mimi 4.3% / 11.1%, the 700k teacher on JSUT 7.0% / 9.3%.

    uv run python scripts/analyze_first_kana.py teacher=runs/B1_4k/eval/step00700000/jsut_rotation_cfg2.0
    uv run python scripts/analyze_first_kana.py --dataset exp/onset/items.jsonl --reference text \
        real=exp/onset/asr_orig/kana.jsonl mimi=exp/onset/asr_orig_mimi/kana.jsonl
"""

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana

POSITIONS = ("1st", "2nd", "3rd-5th", "later")


def position(index: int) -> str:
    return "1st" if index == 0 else "2nd" if index == 1 else "3rd-5th" if index < 5 else "later"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+",
                        help="NAME=result directory (with asr_audio/kana.jsonl) or NAME=kana.jsonl")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/jsut_eval_1k.jsonl"),
                        help="JSONL with key and the reference (kana or yomi; text with --reference text)")
    parser.add_argument("--reference", choices=("kana", "text"), default="kana",
                        help="text: read the reference off the text with pyopenjtalk")
    parser.add_argument("--examples", type=int, default=0, help="print this many first-kana-only errors per result")
    args = parser.parse_args()

    refs: dict[str, str] = {}
    texts: dict[str, str] = {}
    for line in open(args.dataset):
        row = json.loads(line)
        if args.reference == "text":
            import pyopenjtalk

            reading = pyopenjtalk.g2p(row["text"], kana=True)
        else:
            reading = row.get("kana") or row.get("yomi") or ""
        if canonical_kana(reading):
            refs[row["key"]] = canonical_kana(reading)
            texts[row["key"]] = row.get("text", "")

    print("| result | clips | exact | Kana-CER | 1st only | " + " | ".join(POSITIONS) + " |")
    print("|---|---:|---:|---:|---:|" + "---:|" * len(POSITIONS))
    examples: list[str] = []
    for spec in args.results:
        name, path = spec.split("=", 1)
        kana_file = Path(path) if path.endswith(".jsonl") else Path(path) / "asr_audio" / "kana.jsonl"
        asr = {json.loads(line)["key"]: json.loads(line)["yomi"] for line in open(kana_file)}
        clips = exact = only = shown = edits = kana = 0
        wrong = dict.fromkeys(POSITIONS, 0)
        total = dict.fromkeys(POSITIONS, 0)
        for key, ref in refs.items():
            if key not in asr:
                continue
            ops = Levenshtein.editops(ref, canonical_kana(asr[key]))
            clips += 1
            exact += not ops
            edits += len(ops)
            kana += len(ref)
            # an insertion in front of the first kana counts only when it starts the prediction
            first_only = len(ops) == 1 and ops[0].src_pos == 0 and not (ops[0].tag == "insert" and ops[0].dest_pos > 0)
            only += first_only
            if first_only and shown < args.examples:
                examples.append(f"    {name}: {texts[key]}  ->  {asr[key]}")
                shown += 1
            if len(ref) >= 8:
                bad = {op.src_pos for op in ops if op.tag != "insert"}
                for index in range(len(ref)):
                    total[position(index)] += 1
                    wrong[position(index)] += index in bad
        if not clips:
            print(f"| {name} | 0 |" + " |" * (3 + len(POSITIONS)))
            continue
        rates = " | ".join(f"{100 * wrong[p] / max(total[p], 1):.1f}%" for p in POSITIONS)
        print(f"| {name} | {clips} | {100 * exact / clips:.1f}% | {100 * edits / kana:.2f}% | "
              f"{100 * only / max(exact + only, 1):.1f}% | {rates} |")
    if examples:
        print()
        print("\n".join(examples))


if __name__ == "__main__":
    main()
