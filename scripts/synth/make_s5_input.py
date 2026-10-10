"""S5 (C2 hard mining): re-synthesize the S3 sentences whose target the teacher misread.

The S3 QC dropped 40% of the corpus, 84k sentences because Irodori itself read the tagged
word wrong (43% of the LLM sentences written for hard words, 40% of the Aozora ruby rows):
the hardest readings are exactly the ones missing from the training mix. Irodori reads kana
reliably (kana oracle of the Irodori experiments: 85% of its systematic misreadings fixed), so
here the misread words are given to the teacher in kana, while the training transcript stays
the original kanji text (finalize_s3.py --encode takes it from texts.jsonl by id).

A row is selected when it was rejected and at least one tagged target was not exact. Each such
target's word (the kanji run around it) is replaced by its reading; ids and reference voices
are those of S3, the seed is new.

    python3 scripts/synth/make_s5_input.py --kana hira --out data/synth/s5_input_hira.jsonl
    python3 scripts/synth/make_s5_input.py --kana kata --sample 1000 --out data/synth/s5_pilot_kata.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jkyb_kana_oracle import to_hiragana, word_region  # noqa: E402

ROOT = Path("data/synth")
MAX_KANA_PER_CHAR = 6


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pilot_hash(key: str) -> int:
    return int(hashlib.md5(f"pilot:{key}".encode()).hexdigest(), 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=Path, default=ROOT / "texts.jsonl")
    parser.add_argument("--s3-input", type=Path, default=ROOT / "s3_input.jsonl")
    parser.add_argument("--qc-dir", type=Path, default=ROOT / "s3_full" / "qc",
                        help="qc.jsonl and g2p_all.jsonl (qc_g2p/details/all.jsonl) of the S3 QC")
    parser.add_argument("--kana", choices=("hira", "kata"), default="hira")
    parser.add_argument("--sample", type=int, default=0, help="a pilot subset of about N rows (hash of the id)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rejected = {r["id"] for r in read_jsonl(args.qc_dir / "qc.jsonl") if not r["accepted"] and r["target_exact"] is False}
    failed: dict[str, set[str]] = {}  # sentence id -> keys of its targets that were not exact
    with (args.qc_dir / "g2p_all.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            base = r["key"].split("__t")[0]
            if base in rejected and not r["target_exact"]:
                failed.setdefault(base, set()).add(r["key"])
    inputs = {r["id"]: r for r in read_jsonl(args.s3_input)}

    stats: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as out:
        for t in read_jsonl(args.texts):
            keys = failed.get(t["id"])
            if not keys or (args.sample and pilot_hash(t["id"]) % len(failed) >= args.sample):
                continue
            targets = [t] + list(t.get("extra_targets") or [])
            regions = sorted(
                {word_region(x["tagged_text"], x["tagged_yomi"]) for x in targets if x["key"] in keys},
                key=lambda r: (r[0], -r[1]),
            )
            if any(len(reading) > MAX_KANA_PER_CHAR * (end - start) for start, end, reading, _ in regions):
                stats["misaligned"] += 1  # a reading far too long for its word
                continue
            text, words, last = t["text"], [], len(t["text"]) + 1
            for start, end, reading, aligned in reversed(regions):
                if end > last:  # nested in a region already replaced
                    continue
                kana = to_hiragana(reading) if args.kana == "hira" else reading
                words.append({"word": text[start:end], "reading": reading})
                text = text[:start] + kana + text[end:]
                last = start
                stats["words"] += 1
                stats["unaligned"] += not aligned
            inp = inputs[t["id"]]
            seed = int(hashlib.md5(f"seed:s5:{t['id']}".encode()).hexdigest(), 16) % (2**31)
            out.write(json.dumps({**inp, "text": text, "seed": seed, "text_orig": t["text"], "kana_words": words[::-1]},
                                 ensure_ascii=False) + "\n")
            stats["rows"] += 1
            by_source[t["source"]] += 1
    print(f"{stats['rows']} rows, {stats['words']} words in kana ({stats['unaligned']} target-only; "
          f"{stats['misaligned']} rows dropped as misaligned) -> {args.out}")
    print("  " + ", ".join(f"{s} {n}" for s, n in by_source.most_common()))


if __name__ == "__main__":
    main()
