"""Kana-CER of result directories (kana-whisper transcripts from asr_transcribe.py), by gender.

The reference is the item's verified reading (``yomi``: JSUT kana, JVS labels) or else the reading
of its text (pyopenjtalk), canonicalized as JKYB; or, for
items without a text (YODAS3 clips), the transcript of the recording itself (``--ref-dir orig``),
which measures how much the round trip changes what the ASR hears.

    uv run python scripts/eval/kana_cer.py exp/irodori_distill/p0/recon_jvs orig dacvae mimi soup4
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana, reading


def load_asr(root: Path, name: str) -> dict[str, str]:
    path = root / f"asr_{name}" / "kana.jsonl"
    return {row["key"]: canonical_kana(row["yomi"]) for row in map(json.loads, open(path, encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--ref-dir", default=None, help="use this result's transcripts as the reference (no text)")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    if args.ref_dir:
        asr_ref = load_asr(args.root, args.ref_dir)
        refs = {it["key"]: asr_ref[it["key"]] for it in items if asr_ref.get(it["key"])}
        label = f"transcript of {args.ref_dir}"
    else:
        refs = {it["key"]: canonical_kana(it["yomi"]) if it.get("yomi") else reading(it["text"])
                for it in items if it.get("text")}
        label = "verified reading (yomi) where given, else the reading of the text"
    gender = {it["key"]: it.get("gender", "") for it in items}
    groups = ["all"] + [g for g in ("F", "M") if any(gender[k] == g for k in refs) and len({gender[k] for k in refs}) > 1]
    print(f"reference: {label}, {len(refs)} clips")
    print("| result | " + " | ".join(f"Kana-CER {g}" for g in groups) + " | exact |")
    print("|---|" + "---:|" * (len(groups) + 1))
    for name in args.names:
        hyp = load_asr(args.root, name)
        edits = {g: 0 for g in groups}
        total = {g: 0 for g in groups}
        exact = []
        for key, ref in refs.items():
            if key not in hyp:
                continue
            d = Levenshtein.distance(ref, hyp[key])
            exact.append(d == 0)
            for g in groups:
                if g == "all" or gender[key] == g:
                    edits[g] += d
                    total[g] += len(ref)
        cells = " | ".join(f"{100 * edits[g] / max(total[g], 1):.2f}%" for g in groups)
        print(f"| {name} | {cells} | {100 * np.mean(exact):.1f}% |")


if __name__ == "__main__":
    main()
