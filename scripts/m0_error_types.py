"""M0: which kana errors does the Mimi round trip add?

Aligns each kana-whisper transcript (original recording and Mimi reconstruction) to the
reading from the JVS phoneme labels, both canonicalized as in JKYB-Parakeet, and buckets
every edit. Long vowels are already folded by the canonicalization (オウ/オー -> オオ), so
a vowel-length error shows up as an inserted/deleted vowel after the same vowel.

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/m0_error_types.py exp/m0_mimi_recon
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from jkyb_eval.normalization import _KANA_VOWELS, canonicalize_yomi
from rapidfuzz.distance import Levenshtein

VOICED = dict(
    zip(
        "カキクケコサシスセソタチツテトハヒフヘホパピプペポ",
        "ガギグゲゴザジズゼゾダヂヅデドバビブベボバビブベボ",
    )
)
SMALL = set("ャュョァィゥェォ")


def classify(op: str, ref: str, hyp: str, i: int, j: int) -> str:
    r = ref[i] if i < len(ref) else ""
    h = hyp[j] if j < len(hyp) else ""
    if op in ("insert", "delete"):
        ch = h if op == "insert" else r
        ctx = ref if op == "delete" else hyp
        k = i if op == "delete" else j
        prev = ctx[k - 1] if k > 0 else ""
        if ch == "ッ":
            return "sokuon"
        if ch == "ン":
            return "moraic_n"
        if ch in SMALL:
            return "palatal"
        if ch in "アイウエオ" and _KANA_VOWELS.get(prev) == ch:
            return "vowel_length"
        return f"{op}_other"
    if op == "replace":
        if VOICED.get(r) == h or VOICED.get(h) == r:
            return "voicing"
        if r in SMALL or h in SMALL:
            return "palatal"
        if _KANA_VOWELS.get(r) and _KANA_VOWELS.get(r) == _KANA_VOWELS.get(h):
            return "consonant"  # same vowel, different consonant
        if r in "アイウエオ" and h in "アイウエオ":
            return "vowel"
        return "replace_other"
    return "?"


def edits(ref: str, hyp: str) -> Counter:
    c: Counter = Counter()
    for op, i, j in Levenshtein.editops(ref, hyp):
        c[classify(op, ref, hyp, i, j)] += 1
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    jvs = args.root / "jvs"
    items = [json.loads(line) for line in open(jvs / "items.jsonl")]
    asr = {
        kind: {
            json.loads(line)["key"]: json.loads(line)["yomi"]
            for line in open(jvs / f"asr_{kind}" / "kana.jsonl")
        }
        for kind in ("orig", "recon")
    }
    out: dict[str, dict[str, Counter]] = {}
    ref_chars: Counter = Counter()
    for it in items:
        if not it["yomi"]:
            continue
        group = "read" if it["subset"] in ("parallel100", "nonpara30") else it["subset"]
        ref = canonicalize_yomi(it["yomi"])
        ref_chars[group] += len(ref)
        for kind in ("orig", "recon"):
            hyp = canonicalize_yomi(asr[kind][it["key"]])
            out.setdefault(group, {}).setdefault(kind, Counter()).update(edits(ref, hyp))
    lines = ["| group | error type | orig (per 1k ref kana) | recon | Δ |", "|---|---|---:|---:|---:|"]
    for group, kinds in out.items():
        n = ref_chars[group] / 1000
        cats = sorted(set(kinds["orig"]) | set(kinds["recon"]), key=lambda c: -kinds["recon"][c])
        for c in cats:
            o, r = kinds["orig"][c] / n, kinds["recon"][c] / n
            lines.append(f"| {group} | {c} | {o:.2f} | {r:.2f} | {r - o:+.2f} |")
    md = "\n".join(lines)
    (args.root / "error_types.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
