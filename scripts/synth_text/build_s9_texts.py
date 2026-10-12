"""S9 text: very short utterances (2-8 tokens), for "short text -> short audio, then stop".

The stock inference generates every chunk of at most 50 tokens on its own, so the last clause
of a long sentence ("認知されている.") or a short last sentence of a paragraph is generated
alone, and people type one-word texts. The training mix has almost none (0.04% of the rows
have 5 tokens or fewer: every real clip is at least 3 s long and Irodori's duration predictor
stretches short text to 2-3 s), and the teacher goes on talking after a third of such texts.

Kinds of rows:
  tail   the last clause of an S1-S3 sentence (after its last "、")
  head   its first clause, closed with "。" (what a chunk cut at a comma looks like)
  sent   Tatoeba sentences of the S4 list
  word   frequent nouns of the S1-S3 sentences, as one-word utterances
Readings: the clause's part of the sentence yomi for tail/head, the S4 yomi for sent, plain
Sudachi for word. Texts of the short eval set (data/eval/short_ja.jsonl, kind tail) are left
out. Each row gets the clip length for irodori_synth.py: seconds = pad + morae * rate with a
random speaking rate, since Irodori's own duration predictor cannot be used here. In a pilot
on 845 Tatoeba texts the kana-whisper reading was exact for 4-9% of the 4-6 token clips with
the predicted length, and for 75% at 0.13 s per mora (0.20 s: 54%, the spare time fills up).

    uv run python scripts/synth_text/build_s9_texts.py --out data/synth/s9_texts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_filters import japanese_ratio, leak_key, quality_reason, yomi_ok  # noqa: E402
from synth_wiki import plain_sudachi_yomi  # noqa: E402

from pocket_tts_ja.text import normalize  # noqa: E402

SMALL_KANA = set("ャュョァィゥェォヮ")


def morae(yomi: str) -> int:
    return sum(1 for c in yomi if ("ァ" <= c <= "ヺ" or c == "ー") and c not in SMALL_KANA)


def clauses(text: str, yomi: str) -> list[tuple[str, str, str]]:
    """(kind, text, yomi) of the first and the last comma clause when text and yomi split alike."""
    t, y = text.split("、"), yomi.split("、")
    if len(t) < 2 or len(t) != len(y):
        return []
    out = [("tail", t[-1].strip(), y[-1].strip())]
    head = t[0].strip()
    if head and head[-1] not in "。！？!?":
        out.append(("head", head + "。", y[0].strip() + "。"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=Path, default=Path("data/synth/texts.jsonl"))
    parser.add_argument("--short-texts", type=Path, default=Path("data/synth/short_texts.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval/short_ja.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/ja_4k.model"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--tails", type=int, default=40000)
    parser.add_argument("--heads", type=int, default=8000)
    parser.add_argument("--words", type=int, default=8000)
    parser.add_argument("--rate", type=float, nargs=2, default=(0.11, 0.15), help="seconds per mora, range")
    parser.add_argument("--pad", type=float, nargs=2, default=(0.15, 0.35), help="seconds added, range")
    parser.add_argument("--min-seconds", type=float, default=0.6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    held_out = {leak_key(json.loads(line)["text"]) for line in open(args.eval) if json.loads(line)["kind"] == "tail"}
    rng = random.Random(0)
    seen: set[str] = set()
    pools: dict[str, list[dict[str, str]]] = {"tail": [], "head": [], "sent": [], "word": []}
    dropped: Counter[str] = Counter()

    def consider(kind: str, text: str, yomi: str, source: str) -> None:
        key = leak_key(text)
        if not key or key in seen:
            return
        if key in held_out:
            dropped["held_out"] += 1
            return
        if quality_reason(text, min_len=2, max_len=30) or japanese_ratio(text) < 0.9 or not yomi_ok(yomi):
            dropped[f"{kind}_quality"] += 1
            return
        if not 2 <= len(sp.encode(normalize(text))) <= args.max_tokens:
            return
        seen.add(key)
        pools[kind].append({"text": text, "yomi": yomi, "kind": kind, "source": source})

    nouns: Counter[str] = Counter()
    from sudachipy import Dictionary, SplitMode

    sudachi = Dictionary().tokenizer()
    for n, line in enumerate(open(args.texts, encoding="utf-8")):
        row = json.loads(line)
        for kind, text, yomi in clauses(row["text"], row["yomi"]):
            consider(kind, text, yomi, row["source"])
        if n % 5 == 0:  # a fifth of the corpus is plenty for a frequency list
            for m in sudachi.tokenize(row["text"], SplitMode.C):
                pos = m.part_of_speech()
                if pos[0] == "名詞" and pos[1] in ("普通名詞", "固有名詞") and len(m.surface()) >= 2:
                    nouns[m.surface()] += 1
    for line in open(args.short_texts, encoding="utf-8"):
        row = json.loads(line)
        consider("sent", row["text"], row["yomi"], "tatoeba")
    for word, _ in nouns.most_common(3 * args.words):
        yomi, fixes = plain_sudachi_yomi(word)
        for a, b, r in sorted(fixes, reverse=True):
            yomi = yomi[:a] + r + yomi[b:]
        consider("word", word + "。", yomi + "。", "s1-3")

    limits = {"tail": args.tails, "head": args.heads, "sent": len(pools["sent"]), "word": args.words}
    rows = []
    for kind, pool in pools.items():
        if kind != "word":  # words stay in frequency order
            rng.shuffle(pool)
        rows += pool[: limits[kind]]
        print(f"{kind}: {len(pool)} available, {min(len(pool), limits[kind])} kept")
    with open(args.out, "w", encoding="utf-8") as f:
        for k, row in enumerate(rows):
            n = morae(row["yomi"])
            seconds = max(args.min_seconds, rng.uniform(*args.pad) + n * rng.uniform(*args.rate))
            f.write(json.dumps({"id": f"syn_s9_{k}", "key": f"syn_s9_{k}", **row, "morae": n,
                                "seconds": round(seconds, 3)}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(dropped.most_common()), ensure_ascii=False))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
