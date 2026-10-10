"""C0 kana oracle: JKYB-schema rows whose target word is written in kana.

If the model reads the kana version right where it misreads the kanji, the error is on the text
side (it does not know the reading); if it still fails, the problem is acoustic (or the ASR).
Ported from the Irodori experiments (jkyb_kana_oracle.py). For every row, `text` becomes:

  kata / hira  the whole kanji run around the target replaced by its reading (従容 -> ショウヨウ),
               aligned from tagged_yomi; when the alignment fails (digits next to the target
               etc.) only the target kanji is replaced
  char         only the target kanji replaced, in hiragana (交ぜ書き)

The other fields are unchanged, so jkyb-eval scores the audio against the same yomi; the
original sentence is kept as `text_orig` (Text CER is meaningless for these files).

    uv run python scripts/jkyb_kana_oracle.py data/eval/jkyb_mini.jsonl --out-dir data/eval/oracle
    uv run python scripts/eval_tts.py runs/B1_4k --snapshot 300000 --dataset data/eval/oracle/jkyb_mini_kata.jsonl
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KANJI = re.compile(r"[㐀-鿿々〆ヶ]")
PUNCTUATION = set("、。？！「」『』・，．?!")
KANA_PARTICLE_SOUNDS = {"は": "ワ", "へ": "エ", "を": "オ"}
MAX_KANA_PER_CHAR = 12  # a run of digits/latin can read long (A4 -> エーヨン)
YOMI_BREAKS = PUNCTUATION | {"<", ">"}


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


@dataclass(frozen=True)
class Segment:
    kind: str  # "open", "close", "kana", "punct", "run"
    text: str


def segment(tagged_text: str) -> list[Segment]:
    """Split tagged text into tag markers, kana, punctuation, and other runs."""
    segments: list[Segment] = []
    for ch in tagged_text:
        if ch == "<":
            segments.append(Segment("open", ch))
        elif ch == ">":
            segments.append(Segment("close", ch))
        elif ("ぁ" <= ch <= "ゖ" or "ァ" <= ch <= "ヺ" or ch == "ー") and ch != "ヶ":
            segments.append(Segment("kana", ch))
        elif ch in PUNCTUATION:
            segments.append(Segment("punct", ch))
        elif segments and segments[-1].kind == "run":
            segments[-1] = Segment("run", segments[-1].text + ch)
        else:
            segments.append(Segment("run", ch))
    return segments


def align_runs(tagged_text: str, tagged_yomi: str) -> list[tuple[Segment, str]] | None:
    """Pair each segment with its reading, or None when the alignment fails.

    Kana and tags match themselves (は/へ/を also their particle sounds), punctuation is
    optional, and a run takes the shortest reading that lets the rest match. The search is
    memoized on (segment, yomi position): long literary sentences with many kanji runs made
    the equivalent regex backtrack exponentially.
    """
    segments = segment(tagged_text)
    n, m = len(segments), len(tagged_yomi)
    dead: set[tuple[int, int]] = set()
    lengths: list[int] = [0] * n

    def options(seg: Segment, j: int) -> list[int]:
        if seg.kind in ("open", "close"):
            return [1] if tagged_yomi.startswith(seg.text, j) else []
        if seg.kind == "kana":
            kana = to_katakana(seg.text)
            allowed = {kana, *KANA_PARTICLE_SOUNDS.get(seg.text, "")}
            return [1] if j < m and tagged_yomi[j] in allowed else []
        if seg.kind == "punct":
            return ([1] if tagged_yomi.startswith(seg.text, j) else []) + [0]
        limit = min(m - j, MAX_KANA_PER_CHAR * len(seg.text))
        stop = next((k for k in range(limit) if tagged_yomi[j + k] in YOMI_BREAKS), limit)
        return list(range(1, stop + 1))  # a reading never runs across punctuation or a tag

    # iterative depth-first search (sentences have up to ~100 segments, but stay off the recursion limit)
    stack: list[tuple[int, int, list[int]]] = [(0, 0, options(segments[0], 0))] if n else []
    while stack:
        i, j, todo = stack[-1]
        if not todo:
            dead.add((i, j))
            stack.pop()
            continue
        k = todo.pop(0)
        lengths[i] = k
        if i + 1 == n:
            if j + k == m:
                break
            continue
        if (i + 1, j + k) not in dead:
            stack.append((i + 1, j + k, options(segments[i + 1], j + k)))
    else:
        if n or m:
            return None
    out, j = [], 0
    for seg, k in zip(segments, lengths, strict=True):
        out.append((seg, tagged_yomi[j : j + k]))
        j += k
    return out


def word_span(aligned: list[tuple[Segment, str]]) -> tuple[int, int] | None:
    """Segment index range [start, end) of the kanji word containing the target."""
    open_index = next(i for i, (seg, _) in enumerate(aligned) if seg.kind == "open")
    close_index = next(i for i, (seg, _) in enumerate(aligned) if seg.kind == "close")
    start = open_index
    if start > 0 and aligned[start - 1][0].kind == "run":
        if not all(KANJI.match(ch) for ch in aligned[start - 1][0].text):
            return None
        start -= 1
    end = close_index + 1
    if end < len(aligned) and aligned[end][0].kind == "run":
        if not all(KANJI.match(ch) for ch in aligned[end][0].text):
            return None
        end += 1
    inside = aligned[open_index + 1 : close_index]
    if not all(seg.kind == "run" and KANJI.match(seg.text) for seg, _ in inside):
        return None
    return start, end


def substitute_word(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """(before, word, katakana reading, after) around the kanji word, or None if not aligned."""
    aligned = align_runs(row["tagged_text"], row["tagged_yomi"])
    span = word_span(aligned) if aligned else None
    if aligned is None or span is None:
        return None
    start, end = span
    word = "".join(seg.text for seg, _ in aligned[start:end] if seg.kind == "run")
    reading = "".join(y for seg, y in aligned[start:end] if seg.kind == "run")
    before = "".join(seg.text for seg, _ in aligned[:start] if seg.kind not in ("open", "close"))
    after = "".join(seg.text for seg, _ in aligned[end:] if seg.kind not in ("open", "close"))
    return before, word, reading, after


def word_region(tagged_text: str, tagged_yomi: str) -> tuple[int, int, str, bool]:
    """(start, end, katakana reading, aligned) of the target word in the untagged text.

    The word is the kanji run around the target; when the alignment fails it is the tagged
    span itself (aligned = False).
    """
    aligned = align_runs(tagged_text, tagged_yomi)
    span = word_span(aligned) if aligned else None
    if aligned is None or span is None:
        start = tagged_text.index("<")
        reading = re.search(r"<([^>]+)>", tagged_yomi).group(1)  # type: ignore[union-attr]
        return start, tagged_text.index(">") - 1, reading, False
    first, last = span
    start = sum(len(seg.text) for seg, _ in aligned[:first] if seg.kind not in ("open", "close"))
    length = sum(len(seg.text) for seg, _ in aligned[first:last] if seg.kind == "run")
    return start, start + length, "".join(y for seg, y in aligned[first:last] if seg.kind == "run"), True


def substitute_char(row: dict[str, Any], hiragana: bool) -> str:
    reading = re.search(r"<([^>]+)>", row["tagged_yomi"]).group(1)  # type: ignore[union-attr]
    return re.sub(r"<[^>]+>", to_hiragana(reading) if hiragana else reading, row["tagged_text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="JKYB-schema jsonl (e.g. data/eval/jkyb_mini.jsonl)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict[str, Any]]] = {"kata": [], "hira": [], "char": []}
    unaligned = 0
    for row in rows:
        sub = substitute_word(row)
        if sub is None:
            unaligned += 1
            texts = {"kata": substitute_char(row, False), "hira": substitute_char(row, True)}
            kana_word = None
        else:
            before, word, reading, after = sub
            texts = {"kata": before + reading + after, "hira": before + to_hiragana(reading) + after}
            kana_word = {"word": word, "reading": reading}
        texts["char"] = substitute_char(row, True)
        for name, text in texts.items():
            out[name].append({**row, "text": text, "text_orig": row["text"], "kana_word": kana_word})
    for name, variant in out.items():
        path = args.out_dir / f"{args.dataset.stem}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in variant:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(variant)} rows -> {path}")
    print(f"{unaligned} rows not aligned (only the target kanji replaced in kata/hira)")


if __name__ == "__main__":
    main()
