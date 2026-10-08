"""Character classes, quality filters and small helpers for the synthetic-text corpus (S1).

Shared by ``build_texts.py``, ``synth_wiki.py`` and ``synth_leakage.py``. Nothing here
rewrites a kept text: the filters only decide whether a sentence is usable as it is.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

KANJI_CLASS = "一-鿿々〆〇ヶ"
KANJI = re.compile(f"[{KANJI_CLASS}]")
KANA = re.compile(r"[ぁ-ゖァ-ヺー]")
HIRAGANA = re.compile(r"[ぁ-ゖ]")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
DIGIT = re.compile(r"[0-9０-９]")
MULTI_DIGIT = re.compile(r"[0-9０-９]{2,}")
URL = re.compile(r"https?:|www\.|\.(com|org|net|jp)\b", re.IGNORECASE)

# Characters a sentence may contain: kana, the CJK Unified block (no extensions, no
# compatibility ideographs), printable ASCII, full-width ASCII and common Japanese
# punctuation. Iteration marks other than 々 (ゝゞヽヾ) and rare symbols are excluded.
_ALLOWED_RANGES = (
    (0x3041, 0x3096),  # hiragana
    (0x30A1, 0x30FA),  # katakana (incl. ヴ ヵ ヶ)
    (0x30FB, 0x30FC),  # ・ ー
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0x0020, 0x007E),  # printable ASCII
    (0xFF01, 0xFF5E),  # full-width ASCII
)
_ALLOWED_EXTRA = set("々〆〇、。「」『』【】〈〉《》〔〕（）［］｛｝…‥―‐〜～“”‘’　")

# Bracket pairs checked for balance (stack based, so 「…）」 also counts as unbalanced).
BRACKETS = {
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "【": "】",
    "［": "］",
    "[": "]",
    "〈": "〉",
    "《": "》",
    "〔": "〕",
    "｛": "｝",
    "{": "}",
    "“": "”",
}
_CLOSERS = {v: k for k, v in BRACKETS.items()}
BRACKET_CHARS = set(BRACKETS) | set(_CLOSERS)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def is_allowed_char(ch: str) -> bool:
    if ch in _ALLOWED_EXTRA:
        return True
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ALLOWED_RANGES)


def brackets_balanced(text: str) -> bool:
    stack: list[str] = []
    for ch in text:
        if ch in BRACKETS:
            stack.append(ch)
        elif ch in _CLOSERS:
            if not stack or stack[-1] != _CLOSERS[ch]:
                return False
            stack.pop()
    return not stack and text.count('"') % 2 == 0


def strip_lone_quote(text: str) -> str | None:
    """``text`` without a single unmatched leading 「/『 or trailing 」/』, if that balances it.

    Only used with ``--repair-lone-quote`` (off by default): Aozora dialogue lines are
    often cut at 。 inside the quotation, leaving 「… without its 」.
    """
    for opener, closer in (("「", "」"), ("『", "』")):
        if text.startswith(opener) and brackets_balanced(text[1:]):
            return text[1:]
        if text.endswith(closer) and brackets_balanced(text[:-1]):
            return text[:-1]
    return None


def is_yomi_char(ch: str) -> bool:
    """Katakana, the long-vowel mark, or punctuation/space (no kanji, Latin or digits)."""
    if "ァ" <= ch <= "ヺ" or ch == "ー":
        return True
    return unicodedata.category(ch)[0] in ("P", "Z")


def yomi_ok(yomi: str) -> bool:
    return bool(yomi) and any("ァ" <= c <= "ヺ" for c in yomi) and all(map(is_yomi_char, yomi))


def leak_key(text: str) -> str:
    """Comparison form for leakage and dedup: NFKC, casefolded, no punctuation/space/symbols."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in ("P", "Z", "S"))


def quality_reason(text: str, *, min_len: int = 8, max_len: int = 80) -> str | None:
    """First failed quality rule for a raw sentence, or None when it passes."""
    if not min_len <= len(text) <= max_len:
        return "length"
    if not (KANJI.search(text) or KANA.search(text)):
        return "no_kana_kanji"
    if not all(map(is_allowed_char, text)):
        return "rare_chars"
    if any(ch in "<>＜＞" for ch in text):
        return "tag_chars"  # would clash with the <target> tag syntax
    if len(LATIN.findall(text)) > 0.15 * len(text):
        return "latin_heavy"
    if URL.search(text):
        return "url"
    if not brackets_balanced(text):
        return "unbalanced_brackets"
    if MULTI_DIGIT.search(text):
        # Sudachi reads multi-digit numbers digit by digit (100 -> イチレイレイ), so the
        # yomi of such a sentence is wrong.
        return "multi_digit_number"
    return None


def japanese_ratio(text: str) -> float:
    n = sum(1 for ch in text if KANJI.match(ch) or KANA.match(ch) or ch in "、。・「」")
    return n / max(len(text), 1)


def kanji_ratio(text: str) -> float:
    return len(KANJI.findall(text)) / max(len(text), 1)
