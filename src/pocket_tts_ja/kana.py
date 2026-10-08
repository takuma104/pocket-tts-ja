"""Reading-level comparison of two transcripts (caption-agreement filter).

`canonical_kana` mirrors jkyb_eval.normalization.canonicalize_yomi (JKYB-Parakeet, MIT):
NFKC, drop punctuation/symbols/spaces, hiragana -> katakana, ヂヅヲヰヱ -> ジズオイエ,
prolonged sound mark -> preceding vowel, オ-row + ウ -> オ, エ-row + イ -> エ. Readings come
from pyopenjtalk, whose errors hit both sides alike, so agreement stays meaningful.
"""

import unicodedata

import pyopenjtalk
from rapidfuzz.distance import Levenshtein

_SUBST = {"ヂ": "ジ", "ヅ": "ズ", "ヲ": "オ", "ヰ": "イ", "ヱ": "エ"}
_VOWEL: dict[str, str] = {}
for _chars, _v in (
    ("アカサタナハマヤラワガザダバパァャヮ", "ア"),
    ("イキシチニヒミリギジヂビピィ", "イ"),
    ("ウクスツヌフムユルグズヅブプゥュヴ", "ウ"),
    ("エケセテネヘメレゲゼデベペェ", "エ"),
    ("オコソトノホモヨロヲゴゾドボポォョ", "オ"),
):
    for _c in _chars:
        _VOWEL[_c] = _v


def _sep(ch: str) -> bool:
    return unicodedata.category(ch)[0] in {"P", "S", "Z"} or ch.isspace()


def canonical_text(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not _sep(c))


def canonical_kana(kana: str) -> str:
    out: list[str] = []
    prev = "ア"
    for raw in unicodedata.normalize("NFKC", kana):
        if _sep(raw):
            continue
        ch = chr(ord(raw) + 0x60) if "ぁ" <= raw <= "ゖ" else raw
        ch = _SUBST.get(ch, ch)
        if ch == "ー":
            ch = _VOWEL.get(prev, prev)
        elif ch == "イ" and _VOWEL.get(prev) == "エ":
            ch = "エ"
        elif ch == "ウ" and _VOWEL.get(prev) == "オ":
            ch = "オ"
        out.append(ch)
        prev = ch
    return "".join(out)


def reading(text: str) -> str:
    return canonical_kana(pyopenjtalk.g2p(text, kana=True)) if text.strip() else ""


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(1.0, Levenshtein.distance(ref, hyp) / len(ref))
