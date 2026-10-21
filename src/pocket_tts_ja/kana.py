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


# Fillers that one ASR keeps and the other drops; removed from both readings before comparing.
FILLERS = ("エエト", "エット", "アノオ", "アノ", "ソノオ", "マア", "ウウン", "ウン", "エエ", "アア", "オオ")


def strip_fillers(kana: str) -> str:
    for f in FILLERS:
        kana = kana.replace(f, "")
    return kana


def supported_cer(label: str, other: str) -> float:
    """Edit distance of `label` against the best-matching substring of `other`, over len(label).

    Semi-global alignment: `other` (the YouTube caption) may carry extra words at either end
    (words of the neighbouring utterance attached across a segment boundary) without cost;
    everything in `label` (the Whisper transcript used for training) must be matched.
    """
    import numpy as np

    n, m = len(label), len(other)
    if n == 0:
        return 0.0
    if m == 0:
        return 1.0
    o = np.frombuffer(other.encode("utf-32-le"), dtype=np.uint32)
    prev = np.zeros(m + 1, dtype=np.int32)  # free start anywhere in `other`
    for i, ch in enumerate(label, start=1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        sub = prev[:-1] + (o != ord(ch))
        dele = prev[1:] + 1
        best = np.minimum(sub, dele)
        # insertions (skip a char of `other` inside the match) need a running minimum
        for j in range(1, m + 1):
            v = best[j - 1]
            ins = cur[j - 1] + 1
            cur[j] = v if v < ins else ins
        prev = cur
    return min(1.0, float(prev.min()) / n)  # free end anywhere in `other`
