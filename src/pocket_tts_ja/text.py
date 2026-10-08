"""Japanese text normalization shared by data prep, tokenizer training and evaluation.

The rules are chosen so that the official pocket-tts inference path can reproduce them
with its per-character `replace_characters` table (see `INFERENCE_REPLACE_CHARACTERS`):
sentence punctuation is ASCII, so the stock sentence splitter (which looks for the
tokens of ".!...?" and ",;:") and terminal-punctuation completion work unchanged.
"""

import re
import unicodedata

# Applied after NFKC. Brackets and quotes carry no sound; dropping them keeps rarely seen
# characters out of the vocabulary (their embeddings would stay untrained).
_CHAR_MAP = {
    "。": ".",
    "．": ".",
    "、": ",",
    "，": ",",
    "〜": "ー",
    "～": "ー",
    "~": "ー",
    "―": "ー",
    "‐": "ー",
    "「": "",
    "」": "",
    "『": "",
    "』": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "〈": "",
    "〉": "",
    "《": "",
    "》": "",
    "“": "",
    "”": "",
    '"': "",
    "‘": "",
    "’": "",
    "'": "",
    "♪": "",
    "*": "",
    "#": "",
}
_TRANS = str.maketrans(_CHAR_MAP)
_SENTENCE_END = re.compile(r"([.!?…])(?=[^\s.!?…,])")
_REPEATED_PUNCT = re.compile(r"([.,!?])\1+")


def normalize(text: str) -> str:
    """Raw transcript -> model text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANS)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{3,}|…+", "…", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    # The inference chunker joins sentences with a space; do the same here.
    text = _SENTENCE_END.sub(r"\1 ", text)
    # A comma or space right before sentence-final punctuation is noise.
    text = re.sub(r"[ ,]+([.!?])", r"\1", text)
    return text


# Characters a user may type that `normalize` rewrites, for the inference config. NFKC itself
# cannot run inside pocket-tts, so the common full-width forms are listed explicitly.
def inference_replace_characters() -> dict[str, str]:
    table = dict(_CHAR_MAP)
    for code in range(0xFF01, 0xFF5F):  # full-width ASCII block
        ch = chr(code)
        table.setdefault(ch, normalize(ch) if normalize(ch) != "" else "")
    table["？"] = "?"
    table["！"] = "!"
    table["　"] = " "
    return {k: v for k, v in table.items() if k != v}
