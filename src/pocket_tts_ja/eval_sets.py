"""Evaluation sets shared by the teacher (Irodori) and the students, as on main's eval_tts.py.

Every item has a key, the text to speak, the voice prompt (a wav), the speaker / gender of that
voice, an optional verified reading (``yomi``, katakana or hiragana) for Kana-CER, and the
``reference`` wav that speaker similarity is measured against (the speaker's own recording of the
sentence for jvs_bal, the prompt otherwise).

    voices     the ten release voices (exp/release_cc/voices, Irodori voice design) x the first 20
               JSUT sentences of jsut_eval_1k (mimi_ft's "Space voices" check; female_02 / female_05
               are the high voices the Mimi path breaks)
    jsut       1,000 JSUT BASIC5000 sentences with verified kana, JKYB rotation voices
    short      172 texts of 2-8 tokens, rotation voices
    jvs_bal    5 male + 5 female JVS speakers (not the rotation ones) x 10 parallel100 sentences,
               prompted with the speaker's first nonpara30 clip
    jkyb_mini  1,500 JKYB rows (JKYB schema), rotation voices
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pocket_tts_ja.jvs import list_items, read_genders

REPO = Path(__file__).resolve().parents[2]
JVS = REPO / "data" / "jvs_ver1"
EVAL = REPO / "data" / "eval"
VOICES = REPO / "exp" / "release_cc" / "voices"
ROTATION = ("jvs006", "jvs041", "jvs007", "jvs066")  # M low/high F0, F low/high F0
SETS = ("voices", "jsut", "short", "jvs_bal", "jkyb_mini")


@dataclass(frozen=True)
class EvalItem:
    key: str
    text: str
    prompt: str
    speaker: str
    gender: str
    reference: str
    yomi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_nonpara(spk: str) -> Path:
    return sorted((JVS / spk / "nonpara30" / "wav24kHz16bit").glob("*.wav"))[0]


def _rotation(key: str) -> str:
    return ROTATION[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(ROTATION)]


def item_seed(key: str) -> int:
    """A fixed per-item seed (31 bits) from the key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def build(which: str) -> list[EvalItem]:
    genders = read_genders(JVS)
    if which == "voices":
        spec = json.loads((VOICES / "voices.json").read_text(encoding="utf-8"))
        sentences = [json.loads(line) for line in open(EVAL / "jsut_eval_1k.jsonl", encoding="utf-8")][:20]
        out = []
        for v in spec["voices"]:
            wav = str(VOICES / f"{v['id']}.wav")
            gender = "F" if v["id"].startswith("female") else "M"
            for s in sentences:
                out.append(EvalItem(f"{v['id']}_{s['key']}", s["text"], wav, v["id"], gender, wav, s["kana"]))
        return out
    if which in ("jsut", "short"):
        path = EVAL / ("jsut_eval_1k.jsonl" if which == "jsut" else "short_ja.jsonl")
        out = []
        for r in map(json.loads, open(path, encoding="utf-8")):
            spk = _rotation(r["key"])
            prompt = str(first_nonpara(spk))
            out.append(EvalItem(r["key"], r["text"], prompt, spk, genders[spk][0], prompt, r.get("kana")))
        return out
    if which == "jkyb_mini":
        out = []
        for r in map(json.loads, open(EVAL / "jkyb_mini.jsonl", encoding="utf-8")):
            spk = _rotation(r["key"])
            prompt = str(first_nonpara(spk))
            out.append(EvalItem(r["key"], r.get("tts_text") or r["text"], prompt, spk, genders[spk][0], prompt))
        return out
    if which == "jvs_bal":
        rows = {i.key: i for i in list_items(JVS, ("parallel100",))}
        spks: list[str] = []
        for g in ("M", "F"):
            pool = sorted(k for k, v in genders.items() if v[0] == g and k not in ROTATION)
            spks += [pool[i] for i in np.linspace(0, len(pool) - 1, 5).round().astype(int)]
        out = []
        for spk in spks:
            for n in range(5, 100, 10):
                row = rows.get(f"{spk}_parallel100_VOICEACTRESS100_{n:03d}")
                if row is not None:
                    out.append(EvalItem(row.key, row.text, str(first_nonpara(spk)), spk, row.gender, row.wav, row.yomi))
        return out
    raise ValueError(f"unknown set {which!r} (one of {', '.join(SETS)})")
