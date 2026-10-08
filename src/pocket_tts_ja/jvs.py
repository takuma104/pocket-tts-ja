"""JVS corpus helpers: item listing and reference readings from the phoneme labels."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUBSETS = ("parallel100", "nonpara30", "whisper10", "falset10")

_VOWEL_INDEX = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}
# OpenJTalk consonant -> katakana for the vowels a i u e o.
_CV = {
    "": ["ア", "イ", "ウ", "エ", "オ"],
    "k": ["カ", "キ", "ク", "ケ", "コ"],
    "g": ["ガ", "ギ", "グ", "ゲ", "ゴ"],
    "s": ["サ", "スィ", "ス", "セ", "ソ"],
    "sh": ["シャ", "シ", "シュ", "シェ", "ショ"],
    "z": ["ザ", "ズィ", "ズ", "ゼ", "ゾ"],
    "j": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
    "t": ["タ", "ティ", "トゥ", "テ", "ト"],
    "ch": ["チャ", "チ", "チュ", "チェ", "チョ"],
    "ts": ["ツァ", "ツィ", "ツ", "ツェ", "ツォ"],
    "d": ["ダ", "ディ", "ドゥ", "デ", "ド"],
    "n": ["ナ", "ニ", "ヌ", "ネ", "ノ"],
    "h": ["ハ", "ヒ", "フ", "ヘ", "ホ"],
    "f": ["ファ", "フィ", "フ", "フェ", "フォ"],
    "b": ["バ", "ビ", "ブ", "ベ", "ボ"],
    "p": ["パ", "ピ", "プ", "ペ", "ポ"],
    "m": ["マ", "ミ", "ム", "メ", "モ"],
    "y": ["ヤ", "イ", "ユ", "イェ", "ヨ"],
    "r": ["ラ", "リ", "ル", "レ", "ロ"],
    "w": ["ワ", "ウィ", "ウ", "ウェ", "ウォ"],
    "v": ["ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"],
    "ky": ["キャ", "キ", "キュ", "キェ", "キョ"],
    "gy": ["ギャ", "ギ", "ギュ", "ギェ", "ギョ"],
    "ny": ["ニャ", "ニ", "ニュ", "ニェ", "ニョ"],
    "hy": ["ヒャ", "ヒ", "ヒュ", "ヒェ", "ヒョ"],
    "by": ["ビャ", "ビ", "ビュ", "ビェ", "ビョ"],
    "py": ["ピャ", "ピ", "ピュ", "ピェ", "ピョ"],
    "my": ["ミャ", "ミ", "ミュ", "ミェ", "ミョ"],
    "ry": ["リャ", "リ", "リュ", "リェ", "リョ"],
    "dy": ["デャ", "ディ", "デュ", "デェ", "デョ"],
    "ty": ["テャ", "ティ", "テュ", "テェ", "テョ"],
}


def phonemes_to_kana(phonemes: list[str]) -> str:
    """OpenJTalk phonemes (as in JVS lab/mon) -> katakana reading.

    Devoiced vowels are upper case in the labels; pauses become "、" (dropped by
    the JKYB canonicalization anyway).
    """
    out: list[str] = []
    onset = ""
    for raw in phonemes:
        if raw in ("sil", "pau"):
            if raw == "pau":
                out.append("、")
            onset = ""
            continue
        if raw == "N":
            out.append("ン")
            continue
        if raw == "cl":
            out.append("ッ")
            continue
        p = raw.lower() if raw in ("A", "I", "U", "E", "O") else raw
        if p in _VOWEL_INDEX:
            out.append(_CV[onset][_VOWEL_INDEX[p]])
            onset = ""
            continue
        if p not in _CV:
            raise ValueError(f"unknown phoneme {raw!r} in {phonemes}")
        onset = p
    return "".join(out)


def read_mono_lab(path: Path) -> list[str]:
    return [line.split()[2] for line in path.read_text().splitlines() if line.strip()]


@dataclass
class JvsItem:
    key: str  # unique stem: jvs001_parallel100_VOICEACTRESS100_001
    speaker: str
    gender: str  # "M" | "F"
    subset: str
    utt_id: str
    wav: str
    text: str
    yomi: str | None  # reading from the phoneme labels, when the subset has labels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_genders(root: Path) -> dict[str, tuple[str, int, int]]:
    """speaker -> (gender, min f0, max f0) from gender_f0range.txt."""
    out = {}
    for line in (root / "gender_f0range.txt").read_text().splitlines()[1:]:
        spk, gender, lo, hi = line.split()
        out[spk] = (gender, int(lo), int(hi))
    return out


def list_items(root: Path, subsets: tuple[str, ...] = SUBSETS) -> list[JvsItem]:
    genders = read_genders(root)
    items: list[JvsItem] = []
    # Reading labels of the shared sentences, for subsets that ship none (whisper10/falset10
    # reuse VOICEACTRESS100 sentences).
    shared_yomi: dict[str, str] = {}
    for spk in sorted(genders):
        lab_dir = root / spk / "parallel100" / "lab" / "mon"
        for lab in sorted(lab_dir.glob("*.lab")):
            shared_yomi.setdefault(lab.stem, phonemes_to_kana(read_mono_lab(lab)))
    for spk in sorted(genders):
        for subset in subsets:
            sub_dir = root / spk / subset
            transcripts = sub_dir / "transcripts_utf8.txt"
            if not transcripts.exists():
                continue
            for line in transcripts.read_text().splitlines():
                if ":" not in line:
                    continue
                utt_id, text = line.split(":", 1)
                wav = sub_dir / "wav24kHz16bit" / f"{utt_id}.wav"
                if not wav.exists():
                    continue  # a few transcripts have no recording
                lab = sub_dir / "lab" / "mon" / f"{utt_id}.lab"
                if lab.exists():
                    yomi: str | None = phonemes_to_kana(read_mono_lab(lab))
                else:
                    yomi = shared_yomi.get(utt_id)
                items.append(
                    JvsItem(
                        key=f"{spk}_{subset}_{utt_id}",
                        speaker=spk,
                        gender=genders[spk][0],
                        subset=subset,
                        utt_id=utt_id,
                        wav=str(wav),
                        text=text.strip(),
                        yomi=yomi,
                    )
                )
    return items
