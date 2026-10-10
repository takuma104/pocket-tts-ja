"""C0: classify the target-reading errors of a JKYB-schema eval (jkyb_mini, jkyb, aozora).

Ported from the Irodori experiments (analyze_jkyb_errors.py) to this repo's SentencePiece
tokenizer. Every row with target_exact == false gets one error type:

  marginal            matches a marginal reading (correct under Relaxed Accuracy)
  generation_failure  the context around the target is broken, or the target span is empty
  voicing             differs only in voicing (missing/extra rendaku)
  sokuon_or_length    differs only in sokuon or vowel length
  other_joyo_reading  read with another Joyo reading of the same kanji (on/kun confusion)
  other               any other misreading

Rows whose Whisper (kanji-kana) transcript still contains the target word are flagged
text_asr_agrees: the audio may be right and only the kana ASR misheard it (or Whisper's LM
corrected it). The report also breaks accuracy down by how the tokenizer splits the target
(tokenizer_study.target_tokenization) and by how often the target word occurs in the training
transcripts (real speech and synthetic rows counted separately).

    uv run python scripts/analyze_jkyb_errors.py \
        runs/B1_4k/eval/step00300000/jkyb_mini_rotation_cfg2.0/results \
        --train data/mix/B1_train_v3.jsonl --out exp/c0/B1_300k_jkyb_mini.md
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_study import target_tokenization  # noqa: E402

TAG = re.compile(r"<([^>]+)>")
KANJI = re.compile(r"[㐀-鿿々]")
CATEGORIES = {"on_yomi": "音読み", "kun_yomi": "訓読み", "joyo_appendix_reading": "付表の語"}
ERROR_TYPES = {
    "marginal": "許容読みに一致（Relaxed では正解）",
    "generation_failure": "生成破綻（文脈ごと崩れている）",
    "voicing": "清濁の差のみ",
    "sokuon_or_length": "促音・長音の差のみ",
    "other_joyo_reading": "同じ漢字の別の常用読み",
    "other": "その他の読み誤り",
}
TOKEN_TYPES = {
    "multi": "語のトークン（対象＋隣接文字）",
    "split": "単漢字トークンで隣も漢字（熟語が分割）",
    "single": "単漢字トークンで隣は仮名・記号",
    "fragmented": "対象が複数トークン",
    "byte": "バイトフォールバック",
    "unknown": "不明",
}
SWAPS = {"on→kun": "音 → 訓", "on→on": "音 → 別の音", "kun→on": "訓 → 音", "kun→kun": "訓 → 別の訓", "other": "付表の語など"}
FREQ_BINS = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 10**9)]
GENERATION_FAILURE_CONTEXT_CER = 0.3
JKYB_FULL = next(
    (Path.home() / ".cache/huggingface/hub").glob(
        "datasets--Parakeet-Inc--joyo-kanji-yomi-benchmark-parakeet/snapshots/*/data/common_kanji_source.jsonl"
    ),
    None,
)
VOWEL_ROWS = {
    "ア": "アカサタナハマヤラワガザダバパャ",
    "イ": "イキシチニヒミリギジヂビピ",
    "ウ": "ウクスツヌフムユルグズヅブプュ",
    "エ": "エケセテネヘメレゲゼデベペ",
    "オ": "オコソトノホモヨロヲゴゾドボポョ",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def target_kanji(row: dict[str, Any]) -> str:
    m = TAG.search(row["tagged_text"])
    return m.group(1) if m else ""


def strip_voicing(kana: str) -> str:
    decomposed = unicodedata.normalize("NFD", kana)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if ch not in "゙゚"))


def vowel(kana: str) -> str:
    return next((v for v, row in VOWEL_ROWS.items() if kana in row), "")


def strip_sound_change(kana: str) -> str:
    """Drop voicing, sokuon, stem-final ク/キ/ツ/チ and vowel lengthening.

    Lengthening covers the canonical form of the eval details (オオ, エエ) and the plain kana
    of the dataset (オウ, エイ, ー), so readings from either side compare equal.
    """
    text = re.sub(r"[クキツチ]$", "", strip_voicing(kana).replace("ッ", ""))
    out: list[str] = []
    for ch in text:
        prev = vowel(out[-1]) if out else ""
        if prev and (ch == "ー" or ch == prev or (prev, ch) in (("オ", "ウ"), ("エ", "イ"))):
            continue
        out.append(ch)
    return "".join(out)


def target_word(row: dict[str, Any]) -> str:
    """Kanji run containing the target (plus one okurigana for kun readings)."""
    tagged = row["tagged_text"]
    start, end = tagged.index("<"), tagged.index(">") - 1
    text = tagged.replace("<", "").replace(">", "")
    while start > 0 and KANJI.match(text[start - 1]):
        start -= 1
    while end < len(text) and KANJI.match(text[end]):
        end += 1
    if row["reading_category"] == "kun_yomi" and end < len(text) and "ぁ" <= text[end] <= "ゟ":
        end += 1
    return text[start:end]


def classify(row: dict[str, Any], readings: dict[str, set[str]]) -> str:
    predicted = row["mapped_target"]
    expected = [r["normalized"] for r in row["accepted_readings"]]
    if row["relaxed_target_exact"]:
        return "marginal"
    if float(row["alignment_context_error_rate"]) >= GENERATION_FAILURE_CONTEXT_CER or predicted == "":
        return "generation_failure"
    if any(strip_voicing(predicted) == strip_voicing(e) for e in expected):
        return "voicing"
    if any(strip_sound_change(predicted) == strip_sound_change(e) for e in expected):
        return "sokuon_or_length"
    others = readings.get(target_kanji(row), set()) - {strip_sound_change(e) for e in expected}
    if strip_sound_change(predicted) in others:
        return "other_joyo_reading"
    return "other"


def swap_direction(row: dict[str, Any], by_category: dict[str, dict[str, set[str]]]) -> str:
    predicted = strip_sound_change(row["mapped_target"])
    expected = {"on_yomi": "on", "kun_yomi": "kun"}.get(row["reading_category"])
    if expected is None:
        return "other"
    readings = by_category[target_kanji(row)]
    if predicted in readings["on_yomi"]:
        return f"{expected}→on"
    if predicted in readings["kun_yomi"]:
        return f"{expected}→kun"
    return "other"


def reading_tables(reference: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """kanji -> readings, and kanji -> category -> readings, both without sound changes.

    Rows are eval details (accepted_readings) or the JKYB dataset itself (readings).
    """
    readings: dict[str, set[str]] = defaultdict(set)
    by_category: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in reference:
        kanji = TAG.search(row["tagged_text"]).group(1)  # type: ignore[union-attr]
        if "accepted_readings" in row:
            kana = [a["normalized"] for a in row["accepted_readings"]]
        else:
            kana = row["readings"]["natural"] + row["readings"]["marginal"]
        for k in kana:
            readings[kanji].add(strip_sound_change(k))
            by_category[kanji][row["reading_category"]].add(strip_sound_change(k))
    return readings, by_category


def train_counts(manifest: Path, words: set[str]) -> dict[str, dict[str, int]]:
    """Occurrences of each word in the training transcripts: real speech vs synthetic rows."""
    texts: dict[str, list[str]] = {"real": [], "synth": []}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source_type"] == "emilia_gain":  # a level-shifted copy of rows already counted
                continue
            texts["synth" if r["source_type"].startswith("irodori") else "real"].append(r["transcript"])
    joined = {k: "\n".join(v) for k, v in texts.items()}
    return {w: {k: s.count(w) for k, s in joined.items()} for w in words}


def table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in body])


def rate(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "–"


def freq_bin(n: int) -> str:
    lo, hi = next(b for b in FREQ_BINS if b[0] <= n <= b[1])
    return "0" if hi == 0 else f"{lo}+" if hi > 10**8 else f"{lo}–{hi}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="jkyb-eval output (details/all.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/ja_4k.model"))
    parser.add_argument("--train", type=Path, default=None, help="training manifest for the word-frequency table")
    parser.add_argument("--readings-from", type=Path, default=JKYB_FULL,
                        help="JKYB-schema rows listing the other readings of each kanji, for the on/kun swap "
                        "check (default: the full JKYB dataset from the HF cache, if present)")
    parser.add_argument("--examples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.results_dir / "details" / "all.jsonl")
    readings, by_category = reading_tables(rows + (read_jsonl(args.readings_from) if args.readings_from else []))
    if not args.readings_from:
        print("no --readings-from: other readings come from the evaluated rows only")
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    for r in rows:
        r["kanji"] = target_kanji(r)
        r["word"] = target_word(r)
        r["token_type"] = target_tokenization(sp, r)
        r["error_type"] = None if r["target_exact"] else classify(r, readings)
        r["swap"] = swap_direction(r, by_category) if r["error_type"] == "other_joyo_reading" else None
        r["text_asr_agrees"] = r["word"] in (r.get("prediction_text") or "")
    errors = [r for r in rows if r["error_type"]]

    out = [f"# JKYB 読み誤りの分類: {args.results_dir.parent}", ""]
    # from the rows themselves: summary.json may come from another run of the same eval
    out.append(table(["指標", "値"], [
        ["例文数", str(len(rows))],
        ["Accuracy", f"{sum(bool(r['target_exact']) for r in rows) / len(rows):.2%}"],
        ["Relaxed Accuracy", f"{sum(bool(r['relaxed_target_exact']) for r in rows) / len(rows):.2%}"],
    ]))

    out += ["", "## 読み分類別", ""]
    totals = Counter(r["reading_category"] for r in rows)
    wrong = Counter(r["reading_category"] for r in errors)
    out.append(table(["分類", "例文数", "誤り", "誤り率"],
                     [[label, str(totals[c]), str(wrong[c]), rate(wrong[c], totals[c])] for c, label in CATEGORIES.items()]))

    out += ["", "## 誤りタイプ別", ""]
    by_type = Counter(r["error_type"] for r in errors)
    cross = Counter((r["error_type"], r["reading_category"]) for r in errors)
    agrees = Counter(r["error_type"] for r in errors if r["text_asr_agrees"])
    out.append(table(
        ["誤りタイプ", "件数", "割合", *CATEGORIES.values(), "うち Whisper 書き起こしは原文どおり"],
        [[label, str(by_type[t]), rate(by_type[t], len(errors)), *[str(cross[(t, c)]) for c in CATEGORIES], str(agrees[t])]
         for t, label in ERROR_TYPES.items()]
        + [["合計", str(len(errors)), "100%", *[str(wrong[c]) for c in CATEGORIES], str(sum(agrees.values()))]],
    ))
    out += ["", "「Whisper 書き起こしは原文どおり」は、漢字かな交じりの書き起こしに対象語がそのまま含まれていた件数。"
            "カナ ASR の聞き誤りの可能性がある一方、Whisper の言語モデルが文脈から補っただけの場合も含む。"]

    out += ["", "## 対象漢字のトークン化別", ""]
    tok_total = Counter(r["token_type"] for r in rows)
    tok_err = Counter(r["token_type"] for r in errors)
    tok_swap = Counter(r["token_type"] for r in errors if r["error_type"] == "other_joyo_reading")
    tok_other = Counter(r["token_type"] for r in errors if r["error_type"] == "other")
    out.append(table(["トークン化", "例文数", "誤り", "誤り率", "うち別の常用読み", "うちその他"],
                     [[label, str(tok_total[t]), str(tok_err[t]), rate(tok_err[t], tok_total[t]), str(tok_swap[t]), str(tok_other[t])]
                      for t, label in TOKEN_TYPES.items() if tok_total[t]]))

    out += ["", "## 別の常用読みで読んだ誤りの向き", ""]
    swaps = Counter(r["swap"] for r in errors if r["swap"])
    out.append(table(["正解 → 生成の読み", "件数"], [[label, str(swaps[d])] for d, label in SWAPS.items()]))

    if args.train:
        counts = train_counts(args.train, {r["word"] for r in rows})
        for r in rows:
            r["train_real"], r["train_synth"] = counts[r["word"]]["real"], counts[r["word"]]["synth"]
        out += ["", f"## 対象語の学習テキスト中の出現数と正解率（{args.train.name}）", "",
                "対象語は、対象漢字を含む漢字列（訓読みは送り仮名1字を含む）。読みは区別していないので、"
                "多読字では別の読みでの出現も数えている。", ""]
        for key, label in (("train_real", "実音声（Emilia / YODAS）"), ("train_synth", "合成（Irodori）"),
                           ("train_all", "合計")):
            body = []
            for cat in (None, *CATEGORIES):
                pool = [r for r in rows if cat is None or r["reading_category"] == cat]
                n: Counter[str] = Counter()
                ok: Counter[str] = Counter()
                for r in pool:
                    b = freq_bin(r["train_real"] + r["train_synth"] if key == "train_all" else r[key])
                    n[b] += 1
                    ok[b] += bool(r["target_exact"])
                bins = [freq_bin(lo) for lo, _ in FREQ_BINS]
                body.append([CATEGORIES.get(cat, "全体") if cat else "全体",  # type: ignore[arg-type]
                             *[f"{rate(ok[b], n[b])}（{n[b]}）" for b in bins]])
            out += [f"**{label}**: 正解率（例文数）", "",
                    table(["分類", *[f"{freq_bin(lo)} 回" for lo, _ in FREQ_BINS]], body), ""]

    out += ["", "## よく読み誤る漢字（誤り数の上位）", ""]
    k_err = Counter(r["kanji"] for r in errors)
    k_total = Counter(r["kanji"] for r in rows)
    out.append(table(["漢字", "誤り / 例文数"], [[k, f"{n}/{k_total[k]}"] for k, n in k_err.most_common(30)]))

    rng = random.Random(args.seed)
    for t, label in ERROR_TYPES.items():
        pool = [r for r in errors if r["error_type"] == t]
        if not pool:
            continue
        sample = sorted(rng.sample(pool, min(args.examples, len(pool))), key=lambda r: r["key"])
        out += ["", f"## 例: {label}（{len(pool)} 件中 {len(sample)} 件）", ""]
        head = ["分類", "例文（<> が対象）", "正解", "生成の読み", "Whisper 書き起こし", "W一致"]
        if args.train:
            head += ["実", "合成"]
        body = []
        for r in sample:
            expected = " / ".join(a["reading"] + ("（許容）" if a["category"] == "marginal" else "") for a in r["accepted_readings"])
            cells = [CATEGORIES[r["reading_category"]], r["tagged_text"].replace("<", "&lt;"), expected,
                     r["mapped_target"] or "（なし）", r.get("prediction_text") or "", "✓" if r["text_asr_agrees"] else ""]
            if args.train:
                cells += [str(r["train_real"]), str(r["train_synth"])]
            body.append(cells)
        out.append(table(head, body))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    keep = ("key", "kanji", "word", "reading_category", "error_type", "swap", "token_type", "text_asr_agrees",
            "tagged_text", "mapped_target", "prediction_yomi", "prediction_text", "sentence_kana_cer",
            "train_real", "train_synth")
    with args.out.with_suffix(".errors.jsonl").open("w", encoding="utf-8") as f:
        for r in errors:
            f.write(json.dumps({k: r[k] for k in keep if k in r}, ensure_ascii=False) + "\n")
    print(f"{len(errors)} errors of {len(rows)} -> {args.out}")
    print("  " + ", ".join(f"{t} {by_type[t]}" for t in ERROR_TYPES))


if __name__ == "__main__":
    main()
