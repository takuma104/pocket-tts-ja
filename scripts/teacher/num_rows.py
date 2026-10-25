"""num: teacher rows for sentences with multi-digit numbers, and a held-out set to measure them.

None of the 352k training sentences of the distillation (main's synthetic CC BY pool, normalised
as main's training text) has a number of two or more digits, so every student reads "50" digit by
digit ("ゴゼロ"; teacher and release "ゴジュー"). Emilia-YODAS JA (CC BY) has them: the rechecked
training transcripts (data/emilia_ja/train_rechecked_yodas.jsonl) with a number of 2+ digits, less
those with a decimal point or a thousands comma (main's normaliser turned "52.6" into "52. 6"). The
old_kana that qc.py checks the teacher against is the reading of the transcript (pyopenjtalk, which
reads the numbers: "ジュウニジ", "センキュウヒャクハチジュウネン"); the kana-whisper reading of the
recording (asr_recheck_full.jsonl, kept as rec_kana) carries the speaker's fillers and the ASR's
slips (4 in 10 within a Kana-CER of 0.08 of the transcript's reading).

    uv run python scripts/teacher/num_rows.py --per-text 2
    -> data/irodori_distill/teacher/rows_num.jsonl: each training sentence read by --per-text of
       the teacher's reference voices (refs_cc / refs_yodas, P2's held-out voices left out), the
       teacher input with Japanese punctuation, the student's text as the transcript
    -> exp/irodori_distill/eval/num_eval/items.jsonl: --eval-texts other sentences, each read by
       one of the ten release voices (no yomi: kana_cer.py takes the reading of the text), and
       teacher_rows.jsonl for the teacher's own clips of them (irodori_synth.py)
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from pocket_tts_ja.kana import reading

TEACHER = Path("data/irodori_distill/teacher")
EMILIA = Path("data/emilia_ja")
VOICES = Path("exp/irodori_distill/eval/voices/items.jsonl")
MULTI_DIGIT = re.compile(r"[0-9]{2,}")
BROKEN_NUMBER = re.compile(r"[0-9][.,] ?[0-9]")  # a decimal point or thousands comma, split by the normaliser
TO_TEACHER = str.maketrans({",": "、", ".": "。", "?": "？", "!": "！"})


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_text(transcript: str) -> str:
    """The student's text (ASCII sentence punctuation, a space after a sentence) -> the teacher's input."""
    return re.sub(r"\s+", "", transcript.translate(TO_TEACHER))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="num")
    parser.add_argument("--per-text", type=int, default=2, help="teacher voices per training sentence")
    parser.add_argument("--eval-texts", type=int, default=200)
    parser.add_argument("--min-sec", type=float, default=1.5)
    parser.add_argument("--max-sec", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kana = {r["id"]: r["kana"] for r in read(EMILIA / "asr_recheck_full.jsonl")}
    texts: dict[str, dict[str, Any]] = {}
    for r in read(EMILIA / "train_rechecked_yodas.jsonl"):
        t = r["transcript"]
        if (MULTI_DIGIT.search(t) and not BROKEN_NUMBER.search(t) and r["id"] in kana
                and args.min_sec <= r["duration"] <= args.max_sec):
            texts.setdefault(t, r)  # one recording per sentence
    held_voices = {r["ref_speaker"] for r in read(Path("data/irodori_distill/mix/P2_s3_valid.jsonl"))}
    pool: dict[str, dict[str, str]] = {}
    for s in ("s3", "s4", "s9"):
        for r in read(TEACHER / f"rows_{s}.jsonl"):
            if r["ref_speaker"] not in held_voices:
                pool.setdefault(r["ref_speaker"], {k: r[k] for k in ("ref_wav", "ref_speaker", "ref_id")})
    voices = sorted(pool)

    rng = random.Random(args.seed)
    order = sorted(texts)
    rng.shuffle(order)
    held, train = order[: args.eval_texts], order[args.eval_texts :]
    path = TEACHER / f"rows_{args.set}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        i = 0
        for t in train:
            r = texts[t]
            for v in rng.sample(voices, args.per_text):
                row = {"id": f"syn_{args.set}_{i}", "text": teacher_text(t), **pool[v], "seed": rng.randrange(2**31),
                       "transcript": t, "old_kana": reading(t), "rec_kana": kana[r["id"]], "old_duration": r["duration"],
                       "text_from": r["id"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                i += 1
    hours = sum(texts[t]["duration"] for t in train) * args.per_text / 3600
    print(f"{len(texts)} sentences with a 2+ digit number; {len(train)} for training -> {i} rows "
          f"({hours:.1f} h at the recordings' durations) -> {path}; {len(voices)} voices")

    release = sorted({(v["speaker"], v["gender"], v["prompt"]) for v in read(VOICES)})
    out = Path("exp/irodori_distill/eval") / f"{args.set}_eval"
    out.mkdir(parents=True, exist_ok=True)
    eval_rng = random.Random(args.seed + 1)
    with open(out / "items.jsonl", "w", encoding="utf-8") as f, open(out / "teacher_rows.jsonl", "w", encoding="utf-8") as g:
        for j, t in enumerate(held):
            speaker, gender, prompt = release[j % len(release)]
            r = texts[t]
            item = {"key": f"{speaker}_{r['id']}", "text": teacher_text(t), "speaker": speaker, "gender": gender,
                    "rec_kana": kana[r["id"]], "prompt": prompt, "text_from": r["id"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            row = {"id": item["key"], "text": item["text"], "ref_wav": prompt, "seed": eval_rng.randrange(2**31)}
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(held)} held-out sentences -> {out / 'items.jsonl'}")


if __name__ == "__main__":
    main()
