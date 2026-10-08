"""Compare SentencePiece vocab sizes on Japanese transcripts.

For each vocab size, trains a unigram model on normalized transcripts (with every Joyo kanji
forced in as a piece) and reports, on held-out utterances and on JKYB-Parakeet:

- chars/token and tokens per second of speech. pocket-tts sizes its generation budget as
  tokens / 3 + 2 seconds (TTSModel._estimate_max_gen_len), so a tokenizer that is too coarse
  truncates long utterances in the stock inference code;
- byte-fallback rate (characters the vocab does not cover);
- how JKYB target kanji are tokenized. In the Irodori experiments, a target kanji split off
  as a single-character token next to other kanji (a compound broken apart) was the
  strongest predictor of a misreading (13.5% error vs 2.3% when the word was one token).

    uv run python scripts/tokenizer_study.py exp/emilia_meta.tsv --sizes 4000 8000 16000
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

import sentencepiece as spm
from huggingface_hub import hf_hub_download

from pocket_tts_ja.text import normalize

KANJI = re.compile(r"[一-鿿㐀-䶿々〆ヶ]")
KANA = "".join(chr(c) for c in range(0x3041, 0x3097)) + "".join(
    chr(c) for c in range(0x30A1, 0x30FB)
) + "ー"


def load_rows(path: Path, text_col: str, dur_col: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=[text_col, dur_col])
        return list(zip(table[text_col].to_pylist(), map(float, table[dur_col].to_pylist())))
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                rows.append((d[text_col], float(d[dur_col])))
        return rows
    with open(path, newline="") as f:
        for d in csv.DictReader(f, delimiter="\t"):
            rows.append((d[text_col], float(d[dur_col])))
    return rows


def is_heldout(text: str) -> bool:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 50 == 0


def jkyb_rows() -> list[dict[str, Any]]:
    path = hf_hub_download(
        "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet",
        "data/common_kanji_source.jsonl",
        repo_type="dataset",
    )
    return [json.loads(line) for line in open(path)]


def joyo_kanji(rows: list[dict[str, Any]]) -> str:
    chars = set()
    for r in rows:
        if r["reading_category"] != "joyo_appendix_reading":
            m = re.search(r"<(.+?)>", r["tagged_text"])
            if m and len(m.group(1)) == 1:
                chars.add(m.group(1))
    return "".join(sorted(chars))


_JKYB_PUNCT = str.maketrans({"。": ".", "、": ",", "？": "?", "！": "!"})  # 1:1, keeps offsets


def target_tokenization(sp: spm.SentencePieceProcessor, row: dict[str, Any]) -> str:
    """How the JKYB target span is tokenized.

    multi: a piece covers the target plus neighbouring characters (word-level piece);
    split: the target is exactly one single-kanji piece with a kanji neighbour (a compound
           broken apart, the risky case); single: one piece equal to the target otherwise;
    fragmented: the target spans several pieces; byte: byte fallback inside the target.
    """
    tagged = row["tagged_text"].translate(_JKYB_PUNCT)
    s = tagged.index("<")
    e = tagged.index(">") - 1
    text = tagged.replace("<", "").replace(">", "")
    # JKYB sentences contain no spaces, so "\u2581" only marks the dummy prefix and the
    # pieces (bytes regrouped into characters) concatenate back to `text`.
    spans: list[tuple[int, int, str]] = []
    pos, pending = 0, b""
    for piece in sp.encode(text, out_type=str):
        if piece.startswith("<0x") and piece.endswith(">"):
            pending += bytes([int(piece[3:-1], 16)])
            try:
                n = len(pending.decode())
            except UnicodeDecodeError:
                continue
            spans.append((pos, pos + n, "<byte>"))
            pos, pending = pos + n, b""
            continue
        surface = piece.replace("\u2581", "")
        if surface:
            spans.append((pos, pos + len(surface), piece))
            pos += len(surface)
    assert pos == len(text), (text, sp.encode(text, out_type=str))
    covering = [(b, en, p) for b, en, p in spans if b < e and en > s]
    if not covering:
        return "unknown"
    if any(p == "<byte>" for _, _, p in covering):
        return "byte"
    if len(covering) > 1:
        return "fragmented"
    b, en, _ = covering[0]
    if b < s or en > e:
        return "multi"
    left = text[s - 1] if s > 0 else " "
    right = text[e] if e < len(text) else " "
    if e - s == 1 and (KANJI.match(left) or KANJI.match(right)):
        return "split"
    return "single"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--dur-col", default="duration")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4000, 8000, 16000])
    parser.add_argument("--max-train", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, default=Path("exp/tokenizer_study"))
    args = parser.parse_args()

    rows = [(normalize(t), d) for t, d in load_rows(args.table, args.text_col, args.dur_col)]
    rows = [(t, d) for t, d in rows if t]
    train = [t for t, _ in rows if not is_heldout(t)][: args.max_train]
    held = [(t, d) for t, d in rows if is_heldout(t)]
    jk = jkyb_rows()
    required = joyo_kanji(jk) + KANA
    print(f"{len(train)} train / {len(held)} held-out utterances, {len(required)} required chars")

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for t in train:
            tmp.write(t + "\n")
        corpus = tmp.name

    results = []
    for size in args.sizes:
        prefix = str(args.out / f"ja_unigram_{size}")
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=prefix,
            vocab_size=size,
            character_coverage=0.9995,
            model_type="unigram",
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            max_sentencepiece_length=6,
            allow_whitespace_only_pieces=True,
            split_digits=True,
            byte_fallback=True,
            pad_id=3,
            required_chars=required,
            input_sentence_size=10_000_000,
            shuffle_input_sentence=True,
            num_threads=16,
            minloglevel=2,
        )
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        n_tok = [len(sp.encode(t)) for t, _ in held]
        n_chr = [len(t) for t, _ in held]
        tps = [n / d for n, (_, d) in zip(n_tok, held) if d > 0]
        # stock inference budget: tokens / 3 + 2 s; flag utterances longer than that
        truncated = sum(n / 3.0 + 2.0 < d for n, (_, d) in zip(n_tok, held)) / len(held)
        byte_pieces = sum(
            sum(p.startswith("<0x") for p in sp.encode(t, out_type=str)) for t, _ in held
        ) / max(1, sum(n_tok))
        jk_texts = [normalize(r["text"]) for r in jk]
        jk_bytes = sum(
            sum(p.startswith("<0x") for p in sp.encode(t, out_type=str)) for t in jk_texts
        )
        kinds: dict[str, int] = {}
        for r in jk:
            k = target_tokenization(sp, r)
            kinds[k] = kinds.get(k, 0) + 1
        res = {
            "vocab": size,
            "chars_per_token": sum(n_chr) / sum(n_tok),
            "tokens_per_sec_p10_p50_p90": [
                statistics.quantiles(tps, n=10)[0],
                statistics.median(tps),
                statistics.quantiles(tps, n=10)[-1],
            ],
            "stock_budget_truncated_frac": truncated,
            "byte_piece_frac_heldout": byte_pieces,
            "jkyb_byte_pieces_total": jk_bytes,
            "jkyb_mean_tokens": sum(len(sp.encode(t)) for t in jk_texts) / len(jk_texts),
            "jkyb_target_tokenization": {k: v / len(jk) for k, v in sorted(kinds.items())},
        }
        results.append(res)
        print(json.dumps(res, ensure_ascii=False), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
