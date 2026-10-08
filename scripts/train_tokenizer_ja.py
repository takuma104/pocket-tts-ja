"""A1: train the Japanese SentencePiece tokenizer used by the model.

Same trainer spec as upstream training/scripts/train_tokenizer.py (unigram, identity
normalization, no whitespace squashing, pieces <= 6 chars, split digits, byte fallback,
pad id 3) — text is normalized beforehand by pocket_tts_ja.text.normalize, which the
inference config reproduces with `replace_characters`. Differences:

  - required_chars: every Joyo kanji, all kana, printable ASCII, plus any character seen at
    least --min-char-count times in the training text. Characters in the required set never
    fall back to bytes (byte-fallback kanji were the hardest to learn in the Irodori study).
  - --character-coverage (0.9999) for the rest; train only on text that has audio
    (transcripts + the synthetic corpus), so web text does not inflate the alphabet.
  - inputs: the training manifests' `transcript` fields and optional extra text files
    (one normalized or raw sentence per line, or jsonl with a "text" field, e.g. the
    synthetic-data corpus), so rare kanji of the synthetic sentences get their own pieces.

    uv run python scripts/train_tokenizer_ja.py --vocab-size 8000 --out data/tokenizer/ja_8k \
        data/emilia_ja/train.jsonl data/synth/texts.jsonl
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from pocket_tts_ja.text import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_study import KANA, jkyb_rows, joyo_kanji  # noqa: E402

ASCII = "".join(chr(c) for c in range(0x21, 0x7F))


def iter_texts(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    d = json.loads(line)
                    text = d.get("transcript") or normalize(d.get("text", ""))
                else:
                    text = normalize(line)
                if text:
                    out.append(text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--out", type=str, required=True, help="writes <out>.model / <out>.vocab")
    parser.add_argument("--min-char-count", type=int, default=100)
    parser.add_argument("--character-coverage", type=float, default=0.9999)
    args = parser.parse_args()

    texts = iter_texts(args.inputs)
    counts = Counter(ch for t in texts for ch in t)
    required = set(joyo_kanji(jkyb_rows())) | set(KANA) | set(ASCII)
    required |= {ch for ch, n in counts.items() if n >= args.min_char_count and not ch.isspace()}
    required_str = "".join(sorted(required))
    print(f"{len(texts)} sentences, {len(counts)} distinct chars, {len(required)} required")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for t in texts:
            tmp.write(t + "\n")
        corpus = tmp.name
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=args.out,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type="unigram",
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentencepiece_length=6,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        byte_fallback=True,
        pad_id=3,
        required_chars=required_str,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        num_threads=16,
        minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=args.out + ".model")
    n_tok = sum(len(sp.encode(t)) for t in texts[:200_000])
    n_chr = sum(len(t) for t in texts[:200_000])
    byte_chars = sorted(
        {ch for ch in counts if any(p.startswith("<0x") for p in sp.encode(ch, out_type=str))}
    )
    report = {
        "vocab": sp.get_piece_size(),
        "required_chars": len(required),
        "chars_per_token": round(n_chr / n_tok, 3),
        "byte_fallback_chars_in_training_text": len(byte_chars),
        "byte_fallback_char_occurrences": sum(counts[c] for c in byte_chars),
        "multi_char_pieces": sum(
            len(sp.id_to_piece(i).replace("▁", "")) > 1 for i in range(sp.get_piece_size())
        ),
    }
    Path(args.out + ".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
