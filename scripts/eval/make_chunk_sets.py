"""Evaluation sets in the text forms the stock wheel's chunker makes (plan 10/23, P3e).

The wheel (pocket_tts.models.text_chunking.split_into_best_sentences, 50 tokens) packs
consecutive sentences into one chunk joined by a space ("A. B.") and splits a longer sentence at
its commas, joining the pieces with ", ". The teacher's data are single sentences without either,
so these sets measure a student on what the wheel will feed it, through the GPU path
(student_synth.py) as well as through the wheel:

  <out>/pair_probe/         two consecutive items of --pairs-from as one item, where the wheel
                            keeps them in one chunk; the first one's prompt, both verified readings
  <out>/chunk_probe/chunk_sp/    each chunk of the multi-chunk items of --chunks-from, as the
                                 wheel makes it (the reference is the reading of the chunk)
  <out>/chunk_probe/chunk_nosp/  the same chunks with the space after each comma removed

    uv run python scripts/eval/make_chunk_sets.py exp/irodori_distill/package/p3d_20k/japanese_dv.yaml \
        exp/irodori_distill/eval --pairs-from exp/irodori_distill/eval/jsut \
        --chunks-from exp/irodori_distill/eval/jvs_bal exp/irodori_distill/eval/jsut
"""

import argparse
import json
from pathlib import Path

import yaml

from pocket_tts.models.text_chunking import split_into_best_sentences
from pocket_tts.modules.text_conditioner import SentencePieceTokenizer

MAX_TOKENS = 50  # generate_audio's default
KEEP = ("speaker", "gender", "prompt")


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "items.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} items")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="a package's model config (its tokenizer.model beside it)")
    parser.add_argument("out", type=Path, help="the evaluation root")
    parser.add_argument("--pairs-from", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=150)
    parser.add_argument("--chunks-from", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tok = SentencePieceTokenizer(cfg["flow_lm"]["lookup_table"]["n_bins"], str(args.config.parent / "tokenizer.model"))
    # the stock text handling, with the spaces kept (the package's own deletion would hide them)
    chars = {k: v for k, v in cfg["replace_characters"].items() if k not in (" ", "　")}
    kw = dict(remove_semicolons=cfg["remove_semicolons"], append_terminal_punctuation=cfg["append_terminal_punctuation"],
              capitalize_first_letter=cfg["capitalize_first_letter"], replace_characters=chars)

    def chunks(text: str) -> list[str]:
        return split_into_best_sentences(tok, text, MAX_TOKENS, False, **kw)

    items = [json.loads(line) for line in open(args.pairs_from / "items.jsonl", encoding="utf-8")]
    pairs = []
    for a, b in zip(items[0::2], items[1::2], strict=False):
        packed = chunks(a["text"] + b["text"])
        if len(packed) == 1:
            pairs.append({"key": f"{a['key']}+{b['key'].split('_')[-1]}", "text": a["text"] + b["text"],
                          "yomi": a["yomi"] + "、" + b["yomi"], **{k: a[k] for k in KEEP if k in a}, "chunk": packed[0]})
        if len(pairs) >= args.max_pairs:
            break
    write(args.out / "pair_probe", pairs)

    sp, nosp = [], []
    for d in args.chunks_from:
        for it in map(json.loads, open(d / "items.jsonl", encoding="utf-8")):
            pieces = chunks(it["text"])
            if len(pieces) < 2:
                continue
            for i, c in enumerate(pieces):
                base = {"key": f"{it['key']}_c{i}", **{k: it[k] for k in KEEP if k in it}}
                sp.append({**base, "text": c})
                nosp.append({**base, "text": c.replace(", ", ",")})
    write(args.out / "chunk_probe" / "chunk_sp", sp)
    write(args.out / "chunk_probe" / "chunk_nosp", nosp)


if __name__ == "__main__":
    main()
