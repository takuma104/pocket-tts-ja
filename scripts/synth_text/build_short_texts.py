"""S4 text: short everyday sentences from Tatoeba, for tight short-utterance training clips.

Every Emilia clip is at least 3 s long, and its short transcripts often leave out speech the
clip contains (a lesson video saying the sentence twice). Irodori reads short text without
padding, so synthetic short sentences are the clean examples of "short text -> short audio,
then stop". Tatoeba sentences are CC BY 2.0 FR (attribution: Tatoeba, https://tatoeba.org).

Rows pass the S1 quality rules (with short lengths allowed), are deduplicated, checked for
leakage against JKYB / JVS / the Aozora and JSUT evaluation sets exactly as S1, get the
plain Sudachi yomi with the JKYB conventions (for the QC), and are sampled evenly over
lengths.

    uv run python scripts/synth_text/build_short_texts.py data/text/tatoeba/jpn_sentences.tsv \
        --out data/synth/short_texts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_texts import IRODORI, REPO, load_eval_sets  # noqa: E402
from synth_filters import japanese_ratio, leak_key, quality_reason, yomi_ok  # noqa: E402
from synth_leakage import LeakageIndex  # noqa: E402
from synth_wiki import plain_sudachi_yomi  # noqa: E402

BINS = ((2, 5), (6, 10), (11, 15), (16, 20))


def yomi_of(text: str) -> str:
    yomi, fixes = plain_sudachi_yomi(text)
    for a, b, r in sorted(fixes, reverse=True):
        yomi = yomi[:a] + r + yomi[b:]
    return yomi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tatoeba", type=Path, help="jpn_sentences.tsv (id, lang, text)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=12000)
    parser.add_argument("--irodori", type=Path, default=IRODORI)
    parser.add_argument("--jvs-dir", type=Path, default=REPO / "data/jvs_ver1")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    stats: dict[str, int] = {}

    def drop(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    seen: set[str] = set()
    cands: list[tuple[str, str]] = []
    for line in open(args.tatoeba, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        sid, _, text = parts
        text = " ".join(text.split())
        reason = quality_reason(text, min_len=2, max_len=30)
        if reason:
            drop(reason)
            continue
        key = leak_key(text)
        if not BINS[0][0] <= len(key) <= BINS[-1][1]:
            drop("length")
            continue
        if japanese_ratio(text) < 0.9:
            drop("non_japanese")
            continue
        if key in seen:
            drop("duplicate")
            continue
        seen.add(key)
        cands.append((sid, text))
    print(f"{len(cands)} candidates after quality/dedup", flush=True)

    leakage = LeakageIndex()
    leakage.build(load_eval_sets(args.irodori, args.jvs_dir))
    kept: list[tuple[str, str]] = []
    for sid, text in cands:
        leak = leakage.check(text)
        if leak is not None:
            drop(f"leak_{leak[0]}:{leak[1]}")
            continue
        kept.append((sid, text))

    with Pool(args.workers) as pool:
        yomis = pool.map(yomi_of, [t for _, t in kept], chunksize=256)
    rows = []
    for (sid, text), yomi in zip(kept, yomis, strict=True):
        if not yomi_ok(yomi):
            drop("bad_yomi")
            continue
        rows.append({"tatoeba_id": sid, "text": text, "yomi": yomi, "n": len(leak_key(text))})

    rng = random.Random(0)
    out_rows = []
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["n"] <= hi]
        rng.shuffle(sel)
        out_rows += sel[: args.per_bin]
        print(f"chars {lo}-{hi}: {len(sel)} available, {min(len(sel), args.per_bin)} kept")
    with open(args.out, "w", encoding="utf-8") as f:
        for k, r in enumerate(out_rows):
            f.write(json.dumps({"id": f"syn_short_{k}", "key": f"syn_short_{k}", "text": r["text"],
                                "yomi": r["yomi"], "source": "tatoeba", "tatoeba_id": r["tatoeba_id"]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(dict(sorted(stats.items(), key=lambda kv: -kv[1])), ensure_ascii=False, indent=1))
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
