"""Drop training clips whose audio says more than their transcript (from recheck_asr.py output).

The transcript's reading (pyopenjtalk) is aligned to the kana-whisper transcript of the clip
(both canonicalized as in JKYB). Kana the ASR heard that the transcript lacks are insertions;
fillers are removed from both sides first, so a dropped "えー" alone does not count. A clip is
dropped when the insertions reach max(--min-insert, --insert-frac x reference length), when
the ASR output is far longer than the reference (hallucinated loops), or when the CER exceeds
--max-cer. Long clips whose transcript stops at the ASR token limit are scored on the part of
the reference the transcript covers.

With --max-tail N a clip is also dropped when the ASR heard N or more kana after the last
transcript kana, fillers included: a short "うん" / "えー" / "はい" after the sentence (another
speaker's backchannel in YouTube talk) passes the insertion rule but teaches the model to add
it after the text (B-CC's trailing extra speech).

With --drop-fillers a clip is also dropped when the ASR heard a filler or backchannel the
transcript lacks anywhere in the clip ("えー", "うん", "はい", a hum: an insertion of up to
FILLER_INSERT_MAX kana that contains one of BACKCHANNELS). The insertion rule cannot see them
(fillers are stripped first); a model trained on them says "ウン" with no text for it.

    uv run python scripts/filter_asr_recheck.py data/emilia_ja/train.jsonl data/emilia_ja/asr_recheck.jsonl \
        --out data/emilia_ja/train_rechecked.jsonl
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocket_tts_ja.kana import FILLERS, canonical_kana, reading, strip_fillers  # noqa: E402

BINS = ((0, 15), (15, 30), (30, 60), (60, 10_000))


def ref_reading(text: str) -> str:
    return strip_fillers(reading(text))


def compare(ref: str, hyp: str, truncated: bool = False) -> dict[str, float]:
    """Inserted kana and CER of the ASR transcript against the reference reading.

    A truncated transcript (the ASR stopped at its token limit on a long clip) is scored on the
    part of the reference it reaches: the reference tail it never got to is not an error.
    """
    ops = list(Levenshtein.opcodes(ref, hyp))
    ins = sum(o.dest_end - o.dest_start for o in ops if o.tag == "insert")
    if truncated and ops and ops[-1].tag == "delete":
        errors = sum(max(o.src_end - o.src_start, o.dest_end - o.dest_start) for o in ops[:-1] if o.tag != "equal")
        return {"ins": ins, "cer": errors / max(ops[-1].src_start, 1)}
    return {"ins": ins, "cer": Levenshtein.distance(ref, hyp) / max(len(ref), 1)}


def full_reading(text: str) -> str:
    return canonical_kana(reading(text))


def tail_len(ref: str, hyp: str) -> int:
    """ASR kana after the last reference kana (trailing inserts of the alignment)."""
    n = 0
    for o in reversed(Levenshtein.opcodes(ref, hyp)):
        if o.tag != "insert" or o.src_start != len(ref):
            break
        n += o.dest_end - o.dest_start
    return n


BACKCHANNELS = (*FILLERS, "ハイ", "ンン", "ハア", "フウ")
FILLER_INSERT_MAX = 6


def inserted_filler(ref: str, hyp: str) -> bool:
    """Whether the ASR heard a filler / backchannel the reference lacks (a short insertion)."""
    for o in Levenshtein.opcodes(ref, hyp):
        if o.tag == "insert" and o.dest_end - o.dest_start <= FILLER_INSERT_MAX:
            ins = hyp[o.dest_start : o.dest_end]
            if any(f in ins for f in BACKCHANNELS):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("recheck", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-insert", type=int, default=3)
    parser.add_argument("--insert-frac", type=float, default=0.2)
    parser.add_argument("--max-cer", type=float, default=0.3)
    parser.add_argument("--max-hyp-ratio", type=float, default=2.5, help="ASR loop guard: len(hyp)/len(ref)")
    parser.add_argument("--max-tail", type=int, default=None,
                        help="drop clips with this many ASR kana (fillers included) after the transcript")
    parser.add_argument("--drop-fillers", action="store_true",
                        help="drop clips with a filler / backchannel the transcript lacks")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-unchecked", action="store_true", help="pass rows the recheck has not reached")
    parser.add_argument("--truncated-at", type=int, default=115,
                        help="ASR transcripts of this many kana or more ran into recheck_asr.py's token limit "
                        "(128 tokens: the output levels off at ~125 kana for clips over 15 s)")
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()

    hyps = {(r := json.loads(line))["id"]: r["kana"] for line in open(args.recheck)}
    all_rows = [json.loads(line) for line in open(args.manifest)]
    rows = [r for r in all_rows if r["id"] in hyps]
    unchecked = [r for r in all_rows if r["id"] not in hyps]
    with Pool(args.workers) as pool:
        refs = pool.map(ref_reading, [r["transcript"] for r in rows], chunksize=512)
        fulls = (pool.map(full_reading, [r["transcript"] for r in rows], chunksize=512)
                 if args.max_tail is not None or args.drop_fillers else [""] * len(rows))
    stats = []
    for r, ref, full in zip(rows, refs, fulls, strict=True):
        hyp = strip_fillers(canonical_kana(hyps[r["id"]]))
        truncated = len(hyps[r["id"]]) >= args.truncated_at
        s = compare(ref, hyp, truncated=truncated)
        loop = len(hyp) > args.max_hyp_ratio * max(len(ref), 4)
        extra = s["ins"] >= max(args.min_insert, args.insert_frac * len(ref))
        bad_cer = s["cer"] > args.max_cer
        tail = (args.max_tail is not None and not truncated
                and tail_len(full, canonical_kana(hyps[r["id"]])) >= args.max_tail)
        filler = args.drop_fillers and inserted_filler(full, canonical_kana(hyps[r["id"]]))
        reason = ("loop" if loop else "extra" if extra else "cer" if bad_cer else "tail" if tail
                  else "filler" if filler else "")
        stats.append((len(ref), reason, r, ref, hyp))

    tail_rule = f", ASR tail >= {args.max_tail} kana" if args.max_tail is not None else ""
    tail_rule += ", a filler the transcript lacks" if args.drop_fillers else ""
    print(f"{len(stats)} clips rechecked; drop when insertions >= max({args.min_insert}, "
          f"{args.insert_frac} x ref), CER > {args.max_cer}{tail_rule}, or ASR loop\n")
    print("| ref kana | clips | extra speech | CER | ASR loop | tail | filler | kept |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lo, hi in BINS:
        sel = [s for s in stats if lo <= s[0] < hi]
        if not sel:
            continue
        def share(reason: str) -> str:
            return f"{100 * np.mean([s[1] == reason for s in sel]):.1f}%"
        print(f"| {lo}-{hi if hi < 10_000 else ''} | {len(sel)} | {share('extra')} | {share('cer')} | "
              f"{share('loop')} | {share('tail')} | {share('filler')} | {share('')} |")
    kept = [s[2] for s in stats if not s[1]]
    hours = sum(r["lat_len"] for r in kept) * 0.08 / 3600
    print(f"\nkept {len(kept)} / {len(stats)} rechecked clips ({hours:.1f}h)")
    if args.keep_unchecked and unchecked:
        # recheck_asr.py goes shortest first, so a partial run covers the short clips
        kept += unchecked
        print(f"plus {len(unchecked)} clips not rechecked yet ({sum(r['lat_len'] for r in unchecked) * 0.08 / 3600:.1f}h), "
              f"longer than {min(r['lat_len'] for r in unchecked) * 0.08:.1f}s")
    for _, reason, r, ref, hyp in [s for s in stats if s[1] in ("extra", "tail", "filler")][: args.examples]:
        print(f"  [{reason}] {r['transcript']}\n      ref {ref}\n      asr {hyp}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
