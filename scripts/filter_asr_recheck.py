"""Drop training clips whose audio says more than their transcript (from recheck_asr.py output).

The transcript's reading (pyopenjtalk) is aligned to the kana-whisper transcript of the clip
(both canonicalized as in JKYB). Kana the ASR heard that the transcript lacks are insertions;
fillers are removed from both sides first, so a dropped "えー" alone does not count. A clip is
dropped when the insertions reach max(--min-insert, --insert-frac x reference length), when
the ASR ran into its token limit (hallucinated loops), or when the CER exceeds --max-cer.

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
from pocket_tts_ja.kana import canonical_kana, reading, strip_fillers  # noqa: E402

BINS = ((0, 15), (15, 30), (30, 60), (60, 10_000))


def ref_reading(text: str) -> str:
    return strip_fillers(reading(text))


def compare(ref: str, hyp: str) -> dict[str, float]:
    ops = Levenshtein.opcodes(ref, hyp)
    ins = sum(o.dest_end - o.dest_start for o in ops if o.tag == "insert")
    return {"ins": ins, "cer": Levenshtein.distance(ref, hyp) / max(len(ref), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("recheck", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-insert", type=int, default=3)
    parser.add_argument("--insert-frac", type=float, default=0.2)
    parser.add_argument("--max-cer", type=float, default=0.3)
    parser.add_argument("--max-hyp-ratio", type=float, default=2.5, help="ASR loop guard: len(hyp)/len(ref)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()

    hyps = {(r := json.loads(line))["id"]: r["kana"] for line in open(args.recheck)}
    rows = [r for r in map(json.loads, open(args.manifest)) if r["id"] in hyps]
    with Pool(args.workers) as pool:
        refs = pool.map(ref_reading, [r["transcript"] for r in rows], chunksize=512)
    stats = []
    for r, ref in zip(rows, refs, strict=True):
        hyp = strip_fillers(canonical_kana(hyps[r["id"]]))
        s = compare(ref, hyp)
        loop = len(hyp) > args.max_hyp_ratio * max(len(ref), 4)
        extra = s["ins"] >= max(args.min_insert, args.insert_frac * len(ref))
        bad_cer = s["cer"] > args.max_cer
        reason = "loop" if loop else "extra" if extra else "cer" if bad_cer else ""
        stats.append((len(ref), reason, r, ref, hyp))

    print(f"{len(stats)} clips rechecked; drop when insertions >= max({args.min_insert}, "
          f"{args.insert_frac} x ref), CER > {args.max_cer}, or ASR loop\n")
    print("| ref kana | clips | extra speech | CER | ASR loop | kept |")
    print("|---|---:|---:|---:|---:|---:|")
    for lo, hi in BINS:
        sel = [s for s in stats if lo <= s[0] < hi]
        if not sel:
            continue
        def share(reason: str) -> str:
            return f"{100 * np.mean([s[1] == reason for s in sel]):.1f}%"
        print(f"| {lo}-{hi if hi < 10_000 else ''} | {len(sel)} | {share('extra')} | {share('cer')} | "
              f"{share('loop')} | {share('')} |")
    kept = [s[2] for s in stats if not s[1]]
    hours = sum(r["lat_len"] for r in kept) * 0.08 / 3600
    print(f"\nkept {len(kept)} / {len(stats)} clips ({hours:.1f}h)")
    for _, reason, r, ref, hyp in [s for s in stats if s[1] == "extra"][: args.examples]:
        print(f"  [{reason}] {r['transcript']}\n      ref {ref}\n      asr {hyp}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
