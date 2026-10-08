"""Tokens per second of read speech (JVS parallel100 + nonpara30) for trained tokenizers.

Complements tokenizer_study.py, whose held-out set is spontaneous Emilia speech: pocket-tts
caps generation at tokens / 3 + 2 seconds, so the slow tail of real speech decides whether
the stock inference code truncates.

    uv run python scripts/token_rate_jvs.py exp/tokenizer_study/ja_unigram_*.model
"""

import argparse
import json
import statistics
from pathlib import Path

import sentencepiece as spm
import soundfile as sf

from pocket_tts_ja.jvs import list_items
from pocket_tts_ja.text import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    args = parser.parse_args()
    items = [i for i in list_items(args.jvs_root, ("parallel100", "nonpara30"))]
    durs = [sf.info(i.wav).duration for i in items]
    texts = [normalize(i.text) for i in items]
    for m in args.models:
        sp = spm.SentencePieceProcessor(model_file=str(m))
        n = [len(sp.encode(t)) for t in texts]
        tps = sorted(k / d for k, d in zip(n, durs))
        over = sum(k / 3.0 + 2.0 < d for k, d in zip(n, durs)) / len(durs)
        q = statistics.quantiles(tps, n=100)
        print(json.dumps({
            "model": m.name, "n": len(items),
            "tokens_per_sec_p1_p10_p50": [round(q[0], 2), round(q[9], 2), round(statistics.median(tps), 2)],
            "stock_budget_truncated_frac": round(over, 4),
            "chars_per_sec_median": round(statistics.median(len(t) / d for t, d in zip(texts, durs)), 2),
        }))


if __name__ == "__main__":
    main()
