"""Kana-CER / Text CER of generated JVS sentences (JKYB-Parakeet normalization).

Runs in the benchmark environment; called by scripts/eval_tts.py:

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/score_jvs_cer.py items.json asr_dir out.json
"""

import json
import sys
from pathlib import Path

from jkyb_eval.metrics import character_error_rate
from jkyb_eval.normalization import canonicalize_text, canonicalize_yomi


def by_key(path: Path, field: str) -> dict[str, str]:
    return {json.loads(line)["key"]: json.loads(line)[field] for line in open(path)}


def main():
    items = json.loads(Path(sys.argv[1]).read_text())
    asr = Path(sys.argv[2])
    kana, text = by_key(asr / "kana.jsonl", "yomi"), by_key(asr / "text.jsonl", "text")
    rows = []
    for it in items:
        k = it["key"]
        rows.append(
            {
                "gender": it.get("gender"),
                "kana": min(1.0, character_error_rate(canonicalize_yomi(it["yomi"]), canonicalize_yomi(kana[k])))
                if it.get("yomi")
                else None,
                "text": min(1.0, character_error_rate(canonicalize_text(it["text"]), canonicalize_text(text[k]))),
            }
        )

    def mean(vals: list[float]) -> float:
        return sum(vals) / max(1, len(vals))

    out = {
        "kana_cer@1": mean([r["kana"] for r in rows if r["kana"] is not None]),
        "text_cer@1": mean([r["text"] for r in rows]),
    }
    for g in ("M", "F"):
        out[f"kana_cer@1_{g}"] = mean([r["kana"] for r in rows if r["gender"] == g and r["kana"] is not None])
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
