"""M0 scoring: how much does the Mimi round trip cost in intelligibility?

Runs in the benchmark environment (uses its normalization and CER):

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/m0_score.py exp/m0_mimi_recon

JVS part (needs asr_orig/ and asr_recon/ from scripts/asr_transcribe.py):
  - Text CER: whisper-large-v3-turbo transcript vs the JVS transcript (canonicalize_text)
  - Kana-CER: kana-whisper transcript vs the reading from the JVS phoneme labels
    (canonicalize_yomi); rows without labels are skipped for this metric
  - Kana consistency: kana-whisper(recon) vs kana-whisper(orig)
  Means are per-utterance averages (CER@1, like the benchmark's clipped metric), with a
  95% bootstrap interval on the recon - orig difference.

JKYB part (needs jkyb/results from `jkyb-eval tts` on the reconstructions): paired
accuracy comparison against the source audio's results (fixed / broken, exact McNemar).
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from jkyb_eval.metrics import character_error_rate
from jkyb_eval.normalization import canonicalize_text, canonicalize_yomi


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_key(path: Path, field: str) -> dict[str, str]:
    return {r["key"]: r[field] for r in read_jsonl(path)}


def cer1(ref: str, hyp: str) -> float:
    return min(1.0, character_error_rate(ref, hyp))


def bootstrap_ci(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, float] | None:
    pairs = [(r[f"{metric}_orig"], r[f"{metric}_recon"]) for r in rows if r.get(f"{metric}_orig") is not None]
    if not pairs:
        return None
    orig = sum(p[0] for p in pairs) / len(pairs)
    recon = sum(p[1] for p in pairs) / len(pairs)
    diffs = [p[1] - p[0] for p in pairs]
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(pairs),
        "orig": orig,
        "recon": recon,
        "delta": recon - orig,
        "delta_ci95": [lo, hi],
        "worse_frac": sum(d > 0 for d in diffs) / len(diffs),
        "better_frac": sum(d < 0 for d in diffs) / len(diffs),
    }


def score_jvs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jvs = root / "jvs"
    items = read_jsonl(jvs / "items.jsonl")
    asr = {
        kind: {
            "kana": by_key(jvs / f"asr_{kind}" / "kana.jsonl", "yomi"),
            "text": by_key(jvs / f"asr_{kind}" / "text.jsonl", "text"),
        }
        for kind in ("orig", "recon")
    }
    rows = []
    for it in items:
        key = it["key"]
        ref_text = canonicalize_text(it["text"])
        row: dict[str, Any] = {k: it[k] for k in ("key", "speaker", "gender", "subset")}
        for kind in ("orig", "recon"):
            row[f"text_cer_{kind}"] = cer1(ref_text, canonicalize_text(asr[kind]["text"][key]))
            row[f"kana_{kind}"] = asr[kind]["kana"][key]
            row[f"text_{kind}"] = asr[kind]["text"][key]
            if it["yomi"]:
                row[f"kana_cer_{kind}"] = cer1(
                    canonicalize_yomi(it["yomi"]), canonicalize_yomi(asr[kind]["kana"][key])
                )
            else:
                row[f"kana_cer_{kind}"] = None
        row["kana_consistency_orig"] = 0.0
        row["kana_consistency_recon"] = cer1(
            canonicalize_yomi(row["kana_orig"]), canonicalize_yomi(row["kana_recon"])
        )
        rows.append(row)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[f"subset={r['subset']}"].append(r)
        groups[f"gender={r['gender']}"].append(r)
        if r["subset"] in ("parallel100", "nonpara30"):
            groups["read (parallel100+nonpara30)"].append(r)
            groups[f"read, gender={r['gender']}"].append(r)
    summary = {
        name: {m: summarize(g, m) for m in ("text_cer", "kana_cer", "kana_consistency")}
        for name, g in groups.items()
    }
    # Per-speaker deltas on read speech: who suffers most from the codec?
    per_spk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in groups["read (parallel100+nonpara30)"]:
        per_spk[r["speaker"]].append(r)
    spk_rows = []
    for spk, g in sorted(per_spk.items()):
        s = summarize(g, "kana_cer")
        t = summarize(g, "text_cer")
        assert s is not None and t is not None
        spk_rows.append(
            {
                "speaker": spk,
                "gender": g[0]["gender"],
                "kana_cer_orig": s["orig"],
                "kana_cer_recon": s["recon"],
                "kana_delta": s["delta"],
                "text_delta": t["delta"],
            }
        )
    summary["per_speaker_read"] = sorted(spk_rows, key=lambda r: -r["kana_delta"])
    return summary, rows


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def score_jkyb(root: Path, source_results: Path) -> dict[str, Any] | None:
    recon_path = root / "jkyb" / "results" / "details" / "all.jsonl"
    if not recon_path.exists():
        return None
    src = {r["key"]: r for r in read_jsonl(source_results / "details" / "all.jsonl")}
    rec = {r["key"]: r for r in read_jsonl(recon_path)}
    keys = sorted(src.keys() & rec.keys())
    fixed = sum(1 for k in keys if not src[k]["target_exact"] and rec[k]["target_exact"])
    broken = sum(1 for k in keys if src[k]["target_exact"] and not rec[k]["target_exact"])

    def mean(d: dict[str, dict[str, Any]], f: str) -> float:
        return sum(d[k][f] for k in keys) / len(keys)

    out: dict[str, Any] = {"n": len(keys), "fixed": fixed, "broken": broken}
    out["mcnemar_p"] = mcnemar_exact(fixed, broken)
    for f in ("target_exact", "target_kana_cer_clipped", "sentence_kana_cer_clipped", "text_cer_clipped"):
        out[f] = {"source": mean(src, f), "recon": mean(rec, f)}
    cats: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        cats[src[k]["reading_category"]].append(k)
    out["by_category"] = {
        c: {
            "n": len(ks),
            "source": sum(src[k]["target_exact"] for k in ks) / len(ks),
            "recon": sum(rec[k]["target_exact"] for k in ks) / len(ks),
        }
        for c, ks in cats.items()
    }
    out["broken_examples"] = [
        {
            "key": k,
            "text": src[k]["text"],
            "ref": src[k]["reference_target"],
            "source_read": src[k]["mapped_target"],
            "recon_read": rec[k]["mapped_target"],
        }
        for k in keys
        if src[k]["target_exact"] and not rec[k]["target_exact"]
    ][:40]
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def to_markdown(summary: dict[str, Any], jkyb: dict[str, Any] | None) -> str:
    lines = ["# M0: Mimi reconstruction of Japanese speech", ""]
    lines.append("## JVS (orig vs Mimi round trip), per-utterance CER@1 in %")
    lines.append("")
    lines.append(
        "| group | n | Text CER orig | recon | Δ [95% CI] | Kana-CER orig | recon | Δ [95% CI] "
        "| Kana consistency |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in summary.items():
        if name == "per_speaker_read":
            continue
        t, k, c = s["text_cer"], s["kana_cer"], s["kana_consistency"]
        kcells = (
            f"{pct(k['orig'])} | {pct(k['recon'])} | {pct(k['delta'])} "
            f"[{pct(k['delta_ci95'][0])}, {pct(k['delta_ci95'][1])}]"
            if k
            else "– | – | –"
        )
        lines.append(
            f"| {name} | {t['n']} | {pct(t['orig'])} | {pct(t['recon'])} | {pct(t['delta'])} "
            f"[{pct(t['delta_ci95'][0])}, {pct(t['delta_ci95'][1])}] | {kcells} "
            f"| {pct(c['recon'])} |"
        )
    lines += ["", "Kana consistency = Kana-CER of kana-whisper(recon) against kana-whisper(orig).", ""]
    lines.append("### Speakers with the largest Kana-CER increase (read speech)")
    lines.append("")
    lines.append("| speaker | gender | Kana-CER orig | recon | Δ | Text Δ |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in summary["per_speaker_read"][:10]:
        lines.append(
            f"| {r['speaker']} | {r['gender']} | {pct(r['kana_cer_orig'])} | "
            f"{pct(r['kana_cer_recon'])} | {pct(r['kana_delta'])} | {pct(r['text_delta'])} |"
        )
    if jkyb:
        lines += ["", "## JKYB-Parakeet: Irodori-TTS release audio vs its Mimi round trip", ""]
        lines.append("| metric | source | Mimi recon |")
        lines.append("|---|---:|---:|")
        for f in ("target_exact", "target_kana_cer_clipped", "sentence_kana_cer_clipped", "text_cer_clipped"):
            lines.append(f"| {f} | {pct(jkyb[f]['source'])} | {pct(jkyb[f]['recon'])} |")
        lines.append("")
        lines.append(
            f"Paired accuracy: fixed {jkyb['fixed']}, broken {jkyb['broken']} "
            f"(n={jkyb['n']}, exact McNemar p={jkyb['mcnemar_p']:.2g})"
        )
        lines.append("")
        lines.append("| category | n | source acc | recon acc |")
        lines.append("|---|---:|---:|---:|")
        for c, v in jkyb["by_category"].items():
            lines.append(f"| {c} | {v['n']} | {pct(v['source'])} | {pct(v['recon'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--jkyb-source-results",
        type=Path,
        default=Path("/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/release/jkyb/results"),
    )
    args = parser.parse_args()
    summary, rows = score_jvs(args.root)
    jkyb = score_jkyb(args.root, args.jkyb_source_results)
    with open(args.root / "jvs" / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.root / "summary.json").write_text(
        json.dumps({"jvs": summary, "jkyb": jkyb}, ensure_ascii=False, indent=2)
    )
    md = to_markdown(summary, jkyb)
    (args.root / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
