"""High-pitched reference voices for new teacher data (user's listening of P2 60k: female_02 / 05
sound doubled; at 351 / 374 Hz they lie above every one of the 3,000 training references, whose
99th percentile is 289 Hz).

Emilia-YODAS (CC BY) speakers not yet used as references, filtered as main's select_refs.py
(first copy, >= --min-utts utterances, not in the held-out bucket of the source or its twin,
DNSMOS >= --min-dnsmos, >= 56 kbps, 5-12 s, Japanese text), one clip per speaker (the highest DNSMOS):

    candidates  meta parquet -> <out>/candidates.jsonl
    extract     (where the tar shards are) the mp3 members -> <out>/mp3/<speaker>.mp3
    scan        median F0 of each clip (hf_structure.voiced_f0 at 24 kHz, -20 dBFS) -> <out>/f0.jsonl
    select      speakers with median F0 >= --min-f0 -> <out>/refs/<speaker>.wav (24 kHz 16 bit)
                and <out>/refs.jsonl (the fields of refs_cc.jsonl plus f0 / voiced)
    rows        teacher input rows for those voices -> data/irodori_distill/teacher/rows_<--set>.jsonl

    uv run python scripts/teacher/high_f0_refs.py candidates --out data/irodori_distill/synth_src/hi_f0
    ssh galleria2-direct 'cd co/pocket-tts-ja-distill && python3 scripts/teacher/high_f0_refs.py extract \
        --out data/irodori_distill/synth_src/hi_f0 --tar-root /mnt/ext11/Emilia-Dataset/Emilia-YODAS/JA'
    uv run python scripts/teacher/high_f0_refs.py scan --out data/irodori_distill/synth_src/hi_f0
    uv run python scripts/teacher/high_f0_refs.py select --out data/irodori_distill/synth_src/hi_f0 --min-f0 280
    uv run python scripts/teacher/high_f0_refs.py rows --set hi1 --per-voice 80
"""

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

SRC = Path("data/irodori_distill/synth_src")
JA_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿々〆ヶ]")
LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
SR = 24000


def in_valid_bucket(source: str, mod: int = 200) -> bool:
    return int(hashlib.md5(source.encode()).hexdigest(), 16) % mod == 0


def japanese_text(text: str, max_latin_ratio: float = 0.5) -> bool:
    chars = [c for c in text if not c.isspace()]
    return bool(JA_RE.search(text)) and sum(bool(LATIN_RE.match(c)) for c in chars) / max(1, len(chars)) < max_latin_ratio


def used_speakers() -> set[str]:
    used = {json.loads(line)["speaker"] for line in open(SRC / "refs_cc.jsonl", encoding="utf-8")}
    return used | {line.strip().removesuffix(".wav") for line in open(SRC / "refs_yodas_files.txt", encoding="utf-8")}


def candidates(args: argparse.Namespace) -> None:
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = ["subset", "shard", "key", "id", "speaker", "source", "duration", "dnsmos", "text", "audio_offset", "audio_size", "is_first_copy"]
    table = pq.read_table(args.meta, columns=cols).filter(pc.field("is_first_copy"))
    utts = Counter(table.column("speaker").to_pylist())
    table = table.filter(pc.equal(table["subset"], "Emilia-YODAS"))
    rows = table.to_pylist()
    held = {r["source"] for r in rows if in_valid_bucket(r["source"])}
    with open(args.source_map, encoding="utf-8") as f:
        next(f)
        for line in f:
            src, video = line.rstrip("\n").split("\t")
            if in_valid_bucket(src) or in_valid_bucket(video):
                held |= {src, video}
    held_speakers = {r["speaker"] for r in rows if r["source"] in held}
    used = used_speakers()
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        kbps = r["audio_size"] * 8 / max(r["duration"], 1e-6) / 1000
        if (utts[r["speaker"]] < args.min_utts or r["speaker"] in held_speakers or r["speaker"] in used or r["dnsmos"] < args.min_dnsmos
                or kbps < 56 or not 5.0 <= r["duration"] <= 12.0 or not japanese_text(r["text"])):
            continue
        r["kbps"] = round(kbps, 1)
        if r["speaker"] not in best or r["dnsmos"] > best[r["speaker"]]["dnsmos"]:
            best[r["speaker"]] = r
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for spk in sorted(best):
            r = best[spk]
            f.write(json.dumps({k: r[k] for k in ("speaker", "subset", "id", "key", "shard", "source", "duration", "dnsmos", "kbps",
                                                  "text", "audio_offset", "audio_size")}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} first-copy Emilia-YODAS rows, {len(used)} speakers already used, {len(held_speakers)} held out: "
          f"{len(best)} candidate speakers, median DNSMOS {np.median([r['dnsmos'] for r in best.values()]):.2f}")


def extract(args: argparse.Namespace) -> None:
    """Byte ranges only (no decoding), so it costs the tar box no CPU."""
    rows = [json.loads(line) for line in open(args.out / "candidates.jsonl", encoding="utf-8")]
    out = args.out / "mp3"
    out.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    n = 0
    for shard, group in sorted(by_shard.items()):
        with open(args.tar_root / shard, "rb") as f:
            for r in sorted(group, key=lambda r: r["audio_offset"]):
                f.seek(r["audio_offset"])
                (out / f"{r['speaker']}.mp3").write_bytes(f.read(r["audio_size"]))
                n += 1
        print(f"{shard}: {len(group)} clips", flush=True)
    print(f"{n} clips -> {out}")


def load(path: Path) -> "Any":
    import numpy as np
    import soundfile as sf
    import soxr

    x, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ").astype(np.float32)
    return x


def median_f0(path: Path) -> tuple[str, float, float]:
    import numpy as np

    sys.path.insert(0, "scripts/eval")
    from hf_structure import HOP, voiced_f0

    try:
        x = load(path)
    except Exception:  # noqa: BLE001 - a broken member drops the speaker
        return path.stem, float("nan"), 0.0
    x = x * 10 ** (-20 / 20) / max(float(np.sqrt(np.mean(x**2))), 1e-6)
    f = [f0 for _, f0 in voiced_f0(x, SR)]
    return path.stem, float(np.median(f)) if f else float("nan"), len(f) * HOP / max(len(x), 1)


def scan(args: argparse.Namespace) -> None:
    import numpy as np

    paths = sorted((args.out / "mp3").glob("*.mp3"))
    with ProcessPoolExecutor(args.workers) as pool, open(args.out / "f0.jsonl", "w", encoding="utf-8") as f:
        for i, (spk, f0, voiced) in enumerate(pool.map(median_f0, paths, chunksize=16)):
            f.write(json.dumps({"speaker": spk, "f0": round(f0, 1), "voiced": round(voiced, 3)}) + "\n")
            if i % 2000 == 0:
                print(f"{i}/{len(paths)}", flush=True)
    rows = [json.loads(line) for line in open(args.out / "f0.jsonl", encoding="utf-8")]
    f0s = np.array([r["f0"] for r in rows if np.isfinite(r["f0"]) and r["voiced"] >= 0.2])
    print(f"{len(rows)} clips, {len(f0s)} with >= 20% voiced frames; median F0 percentiles 50/90/95/99: "
          f"{np.percentile(f0s, [50, 90, 95, 99]).round(0)}")
    for t in (260, 280, 300, 320, 350):
        print(f">= {t} Hz: {np.sum(f0s >= t)} speakers")


def select(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf

    f0 = {r["speaker"]: r for r in map(json.loads, open(args.out / "f0.jsonl", encoding="utf-8"))}
    keep = [r for r in map(json.loads, open(args.out / "candidates.jsonl", encoding="utf-8"))
            if r["speaker"] in f0 and f0[r["speaker"]]["f0"] >= args.min_f0 and f0[r["speaker"]]["voiced"] >= 0.2]
    keep.sort(key=lambda r: -f0[r["speaker"]]["f0"])
    keep = keep[: args.max_speakers]
    out = args.out / "refs"
    out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "refs.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            x = load(args.out / "mp3" / f"{r['speaker']}.mp3")
            sf.write(out / f"{r['speaker']}.wav", np.clip(x, -1, 1), SR, subtype="PCM_16")
            f.write(json.dumps(r | {"f0": f0[r["speaker"]]["f0"], "voiced": f0[r["speaker"]]["voiced"]}, ensure_ascii=False) + "\n")
    print(f"{len(keep)} speakers with median F0 >= {args.min_f0} Hz -> {out}")


def rows(args: argparse.Namespace) -> None:
    """Teacher input rows for the selected voices: sentences of the packed sets (so each carries
    main's accepted transcript as old_kana for qc.py), drawn at random over all their lengths, with
    a new voice and seed. Sentences of the P2 validation rows and of its held-out voices stay out."""
    import random

    teacher = Path("data/irodori_distill/teacher")
    valid = [json.loads(line) for line in open("data/irodori_distill/mix/P2_s3_valid.jsonl", encoding="utf-8")]
    held_voices = {r["ref_speaker"] for r in valid}
    held_texts = {r["text"] for r in valid}
    pool: dict[str, dict[str, Any]] = {}
    for s in ("s3", "s3b", "s3c", "s4", "s5", "s6a", "s6b", "s7", "s9"):
        packed = {json.loads(line)["id"] for line in open(teacher / s / "manifest.jsonl", encoding="utf-8")}
        for line in open(teacher / f"rows_{s}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in packed and r["ref_speaker"] not in held_voices and r["text"] not in held_texts:
                pool.setdefault(r["id"], r)
    voices = [json.loads(line) for line in open(args.out / "refs.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    picks = rng.sample(sorted(pool), len(voices) * args.per_voice)
    path = teacher / f"rows_{args.set}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, rid in enumerate(picks):
            r, v = pool[rid], voices[i % len(voices)]
            row = {"id": f"syn_{args.set}_{i}", "text": r["text"], "ref_wav": str(args.out / "refs" / f"{v['speaker']}.wav"),
                   "seed": rng.randrange(2**31), "transcript": r["transcript"], "ref_speaker": v["speaker"], "ref_id": v["id"],
                   "old_kana": r["old_kana"], "old_duration": r["old_duration"], "text_from": rid}
            if "seconds" in r:  # a fixed length (main's short words), else the duration predictor for the new voice
                row["seconds"] = r["seconds"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(pool)} sentences in the pool; {len(picks)} rows ({args.per_voice} per voice, "
          f"{sum(pool[p]['old_duration'] for p in picks) / 3600:.1f} h at the old durations) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["candidates", "extract", "scan", "select", "rows"])
    parser.add_argument("--out", type=Path, default=SRC / "hi_f0")
    parser.add_argument("--meta", type=Path, default=Path("exp/emilia_survey/emilia_ja_meta.parquet"))
    parser.add_argument("--source-map", type=Path, default=Path("exp/emilia_survey/emilia_source_to_yodas_video.tsv"))
    parser.add_argument("--tar-root", type=Path, default=Path("/mnt/ext11/Emilia-Dataset/Emilia-YODAS/JA"))
    parser.add_argument("--min-dnsmos", type=float, default=3.3)
    parser.add_argument("--min-utts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-f0", type=float, default=270.0)
    parser.add_argument("--max-speakers", type=int, default=600)
    parser.add_argument("--set", default="hi1", help="rows: the teacher set name (rows_<set>.jsonl)")
    parser.add_argument("--per-voice", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"candidates": candidates, "extract": extract, "scan": scan, "select": select, "rows": rows}[args.stage](args)


if __name__ == "__main__":
    main()
