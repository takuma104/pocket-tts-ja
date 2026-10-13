#!/usr/bin/env python3
"""Select and extract reference voices for the synthetic corpus (plan §4.4, S2).

Selection (seeded):

- first-copy rows only (``is_first_copy``; cross-subset exact duplicates removed,
  so ``Emilia-YODAS`` rows here are YODAS-only material);
- the speaker has >= ``--min-utts`` first-copy utterances;
- the speaker is not held out: no row of the speaker has a ``source`` in the
  validation bucket ``int(md5(source), 16) % 200 == 0`` (the rule of the main
  valid split). With ``--source-map`` (Emilia source -> YODAS video), a speaker
  is also excluded when the twin source on the other side is in the bucket;
- clip filters: DNSMOS >= 3.3, bitrate (audio_size * 8 / duration) >= 56 kbps
  (drops the 32 kbps band-limited files), 5-12 s, the text contains kana/kanji
  and is not mostly Latin letters;
- one clip per speaker, the highest DNSMOS (ties broken randomly);
- speakers per subset in proportion to the subset's first-copy hours, sampled
  uniformly at random among the eligible speakers.

Extraction: the mp3 member is read from the tar shard at ``audio_offset``,
decoded, downmixed, resampled to 24 kHz (soxr HQ) and written as 16-bit PCM to
``<out-dir>/<speaker>.wav``. A clip that fails to decode, is shorter than 4 s or
has RMS <= -40 dBFS is replaced by the speaker's next-best clip, and a speaker
without a usable clip by the next sampled speaker of the same subset.

Writes ``--out-jsonl`` (one line per reference) and ``<out-jsonl stem>_summary.json``.
Needs numpy, pyarrow, soundfile (libsndfile >= 1.1 for mp3) and soxr.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
import soxr

COLUMNS = [
    "subset",
    "shard",
    "key",
    "id",
    "speaker",
    "source",
    "duration",
    "dnsmos",
    "text",
    "audio_offset",
    "audio_size",
    "is_first_copy",
]
JA_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿々〆ヶ]")
LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
SUBSET_DIRS = {"Emilia": "Emilia", "Emilia-YODAS": "Emilia-YODAS"}


def in_valid_bucket(source: str, mod: int = 200) -> bool:
    """The held-out rule shared with the main valid split."""
    return int(hashlib.md5(source.encode()).hexdigest(), 16) % mod == 0


def japanese_text(text: str, max_latin_ratio: float) -> bool:
    if not JA_RE.search(text):
        return False
    chars = [c for c in text if not c.isspace()]
    latin = sum(1 for c in chars if LATIN_RE.match(c))
    return latin / max(1, len(chars)) < max_latin_ratio


@dataclass(frozen=True)
class Clip:
    row: int
    subset: str
    shard: str
    key: str
    id: str
    speaker: str
    source: str
    duration: float
    dnsmos: float
    kbps: float
    text: str
    offset: int
    size: int


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    norm = sum(weights.values())
    raw = {k: total * w / norm for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[
        : total - sum(out.values())
    ]:
        out[k] += 1
    return out


def percentiles(
    values: list[float], qs: tuple[int, ...] = (0, 5, 25, 50, 75, 95, 100)
) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs} | {
        "mean": round(float(arr.mean()), 3)
    }


# --------------------------------------------------------------------------- selection


def select_candidates(
    meta: Path,
    *,
    source_map: Path | None,
    min_utts: int,
    min_dnsmos: float,
    min_kbps: float,
    min_dur: float,
    max_dur: float,
    max_latin_ratio: float,
    valid_mod: int,
    per_speaker: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, list[Clip]]], dict[str, float], dict[str, Any]]:
    """Returns subset -> speaker -> ranked clips, first-copy hours per subset, and stats."""
    table = pq.read_table(meta, columns=COLUMNS)
    stats: dict[str, Any] = {"rows": table.num_rows}
    table = table.filter(pc.field("is_first_copy"))
    stats["first_copy_rows"] = table.num_rows
    cols = {
        name: table.column(name).to_pylist()
        for name in COLUMNS
        if name != "is_first_copy"
    }
    n = table.num_rows
    subset = cols["subset"]
    speaker = cols["speaker"]
    source = cols["source"]
    duration = np.asarray(cols["duration"], dtype=np.float64)
    dnsmos = np.asarray(cols["dnsmos"], dtype=np.float64)
    size = np.asarray(cols["audio_size"], dtype=np.float64)
    kbps = size * 8.0 / np.maximum(duration, 1e-6) / 1000.0

    hours: dict[str, float] = defaultdict(float)
    for s, d in zip(subset, duration, strict=True):
        hours[s] += float(d) / 3600.0
    stats["first_copy_hours"] = {k: round(v, 1) for k, v in hours.items()}

    # Speakers: utterance counts and held-out exclusion.
    utt_count = Counter(speaker)
    held_sources = {s for s in set(source) if in_valid_bucket(s, valid_mod)}
    twin_held: set[str] = set()
    if source_map is not None and source_map.is_file():
        with source_map.open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                src, video = line.rstrip("\n").split("\t")
                if in_valid_bucket(video, valid_mod) or in_valid_bucket(src, valid_mod):
                    twin_held.update((src, video))
    held_speakers = {
        spk for spk, src in zip(speaker, source, strict=True) if src in held_sources
    }
    twin_speakers = {
        spk for spk, src in zip(speaker, source, strict=True) if src in twin_held
    } - held_speakers
    excluded = held_speakers | twin_speakers
    spk_subset: dict[str, str] = {}
    for spk, s in zip(speaker, subset, strict=True):
        spk_subset.setdefault(spk, s)
    all_speakers = set(speaker)
    stats["speakers_first_copy"] = len(all_speakers)
    stats["speakers_min_utts"] = sum(
        1 for spk in all_speakers if utt_count[spk] >= min_utts
    )
    stats["speakers_held_out_valid_bucket"] = len(held_speakers)
    stats["speakers_held_out_twin_source"] = len(twin_speakers)

    # Clip filters (stage counts on first-copy rows).
    ok = np.ones(n, dtype=bool)
    stage: dict[str, int] = {}
    for name, mask in (
        ("speaker_min_utts", np.array([utt_count[s] >= min_utts for s in speaker])),
        ("not_held_out", np.array([s not in excluded for s in speaker])),
        (f"dnsmos>={min_dnsmos}", dnsmos >= min_dnsmos),
        (f"kbps>={min_kbps}", kbps >= min_kbps),
        (
            f"duration_{min_dur}-{max_dur}s",
            (duration >= min_dur) & (duration <= max_dur),
        ),
    ):
        ok &= mask
        stage[name] = int(ok.sum())
    texts = cols["text"]
    ja = np.array(
        [bool(ok[i]) and japanese_text(texts[i], max_latin_ratio) for i in range(n)]
    )
    ok &= ja
    stage["japanese_text"] = int(ok.sum())
    stats["clip_filter_rows_remaining"] = stage

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(ok):
        by_speaker[speaker[i]].append(int(i))
    tiebreak = rng.random(n)
    pools: dict[str, dict[str, list[Clip]]] = defaultdict(dict)
    for spk in sorted(by_speaker):
        idx = sorted(by_speaker[spk], key=lambda i: (-dnsmos[i], tiebreak[i]))[
            :per_speaker
        ]
        pools[spk_subset[spk]][spk] = [
            Clip(
                row=i,
                subset=subset[i],
                shard=cols["shard"][i],
                key=cols["key"][i],
                id=cols["id"][i],
                speaker=spk,
                source=source[i],
                duration=float(duration[i]),
                dnsmos=float(dnsmos[i]),
                kbps=float(kbps[i]),
                text=texts[i],
                offset=int(cols["audio_offset"][i]),
                size=int(cols["audio_size"][i]),
            )
            for i in idx
        ]
    stats["eligible_speakers"] = {k: len(v) for k, v in pools.items()}
    return pools, dict(hours), stats


# --------------------------------------------------------------------------- extraction


def extract(
    clip: Clip,
    *,
    root: Path,
    out_dir: Path,
    sample_rate: int,
    min_rms_dbfs: float,
    min_sec: float,
) -> dict[str, Any] | str:
    """Write ``<speaker>.wav``; returns stats, or an error string."""
    path = root / SUBSET_DIRS[clip.subset] / "JA" / clip.shard
    try:
        with path.open("rb") as handle:
            handle.seek(clip.offset)
            data = handle.read(clip.size)
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return f"decode: {type(exc).__name__}: {exc}"
    x = audio.mean(axis=1)
    if sr != sample_rate:
        x = soxr.resample(x, sr, sample_rate, quality="HQ")
    seconds = len(x) / sample_rate
    if seconds < min_sec:
        return f"too short: {seconds:.2f}s"
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    if rms_dbfs <= min_rms_dbfs:
        return f"silent: rms {rms_dbfs:.1f} dBFS"
    peak = float(np.max(np.abs(x)))
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = out_dir / f"{clip.speaker}.wav"
    tmp = out.with_name(f".{out.name}.tmp")
    sf.write(str(tmp), pcm, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, out)
    return {
        "source_sr": int(sr),
        "wav_seconds": round(seconds, 3),
        "rms_dbfs": round(rms_dbfs, 2),
        "peak": round(peak, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--meta", type=Path, default=Path("exp/emilia_survey/emilia_ja_meta.parquet")
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("exp/emilia_survey/emilia_source_to_yodas_video.tsv"),
        help="Emilia source -> YODAS video tsv; twins of held-out sources are excluded too.",
    )
    parser.add_argument(
        "--emilia-root", type=Path, default=Path("/mnt/ext11/Emilia-Dataset")
    )
    parser.add_argument(
        "--only-subset",
        default=None,
        help="take all speakers from this subset (Emilia-YODAS: a set of CC BY voices)",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        default=None,
        help="references selected earlier; their speakers are not selected again",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/synth/refs"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/synth/refs.jsonl"))
    parser.add_argument("--n-speakers", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-utts", type=int, default=5)
    parser.add_argument("--min-dnsmos", type=float, default=3.3)
    parser.add_argument("--min-kbps", type=float, default=56.0)
    parser.add_argument("--min-dur", type=float, default=5.0)
    parser.add_argument("--max-dur", type=float, default=12.0)
    parser.add_argument("--max-latin-ratio", type=float, default=0.5)
    parser.add_argument("--valid-mod", type=int, default=200)
    parser.add_argument("--per-speaker-fallbacks", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-rms-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-sec", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pools, hours, stats = select_candidates(
        args.meta,
        source_map=args.source_map,
        min_utts=args.min_utts,
        min_dnsmos=args.min_dnsmos,
        min_kbps=args.min_kbps,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        max_latin_ratio=args.max_latin_ratio,
        valid_mod=args.valid_mod,
        per_speaker=args.per_speaker_fallbacks,
        rng=rng,
    )
    if args.exclude_jsonl is not None:
        with open(args.exclude_jsonl, encoding="utf-8") as f:
            taken = {json.loads(line)["speaker"] for line in f}
        pools = {s: {spk: c for spk, c in pool.items() if spk not in taken} for s, pool in pools.items()}
        stats["eligible_speakers_not_selected_before"] = {k: len(v) for k, v in pools.items()}
    if args.only_subset is not None:
        pools = {args.only_subset: pools[args.only_subset]}
        quota = {args.only_subset: min(args.n_speakers, len(pools[args.only_subset]))}
    else:
        quota = largest_remainder({s: hours[s] for s in pools}, args.n_speakers)
        for s in quota:  # shift any shortfall to the other subset
            short = quota[s] - len(pools[s])
            if short > 0:
                quota[s] -= short
                other = next(o for o in quota if o != s)
                quota[other] += short
    stats["quota"] = quota
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pwd = Path(os.environ.get("PWD", os.getcwd()))
    out_abs = args.out_dir if args.out_dir.is_absolute() else pwd / args.out_dir

    def run(clip: Clip) -> dict[str, Any] | str:
        return extract(
            clip,
            root=args.emilia_root,
            out_dir=args.out_dir,
            sample_rate=args.sample_rate,
            min_rms_dbfs=args.min_rms_dbfs,
            min_sec=args.min_sec,
        )

    records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for subset_name in sorted(quota):
            speakers = sorted(pools[subset_name])
            order = [speakers[i] for i in rng.permutation(len(speakers))]
            queue = iter(order)
            chosen: list[dict[str, Any]] = []
            # Round r tries the r-th clip of every pending speaker; replacements come from the queue.
            pending = {next(queue): 0 for _ in range(quota[subset_name])}
            while pending and len(chosen) < quota[subset_name]:
                items = list(pending.items())
                clips = [pools[subset_name][spk][rank] for spk, rank in items]
                results = list(pool.map(run, clips))
                next_pending: dict[str, int] = {}
                for (spk, rank), result in zip(items, results, strict=True):
                    clip = pools[subset_name][spk][rank]
                    if isinstance(result, str):
                        rejects.append(
                            {"speaker": spk, "key": clip.key, "reason": result}
                        )
                        if rank + 1 < len(pools[subset_name][spk]):
                            next_pending[spk] = rank + 1
                        else:
                            replacement = next(queue, None)
                            if replacement is not None:
                                next_pending[replacement] = 0
                        continue
                    chosen.append(
                        {
                            "speaker": spk,
                            "subset": clip.subset,
                            "id": clip.id,
                            "key": clip.key,
                            "shard": clip.shard,
                            "source": clip.source,
                            "duration": round(clip.duration, 3),
                            "dnsmos": round(clip.dnsmos, 4),
                            "kbps": round(clip.kbps, 1),
                            "path_galleria2": str(out_abs / f"{spk}.wav"),
                            "text": clip.text,
                            "clip_rank": rank,
                        }
                        | result
                    )
                pending = next_pending
            records.extend(sorted(chosen, key=lambda r: r["speaker"]))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subset[record["subset"]].append(record)
    summary = stats | {
        "selected": {k: len(v) for k, v in sorted(by_subset.items())},
        "selected_total": len(records),
        "rejected_clips": len(rejects),
        "reject_examples": rejects[:20],
        "duration": percentiles([r["wav_seconds"] for r in records]),
        "duration_by_subset": {
            k: percentiles([r["wav_seconds"] for r in v]) for k, v in by_subset.items()
        },
        "dnsmos": percentiles([r["dnsmos"] for r in records]),
        "kbps_by_subset": {
            k: dict(Counter(round(r["kbps"] / 8) * 8 for r in v).most_common(6))
            for k, v in by_subset.items()
        },
        "rms_dbfs": percentiles([r["rms_dbfs"] for r in records]),
        "peak": percentiles([r["peak"] for r in records]),
        "source_sr": dict(Counter(r["source_sr"] for r in records)),
        "total_ref_minutes": round(sum(r["wav_seconds"] for r in records) / 60.0, 1),
    }
    summary_path = args.out_jsonl.with_name(args.out_jsonl.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("selected", "rejected_clips", "duration", "rms_dbfs")
            },
            indent=1,
        )
    )
    print(f"wrote {len(records)} refs to {args.out_jsonl} and {args.out_dir}")


if __name__ == "__main__":
    main()
