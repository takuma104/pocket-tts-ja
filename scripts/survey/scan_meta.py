"""Stream the .json members of Emilia JA WebDataset shards and build a metadata index.

Also records the byte offset/size of each audio member so audio can be read later by
random access (f.seek(offset); f.read(size)) without re-scanning the tars.
"""

from __future__ import annotations

import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/mnt/ext11/Emilia-Dataset")
OUT = Path(__file__).resolve().parent
SUBSETS: dict[str, Path] = {
    "Emilia": ROOT / "Emilia" / "JA",
    "Emilia-YODAS": ROOT / "Emilia-YODAS" / "JA",
}
BUF = 4 * 1024 * 1024
KNOWN = {"id", "wav", "text", "duration", "speaker", "language", "dnsmos"}


def scan_shard(subset: str, path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    rows: dict[str, dict[str, Any]] = {}
    other_members: list[str] = []
    extra_keys: dict[str, int] = {}
    audio_ext: dict[str, int] = {}
    n_members = 0
    with open(path, "rb", buffering=BUF) as f:
        tf = tarfile.open(fileobj=f, mode="r:")
        for m in tf:
            n_members += 1
            tf.members = []  # avoid accumulating TarInfo objects
            if not m.isfile():
                continue
            name = m.name
            key, _, ext = name.rpartition(".")
            r = rows.setdefault(key, {"key": key})
            if ext == "json":
                fobj = tf.extractfile(m)
                assert fobj is not None
                d: dict[str, Any] = json.loads(fobj.read())
                for k in d:
                    if k not in KNOWN:
                        extra_keys[k] = extra_keys.get(k, 0) + 1
                r["json"] = d
            elif ext in ("mp3", "wav", "flac", "opus", "ogg"):
                r["audio_ext"] = ext
                r["audio_offset"] = m.offset_data
                r["audio_size"] = m.size
                audio_ext[ext] = audio_ext.get(ext, 0) + 1
            else:
                other_members.append(name)
    out: list[dict[str, Any]] = []
    shard = Path(path).name
    n_no_json = 0
    n_no_audio = 0
    for key, r in rows.items():
        d = r.get("json")
        if d is None:
            n_no_json += 1
            d = {}
        if "audio_offset" not in r:
            n_no_audio += 1
        out.append(
            {
                "subset": subset,
                "shard": shard,
                "key": key,
                "id": d.get("id"),
                "speaker": d.get("speaker"),
                "duration": d.get("duration"),
                "dnsmos": d.get("dnsmos"),
                "language": d.get("language"),
                "text": d.get("text"),
                "wav": d.get("wav"),
                "audio_ext": r.get("audio_ext"),
                "audio_offset": r.get("audio_offset"),
                "audio_size": r.get("audio_size"),
                "extra_json": json.dumps({k: v for k, v in d.items() if k not in KNOWN}, ensure_ascii=False)
                if any(k not in KNOWN for k in d)
                else None,
            }
        )
    info: dict[str, Any] = {
        "subset": subset,
        "shard": shard,
        "bytes": Path(path).stat().st_size,
        "members": n_members,
        "samples": len(out),
        "no_json": n_no_json,
        "no_audio": n_no_audio,
        "other_members": other_members[:20],
        "n_other_members": len(other_members),
        "extra_keys": extra_keys,
        "audio_ext": audio_ext,
        "sec": time.time() - t0,
    }
    return out, info


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs: list[tuple[str, str]] = []
    for subset, d in SUBSETS.items():
        for p in sorted(d.glob("*.tar")):
            jobs.append((subset, str(p)))
    # biggest first for better load balance
    jobs.sort(key=lambda j: -Path(j[1]).stat().st_size)
    t0 = time.time()
    all_rows: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_shard, s, p): (s, p) for s, p in jobs}
        for i, fut in enumerate(as_completed(futs)):
            rows, info = fut.result()
            all_rows.extend(rows)
            infos.append(info)
            el = time.time() - t0
            print(
                f"[{i + 1}/{len(jobs)}] {info['subset']}/{info['shard']} samples={info['samples']} "
                f"{info['bytes'] / 1e9:.2f}GB {info['sec']:.1f}s elapsed={el:.0f}s",
                flush=True,
            )
    total = time.time() - t0
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["subset", "shard", "key"]).reset_index(drop=True)
    df.to_parquet(OUT / "emilia_ja_meta.parquet", index=False)
    pd.DataFrame(infos).to_json(OUT / "shard_info.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"TOTAL scan time {total:.1f}s rows={len(df)} workers={workers}", flush=True)
    (OUT / "scan_time.txt").write_text(f"{total:.1f} sec, workers={workers}, rows={len(df)}\n")


if __name__ == "__main__":
    main()
