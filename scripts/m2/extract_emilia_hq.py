"""M2 data: the high-bitrate, clean part of Emilia JA as 24 kHz wavs.

The Mimi decoder fine-tune needs waveforms (the main training only keeps latents), and only
audio that is not band-limited by a low mp3 bitrate: training rows of at least --min-kbps and
--min-dnsmos (defaults: 128 kbps, 3.4; 85.7 h, 52,519 clips of 8,416 speakers in 40 shards).

    uv run python scripts/m2/extract_emilia_hq.py --select   # writes <out>/manifest.jsonl
    uv run python scripts/m2/extract_emilia_hq.py            # reads it, writes <out>/audio/<id>.wav

The two steps can run on different machines (the selection needs the training list, the
extraction the tar shards). Shards are streamed sequentially (scripts/precompute_emilia_ja.py);
clips already written are skipped.
"""

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precompute_emilia_ja import SR, read_shard  # noqa: E402


class WavSink:
    """Takes what read_shard emits and writes the clips."""

    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir
        self.written = 0
        self.errors = 0

    def put(self, item: tuple[Any, ...]) -> None:
        kind = item[0]
        if kind == "item":
            _, _, utt_id, wav, _ = item
            sf.write(str(self.audio_dir / f"{utt_id}.wav"), np.clip(wav, -1.0, 1.0), SR, subtype="PCM_16")
            self.written += 1
        elif kind == "error":
            self.errors += 1


def extract(jobs: list[tuple[str, list[dict[str, Any]]]], audio_dir: Path) -> tuple[int, int]:
    sink = WavSink(audio_dir)
    for shard, entries in jobs:
        entries = [e for e in entries if not (audio_dir / f"{e['id']}.wav").exists()]
        if entries:
            read_shard("Emilia", shard, entries, sink)  # type: ignore[arg-type]
            print(f"{shard}: {len(entries)} clips", flush=True)
    return sink.written, sink.errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/emilia_ja/manifest_all.jsonl"))
    parser.add_argument("--ids", type=Path, default=Path("data/emilia_ja/train_rechecked_full.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/m2/emilia_hq"))
    parser.add_argument("--select", action="store_true", help="write <out>/manifest.jsonl and stop")
    parser.add_argument("--min-kbps", type=float, default=128.0)
    parser.add_argument("--min-dnsmos", type=float, default=3.4)
    parser.add_argument("--seconds", type=float, nargs=2, default=(3.0, 12.0), metavar=("MIN", "MAX"))
    parser.add_argument("--readers", type=int, default=4)
    args = parser.parse_args()

    selected = args.out / "manifest.jsonl"
    if args.select:
        ids = {json.loads(line)["id"] for line in open(args.ids)}
        args.out.mkdir(parents=True, exist_ok=True)
        hours = 0.0
        count = 0
        with open(selected, "w") as f:
            for line in open(args.manifest):
                row = json.loads(line)
                if (row["subset"] == "Emilia" and row["id"] in ids and row["kbps"] >= args.min_kbps
                        and row["dnsmos"] >= args.min_dnsmos and args.seconds[0] <= row["duration"] <= args.seconds[1]):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    hours += row["duration"] / 3600
                    count += 1
        print(f"{count} clips, {hours:.1f} h -> {selected}")
        return

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in open(selected):
        row = json.loads(line)
        by_shard[row["shard"]].append(row)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(by_shard.items(), key=lambda kv: -len(kv[1]))
    parts = [shards[i :: args.readers] for i in range(args.readers)]
    with mp.Pool(args.readers) as pool:
        results = pool.starmap(extract, [(part, audio_dir) for part in parts])
    print(f"written {sum(r[0] for r in results)}, errors {sum(r[1] for r in results)}; EXTRACT_DONE")


if __name__ == "__main__":
    main()
