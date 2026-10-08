"""A0 (1/2): filter the Emilia JA metadata index into one manifest of utterances to encode.

Input is the survey's metadata table (every utterance of Emilia/JA and Emilia-YODAS/JA with
its tar offset). Rules, in order (plan §4.1):

  1. dedup: keep the first copy of each (text, duration@10ms, mp3 size) key
     (Emilia B00004-B00006 duplicate most of Emilia-YODAS; Emilia B0-3 repeats itself)
  2. text contains kana or kanji (drops English-only clips labelled ja)
  3. 1.5 <= chars/sec <= 15 (truncated transcripts, ASR repetition loops)
  4. dnsmos >= 3.0
  5. normalized text is non-empty and is not a JKYB-Parakeet sentence
  6. the speaker keeps >= 2 utterances in its split (a voice prompt needs another utterance)

The split is by video: Emilia speaker ids are mapped to their YouTube video where the survey
found one (Emilia B4-6 == YODAS), so one video's speakers never straddle train and valid.
valid = md5(group) % 200 == 0 (about 0.5%).

    uv run python scripts/prepare_emilia_ja.py
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from pocket_tts_ja.text import normalize

KANA_KANJI = re.compile(r"[ぁ-ゖァ-ヺ一-鿿㐀-䶿々]")
COLUMNS = [
    "subset", "shard", "id", "speaker", "source", "duration", "dnsmos", "text",
    "audio_offset", "audio_size", "is_first_copy",
]


def group_of(subset: str, source: str, video_of: dict[str, str]) -> str:
    if subset == "Emilia-YODAS":
        return source  # the YouTube video id
    return video_of.get(source, f"emilia:{source}")


def is_valid_group(group: str) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 200 == 0


def jkyb_texts() -> set[str]:
    path = hf_hub_download(
        "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet",
        "data/common_kanji_source.jsonl",
        repo_type="dataset",
    )
    return {normalize(json.loads(line)["text"]) for line in open(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=Path("exp/emilia_survey/emilia_ja_meta.parquet"))
    parser.add_argument(
        "--video-map", type=Path, default=Path("exp/emilia_survey/emilia_source_to_yodas_video.tsv")
    )
    parser.add_argument("--out", type=Path, default=Path("data/emilia_ja/manifest_all.jsonl"))
    args = parser.parse_args()

    video_of: dict[str, str] = {}
    with open(args.video_map) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            video_of.setdefault(row["source"], row["video"])
    jkyb = jkyb_texts()

    table = pq.read_table(args.meta, columns=COLUMNS).to_pydict()
    n = len(table["id"])
    dropped: Counter = Counter()
    kept: list[dict] = []
    for i in range(n):
        if not table["is_first_copy"][i]:
            dropped["1_duplicate"] += 1
            continue
        text = table["text"][i]
        dur = float(table["duration"][i])
        if not KANA_KANJI.search(text):
            dropped["2_no_kana_kanji"] += 1
            continue
        cps = len(re.sub(r"\s", "", text)) / dur
        if not 1.5 <= cps <= 15.0:
            dropped["3_chars_per_sec"] += 1
            continue
        if float(table["dnsmos"][i]) < 3.0:
            dropped["4_dnsmos"] += 1
            continue
        transcript = normalize(text)
        if not transcript:
            dropped["5_empty_after_normalize"] += 1
            continue
        if transcript in jkyb:
            dropped["5_jkyb_sentence"] += 1
            continue
        subset = table["subset"][i]
        group = group_of(subset, table["source"][i], video_of)
        kept.append(
            {
                "id": table["id"][i],
                "subset": subset,
                "source_type": "yodas" if subset == "Emilia-YODAS" else "emilia",
                "shard": table["shard"][i],
                "offset": int(table["audio_offset"][i]),
                "size": int(table["audio_size"][i]),
                "duration": dur,
                "kbps": round(int(table["audio_size"][i]) * 8 / dur / 1000, 1),
                "dnsmos": float(table["dnsmos"][i]),
                "speaker": table["speaker"][i],
                "group": group,
                "split": "valid" if is_valid_group(group) else "train",
                "text_raw": text,
                "transcript": transcript,
            }
        )

    per_spk = Counter((r["split"], r["speaker"]) for r in kept)
    final = []
    for r in kept:
        if per_spk[(r["split"], r["speaker"])] < 2:
            dropped["6_single_utt_speaker"] += 1
            continue
        final.append(r)
    final.sort(key=lambda r: (r["subset"], r["shard"], r["offset"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours: dict[str, float] = defaultdict(float)
    spk: dict[str, set[str]] = defaultdict(set)
    for r in final:
        key = f"{r['split']}/{r['source_type']}"
        hours[key] += r["duration"] / 3600
        spk[key].add(r["speaker"])
    stats = {
        "input_rows": n,
        "dropped": dict(sorted(dropped.items())),
        "kept_rows": len(final),
        "hours": {k: round(v, 1) for k, v in sorted(hours.items())},
        "speakers": {k: len(v) for k, v in sorted(spk.items())},
        "kbps_hours": {
            b: round(sum(r["duration"] for r in final if lo <= r["kbps"] < hi) / 3600, 1)
            for b, (lo, hi) in {"<48": (0, 48), "48-100": (48, 100), ">=100": (100, 1e9)}.items()
        },
    }
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
