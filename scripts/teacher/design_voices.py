"""High-pitched reference voices designed by Irodori-TTS (hi2), for new teacher data.

The students lose the harmonics of female_02 / 05 (median F0 351 / 374 Hz) and the loss grows
with the frame's F0: none below 250 Hz, about 2 dB at 300-400 Hz, 3-4 dB at 400-450 Hz (comb
2.5-6 kHz, 10/24). The training references stop there: of the 3,000 Emilia references the 99th
percentile is 289 Hz, hi1's 270 voices are 280-412 Hz (median 301), and Emilia-YODAS has only 19
more speakers above 350 Hz. The release voices themselves were designed by Irodori-TTS
v4.1-Small from a caption (exp/release_cc/voices/voices.json), so new voices of the same kind
come from captions that ask for a high voice, each with many seeds: Irodori's voice design
(caption, no reference) reading the release voices' text.

    rows    captions x seeds -> <out>/design_rows.jsonl
            (galleria2: irodori_synth.py --checkpoint Irodori-TTS-v4.1-Small -> <out>/audio)
    scan    median F0 (as high_f0_refs.py: hf_structure.voiced_f0 at 24 kHz, -20 dBFS) of every
            designed clip -> <out>/f0.jsonl
    select  the clips with median F0 in [--min-f0, --max-f0], at most --per-caption of each
            caption (the highest first) -> <out>/refs/<id>.wav and <out>/refs.jsonl, the fields
            high_f0_refs.py rows reads (speaker, id) plus f0, voiced, caption, seed

    uv run python scripts/teacher/design_voices.py rows
    uv run python scripts/teacher/design_voices.py scan
    uv run python scripts/teacher/design_voices.py select --min-f0 330
    uv run python scripts/teacher/high_f0_refs.py rows --out data/irodori_distill/synth_src/hi_design --set hi2

The captions paraphrase the release's high voices (female_02 明るく元気, female_05 高めで可愛らしい)
without repeating them, and the seeds start at 1000, so female_02 / 05 stay unseen voices.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from high_f0_refs import SRC, median_f0

RELEASE_VOICES = Path("exp/release_cc/voices/voices.json")
CAPTIONS = (
    "明るく元気な若い女性の声。高めのトーンで、はきはきと話している。",
    "高く澄んだ少女の声。素直に、はっきりと読み上げている。",
    "アニメのヒロインのような、高くて可愛らしい女性の声。明るく話している。",
    "甲高く元気な女の子の声。楽しそうに、はずむように話している。",
    "高めで優しい若い女性の声。ふんわりと、やわらかく話している。",
    "高い声の若い女性。落ち着いて、丁寧に読み上げている。",
    "高めで透明感のある女性の声。さわやかに、はっきりと話している。",
    "小柄な女の子のような高い声。元気いっぱいに話している。",
    "高めの声の女性アナウンサー。明るく、聞き取りやすく読み上げている。",
    "高めで甘い声の若い女性。にこやかに、親しみやすく話している。",
    "高く細い声の女性。少し恥ずかしそうに、でもはっきりと話している。",
    "可愛らしい高い声の女性。無邪気に、楽しそうに話している。",
)


def rows(args: argparse.Namespace) -> None:
    text = json.loads(RELEASE_VOICES.read_text(encoding="utf-8"))["text"]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "design_rows.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c, caption in enumerate(CAPTIONS):
            for s in range(args.seeds):
                seed = 1000 + s
                f.write(json.dumps({"id": f"hd{c:02d}_{seed}", "text": text, "ref_wav": None, "seed": seed,
                                    "caption": caption}, ensure_ascii=False) + "\n")
    print(f"{len(CAPTIONS)} captions x {args.seeds} seeds -> {path}")


def scan(args: argparse.Namespace) -> None:
    import numpy as np

    paths = sorted((args.out / "audio").glob("*.wav"))
    with ProcessPoolExecutor(args.workers) as pool, open(args.out / "f0.jsonl", "w", encoding="utf-8") as f:
        for spk, f0, voiced in pool.map(median_f0, paths, chunksize=8):
            f.write(json.dumps({"speaker": spk, "f0": round(f0, 1), "voiced": round(voiced, 3)}) + "\n")
    recs = [json.loads(line) for line in open(args.out / "f0.jsonl", encoding="utf-8")]
    by: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        if np.isfinite(r["f0"]) and r["voiced"] >= 0.2:
            by[r["speaker"][:4]].append(r["f0"])
    print(f"{len(recs)} clips; median F0 per caption (10/50/90th percentile, share >= 330 Hz):")
    for c in sorted(by):
        v = np.array(by[c])
        print(f"  {c} {CAPTIONS[int(c[2:])][:24]}: {np.percentile(v, [10, 50, 90]).round(0)} {100 * (v >= 330).mean():.0f}%")


def select(args: argparse.Namespace) -> None:
    import shutil

    design = {r["id"]: r for r in map(json.loads, open(args.out / "design_rows.jsonl", encoding="utf-8"))}
    recs = [r for r in map(json.loads, open(args.out / "f0.jsonl", encoding="utf-8"))
            if args.min_f0 <= r["f0"] <= args.max_f0 and r["voiced"] >= 0.2]
    recs.sort(key=lambda r: -r["f0"])
    count: dict[str, int] = defaultdict(int)
    out = args.out / "refs"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out / "refs.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            cap = r["speaker"][:4]
            if count[cap] >= args.per_caption:
                continue
            count[cap] += 1
            d = design[r["speaker"]]
            shutil.copyfile(args.out / "audio" / f"{r['speaker']}.wav", out / f"{r['speaker']}.wav")
            f.write(json.dumps({"speaker": r["speaker"], "id": r["speaker"], "f0": r["f0"], "voiced": r["voiced"],
                                "caption": d["caption"], "seed": d["seed"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} voices with median F0 in [{args.min_f0:g}, {args.max_f0:g}] Hz "
          f"(per caption: {dict(sorted(count.items()))}) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["rows", "scan", "select"])
    parser.add_argument("--out", type=Path, default=SRC / "hi_design")
    parser.add_argument("--seeds", type=int, default=140, help="rows: seeds per caption")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-f0", type=float, default=330.0)
    parser.add_argument("--max-f0", type=float, default=480.0)
    parser.add_argument("--per-caption", type=int, default=30)
    args = parser.parse_args()
    {"rows": rows, "scan": scan, "select": select}[args.stage](args)


if __name__ == "__main__":
    main()
