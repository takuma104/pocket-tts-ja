"""High-pitched ("anime") voices from Irodori-TTS voice design, as a training set for the Mimi decoder (CC plan C6).

The Space's presets female_02 / female_05 (F0 350 / 406 Hz) are the voices the HF-stage decoder renders worst, and
YODAS3 has next to none that high. Irodori's voice design makes them from a caption alone (no reference voice, so
nothing but Irodori's MIT license applies). Two steps:

    uv run python scripts/m2/anime_set.py rows --texts TEXTS.jsonl --out data/m2/anime --per-caption 100
        writes data/m2/anime/rows.jsonl for scripts/synth/irodori_synth.py: every caption family x the sentences,
        ref_wav null, one seed per row. The last caption family and the last --heldout-sentences sentences of
        the others are the held-out part (ids start with "dev_").
    uv run python scripts/m2/anime_set.py manifests --out data/m2/anime
        after the synthesis (audio in data/m2/anime/audio): train.jsonl and dev.jsonl (id, duration, caption) for
        train_mimi_decoder.py --data, plus dev/ with copies of the held-out clips for the reconstruction checks.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import soundfile as sf

# The first two are the Space's female_02 / female_05 captions; the rest are variations of the same kind of voice.
# The last one is held out (a voice family the decoder never trains on).
CAPTIONS = [
    "明るく元気な若い女性の声。親しみやすく、はきはきと楽しそうに話している。",
    "高めで可愛らしい若い女性の声。無邪気に、弾むように話している。",
    "高く澄んだ少女の声。元気いっぱいに、楽しそうに話している。",
    "甲高くてかわいらしいアニメのヒロインのような声。明るく、少し早口で話している。",
    "高めで透き通った若い女性の声。やさしく、軽やかに語りかけている。",
    "幼さの残る高い女の子の声。はしゃぐように、にぎやかに話している。",
]


def read_texts(path: Path, field: str, lo: int, hi: int) -> list[str]:
    texts = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        t = str(row[field]).strip()
        if lo <= len(t) <= hi:
            texts.append(t)
    return list(dict.fromkeys(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("rows", "manifests"))
    parser.add_argument("--out", type=Path, default=Path("data/m2/anime"))
    parser.add_argument("--texts", type=Path, default=None, help="jsonl of sentences (rows mode)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--per-caption", type=int, default=100, help="sentences per caption family")
    parser.add_argument("--heldout-sentences", type=int, default=10, help="per training family, held out")
    parser.add_argument("--length", type=int, nargs=2, default=(15, 60), metavar=("MIN", "MAX"), help="characters")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.what == "rows":
        assert args.texts is not None, "--texts"
        rng = random.Random(args.seed)
        texts = read_texts(args.texts, args.text_field, *args.length)
        rng.shuffle(texts)
        texts = texts[: args.per_caption]
        rows = []
        for c, caption in enumerate(CAPTIONS):
            heldout_family = c == len(CAPTIONS) - 1
            for i, text in enumerate(texts):
                dev = heldout_family or i >= len(texts) - args.heldout_sentences
                rows.append({"id": f"{'dev' if dev else 'train'}_c{c}_{i:03d}", "text": text, "ref_wav": None,
                             "seed": rng.randrange(1 << 31), "caption": caption, "family": c})
        with open(args.out / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{args.out / 'rows.jsonl'}: {len(rows)} rows, {len(CAPTIONS)} captions x {len(texts)} sentences, "
              f"{sum(r['id'].startswith('dev') for r in rows)} held out")
        return

    rows = [json.loads(line) for line in open(args.out / "rows.jsonl", encoding="utf-8")]
    audio = args.out / "audio"
    dev_dir = args.out / "dev"
    dev_dir.mkdir(exist_ok=True)
    counts = {"train": 0, "dev": 0, "missing": 0}
    with open(args.out / "train.jsonl", "w", encoding="utf-8") as train, open(args.out / "dev.jsonl", "w", encoding="utf-8") as dev:
        for r in rows:
            wav = audio / f"{r['id']}.wav"
            if not wav.exists():
                counts["missing"] += 1
                continue
            info = sf.info(str(wav))
            row = {"id": r["id"], "duration": round(info.frames / info.samplerate, 3), "caption": r["caption"], "family": r["family"]}
            if r["id"].startswith("dev"):
                dev.write(json.dumps(row, ensure_ascii=False) + "\n")
                shutil.copy(wav, dev_dir / wav.name)
                counts["dev"] += 1
            else:
                train.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["train"] += 1
    print(f"{args.out}: {counts}")


if __name__ == "__main__":
    main()
