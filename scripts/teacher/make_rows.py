"""P1 input rows: main's CC BY synthetic sentences again, now keeping the teacher's latents.

Every clip of main's CC BY line that passed QC is regenerated with the same text, reference and
seed (so with the same audio up to the numerics of the batch), this time with the latents saved:

    <set>_cc   main's re-synthesis of the NC-voice sentences in CC BY (Emilia-YODAS) voices
               (data/synth/<set>_cc/manifest.jsonl, inputs <set>_cc_input.jsonl)
    <set>      the original sets' clips whose voice was already an Emilia-YODAS one
               (data/synth/<set>/manifest.jsonl, s3_full for s3)

Inputs come from data/irodori_distill/synth_src (fetched read-only from galleria2): the input
jsonl files, refs_cc/ and refs_yodas/ (the reference wavs) and old_asr/<dir>/kana.jsonl (the
kana-whisper transcripts the accepted clips had, for the QC by agreement). Each output row keeps
the training transcript of the manifest (kanji, normalised as main's training text; the TTS
input may carry katakana for hard readings) and the old transcript.

    uv run python scripts/teacher/make_rows.py --out data/irodori_distill/teacher
    -> <out>/rows_<set>.jsonl for s3 s4 s5 s6a s6b s7 s9
"""

import argparse
import json
from pathlib import Path
from typing import Any

SYNTH = Path("data/synth")
SRC = Path("data/irodori_distill/synth_src")
SETS = ("s3", "s4", "s5", "s6a", "s6b", "s7", "s9")


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def original_input(name: str, form: str) -> Path:
    return SRC / (f"{name}_input.jsonl" if name in ("s3", "s4", "s9") else f"{name}_input_{form}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/irodori_distill/teacher"))
    args = parser.parse_args()
    form = (SRC / "c2_kana_form").read_text().strip()
    yodas = {r["speaker"] for r in read(SYNTH / "refs.jsonl") if r["subset"] == "Emilia-YODAS"}
    args.out.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_hours = 0.0
    for name in SETS:
        sources = [  # (manifest of accepted clips, input rows, old transcripts, reference dir, voice filter)
            (SYNTH / f"{name}_cc" / "manifest.jsonl", SRC / f"{name}_cc_input.jsonl",
             SRC / "old_asr" / f"{name}_cc" / "kana.jsonl", SRC / "refs_cc", None),
            (SYNTH / ("s3_full" if name == "s3" else name) / "manifest.jsonl", original_input(name, form),
             SRC / "old_asr" / ("s3_full" if name == "s3" else name) / "kana.jsonl", SRC / "refs_yodas", yodas),
        ]
        out: list[dict[str, Any]] = []
        hours = 0.0
        for manifest, inputs, asr, ref_dir, voices in sources:
            rows = {r["id"]: r for r in read(inputs)}
            old = {r["key"]: r["yomi"] for r in read(asr)}
            for m in read(manifest):
                speaker = m["speaker"].removeprefix("syn:")
                if voices is not None and speaker not in voices:
                    continue
                row = rows[m["id"]]
                assert row["ref_speaker"] == speaker, (m["id"], row["ref_speaker"], speaker)
                ref = ref_dir / f"{speaker}.wav"
                assert ref.exists(), ref
                item = {"id": m["id"], "text": row["text"], "ref_wav": str(ref), "seed": row["seed"],
                        "transcript": m["transcript"], "ref_speaker": speaker, "ref_id": row["ref_id"],
                        "old_kana": old[m["id"]], "old_duration": m["duration"]}
                if row.get("seconds") is not None:
                    item["seconds"] = row["seconds"]
                out.append(item)
                hours += m["duration"] / 3600
        with open(args.out / f"rows_{name}.jsonl", "w", encoding="utf-8") as f:
            for item in out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        voices_used = len({r["ref_speaker"] for r in out})
        print(f"{name}: {len(out)} rows, {hours:.1f} h, {voices_used} voices")
        total_rows += len(out)
        total_hours += hours
    print(f"total: {total_rows} rows, {total_hours:.1f} h")


if __name__ == "__main__":
    main()
