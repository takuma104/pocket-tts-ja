"""CC-BY line (plan B-CC): synthesis inputs that give the accepted NC-voice sentences a CC BY voice.

Three quarters of the synthetic speech was cloned from reference utterances of Emilia proper
(CC BY-NC). For every set this takes the sentences that passed QC with such a voice and writes
them again as synthesis input, with the same id, text and seed but the reference of a
CC BY speaker (Emilia-YODAS, data/synth/refs_cc.jsonl from select_refs.py --only-subset
Emilia-YODAS --exclude-jsonl data/synth/refs.jsonl). Each NC voice is replaced by one fixed
CC BY voice, so a voice keeps its share of the texts.

    uv run python scripts/synth/make_cc_input.py        # on the synthesis host
    -> data/synth/<set>_cc_input.jsonl for s3, s4, s5, s6a, s6b, s7, s9
"""

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path("data/synth")


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refs", type=Path, default=ROOT / "refs.jsonl")
    parser.add_argument("--refs-cc", type=Path, default=ROOT / "refs_cc.jsonl")
    parser.add_argument("--kana-form", default=None, help="hira / kata (default: data/synth/c2_kana_form)")
    args = parser.parse_args()
    form = args.kana_form or (ROOT / "c2_kana_form").read_text().strip()

    nc = sorted(r["speaker"] for r in read(args.refs) if r["subset"] == "Emilia")
    cc = sorted(read(args.refs_cc), key=lambda r: r["speaker"])
    assert cc and all(r["subset"] == "Emilia-YODAS" for r in cc)
    voice = {speaker: cc[i % len(cc)] for i, speaker in enumerate(nc)}
    print(f"{len(nc)} NC voices -> {len(cc)} CC BY voices")

    sets = {  # name: (accepted rows, the input they were synthesized from)
        "s3": ("s3_full/manifest.jsonl", "s3_input.jsonl"),
        "s4": ("s4/manifest.jsonl", "s4_input.jsonl"),
        "s5": ("s5/manifest.jsonl", f"s5_input_{form}.jsonl"),
        "s6a": ("s6a/manifest.jsonl", f"s6a_input_{form}.jsonl"),
        "s6b": ("s6b/manifest.jsonl", f"s6b_input_{form}.jsonl"),
        "s7": ("s7/manifest.jsonl", f"s7_input_{form}.jsonl"),
        "s9": ("s9/manifest.jsonl", "s9_input.jsonl"),
    }
    for name, (manifest, source) in sets.items():
        rows = {r["id"]: r for r in read(ROOT / source)}
        out = []
        hours = 0.0
        for m in read(ROOT / manifest):
            row = rows[m["id"]]
            if row["ref_speaker"] not in voice:
                continue
            ref = voice[row["ref_speaker"]]
            out.append({**row, "ref_wav": f"data/synth/refs_cc/{ref['speaker']}.wav", "ref_speaker": ref["speaker"], "ref_id": ref["id"]})
            hours += m["duration"] / 3600
        path = ROOT / f"{name}_cc_input.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(out)} sentences ({hours:.1f} h with the NC voices)")


if __name__ == "__main__":
    main()
