"""Listening page of the student's free generation next to the teacher and the release (local).

For every speaker of a p2_eval.sh set it takes a few sentences spread over the text length (the
same ones in every column) and shows the voice prompt, then each result directory's clip with its
kana-whisper reading, Kana-CER, whether EOS fired (<name>.jsonl) and the cached NISQA / UTMOSv2
scores. Clips are copied next to the page at -23 LUFS, so loudness does not steer the comparison.
JVS audio is not redistributable: keep pages with the jsut / jvs_bal sets local.

    uv run python scripts/eval/make_student_page.py --out exp/listening/p2_20k \
        --section "10 voices=exp/irodori_distill/eval/voices" --section "JSUT (JVS voices)=exp/irodori_distill/eval/jsut" \
        teacher release P2_10k P2_20k
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from make_recon_page import load_scores
from rapidfuzz.distance import Levenshtein

from pocket_tts_ja.kana import canonical_kana, reading

LABELS = {"teacher": "教師 (Irodori v4.1-Small-Yomi)", "release": "リリース (D1-CC + soup4)"}
TARGET_LUFS = -23.0


def copy_normalized(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    x, sr = sf.read(str(src), dtype="float32")
    meter = pyloudnorm.Meter(sr)
    if len(x) > sr // 2:
        loud = meter.integrated_loudness(x)
        if np.isfinite(loud):
            x = pyloudnorm.normalize.loudness(x, loud, TARGET_LUFS)
    sf.write(str(dst), np.clip(x, -1, 1), sr, subtype="PCM_16")


def load_rows(root: Path, name: str, filename: str) -> dict[str, dict]:
    path = root / filename
    if not path.exists():
        return {}
    return {row["key"]: row for row in map(json.loads, open(path, encoding="utf-8"))}


def pick(items: list[dict], n: int) -> list[dict]:
    """n sentences spread over the text length (short, middle, long ...)."""
    items = sorted(items, key=lambda it: (len(it["text"]), it["key"]))
    if len(items) <= n:
        return items
    return [items[round(q * (len(items) - 1))] for q in np.linspace(0.1, 0.9, n)]


def section(title: str, root: Path, names: list[str], audio: Path, per_speaker: int) -> str:
    items = [json.loads(line) for line in open(root / "items.jsonl", encoding="utf-8")]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_speaker[it["speaker"]].append(it)
    gender = {it["speaker"]: it["gender"] for it in items}
    scores = {name: load_scores(root, name) for name in names}
    asr = {name: load_rows(root, name, f"asr_{name}/kana.jsonl") for name in names}
    gen = {name: load_rows(root, name, f"{name}.jsonl") for name in names}
    head = "<tr><th>文</th><th>prompt</th>" + "".join(f"<th>{html.escape(LABELS.get(n, n))}</th>" for n in names) + "</tr>"
    rows = []
    for spk in sorted(by_speaker, key=lambda s: (gender[s] != "F", s)):
        chosen = pick(by_speaker[spk], per_speaker)
        prompt = Path(chosen[0]["prompt"])
        pdst = audio / f"{root.name}_prompt_{spk}.wav"
        copy_normalized(prompt, pdst)
        rows.append(f"<tr><td class='spk' colspan='{len(names) + 2}'>{html.escape(spk)} ({gender[spk]})</td></tr>")
        for i, it in enumerate(chosen):
            ref = canonical_kana(it["yomi"]) if it.get("yomi") else reading(it["text"])
            cells = [f"<td class='text'>{html.escape(it['text'])}<br><small>{html.escape(ref)}</small></td>"]
            cells.append(f'<td><audio controls preload="none" src="audio/{pdst.name}"></audio></td>' if i == 0 else "<td></td>")
            for name in names:
                src = root / name / f"{it['key']}.wav"
                if not src.exists():
                    cells.append("<td>–</td>")
                    continue
                dst = audio / f"{root.name}_{name}_{it['key']}.wav"
                copy_normalized(src, dst)
                notes = []
                hyp = asr[name].get(it["key"])
                if hyp is not None:
                    kana = canonical_kana(hyp["yomi"])
                    cer = Levenshtein.distance(ref, kana) / max(len(ref), 1)
                    notes.append(f"CER {100 * cer:.0f}%: {html.escape(kana[:60])}{'…' if len(kana) > 60 else ''}")
                row = gen[name].get(it["key"])
                if row is not None and not row["eos"]:
                    notes.append("<b>EOS なし</b>")
                if scores[name].get(it["key"]):
                    notes.append(scores[name][it["key"]])
                cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio><br><small>{"<br>".join(notes)}</small></td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<h2>{html.escape(title)}</h2><table>{head}{''.join(rows)}</table>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="result directories, in column order")
    parser.add_argument("--section", action="append", required=True, help="TITLE=p2_eval.sh set directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=3)
    parser.add_argument("--title", default="P2: 生徒(DACVAE 25 Hz)の自由生成")
    parser.add_argument("--intro", default="")
    args = parser.parse_args()

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    parts = []
    for spec in args.section:
        title, root = spec.split("=", 1)
        parts.append(section(title, Path(root), args.names, audio, args.per_speaker))
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse;margin-bottom:2em}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}td.text{{max-width:18em}}td.spk{{background:#eef;font-weight:bold}}
audio{{width:210px;height:32px}}small{{color:#666}}</style></head><body>
<h1>{html.escape(args.title)}</h1><p>{html.escape(args.intro)}</p>
{''.join(parts)}</body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{args.out / 'index.html'}: {len(args.section)} sections x {len(args.names)} columns")


if __name__ == "__main__":
    main()
