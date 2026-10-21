"""Listening page of codec round trips: the same clips through each codec, side by side (local).

For every speaker of a recon.py directory it takes the clips with the largest share of voiced
frames (where the high-band harmonics live; mimi_ft's hard cases) and copies them with each
result directory's version. Under each player it prints the cached scores of that file when they
exist: NISQA-TTS naturalness / NISQA v2 coloration (nisqa_<name>.json) and UTMOSv2 after the
40 Hz high-pass (mos_hp40_<name>.json). Several recon directories can share one page (--section).
JVS audio is not redistributable: keep such pages local.

    uv run python scripts/eval/make_recon_page.py --out exp/listening/p0_codec \
        --section "JVS=exp/irodori_distill/p0/recon_jvs" --section "Space=exp/irodori_distill/p0/recon_space" \
        orig dacvae48 dacvae mimi keep4k soup4
"""

import argparse
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from hf_structure import HOP, voiced_f0

LABELS = {
    "orig": "元音声",
    "dacvae48": "DACVAE (48k)",
    "dacvae": "DACVAE (24k)",
    "mimi": "元の Mimi",
    "keep4k": "keep4k (NC)",
    "soup4": "M2CH2_soup4 (CC)",
    "base2": "M2C_base2 (CC)",
}


def load_scores(root: Path, name: str) -> dict[str, str]:
    out: dict[str, list[str]] = defaultdict(list)
    nisqa = root / f"nisqa_{name}.json"
    if nisqa.exists():
        for key, s in json.loads(nisqa.read_text()).items():
            out[key].append(f"N-TTS {s['tts']:.2f} / col {s['col']:.2f}")
    mos = root / f"mos_hp40_{name}.json"
    if mos.exists():
        for key, s in json.loads(mos.read_text()).items():
            out[key].append(f"UTMOSv2 {s['utmosv2']:.2f}")
    return {k: "<br>".join(v) for k, v in out.items()}


def section(title: str, root: Path, names: list[str], audio: Path, per_speaker: int, max_speakers: int) -> str:
    items = [json.loads(line) for line in open(root / "items.jsonl", encoding="utf-8")]
    by_speaker: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for it in items:
        x, sr = sf.read(str(root / "orig" / f"{it['key']}.wav"), dtype="float32")
        by_speaker[it["speaker"]].append((len(voiced_f0(x, sr)) * HOP / max(len(x), 1), it))
    gender = {it["speaker"]: it["gender"] for it in items}
    speakers = sorted(by_speaker, key=lambda s: (gender[s] != "F", s))[:max_speakers]
    scores = {name: load_scores(root, name) for name in names}
    head = "<tr><th>clip</th>" + "".join(f"<th>{html.escape(LABELS.get(n, n))}</th>" for n in names) + "</tr>"
    rows = []
    for spk in speakers:
        if per_speaker > 1 or len(speakers) < len(by_speaker):
            rows.append(f"<tr><td class='spk' colspan='{len(names) + 1}'>{html.escape(spk)} ({gender[spk]})</td></tr>")
        for share, it in sorted(by_speaker[spk], key=lambda p: -p[0])[:per_speaker]:
            text = html.escape(it.get("text") or it["key"])
            cells = [f"<td class='text'>{text}<br><small>{html.escape(it['key'])}, 有声 {100 * share:.0f}%</small></td>"]
            for name in names:
                src = root / name / f"{it['key']}.wav"
                dst = audio / f"{root.name}_{name}_{it['key']}.wav"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                note = scores[name].get(it["key"], "")
                cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio><br><small>{note}</small></td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<h2>{html.escape(title)}</h2><table>{head}{''.join(rows)}</table>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="result directories, in column order")
    parser.add_argument("--section", action="append", required=True, help="TITLE=recon directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=12)
    parser.add_argument("--title", default="P0b: codec round trips (same clips through each codec)")
    parser.add_argument("--intro", default="")
    args = parser.parse_args()

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    parts = []
    for spec in args.section:
        title, root = spec.split("=", 1)
        parts.append(section(title, Path(root), args.names, audio, args.per_speaker, args.max_speakers))
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse;margin-bottom:2em}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}td.text{{max-width:22em}}td.spk{{background:#eef;font-weight:bold}}
audio{{width:210px;height:32px}}small{{color:#666}}</style></head><body>
<h1>{html.escape(args.title)}</h1><p>{html.escape(args.intro)}</p>
{''.join(parts)}</body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{args.out / 'index.html'}: {len(args.section)} sections x {len(args.names)} columns")


if __name__ == "__main__":
    main()
