"""Listening page for scripts/m2/space_voices.py's output: one voice per block, the prompt first, then every sentence
with one column per student x decoder cell, and the reconstruction of the prompt by every decoder.

    uv run python scripts/m2/make_space_page.py exp/m2/space_voices --out exp/listening/space_voices \
        --cell nc6__ja_release --cell cc6__cc_release --cell cc6__mimi --voices female_02 female_04 female_03 --sentences 5

The voices are synthetic (Irodori-TTS voice design, the Space's presets), so the page can be shared. Audio is copied
next to the page; --mos adds UTMOSv2 (hp40) from the MOS caches under every player.
"""

import argparse
import html
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=[], help="<student>__<decoder> directory of root/tts, in column order")
    parser.add_argument("--recon", nargs="*", default=None, help="decoders of root/recon to show (default: the cells' decoders)")
    parser.add_argument("--voices", nargs="*", default=None)
    parser.add_argument("--voices-dir", type=Path, default=Path("data/space_voices"),
                        help="the prompts, shown when root/recon/orig is missing (a run with --no-recon)")
    parser.add_argument("--sentences", type=int, default=5)
    parser.add_argument("--title", default="Space voices: students x decoders")
    parser.add_argument("--note", action="append", default=[], help="NAME=TEXT: what a column is")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.root / "items.jsonl", encoding="utf-8")]
    voices = args.voices or sorted({it["voice"] for it in items})
    recon = args.recon if args.recon is not None else list(dict.fromkeys(c.split("__", 1)[1] for c in args.cell))
    recon = [d for d in recon if (args.root / "recon" / d).is_dir()]
    orig_dir = args.root / "recon" / "orig" if (args.root / "recon" / "orig").is_dir() else args.voices_dir
    mos: dict[str, dict[str, dict[str, float]]] = {}
    for c in args.cell:
        p = args.root / "tts" / f"mos_hp40_{c}.json"
        if p.exists():
            mos[c] = json.loads(p.read_text())
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)

    def player(src: Path, name: str, score: float | None = None) -> str:
        dst = audio / name
        if not dst.exists():
            shutil.copy(src, dst)
        tag = f'<audio controls preload="none" src="audio/{name}"></audio>'
        return tag + (f"<br><small>UTMOSv2 {score:.2f}</small>" if score is not None else "")

    rows = []
    for voice in voices:
        rows.append(f"<tr><td class='spk' colspan='{len(args.cell) + 1}'>{voice}</td></tr>")
        cells = [f"<td class='text'>参照音声（{voice}）</td>", f"<td>{player(orig_dir / f'{voice}.wav', f'orig_{voice}.wav')}</td>"]
        cells += ["<td></td>"] * (len(args.cell) - 1)
        rows.append("<tr>" + "".join(cells) + "</tr>")
        if recon:
            cells = ["<td class='text'>参照音声の再構成（encoder → 各 decoder）<br><small>"
                     + " / ".join(html.escape(d) for d in recon) + "</small></td>"]
            for d in recon:
                cells.append(f"<td>{html.escape(d)}<br>{player(args.root / 'recon' / d / f'{voice}.wav', f'recon_{d}_{voice}.wav')}</td>")
            cells += ["<td></td>"] * max(0, len(args.cell) - len(recon))
            rows.append("<tr>" + "".join(cells) + "</tr>")
        for it in [it for it in items if it["voice"] == voice][: args.sentences]:
            cells = [f"<td class='text'>{html.escape(it['text'])}</td>"]
            for c in args.cell:
                src = args.root / "tts" / c / f"{it['key']}.wav"
                score = mos.get(c, {}).get(it["key"], {}).get("utmosv2")
                cells.append(f"<td>{player(src, f'{c}_{it['key']}.wav', score)}</td>" if src.exists() else "<td>—</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr><th>sentence</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in args.cell) + "</tr>"
    notes = "".join(f"<li><b>{html.escape(n)}</b>: {html.escape(t)}</li>" for n, t in (x.split("=", 1) for x in args.note))
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}
td.text{{max-width:24em}}td.spk{{background:#eef;font-weight:bold}}audio{{width:230px;height:32px}}small{{color:#666}}</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p>列 = 生徒（言語モデル）__decoder。同じ生徒の列どうしは同じ潜在表現を復号している（違いは decoder だけ）。</p>
{"<ul>" + notes + "</ul>" if notes else ""}
<table>{head}{''.join(rows)}</table></body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{args.out / 'index.html'}: {len(voices)} voices x {args.sentences} sentences x {len(args.cell)} columns")


if __name__ == "__main__":
    main()
