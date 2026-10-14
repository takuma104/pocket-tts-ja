"""M2 listening page of the decoders alone: JVS recordings through each decoder, side by side.

The TTS pages mix the decoder with what the language model generated. Here every column decodes
the same latents (the recording through the Mimi encoder), so a difference between two columns
is the decoder. For every speaker of scripts/m2/recon_jvs.py's output it takes the sentences with
the largest share of unvoiced-consonant frames (where the periodic noise was heard) and fetches
the clips of the given result directories from --remote. JVS audio is not redistributable: keep
the page local.

    uv run python scripts/m2/make_recon_page.py exp/m2/recon orig mimi gan60k ganspec60k gs2_80k \
        --remote galleria2-direct --out exp/listening/m2_recon
"""

import argparse
import html
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from hf_structure import HOP, voiced_f0
from periodicity import unvoiced_consonants

REMOTE_REPO = "~/co/pocket-tts-ja"
LABELS = {"orig": "録音", "mimi": "元の Mimi"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon", type=Path, help="directory written by recon_jvs.py (on --remote if given)")
    parser.add_argument("names", nargs="+", help="result directories, in column order")
    parser.add_argument("--remote", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sentences", type=int, default=3, help="per speaker")
    parser.add_argument("--title", default="M2: decoders on the same latents (JVS reconstruction)")
    parser.add_argument("--rank", choices=("unvoiced", "voiced"), default="unvoiced",
                        help="pick the sentences with the largest share of unvoiced-consonant frames (the periodic "
                             "noise) or of voiced frames (the harmonics of the high band)")
    args = parser.parse_args()

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    src = f"{args.remote}:{REMOTE_REPO}/{args.recon.as_posix()}" if args.remote else str(args.recon)
    subprocess.run(["rsync", "-a", f"{src}/items.jsonl", str(args.out / "items.jsonl")], check=True)
    items = [json.loads(line) for line in open(args.out / "items.jsonl", encoding="utf-8")]

    # the recordings rank the sentences: fetch them all once (they are the first column anyway)
    orig = args.out / "orig_all"
    subprocess.run(["rsync", "-a", f"{src}/orig/", f"{orig}/"], check=True)
    by_speaker: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for it in items:
        x, _ = sf.read(str(orig / f"{it['key']}.wav"), dtype="float32")
        if args.rank == "voiced":
            share = len(voiced_f0(x, 24000)) * HOP / max(len(x), 1)
        else:
            share = float(unvoiced_consonants(x).mean())
        by_speaker[it["speaker"]].append((share, it))

    gender = {it["speaker"]: it["gender"] for it in items}
    speakers = sorted(by_speaker, key=lambda s: (gender[s] != "F", s))  # women first: the noise was worst there
    order = [s for pair in zip(speakers[: len(speakers) // 2], speakers[len(speakers) // 2 :]) for s in pair]
    order += [s for s in speakers if s not in order]
    head = "<tr><th>sentence</th>" + "".join(f"<th>{html.escape(LABELS.get(n, n))}</th>" for n in args.names) + "</tr>"
    rows = []
    for spk in order:
        rows.append(f"<tr><td class='spk' colspan='{len(args.names) + 1}'>{spk} ({gender[spk]})</td></tr>")
        for share, it in sorted(by_speaker[spk], key=lambda p: -p[0])[: args.sentences]:
            kind = "有声" if args.rank == "voiced" else "無声子音"
            cells = [f"<td class='text'>{html.escape(it['text'])}<br><small>{kind}のフレーム {100 * share:.0f}%</small></td>"]
            for name in args.names:
                dst = audio / f"{name}_{it['key']}.wav"
                if not dst.exists():
                    subprocess.run(["rsync", "-a", f"{src}/{name}/{it['key']}.wav", str(dst)], check=True)
                cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio></td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}
td.text{{max-width:26em}}td.spk{{background:#eef;font-weight:bold}}audio{{width:230px;height:32px}}small{{color:#666}}</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p>どの列も同じ潜在表現（録音を Mimi の encoder に通したもの）を復号している。列の違い = decoder の違い。</p>
<table>{head}{''.join(rows)}</table></body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{args.out / 'index.html'}: {len(order)} speakers x {args.sentences} sentences x {len(args.names)} columns")


if __name__ == "__main__":
    main()
