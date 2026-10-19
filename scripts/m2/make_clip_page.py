"""M2 CC data: a listening page of YODAS3 clips from select_yodas3.py's sets, with each clip's measures.

For each set file (a select_yodas3.py output by name, any list of {"id"} rows by its .jsonl path, or "dropped" for the
clips the pool's limits removed), --per-set clips
drawn at random (seeded) with their F0, high-band comb, band-top levels, DNSMOS and, when cached, UTMOSv2 and UTMOS22. To check
by ear what the numbers let through: music, reverberation, noise-suppression artefacts, synthetic voices. YODAS3 is
CC BY 3.0, but the page stays local like the others.

    uv run python scripts/m2/make_clip_page.py data/m2/yodas3 hf_f hf_m all_f dropped --per-set 12 \
        --out exp/listening/y3_sets
"""

import argparse
import html
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path)
    parser.add_argument("sets", nargs="+")
    parser.add_argument("--per-set", type=int, default=12)
    parser.add_argument("--mos-root", type=Path, default=Path("exp/m2/y3mos"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="M2 CC: YODAS3 clips of the training sets")
    args = parser.parse_args()
    manifest = {r["id"]: r for r in map(json.loads, open(args.dir / "manifest.jsonl"))}
    scores = {r["id"]: r for r in map(json.loads, open(args.dir / "scores.jsonl"))}
    mos: dict[str, dict[str, float]] = {}
    for cache in sorted(args.mos_root.glob("mos_hp40_*.json")):
        mos.update(json.loads(cache.read_text()))
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    body = [f"<h1>{html.escape(args.title)}</h1>",
            "<p>comb: 倍音の位置と倍音の間のレベル差 (dB、雑音なら 0)。top: 長時間スペクトルの 8–11kHz (1–4kHz 比) と "
            "11–12kHz (8–11kHz 比)。UTMOSv2 / UTMOS22 は 40Hz high-pass 後。</p>"]
    for name in args.sets:
        if name == "dropped":
            chosen = {json.loads(line)["id"] for path in args.dir.glob("sets/all_*.jsonl") for line in open(path)}
            ids = sorted(k for k in scores if k not in chosen)
        else:
            path = Path(name) if name.endswith(".jsonl") else args.dir / "sets" / f"{name}.jsonl"
            ids = [json.loads(line)["id"] for line in open(path)]
            name = path.stem
        picked = rng.sample(ids, min(args.per_set, len(ids)))
        body.append(f"<h2>{html.escape(name)} ({len(ids)} clips, {len(picked)} at random)</h2><table>")
        body.append("<tr><th>clip</th><th>F0</th><th>comb 4–6k / 6–8k</th><th>top</th><th>DNSMOS</th>"
                    "<th>UTMOSv2 / 22</th><th>clipped</th><th></th></tr>")
        for clip_id in picked:
            s, m = scores[clip_id], manifest[clip_id]
            shutil.copy(args.dir / "audio" / f"{clip_id}.wav", audio / f"{clip_id}.wav")
            comb = f"{s['comb'][2]:.1f} / {s['comb'][3]:.1f}" if "comb" in s else "-"
            utmos = f"{mos[clip_id]['utmosv2']:.2f} / {mos[clip_id]['utmos22']:.2f}" if clip_id in mos else "-"
            body.append(
                f"<tr><td>{html.escape(clip_id)}</td><td>{s.get('f0', 0):.0f}</td><td>{comb}</td>"
                f"<td>{s['top'][0]:.0f} / {s['top'][1] - s['top'][0]:.0f}</td><td>{m['dnsmos_ovrl']:.2f}</td>"
                f"<td>{utmos}</td><td>{s['clipped'] * 1e4:.1f}e-4</td>"
                f"<td><audio controls preload=none src=\"audio/{html.escape(clip_id)}.wav\"></audio></td></tr>")
        body.append("</table>")
    page = ("<!doctype html><html lang=ja><head><meta charset=utf-8><title>YODAS3 sets</title><style>"
            "body{font-family:sans-serif;margin:16px}td,th{padding:2px 8px;font-size:13px;text-align:left}"
            "</style></head><body>" + "\n".join(body) + "</body></html>")
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
