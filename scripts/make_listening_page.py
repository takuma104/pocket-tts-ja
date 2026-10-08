"""Build a local listening page from eval_tts.py outputs of one or more runs at a given step.

For the JVS zero-shot set it shows, per speaker: the voice prompt (real nonpara30 recording),
then for a few sentences the generated audio next to the speaker's real recording of the same
sentence. For JKYB-mini it shows a sample of sentences with the target reading and whether the
benchmark judged it correct. Audio is copied (or fetched with rsync from --remote) next to the
page; JVS audio is not redistributable, so keep the page local.

    uv run python scripts/make_listening_page.py --step 15000 runs/A5_8k runs/A5_4k \
        --remote galleria2 --out exp/listening/A5_step15000
"""

import argparse
import html
import json
import random
import shutil
import subprocess
from pathlib import Path

REMOTE_REPO = "~/co/pocket-tts-ja"
JVS = Path("data/jvs_ver1")


def local_path(p: str) -> Path:
    """Paths recorded on the eval host point into its checkout; map them into this one."""
    path = Path(p)
    if path.exists() or "pocket-tts-ja/" not in p:
        return path
    return Path(__file__).resolve().parents[1] / p.split("pocket-tts-ja/", 1)[1]


def fetch(remote: str | None, run: Path, rel: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    local = run / rel
    if local.exists():
        shutil.copyfile(local, dst)
    elif remote:
        subprocess.run(["rsync", "-a", f"{remote}:{REMOTE_REPO}/{run.as_posix()}/{rel}", str(dst)], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--remote", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jvs-sentences", type=int, default=3)
    parser.add_argument("--jkyb-sentences", type=int, default=12)
    args = parser.parse_args()
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    step_dir = f"eval/step{args.step:08d}"
    sections: list[str] = []

    # JVS: speakers/sentences from the first run's eval items
    items_path = None
    for run in args.runs:
        cand = list((run / step_dir).glob("jvs_*/items.json"))
        if not cand and args.remote:
            tag = "jvs_rotation_cfg2.0_n10"
            (run / step_dir / tag).mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{args.remote}:{REMOTE_REPO}/{run.as_posix()}/{step_dir}/{tag}/items.json",
                            str(run / step_dir / tag / "items.json")], check=False)
            cand = list((run / step_dir).glob("jvs_*/items.json"))
        if cand:
            items_path = cand[0]
            break
    if items_path is not None:
        items = json.loads(items_path.read_text())
        tag = items_path.parent.name
        by_spk: dict[str, list[dict]] = {}
        for it in items:
            by_spk.setdefault(it["key"].split("_")[0], []).append(it)
        rows = []
        for spk, its in sorted(by_spk.items()):
            gender = its[0].get("gender", "?")
            prompt_dst = audio / f"{spk}_prompt.wav"
            shutil.copyfile(local_path(its[0]["prompt"]), prompt_dst)
            rows.append(f'<tr class="spk"><th colspan="{2 + len(args.runs)}">{spk} ({gender}) — prompt '
                        f'<audio controls preload="none" src="audio/{prompt_dst.name}"></audio></th></tr>')
            for it in its[: args.jvs_sentences]:
                real = audio / f"{it['key']}_real.wav"
                shutil.copyfile(local_path(it["reference"]), real)
                cells = [f"<td class='text'>{html.escape(it['text'])}</td>",
                         f'<td><audio controls preload="none" src="audio/{real.name}"></audio></td>']
                for run in args.runs:
                    dst = audio / f"{run.name}_{it['key']}.wav"
                    fetch(args.remote, run, f"{step_dir}/{tag}/audio/{it['key']}.wav", dst)
                    cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio></td>')
                rows.append("<tr>" + "".join(cells) + "</tr>")
        head = "<tr><th>text</th><th>real recording</th>" + "".join(f"<th>{r.name}</th>" for r in args.runs) + "</tr>"
        sections.append(f"<h2>JVS zero-shot (step {args.step})</h2><table>{head}{''.join(rows)}</table>")

    # JKYB-mini sample
    rng = random.Random(0)
    keys = None
    details: dict[str, dict[str, dict]] = {}
    for run in args.runs:
        tag = "jkyb_mini_rotation_cfg2.0"
        p = run / step_dir / tag / "results" / "details" / "all.jsonl"
        if not p.exists() and args.remote:
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{args.remote}:{REMOTE_REPO}/{run.as_posix()}/{step_dir}/{tag}/results/details/all.jsonl",
                            str(p)], check=False)
        if p.exists():
            details[run.name] = {json.loads(l)["key"]: json.loads(l) for l in open(p)}
    if details:
        first = next(iter(details.values()))
        keys = sorted(first)
        rng.shuffle(keys)
        rows = []
        for key in keys[: args.jkyb_sentences]:
            r = first[key]
            cells = [f"<td class='text'>{html.escape(r['tagged_text'])}<br><small>正解: {html.escape(r['reference_target'])}</small></td>"]
            for run in args.runs:
                d = details.get(run.name, {}).get(key)
                dst = audio / f"{run.name}_{key}.wav"
                fetch(args.remote, run, f"{step_dir}/jkyb_mini_rotation_cfg2.0/audio/{key}.wav", dst)
                mark = "?" if d is None else ("○" if d["target_exact"] else f"× ({html.escape(str(d.get('mapped_target')))})")
                cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio> {mark}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        head = "<tr><th>sentence</th>" + "".join(f"<th>{r.name}</th>" for r in args.runs) + "</tr>"
        sections.append(f"<h2>JKYB-mini sample (step {args.step})</h2><table>{head}{''.join(rows)}</table>")

    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Listening step {args.step}</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}}tr.spk th{{background:#eef}}
td.text{{max-width:480px;font-size:14px}}audio{{width:240px}}</style></head><body>
<h1>Listening page — {' / '.join(r.name for r in args.runs)} @ step {args.step}</h1>
<p>Local only (JVS audio is not redistributable). Generation: temp 0.3, CFG 2.0, eos -4.</p>
{''.join(sections)}</body></html>"""
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
