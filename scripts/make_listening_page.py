"""Build a local listening page from eval_tts.py outputs of one or more runs at a given step.

For the JVS zero-shot set it shows, per speaker: the voice prompt (real nonpara30 recording),
then for a few sentences the generated audio next to the speaker's real recording of the same
sentence. For JKYB-mini it shows a sample of sentences with the target reading and whether the
benchmark judged it correct. Audio is copied (or fetched with rsync from --remote) next to the
page; JVS audio is not redistributable, so keep the page local.

    uv run python scripts/make_listening_page.py --step 15000 runs/A5_8k runs/A5_4k \
        --remote galleria2-direct --out exp/listening/A5_step15000

A run can carry its own step (`runs/B1_4k@155000`), so different stages can sit side by side.
JKYB cells show the kana-whisper reading; `--short` adds a section of short sentences (where
extra speech after the end of the text was most frequent) and `--include` adds the sentences
containing the given words.
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


class Run:
    def __init__(self, spec: str, default_step: int) -> None:
        path, _, step = spec.partition("@")
        self.path = Path(path)
        self.step = int(step) if step else default_step
        self.label = f"{self.path.name}@{self.step // 1000}k"
        self.step_dir = f"eval/step{self.step:08d}"


def fetch(remote: str | None, run: Path, rel: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    local = run / rel
    if local.exists():
        shutil.copyfile(local, dst)
    elif remote:
        subprocess.run(["rsync", "-a", f"{remote}:{REMOTE_REPO}/{run.as_posix()}/{rel}", str(dst)], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run dir, optionally RUN@STEP")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--remote", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jvs-sentences", type=int, default=3)
    parser.add_argument("--jvs-tags", nargs="+", default=["jvs_bal_rotation_cfg2.0", "jvs_rotation_cfg2.0_n10"],
                        help="JVS eval sets to use, first available wins (jvs_bal: 5 M + 5 F)")
    parser.add_argument("--jkyb-sentences", type=int, default=12)
    parser.add_argument("--short", type=int, default=0, help="number of short sentences (< 15 kana) to add")
    parser.add_argument("--include", nargs="*", default=[], help="add JKYB-mini sentences containing these")
    args = parser.parse_args()
    runs = [Run(spec, args.step) for spec in args.runs]
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    # JVS: speakers/sentences from the first run's eval items
    items_path = None
    for run in runs:
        for tag in args.jvs_tags:
            cand = run.path / run.step_dir / tag / "items.json"
            if not cand.exists() and args.remote:
                cand.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", f"{args.remote}:{REMOTE_REPO}/{run.path.as_posix()}/{run.step_dir}/{tag}/items.json",
                                str(cand)], check=False, stderr=subprocess.DEVNULL)
            if cand.exists():
                items_path = cand
                break
        if items_path is not None:
            break
    if items_path is not None:
        items = json.loads(items_path.read_text())
        tag = items_path.parent.name
        by_spk: dict[str, list[dict]] = {}
        for it in items:
            by_spk.setdefault(it["key"].split("_")[0], []).append(it)
        heard: dict[str, dict[str, str]] = {}
        for run in runs:
            kana = run.path / run.step_dir / tag / "asr_audio" / "kana.jsonl"
            if not kana.exists() and args.remote:
                kana.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", f"{args.remote}:{REMOTE_REPO}/{run.path.as_posix()}/{run.step_dir}/{tag}/asr_audio/kana.jsonl",
                                str(kana)], check=False)
            if kana.exists():
                heard[run.label] = {(r := json.loads(l))["key"]: r["yomi"] for l in open(kana)}

        def jvs_row(it: dict) -> str:
            real = audio / f"{it['key']}_real.wav"
            shutil.copyfile(local_path(it["reference"]), real)
            cells = [f"<td class='text'>{html.escape(it['text'])}<br><small>{html.escape(it.get('yomi', ''))}</small></td>",
                     f'<td><audio controls preload="none" src="audio/{real.name}"></audio></td>']
            for run in runs:
                dst = audio / f"{run.label}_{it['key']}.wav"
                fetch(args.remote, run.path, f"{run.step_dir}/{tag}/audio/{it['key']}.wav", dst)
                h = html.escape(heard.get(run.label, {}).get(it["key"], ""))
                cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio><br><small>{h}</small></td>')
            return "<tr>" + "".join(cells) + "</tr>"

        def prompt_row(spk: str, its: list[dict]) -> str:
            prompt_dst = audio / f"{spk}_prompt.wav"
            shutil.copyfile(local_path(its[0]["prompt"]), prompt_dst)
            return (f'<tr class="spk"><th colspan="{2 + len(runs)}">{spk} ({its[0].get("gender", "?")}) — prompt '
                    f'<audio controls preload="none" src="audio/{prompt_dst.name}"></audio></th></tr>')

        head = "<tr><th>text</th><th>real recording</th>" + "".join(f"<th>{r.label}</th>" for r in runs) + "</tr>"
        rows = []
        for spk, its in sorted(by_spk.items()):
            rows.append(prompt_row(spk, its))
            rows += [jvs_row(it) for it in its[: args.jvs_sentences]]
        sections.append(f"<h2>JVS zero-shot</h2><table>{head}{''.join(rows)}</table>")
        if args.include:
            rows = []
            for spk, its in sorted(by_spk.items()):
                sel = [it for it in its if any(w in it["text"] for w in args.include)]
                if sel:
                    rows.append(prompt_row(spk, its))
                    rows += [jvs_row(it) for it in sel]
            if rows:
                sections.append(f"<h2>JVS: sentences with {' / '.join(map(html.escape, args.include))}</h2>"
                                f"<table>{head}{''.join(rows)}</table>")

    # JKYB-mini: random sample, short sentences, requested words
    tag = "jkyb_mini_rotation_cfg2.0"
    details: dict[str, dict[str, dict]] = {}
    for run in runs:
        p = run.path / run.step_dir / tag / "results" / "details" / "all.jsonl"
        if not p.exists() and args.remote:
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{args.remote}:{REMOTE_REPO}/{run.path.as_posix()}/{run.step_dir}/{tag}/results/details/all.jsonl",
                            str(p)], check=False)
        if p.exists():
            details[run.label] = {json.loads(l)["key"]: json.loads(l) for l in open(p)}
    if details:
        first = next(iter(details.values()))
        keys = sorted(first)
        random.Random(0).shuffle(keys)
        short = [k for k in keys if len(first[k]["reference_yomi_normalized"]) < 15]
        wanted = [k for k in sorted(first) if any(w in first[k]["text"] for w in args.include)]

        def table(title: str, sel: list[str]) -> str:
            rows = []
            for key in sel:
                r = first[key]
                cells = [f"<td class='text'>{html.escape(r['tagged_text'])}<br><small>正解: {html.escape(r['reference_target'])}"
                         f" / {html.escape(r['reference_yomi'])}</small></td>"]
                for run in runs:
                    d = details.get(run.label, {}).get(key)
                    dst = audio / f"{run.label}_{key}.wav"
                    fetch(args.remote, run.path, f"{run.step_dir}/{tag}/audio/{key}.wav", dst)
                    mark = "?" if d is None else ("○" if d["target_exact"] else f"× ({html.escape(str(d.get('mapped_target')))})")
                    heard = "" if d is None else html.escape(str(d.get("prediction_yomi") or ""))
                    cells.append(f'<td><audio controls preload="none" src="audio/{dst.name}"></audio> {mark}'
                                 f"<br><small>{heard}</small></td>")
                rows.append("<tr>" + "".join(cells) + "</tr>")
            head = "<tr><th>sentence</th>" + "".join(f"<th>{r.label}</th>" for r in runs) + "</tr>"
            return f"<h2>{title}</h2><table>{head}{''.join(rows)}</table>"

        sections.append(table("JKYB-mini sample", keys[: args.jkyb_sentences]))
        if args.short:
            sections.append(table("Short sentences (&lt; 15 kana): extra speech after the text?", short[: args.short]))
        if wanted:
            sections.append(table("JKYB-mini: sentences with " + " / ".join(map(html.escape, args.include)), wanted))

    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Listening page</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}}tr.spk th{{background:#eef}}
td.text{{max-width:480px;font-size:14px}}audio{{width:240px}}</style></head><body>
<h1>Listening page — {' / '.join(r.label for r in runs)}</h1>
<p>Local only (JVS audio is not redistributable). Generation: temp 0.3, CFG 2.0, eos -4.
Small text under each JKYB sample: what kana-whisper heard.</p>
{''.join(sections)}</body></html>"""
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
