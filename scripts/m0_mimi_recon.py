"""M0: round-trip Japanese speech through the frozen Pocket TTS Mimi.

Writes, under --out:
  jvs/items.jsonl            one row per JVS utterance (key, speaker, subset, text, yomi, ...)
  jvs/orig/<key>.wav         symlinks to the original 24 kHz recordings
  jvs/recon/<key>.wav        Mimi reconstructions (24 kHz)
  jkyb/recon/<key>.wav       Mimi reconstructions of an existing JKYB-Parakeet audio set
  listening/index.html       side-by-side originals and reconstructions (5 male + 5 female)

The JKYB set measures how much the codec alone moves the benchmark: score jkyb/recon with
`jkyb-eval tts` and compare with the score of the source audio.
"""

import argparse
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pocket_tts_ja.jvs import JvsItem, list_items, read_genders
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config, reconstruct, to_mono_float

LISTENING_UTTS = (
    ("parallel100", "VOICEACTRESS100_001"),
    ("parallel100", "VOICEACTRESS100_010"),
    ("nonpara30", None),  # the speaker's first nonpara30 utterance
    ("whisper10", None),
    ("falset10", None),
)


def reconstruct_file(src: Path, dst: Path, mimi: torch.nn.Module, device: torch.device) -> float:
    wav, sr = sf.read(str(src), dtype="float32")
    wav = to_mono_float(wav, sr, mimi.sample_rate)
    out = reconstruct(mimi, wav, device)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.clip(out, -1.0, 1.0), mimi.sample_rate, subtype="PCM_16")
    return len(wav) / mimi.sample_rate


def run_jvs(root: Path, out: Path, mimi: torch.nn.Module, device: torch.device) -> list[JvsItem]:
    items = list_items(root)
    (out / "orig").mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    total = 0.0
    start = time.time()
    for item in tqdm(items, desc="jvs"):
        link = out / "orig" / f"{item.key}.wav"
        if not link.exists():
            link.symlink_to(Path(item.wav).resolve())
        dst = out / "recon" / f"{item.key}.wav"
        if dst.exists():
            continue
        total += reconstruct_file(Path(item.wav), dst, mimi, device)
    print(f"jvs: {len(items)} items, {total / 3600:.2f} h encoded in {time.time() - start:.0f}s")
    return items


def run_jkyb(src_dir: Path, out: Path, mimi: torch.nn.Module, device: torch.device):
    wavs = sorted(src_dir.glob("*.wav"))
    for wav in tqdm(wavs, desc="jkyb"):
        dst = out / "recon" / wav.name
        if not dst.exists():
            reconstruct_file(wav, dst, mimi, device)
    (out / "source.txt").write_text(f"{src_dir}\n")
    print(f"jkyb: {len(wavs)} files")


def pick_listening_speakers(root: Path, per_gender: int) -> list[str]:
    """Speakers spread over the F0 range of each gender (low to high max-F0)."""
    genders = read_genders(root)
    picked: list[str] = []
    for g in ("M", "F"):
        spks = sorted((s for s, v in genders.items() if v[0] == g), key=lambda s: genders[s][2])
        idx = np.linspace(0, len(spks) - 1, per_gender).round().astype(int)
        picked.extend(spks[i] for i in idx)
    return picked


def build_listening(root: Path, items: list[JvsItem], out: Path, jvs_out: Path, per_gender: int):
    genders = read_genders(root)
    by_key = {i.key: i for i in items}
    first_of = {}
    for i in items:
        first_of.setdefault((i.speaker, i.subset), i)
    rows: list[str] = []
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for spk in pick_listening_speakers(root, per_gender):
        g, lo, hi = genders[spk]
        rows.append(
            f'<tr class="spk"><th colspan="4">{spk} ({"male" if g == "M" else "female"}, '
            f"F0 {lo}-{hi} Hz)</th></tr>"
        )
        for subset, utt in LISTENING_UTTS:
            item = (
                by_key.get(f"{spk}_{subset}_{utt}") if utt else first_of.get((spk, subset))
            )
            if item is None:
                continue
            cells = []
            for kind in ("orig", "recon"):
                name = f"{item.key}_{kind}.wav"
                src = Path(item.wav) if kind == "orig" else jvs_out / "recon" / f"{item.key}.wav"
                shutil.copyfile(src, audio_dir / name)
                cells.append(f'<td><audio controls preload="none" src="audio/{name}"></audio></td>')
            rows.append(
                f"<tr><td>{subset}</td><td class='text'>{html.escape(item.text)}</td>"
                + "".join(cells)
                + "</tr>"
            )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Mimi reconstruction (JVS)</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: middle; }}
tr.spk th {{ background: #eef; padding-top: 12px; }}
td.text {{ max-width: 520px; font-size: 14px; }}
audio {{ width: 260px; }}
</style></head><body>
<h1>Mimi reconstruction listening test (JVS, 5 male + 5 female)</h1>
<p>Left: original 24 kHz recording. Right: Pocket TTS Mimi encode→decode (32-dim continuous
latent at 12.5 Hz, the representation the FlowLM models). JVS audio is not redistributable;
keep this page local.</p>
<table><tr><th>subset</th><th>text</th><th>original</th><th>Mimi reconstruction</th></tr>
{chr(10).join(rows)}
</table></body></html>
"""
    (out / "index.html").write_text(page)
    print(f"listening page: {out / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--jkyb-audio", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("exp/m0_mimi_recon"))
    parser.add_argument("--per-gender", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mimi = load_mimi(load_pocket_config(), device)
    items = run_jvs(args.jvs_root, args.out / "jvs", mimi, device)
    build_listening(args.jvs_root, items, args.out / "listening", args.out / "jvs", args.per_gender)
    if args.jkyb_audio is not None:
        run_jkyb(args.jkyb_audio, args.out / "jkyb", mimi, device)


if __name__ == "__main__":
    main()
