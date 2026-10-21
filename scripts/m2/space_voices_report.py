"""Per-voice tables for scripts/m2/space_voices.py's output: which voice breaks, under which student and decoder.

Reads the directories of <out>/tts (<student>__<decoder>) and <out>/recon (<decoder>), the MOS caches of
scripts/m2/mos_eval.py (mos_hp40_<name>.json next to them), the kana transcripts of scripts/asr_transcribe.py
(asr_<name>/kana.jsonl), and <out>/stats.jsonl; computes and caches (sim_<name>.json, spec_<name>.json) the ECAPA
similarity of every clip to its voice prompt (recon: to the prompt as recorded) and spectral measures of its
voiced frames (F0 from WORLD, refined on the comb of 1-2.5 kHz as in hf_structure.py --own-f0):

    comb46, comb68   harmonic contrast in 4-6 / 6-8 kHz at the clip's own F0, dB (noise 0)
    lvl46, lvl810    power of 4-6 / 8-10 kHz relative to 1-2.5 kHz, dB
    cer              Kana-CER of the kana-whisper transcript against the JSUT kana (tts only)
    sim              ECAPA cosine similarity to the voice prompt
    utmosv2, utmos22 from the MOS caches (after the 40 Hz high-pass)
    clipped, peak    share of samples at full scale before the 16-bit write, and the peak

    uv run python scripts/m2/space_voices_report.py exp/m2/space_voices --metrics utmosv2 cer sim comb46 lvl46

prints, per metric, one table per student (rows: voices, columns: decoders) and the reconstruction table.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_structure import N_FFT, PAD, band_stats, refine_f0, spectrum  # noqa: E402

from pocket_tts_ja.kana import canonical_kana, cer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS = ("utmosv2", "utmos22", "cer", "sim", "comb46", "comb68", "lvl46", "lvl810", "clipped", "peak", "frames", "no_eos")


def voiced_frames(x: npt.NDArray[np.float64], sr: int) -> list[tuple[int, float]]:
    """(frame start, F0) for the 85 ms frames WORLD finds voiced over at least 80% of their 5 ms steps."""
    import pyworld as pw

    f0, t = pw.dio(x, sr, f0_floor=60, f0_ceil=650, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    hop = int(0.005 * sr)
    out = []
    for start in range(0, len(x) - N_FFT, N_FFT // 2):
        seg = f0[start // hop : (start + N_FFT) // hop]
        if len(seg) == 0 or np.mean(seg > 0) < 0.8:
            continue
        out.append((start, float(np.median(seg[seg > 0]))))
    return out


def spectral(path: Path) -> dict[str, float]:
    x, sr = sf.read(str(path), dtype="float64")
    step = sr / (N_FFT * PAD)
    combs, powers = [], []
    for start, f0 in voiced_frames(x, sr):
        spec = spectrum(x[start : start + N_FFT].astype(np.float32))
        f0_own = refine_f0(np.log(spec + 1e-9), f0, step, span=0.04, lo=1000, hi=2500)
        stats = band_stats(spec, f0_own, step)
        combs.append([s[0] for s in stats])
        powers.append([s[1] for s in stats])
    if not combs:
        return {"voiced": 0}
    c = np.mean(combs, axis=0)
    p = np.mean(powers, axis=0)
    lvl = 10 * np.log10(p / p[0])
    return {"voiced": len(combs), "comb46": float(c[2]), "comb68": float(c[3]), "lvl46": float(lvl[2]), "lvl810": float(lvl[4])}


def cached(path: Path, keys: list[str], compute) -> dict[str, dict[str, float]]:  # noqa: ANN001
    data = json.loads(path.read_text()) if path.exists() else {}
    missing = [k for k in keys if k not in data]
    if missing:
        data.update(compute(missing))
        path.write_text(json.dumps(data))
    return data


class Similarity:
    def __init__(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.enc = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
            savedir=str(REPO / "data/models/spkrec-ecapa-voxceleb"),
        )
        self.cache: dict[str, object] = {}

    def emb(self, path: Path):  # noqa: ANN201
        import torchaudio.functional as AF

        key = str(path)
        if key not in self.cache:
            wav, sr = sf.read(key, dtype="float32")
            x = AF.resample(self.torch.from_numpy(wav)[None].cuda(), sr, 16000)
            with self.torch.no_grad():
                self.cache[key] = self.torch.nn.functional.normalize(self.enc.encode_batch(x)[0, 0], dim=-1)
        return self.cache[key]

    def __call__(self, a: Path, b: Path) -> float:
        return float(self.emb(a) @ self.emb(b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--voices", type=Path, default=Path("data/space_voices"))
    parser.add_argument("--metrics", nargs="+", default=["utmosv2", "utmos22", "cer", "sim", "comb46", "lvl46", "lvl810", "clipped"])
    parser.add_argument("--students", nargs="*", default=None)
    parser.add_argument("--decoders", nargs="*", default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write every per-clip measure here")
    args = parser.parse_args()

    items = {it["key"]: it for it in map(json.loads, open(args.out / "items.jsonl", encoding="utf-8"))}
    voices = sorted({it["voice"] for it in items.values()})
    tts_dirs = sorted(p.name for p in (args.out / "tts").iterdir() if p.is_dir() and "__" in p.name)
    cells = [tuple(d.split("__", 1)) for d in tts_dirs]
    students = [s for s in dict.fromkeys(c[0] for c in cells) if not args.students or s in args.students]
    decoders = [d for d in dict.fromkeys(c[1] for c in cells) if not args.decoders or d in args.decoders]
    recon_dirs = [p.name for p in (args.out / "recon").iterdir() if p.is_dir() and p.name != "orig"] if (args.out / "recon").is_dir() else []
    if recon_dirs:
        recon_dirs = ["orig"] + [d for d in decoders if d in recon_dirs] + [d for d in recon_dirs if d not in decoders]

    stats: dict[tuple[str, str], dict[str, float]] = {}
    for row in map(json.loads, open(args.out / "stats.jsonl", encoding="utf-8")):
        stats[row["dir"], row["key"]] = row
    sim = None
    per_clip: dict[str, dict[str, dict[str, float]]] = {}  # dir -> key -> metric -> value

    def gather(dirname: str, keys: list[str], prompt_of) -> None:  # noqa: ANN001
        nonlocal sim
        root = args.out / dirname
        name = root.name
        clips: dict[str, dict[str, float]] = {k: {} for k in keys}
        mos_path = root.parent / f"mos_hp40_{name}.json"
        if mos_path.exists():
            for k, v in json.loads(mos_path.read_text()).items():
                if k in clips:
                    clips[k].update(v)
        kana_path = root.parent / f"asr_{name}" / "kana.jsonl"
        if kana_path.exists():
            for row in map(json.loads, open(kana_path, encoding="utf-8")):
                k = row["key"]
                if k in clips and items.get(k, {}).get("yomi"):
                    clips[k]["cer"] = cer(canonical_kana(items[k]["yomi"]), canonical_kana(row["yomi"]))
        needs_sim = [k for k in keys if not (root.parent / f"sim_{name}.json").exists()
                     or k not in json.loads((root.parent / f"sim_{name}.json").read_text())]
        if needs_sim:
            sim = sim or Similarity()
        sims = cached(root.parent / f"sim_{name}.json", keys,
                      lambda ks: {k: {"sim": sim(root / f"{k}.wav", prompt_of(k))} for k in ks})
        specs = cached(root.parent / f"spec_{name}.json", keys, lambda ks: {k: spectral(root / f"{k}.wav") for k in ks})
        for k in keys:
            clips[k].update(sims[k])
            clips[k].update(specs[k])
            st = stats.get((dirname, k))
            if st:
                clips[k].update({m: float(st[m]) for m in ("clipped", "peak", "frames", "no_eos") if m in st})
        per_clip[dirname] = clips

    for student in students:
        for dec in decoders:
            d = f"tts/{student}__{dec}"
            if (args.out / d).is_dir():
                print(f"gathering {d}", file=sys.stderr, flush=True)
                gather(d, sorted(items), lambda k: args.voices / f"{items[k]['voice']}.wav")
    for dec in recon_dirs:
        print(f"gathering recon/{dec}", file=sys.stderr, flush=True)
        gather(f"recon/{dec}", voices, lambda k: args.out / "recon" / "orig" / f"{k}.wav")
    if args.json:
        args.json.write_text(json.dumps(per_clip))

    def fmt(metric: str, vals: list[float]) -> str:
        if not vals:
            return "—"
        v = float(np.mean(vals))
        if metric == "cer":
            return f"{100 * v:.1f}%"
        if metric == "clipped":
            return f"{100 * v:.2f}%"
        if metric in ("comb46", "comb68", "lvl46", "lvl810"):
            return f"{v:+.1f}" if metric.startswith("lvl") else f"{v:.1f}"
        if metric in ("frames", "no_eos"):
            return f"{v:.2f}"
        return f"{v:.3f}"

    def table(title: str, dirs: list[tuple[str, str]], metric: str, rows: list[str], by_voice: bool) -> None:
        have = [(label, d) for label, d in dirs if d in per_clip and any(metric in c for c in per_clip[d].values())]
        if not have:
            return
        print(f"\n**{title}: {metric}**\n\n| voice | " + " | ".join(label for label, _ in have) + " |")
        print("|---|" + "---:|" * len(have))
        for r in rows + ["F mean", "M mean", "all"]:
            cells_out = []
            for _, d in have:
                if r == "all":
                    sel = [c[metric] for c in per_clip[d].values() if metric in c]
                elif r.endswith(" mean"):
                    g = r[0]
                    sel = [c[metric] for k, c in per_clip[d].items() if metric in c and (items.get(k, {}).get("gender") or ("F" if k.startswith("female") else "M")) == g]
                else:
                    sel = [c[metric] for k, c in per_clip[d].items() if metric in c and (items[k]["voice"] if k in items else k) == r]
                cells_out.append(fmt(metric, sel))
            print(f"| {r} | " + " | ".join(cells_out) + " |")

    for metric in args.metrics:
        for student in students:
            table(f"TTS {student}", [(dec, f"tts/{student}__{dec}") for dec in decoders], metric, voices, True)
        table("Reconstruction of the prompts", [(dec, f"recon/{dec}") for dec in recon_dirs], metric, voices, True)


if __name__ == "__main__":
    main()
