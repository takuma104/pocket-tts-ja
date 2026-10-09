"""Two listening complaints, measured on generated eval audio.

tail: energy in the frames emitted after EOS fired. eval_tts.py (like the official
      `generate_audio`) keeps `frames_after_eos` frames from the EOS step on, so the last F
      frames of every clip that reached EOS are post-EOS frames the model never trained on.
dyn:  within-utterance loudness movement: 1 s RMS windows (hop 0.25 s) over the speech span,
      reported as the p90-p10 range in dB, paired with the speaker's real JVS recording of the
      same sentence when an eval items.json is given.
onset: level of each 0.4 s bin from the speech onset relative to the rest of the utterance
      (voiced frames only), for generated clips, the real JVS takes and e.g. their Mimi
      reconstructions, all paired by eval item key.

    uv run python scripts/analyze_audio_dynamics.py tail A5_4k=runs/A5_4k/eval/step00015000/jkyb_mini_rotation_cfg2.0
    uv run python scripts/analyze_audio_dynamics.py dyn --items <jvs eval dir>/items.json B1=<jvs eval dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

SR = 24000
FRAME = 1920  # one Mimi frame (80 ms)


def frame_db(wav: npt.NDArray[np.float32], size: int, hop: int) -> npt.NDArray[np.float64]:
    if len(wav) < size:
        return np.zeros(0)
    n = 1 + (len(wav) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10((wav[idx].astype(np.float64) ** 2).mean(axis=1) + 1e-12)


def local_path(p: str) -> Path:
    """Eval items record paths on the eval host; map them into this checkout."""
    path = Path(p)
    if path.exists() or "pocket-tts-ja/" not in p:
        return path
    return Path(__file__).resolve().parents[1] / p.split("pocket-tts-ja/", 1)[1]


def load(path: Path) -> npt.NDArray[np.float32]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import soxr

        wav = soxr.resample(wav, sr, SR).astype(np.float32)
    return wav


def tail_stats(wav: npt.NDArray[np.float32], frames_after_eos: int) -> dict[str, float] | None:
    db = frame_db(wav, FRAME, FRAME)
    if len(db) <= frames_after_eos + 2:
        return None
    peak = float(db[:-frames_after_eos].max())
    post = db[-frames_after_eos:]
    return {"post_max": float(post.max()) - peak, "pre_last": float(db[-frames_after_eos - 1]) - peak,
            "post_mean": float(post.mean()) - peak}


def loudness_range(wav: npt.NDArray[np.float32]) -> float | None:
    """p90-p10 of 1 s window loudness over windows inside the speech span."""
    fdb = frame_db(wav, FRAME, FRAME)
    if len(fdb) == 0:
        return None
    voiced = np.nonzero(fdb >= fdb.max() - 35)[0]
    start, end = voiced[0] * FRAME, (voiced[-1] + 1) * FRAME
    win = frame_db(wav[start:end], SR, SR // 4)
    if len(win) < 3:
        return None
    return float(np.percentile(win, 90) - np.percentile(win, 10))


ONSET_BINS = (0.0, 0.4, 0.8, 1.2, 1.6)


def onset_profile(wav: npt.NDArray[np.float32]) -> list[float] | None:
    """dB of voiced frames in each onset bin minus dB of voiced frames after the last bin."""
    fdb = frame_db(wav, FRAME, FRAME)
    if len(fdb) == 0:
        return None
    peak = fdb.max()
    above = np.nonzero(fdb >= peak - 35)[0]
    start, end = above[0], above[-1] + 1
    sec = FRAME / SR
    edges = [start + int(round(t / sec)) for t in ONSET_BINS]
    if edges[-1] + 5 >= end:
        return None
    power = 10 ** (fdb / 10)
    voiced = fdb >= peak - 40

    def level(a: int, b: int) -> float | None:
        m = voiced[a:b]
        return float(10 * np.log10(power[a:b][m].mean())) if m.any() else None

    body = level(edges[-1], end)
    bins = [level(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    if body is None or any(b is None for b in bins):
        return None
    return [b - body for b in bins]  # type: ignore[operator]


def cmd_onset(args: argparse.Namespace) -> None:
    if not args.items:  # unpaired: every wav of each directory
        print("| set | clips | " + " | ".join(f"{a:.1f}-{b:.1f}s" for a, b in zip(ONSET_BINS[:-1], ONSET_BINS[1:], strict=True)) + " |")
        print("|---|---:|" + "---:|" * (len(ONSET_BINS) - 1))
        for spec in args.sets:
            label, d = spec.split("=", 1)
            profs = [p for w in sorted(Path(d).glob("*.wav"))[: args.limit] if (p := onset_profile(load(w))) is not None]
            med = np.median(np.array(profs), axis=0)
            print(f"| {label} | {len(profs)} | " + " | ".join(f"{m:+.1f}" for m in med) + " |")
        return
    items = json.loads(Path(args.items).read_text())
    sets: list[tuple[str, Path]] = [("real", Path("-"))] + [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1]))
                                                         for spec in args.sets]
    profiles: dict[str, dict[str, list[float]]] = {label: {} for label, _ in sets}
    for it in items:
        if args.match and args.match not in it["key"]:
            continue
        for label, root in sets:
            if label == "real":
                path = local_path(it["reference"])
            else:
                path = root / "audio" / f"{it['key']}.wav"
                if not path.exists():
                    path = root / f"{it['key']}.wav"
            if path.exists():
                prof = onset_profile(load(path))
                if prof is not None:
                    profiles[label][it["key"]] = prof
    bins = [f"{a:.1f}-{b:.1f}s" for a, b in zip(ONSET_BINS[:-1], ONSET_BINS[1:], strict=True)]
    print("dB relative to the rest of the utterance (median over items); paired: minus the real take\n")
    print("| set | items | " + " | ".join(bins) + " | " + " | ".join(f"paired {b}" for b in bins) + " |")
    print("|---|---:|" + "---:|" * (2 * len(bins)))
    real = profiles["real"]
    for label, _ in sets:
        prof = profiles[label]
        keys = sorted(prof)
        if not keys:
            continue
        med = np.median(np.array([prof[k] for k in keys]), axis=0)
        common = [k for k in keys if k in real]
        if label != "real" and common:
            diff = np.median(np.array([np.subtract(prof[k], real[k]) for k in common]), axis=0)
            paired = " | ".join(f"{d:+.1f}" for d in diff)
        else:
            paired = " | ".join("–" for _ in bins)
        print(f"| {label} | {len(keys)} | " + " | ".join(f"{m:+.1f}" for m in med) + f" | {paired} |")


def cmd_tail(args: argparse.Namespace) -> None:
    print(f"post-EOS frames = last {args.frames_after_eos}; dB relative to the clip's pre-EOS peak frame\n")
    print("| set | clips | post max (median) | post max > -25 dB | post max > -15 dB | last pre-EOS frame (median) |")
    print("|---|---:|---:|---:|---:|---:|")
    for spec in args.sets:
        label, d = spec.split("=", 1)
        root = Path(d)
        gen = root / "generation.jsonl" if (root / "generation.jsonl").exists() else root.parent / "generation.jsonl"
        no_eos = set()
        if gen.exists():
            no_eos = {r["key"] for r in map(json.loads, open(gen)) if r.get("no_eos")}
        rows = []
        for p in sorted((root / "audio").glob("*.wav")):
            if p.stem in no_eos:
                continue
            s = tail_stats(load(p), args.frames_after_eos)
            if s is not None:
                rows.append(s)
        pm = np.array([r["post_max"] for r in rows])
        pl = np.array([r["pre_last"] for r in rows])
        print(f"| {label} | {len(rows)} | {np.median(pm):.1f} | {100 * (pm > -25).mean():.1f}% | "
              f"{100 * (pm > -15).mean():.1f}% | {np.median(pl):.1f} |")


def cmd_dyn(args: argparse.Namespace) -> None:
    pairs: dict[str, Path] = {}
    if args.items:
        items = json.loads(Path(args.items).read_text())
        pairs = {it["key"]: local_path(it["reference"]) for it in items}
    print("loudness range = p90-p10 of 1 s window RMS (dB) within the speech span\n")
    print("| set | clips | range (median) | range (p75) | real same sentences (median) |")
    print("|---|---:|---:|---:|---:|")
    for spec in args.sets:
        label, d = spec.split("=", 1)
        root = Path(d)
        wavs = sorted((root / "audio").glob("*.wav")) if (root / "audio").exists() else sorted(root.glob("*.wav"))
        wavs = wavs[: args.limit]
        gen, real = [], []
        for p in wavs:
            r = loudness_range(load(p))
            if r is None:
                continue
            gen.append(r)
            ref = pairs.get(p.stem)
            if ref is not None and ref.exists():
                rr = loudness_range(load(ref))
                if rr is not None:
                    real.append(rr)
        real_s = f"{np.median(real):.1f}" if real else "–"
        print(f"| {label} | {len(gen)} | {np.median(gen):.1f} | {np.percentile(gen, 75):.1f} | {real_s} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tail")
    t.add_argument("sets", nargs="+", help="LABEL=eval set dir (with audio/)")
    t.add_argument("--frames-after-eos", type=int, default=5)
    d = sub.add_parser("dyn")
    d.add_argument("sets", nargs="+", help="LABEL=dir with audio/ or wavs")
    d.add_argument("--items", default=None, help="JVS eval items.json to pair with real recordings")
    d.add_argument("--limit", type=int, default=2000)
    o = sub.add_parser("onset")
    o.add_argument("sets", nargs="+", help="LABEL=dir with audio/<key>.wav or <key>.wav")
    o.add_argument("--items", default=None, help="JVS eval items.json (keys and real recordings); without it, unpaired")
    o.add_argument("--limit", type=int, default=1000)
    o.add_argument("--match", default=None, help="only items whose key contains this")
    args = parser.parse_args()
    {"tail": cmd_tail, "dyn": cmd_dyn, "onset": cmd_onset}[args.cmd](args)


if __name__ == "__main__":
    main()
