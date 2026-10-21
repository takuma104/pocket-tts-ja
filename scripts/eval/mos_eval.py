"""Neural MOS (UTMOS22 strong, UTMOSv2) of decoded audio, per directory, by gender.

Both predictors are trained on listening tests of (mostly English) synthetic speech, so the
absolute values for Japanese mean little; what is compared is the same utterances decoded by
different decoders. Each directory's scores are cached in <root>/mos_<name>.json and reused.

    uv run python scripts/eval/mos_eval.py exp/irodori_distill/p0/recon_jvs orig mimi keep4k M2H_H2 --ref keep4k

UTMOS22 strong (SpeechMOS, torch.hub) sees the whole utterance at 16 kHz. UTMOSv2 (fusion_stage3,
fold 0) sees a wav2vec 2.0 branch at 16 kHz and mel spectrograms of random crops; its "sarulab"
domain (the default) is used, and the crops are seeded so reruns agree.

--highpass HZ filters every file first (zero phase, 4th order): UTMOSv2 punishes the decoders'
content below 40 Hz (slow wander, -34 dB under the speech band, not a DC offset) by about 0.3,
more than anything audible between them; the recordings lose 0.09 by the same filter. Scores
with it are cached as mos_hp<HZ>_<name>.json.
"""

import argparse
import json
import multiprocessing
import os
import random
from pathlib import Path

# UTMOSv2 makes its spectrograms on the CPU in data loader workers; with multithreaded BLAS / OpenMP
# in each worker they oversubscribe the cores and run ~10x slower than single-threaded workers
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

SR = 16000


def load_16k(path: Path, highpass: float = 0.0) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float64", always_2d=True)
    x = wav.mean(1)
    if highpass > 0:
        x = sosfiltfilt(butter(4, highpass, "highpass", fs=sr, output="sos"), x)
    t = torch.from_numpy(x.astype(np.float32))
    return AF.resample(t, sr, SR) if sr != SR else t


class Scorers:
    def __init__(self, device: torch.device) -> None:
        import utmosv2

        self.device = device
        self.utmos22 = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
        self.v2 = utmosv2.create_model(pretrained=True, device=device)


@torch.no_grad()
def score_dir(paths: list[Path], scorers: Scorers, workers: int, highpass: float, tmp: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    tmp.mkdir(parents=True, exist_ok=True)
    for path in paths:
        wav = load_16k(path, highpass)
        out[path.stem] = {"utmos22": float(scorers.utmos22(wav[None].to(scorers.device), SR)[0])}
        if highpass > 0:  # UTMOSv2 reads files: give it the filtered ones
            sf.write(tmp / path.name, wav.numpy(), SR)
    # UTMOSv2's spectrograms are made on the CPU in the data loader: one call per directory
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    res = scorers.v2.predict(
        input_dir=tmp if highpass > 0 else paths[0].parent, val_list=[p.stem for p in paths], device=scorers.device,
        num_workers=workers, batch_size=16, num_repetitions=3, verbose=False,
    )
    for f in tmp.glob("*.wav"):
        f.unlink()
    tmp.rmdir()
    for r in res:
        out[Path(r["file_path"]).stem]["utmosv2"] = float(r["predicted_mos"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory holding one subdirectory of wavs per decoder and items.jsonl")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--ref", default=None, help="also print the paired difference from this name")
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--highpass", type=float, default=0.0, help="Hz; filter every file first (see above)")
    parser.add_argument("--keys", type=Path, default=None, help="only the keys listed in this file (one per line)")
    args = parser.parse_args()

    # Python 3.14 defaults to forkserver, which cannot pickle UTMOSv2's dataset (it holds modules)
    multiprocessing.set_start_method("fork", force=True)
    device = torch.device("cuda")
    items = {}
    items_path = args.root / "items.jsonl"
    if items_path.exists():
        items = {it["key"]: it for it in map(json.loads, open(items_path, encoding="utf-8"))}
    keys = set(args.keys.read_text().split()) if args.keys else None
    names = list(dict.fromkeys(args.names + ([args.ref] if args.ref else [])))
    scores: dict[str, dict[str, dict[str, float]]] = {}
    scorers: Scorers | None = None
    for name in names:
        cache = args.root / (f"mos_hp{args.highpass:g}_{name}.json" if args.highpass > 0 else f"mos_{name}.json")
        if cache.exists():
            scores[name] = json.loads(cache.read_text())
            continue
        paths = sorted((args.root / name).glob("*.wav"))
        print(f"{name}: {len(paths)} files", flush=True)
        scorers = scorers or Scorers(device)
        scores[name] = score_dir(paths, scorers, args.workers, args.highpass, args.root / f".mos_tmp_{os.getpid()}")
        cache.write_text(json.dumps(scores[name]))

    common = set.intersection(*(set(s) for s in scores.values()))
    if keys is not None:
        common &= keys
    common_sorted = sorted(common)
    groups = {"all": common_sorted}
    for g in ("F", "M"):
        sel = [k for k in common_sorted if items.get(k, {}).get("gender") == g]
        if sel:
            groups[g] = sel
    print(f"{len(common_sorted)} utterances in every directory")
    for metric in ("utmos22", "utmosv2"):
        header = f"| {metric} | " + " | ".join(groups) + (" | Δ" + args.ref + " (all) | Δ (F) |" if args.ref else " |")
        print("\n" + header)
        print("|---" * (header.count("|") - 1) + "|")
        for name in names:
            row = [f"{np.mean([scores[name][k][metric] for k in sel]):.3f}" for sel in groups.values()]
            if args.ref:
                for g in ("all", "F"):
                    sel = groups.get(g, [])
                    d = np.array([scores[name][k][metric] - scores[args.ref][k][metric] for k in sel])
                    row.append(f"{d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}" if len(d) > 1 else "")
            print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
