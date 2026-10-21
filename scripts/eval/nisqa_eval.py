"""NISQA scores per result directory, by gender, with paired differences (alongside mos_eval.py).

Two pretrained models of third-party/NISQA (gabrielmittag/NISQA; weights CC BY-NC-SA 4.0, used for
evaluation only):

    nisqa      NISQA v2: overall MOS and the dimensions noisiness / coloration / discontinuity / loudness
    nisqa_tts  NISQA-TTS: naturalness of synthetic speech

Both read the file at 48 kHz with mel bands up to 16 kHz, so unlike UTMOS22 / UTMOSv2 (16 kHz) they
see the high band. On mimi_ft (C6, 10/22) NISQA-TTS naturalness was the score closest to listening
(it singled out the CC decoder's female_02 / female_05, where UTMOSv2 went the other way), and the
NISQA v2 coloration singled out the original Mimi on those two voices.

Scores are cached per directory in <root>/nisqa_<name>.json.

    uv run python scripts/eval/nisqa_eval.py exp/irodori_distill/p0/recon_jvs orig dacvae mimi soup4 --ref mimi
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NISQA_ROOT = Path(__file__).resolve().parents[2] / "third-party" / "NISQA"
MODELS = {"nisqa": "nisqa.tar", "nisqa_tts": "nisqa_tts.tar"}
# NISQA v2 output columns -> short names; NISQA-TTS gives mos_pred only.
V2_COLUMNS = {"mos_pred": "mos", "noi_pred": "noi", "dis_pred": "dis", "col_pred": "col", "loud_pred": "loud"}
METRICS = ("tts", "mos", "col", "noi", "dis", "loud")


def predict_dir(model_file: str, data_dir: Path, workers: int) -> dict[str, dict[str, float]]:
    sys.path.insert(0, str(NISQA_ROOT))
    from nisqa.NISQA_model import nisqaModel

    args = {
        "mode": "predict_dir", "pretrained_model": str(NISQA_ROOT / "weights" / model_file),
        "data_dir": str(data_dir), "output_dir": None, "deg": None, "csv_file": None, "csv_deg": None,
        "num_workers": workers, "bs": 16, "ms_channel": None, "tr_bs_val": 16, "tr_num_workers": workers,
    }
    model = nisqaModel(args)
    model.predict()
    df = model.ds_val.df
    return {Path(row["deg"]).stem: {c: float(row[c]) for c in V2_COLUMNS if c in row} for _, row in df.iterrows()}


def score_dir(data_dir: Path, workers: int) -> dict[str, dict[str, float]]:
    v2 = predict_dir(MODELS["nisqa"], data_dir, workers)
    tts = predict_dir(MODELS["nisqa_tts"], data_dir, workers)
    out: dict[str, dict[str, float]] = {}
    for key, row in v2.items():
        out[key] = {V2_COLUMNS[c]: v for c, v in row.items()}
        out[key]["tts"] = tts[key]["mos_pred"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory holding one subdirectory of wavs per result and items.jsonl")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--ref", default=None, help="also print the paired difference from this name")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metrics", default="tts,mos,col", help=f"comma-separated, of {','.join(METRICS)}")
    args = parser.parse_args()

    items = {}
    if (args.root / "items.jsonl").exists():
        items = {it["key"]: it for it in map(json.loads, open(args.root / "items.jsonl", encoding="utf-8"))}
    names = list(dict.fromkeys(args.names + ([args.ref] if args.ref else [])))
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        cache = args.root / f"nisqa_{name}.json"
        if not cache.exists():
            print(f"{name}: scoring {args.root / name}", flush=True)
            cache.write_text(json.dumps(score_dir(args.root / name, args.workers)))
        scores[name] = json.loads(cache.read_text())

    common = sorted(set.intersection(*(set(s) for s in scores.values())))
    groups = {"all": common}
    for g in ("F", "M"):
        sel = [k for k in common if items.get(k, {}).get("gender") == g]
        if sel and len(sel) < len(common):
            groups[g] = sel
    print(f"{len(common)} utterances in every directory")
    for metric in args.metrics.split(","):
        header = f"| NISQA {metric} | " + " | ".join(groups)
        if args.ref:
            header += " | " + " | ".join(f"Δ{args.ref} ({g})" for g in groups)
        print("\n" + header + " |")
        print("|---" * (header.count("|")) + "|")
        for name in names:
            row = [f"{np.mean([scores[name][k][metric] for k in sel]):.3f}" for sel in groups.values()]
            if args.ref:
                for sel in groups.values():
                    d = np.array([scores[name][k][metric] - scores[args.ref][k][metric] for k in sel])
                    row.append(f"{d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}" if len(d) > 1 else "")
            print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
