"""Generate one eval set under several (temperature, CFG) settings and compare the onset level.

Listening found the first phrase of a sentence quieter than the rest ("一方で、" soft, then
"漁業と商業で" louder). The real takes and their Mimi reconstructions do not do this, nor do
the training targets, so it comes from sampling: at the start the model is least sure of the
level, and a low temperature pulls those first frames towards a quieter average. This sweeps
temperature and CFG (1.0 = the official inference, which has no CFG) on a snapshot, without
ASR scoring, and prints the onset profile against the real recordings.

    uv run python scripts/generation_sweep.py runs/B1_4k --snapshot 180000 --device cpu \
        --configs 0.3:2.0 0.7:2.0 0.3:1.0 0.7:1.0 --out exp/sweep/B1_180k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_audio_dynamics import ONSET_BINS, load, local_path, onset_profile  # noqa: E402
from eval_tts import build_items, generate, load_model, snapshot_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", default="latest")
    parser.add_argument("--set", default="jvs_bal")
    parser.add_argument("--configs", nargs="+", default=["0.3:2.0", "0.7:2.0", "0.3:1.0", "0.7:1.0"],
                        help="TEMP:CFG pairs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    snap, step = snapshot_path(args.run_dir, args.snapshot)
    model, mimi = load_model(args.run_dir, snap, torch.device(args.device))
    items, _ = build_items(args.set, "rotation", 10)
    real = {it["key"]: onset_profile(load(local_path(it["reference"]))) for it in items}
    bins = [f"{a:.1f}-{b:.1f}s" for a, b in zip(ONSET_BINS[:-1], ONSET_BINS[1:], strict=True)]
    rows = []
    for cfg_spec in args.configs:
        temp, cfg = map(float, cfg_spec.split(":"))
        out_dir = args.out / f"t{temp}_cfg{cfg}" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_args = argparse.Namespace(seed=0, batch_size=args.batch_size, temp=temp, cfg=cfg,
                                      eos_threshold=-4.0, frames_after_eos=5)
        generate(model, mimi, items, out_dir, gen_args)
        profs = {it["key"]: onset_profile(load(out_dir / f"{it['key']}.wav")) for it in items}
        keys = [k for k, p in profs.items() if p is not None and real.get(k) is not None]
        med = np.median(np.array([profs[k] for k in keys]), axis=0)
        paired = np.median(np.array([np.subtract(profs[k], real[k]) for k in keys]), axis=0)  # type: ignore[arg-type]
        rows.append((f"temp {temp} / CFG {cfg}", len(keys), med, paired))
        print(f"done temp {temp} cfg {cfg}", flush=True)
    real_keys = [k for k, p in real.items() if p is not None]
    real_med = np.median(np.array([real[k] for k in real_keys]), axis=0)
    lines = [f"onset level, {args.run_dir.name} step {step}, set {args.set} "
             "(dB vs the rest of the utterance; paired = minus the real take)", "",
             "| setting | items | " + " | ".join(bins) + " | " + " | ".join(f"paired {b}" for b in bins) + " |",
             "|---|---:|" + "---:|" * (2 * len(bins)),
             f"| real | {len(real_keys)} | " + " | ".join(f"{m:+.1f}" for m in real_med) + " |" + " – |" * len(bins)]
    for label, n, med, paired in rows:
        lines.append(f"| {label} | {n} | " + " | ".join(f"{m:+.1f}" for m in med) + " | "
                     + " | ".join(f"{p:+.1f}" for p in paired) + " |")
    (args.out / "onset.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (args.out / "config.json").write_text(json.dumps({"snapshot": str(snap), "configs": args.configs}))


if __name__ == "__main__":
    main()
