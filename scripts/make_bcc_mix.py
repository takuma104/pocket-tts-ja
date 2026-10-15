"""CC-BY line (plan B-CC): the training mix from the synthetic speech with CC BY voices that exists.

The natural speech (YODAS3, Emilia-YODAS and its gain copies) is taken whole. Each synthetic set
is aimed at its share of the hours in the main line's mix v9 (TARGET), but gets no more than
--max-repeat times its unique hours: at first only the clips that were cloned from Emilia-YODAS
voices are there (<set>/manifest_ccref.jsonl); the sentences that had an NC voice come back set
by set from the re-synthesis (<set>_cc/manifest.jsonl, scripts/synth/run_cc_sets.sh). Both parts
of a set are repeated equally. Calls scripts/build_train_mix.py.

    uv run python scripts/make_bcc_mix.py --out data/mix/BCC_train_v2.jsonl
    uv run python scripts/make_bcc_mix.py --out data/mix/BCC_train_v7.jsonl \
        --yodas3 yodas3_kept/train_punct_trim.jsonl   # speech ends of YODAS3 moved (yodas3/trim_tails.py)
    uv run python scripts/make_bcc_mix.py --out data/mix/BCC_train_v8.jsonl \
        --yodas3 yodas3_kept/train_punct_trim_rechecked.jsonl \
        --emilia emilia_ja/train_rechecked_yodas_tail.jsonl \
        --emilia-gain emilia_ja_gain/train_yodas_tail.jsonl   # filter_asr_recheck.py --max-tail 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

NATURAL = {
    "yodas3": "yodas3_kept/train_punct.jsonl",
    "emilia": "emilia_ja/train_rechecked_yodas.jsonl",
    "emilia_gain": "emilia_ja_gain/train_yodas.jsonl",
}
TARGET = {"s3": 0.12, "s4": 0.03, "s6a": 0.05, "s5": 0.04, "s6b": 0.05, "s7": 0.02, "s9": 0.01}


def hours(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        return sum(json.loads(line)["duration"] for line in f if line.strip()) / 3600


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--max-repeat", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="print the shares only")
    parser.add_argument("--yodas3", default=NATURAL["yodas3"],
                        help="YODAS3 manifest under --data (mix v7: yodas3_kept/train_punct_trim.jsonl), "
                        "or none")
    parser.add_argument("--emilia", default=NATURAL["emilia"], help="Emilia-YODAS manifest under --data")
    parser.add_argument("--emilia-gain", default=NATURAL["emilia_gain"], help="its gain copies under --data")
    args = parser.parse_args()
    NATURAL["yodas3"] = args.yodas3
    if args.yodas3 == "none":
        del NATURAL["yodas3"]
    NATURAL["emilia"] = args.emilia
    NATURAL["emilia_gain"] = args.emilia_gain

    natural = sum(hours(args.data / p) for p in NATURAL.values())
    parts: dict[str, list[tuple[str, Path, float]]] = {}
    for name in TARGET:
        first = args.data / "synth" / ("s3_full" if name == "s3" else name) / "manifest_ccref.jsonl"
        again = args.data / "synth" / f"{name}_cc" / "manifest.jsonl"
        parts[name] = [(name, first, hours(first))]
        if again.exists():
            parts[name].append((f"{name}_cc", again, hours(again)))
    unique = {name: sum(h for _, _, h in p) for name, p in parts.items()}

    share = dict(TARGET)
    for _ in range(100):  # the total depends on the shares: total = natural / (1 - sum of shares)
        total = natural / (1 - sum(share.values()))
        share = {name: min(TARGET[name], args.max_repeat * unique[name] / total) for name in TARGET}
    total = natural / (1 - sum(share.values()))
    print(f"natural {natural:.1f} h, total {total:.1f} h, synthetic {100 * sum(share.values()):.1f}%")
    sources = [f"{name}={args.data / path}" for name, path in NATURAL.items()]
    for name, p in parts.items():
        repeat = share[name] * total / unique[name]
        print(f"  {name}: {share[name]:.4f} of the hours (target {TARGET[name]}), {unique[name]:.1f} h unique"
              f" ({' + '.join(f'{h:.1f}' for _, _, h in p)}), each clip {repeat:.2f} times")
        sources += [f"irodori_{part}={path}:{share[name] * h / unique[name]:.6f}" for part, path, h in p]
    if args.dry_run:
        return
    cmd = [sys.executable, "scripts/build_train_mix.py", "--out", str(args.out)]
    for s in sources:
        cmd += ["--source", s]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
