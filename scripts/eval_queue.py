"""Evaluate new training snapshots asynchronously on a second machine (plan §6).

Runs on the training machine. Every --interval seconds, for each run directory given, it
finds EMA snapshots (runs/<run>/snapshots/step_*.safetensors) not yet evaluated, copies the
snapshot plus args.yaml / wandb_id.txt to the eval host with rsync, runs
scripts/eval_tts.py there for each requested set (scalars go to the run's wandb run), then
copies the small result files (metrics.json, JKYB summary) back. The remote copy of the
snapshot is deleted afterwards unless --keep-remote. Evaluated steps are recorded in
runs/<run>/eval_done.txt.

    uv run python scripts/eval_queue.py runs/A5_8k runs/A5_4k --sets jkyb_mini jvs
"""

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path

REMOTE_REPO = "~/co/pocket-tts-ja"
UV = "~/.local/bin/uv"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4"]


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    if cmd[0] == "ssh":
        cmd = ["ssh", *SSH_OPTS, *cmd[1:]]
    elif cmd[0] == "rsync":
        cmd = ["rsync", "-e", "ssh " + " ".join(SSH_OPTS), *cmd[1:]]
    return subprocess.run(cmd, check=check, text=True, capture_output=True, timeout=3 * 3600)


def pending(run: Path) -> list[int]:
    done_file = run / "eval_done.txt"
    done = {int(x) for x in done_file.read_text().split()} if done_file.exists() else set()
    steps = []
    for p in sorted((run / "snapshots").glob("step_*.safetensors")):
        step = int(re.search(r"(\d+)", p.stem).group(1))  # type: ignore[union-attr]
        if step not in done:
            steps.append(step)
    return sorted(steps, reverse=True)  # newest first: the latest snapshot matters most


def evaluate(run: Path, step: int, host: str, sets: list[str], jvs_speakers: int, keep: bool):
    rrun = f"{REMOTE_REPO}/{run.as_posix()}"
    snap = run / "snapshots" / f"step_{step:08d}.safetensors"
    sh(["ssh", host, f"mkdir -p {rrun}/snapshots"])
    sh(["rsync", "-a", str(run / "args.yaml"), str(run / "wandb_id.txt"), f"{host}:{rrun}/"])
    sh(["rsync", "-a", str(snap), f"{host}:{rrun}/snapshots/"])
    for s in sets:
        extra = f"--jvs-speakers {jvs_speakers}" if s == "jvs" else ""
        cmd = f"cd {REMOTE_REPO} && {UV} run python scripts/eval_tts.py {shlex.quote(run.as_posix())} --set {s} --snapshot {step} {extra}"
        res = sh(["ssh", host, cmd], check=False)
        log = run / "eval_logs"
        log.mkdir(exist_ok=True)
        (log / f"step{step:08d}_{s}.log").write_text(res.stdout[-20000:] + "\n--- stderr ---\n" + res.stderr[-20000:])
        if res.returncode != 0:
            print(f"{run.name} step {step} {s}: FAILED (see {log})", flush=True)
            return False
    sh(["rsync", "-a", "--include=*/", "--include=metrics.json", "--include=summary.json",
        "--include=summary.md", "--include=cer.json", "--include=generation.jsonl", "--exclude=*",
        f"{host}:{rrun}/eval/", str(run / "eval") + "/"], check=False)
    if not keep:
        sh(["ssh", host, f"rm -f {rrun}/snapshots/step_{step:08d}.safetensors"], check=False)
    with open(run / "eval_done.txt", "a") as f:
        f.write(f"{step}\n")
    print(f"{run.name} step {step}: evaluated {sets}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--host", default="galleria2")
    parser.add_argument("--sets", nargs="+", default=["jkyb_mini", "jvs"])
    parser.add_argument("--jvs-speakers", type=int, default=10)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        for run in args.runs:
            if not (run / "snapshots").exists():
                continue
            for step in pending(run):
                try:
                    ok = evaluate(run, step, args.host, args.sets, args.jvs_speakers, args.keep_remote)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    print(f"{run.name} step {step}: remote unavailable ({exc}); retrying later", flush=True)
                    ok = False
                if not ok:
                    break
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
