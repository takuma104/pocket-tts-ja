"""Generate an evaluation set with the stock pocket-tts wheel on CPU (W1: the drop-in check).

main's scripts/release/wheel_generate.py for this branch's evaluation sets
(scripts/eval/make_eval_dir.py layout), so that the GPU evaluation's scoring scripts read the
result like any student_synth.py output. Runs in the published wheel's environment, not in this
project's, and uses only its public API (TTSModel.load_model, get_state_for_audio_prompt,
generate_audio) with the package's defaults:

    uvx --from pocket-tts==4.0.1 python scripts/wheel/wheel_generate.py \
        exp/irodori_distill/package/p3d_20k/japanese_dv.yaml exp/irodori_distill/eval/voices --name P3d_20k_wheel

Reads <eval dir>/items.jsonl (key, text, prompt wav), writes <eval dir>/<name>/<key>.wav (16-bit,
the wheel's 24 kHz) and appends one row per item to <eval dir>/<name>.jsonl: the frames, whether
every chunk the stock code split the text into ended on EOS ("eos"), the chunk count and the
generation time. Items that already have a wav are skipped.

--snake <package>/<name>/snake.safetensors (package_w1.py, from a decoder trained with Snake's term)
puts each alpha / beta pair back on the ELU it belongs to after the stock loader has built the model:
an internal preview of W2 (the stock SEANet hard-codes ELU), not something the wheel can do itself.
"""

import argparse
import json
import logging
import multiprocessing
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import safetensors.torch
import torch
import torch.nn.functional as F
import yaml

NO_EOS_MESSAGE = "Maximum generation length reached without EOS"


class Counter(logging.Handler):
    """Counts what the wheel only reports through its log."""

    def __init__(self) -> None:
        super().__init__()
        self.no_eos = 0
        self.chunks = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if NO_EOS_MESSAGE in message:
            self.no_eos += 1
        elif message.startswith("starting timer now"):
            self.chunks += 1


class SnakeELU(torch.nn.Module):
    """elu(x) + beta sin^2(alpha x) / alpha per channel (pocket_tts_ja.wheel_codec.SnakeELU, forward only)."""

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + self.beta * torch.sin(self.alpha * x).pow(2) / (self.alpha + 1e-9)


def add_snake(model: torch.nn.Module, path: str) -> None:
    """Swaps every ELU that `path` has an alpha / beta for (named as in the model's state dict)."""
    terms = safetensors.torch.load_file(path)
    swapped = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, torch.nn.ELU) and f"{full}.alpha" in terms:
                setattr(parent, child_name, SnakeELU(terms[f"{full}.alpha"].float(), terms[f"{full}.beta"].float()))
                swapped += 1
    assert 2 * swapped == len(terms), (swapped, len(terms))


def work(config: str, eval_dir: str, name: str, shard: int, shards: int, threads: int, seed: int,
         snake: str | None = None) -> None:
    from pocket_tts import TTSModel

    torch.set_num_threads(threads)
    torch.manual_seed(seed + shard)
    counter = Counter()
    wheel_log = logging.getLogger("pocket_tts")
    wheel_log.setLevel(logging.INFO)
    wheel_log.addHandler(counter)
    wheel_log.propagate = False

    root = Path(eval_dir)
    out = root / name
    items = [json.loads(line) for line in open(root / "items.jsonl", encoding="utf-8")][shard::shards]
    items = [it for it in items if not (out / f"{it['key']}.wav").exists()]
    model = TTSModel.load_model(config=config)
    if snake:
        add_snake(model, snake)
    frame = round(model.sample_rate / model.config.mimi.frame_rate)
    states: dict[str, dict] = {}
    with open(root / f"{name}.jsonl", "a", encoding="utf-8") as log:
        for it in items:
            if it["prompt"] not in states:
                states[it["prompt"]] = model.get_state_for_audio_prompt(Path(it["prompt"]))
            counter.no_eos = counter.chunks = 0
            start = time.monotonic()
            audio = model.generate_audio(states[it["prompt"]], it["text"])
            seconds = time.monotonic() - start
            pcm = (audio.clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
            tmp = out / f".{it['key']}.tmp"
            scipy.io.wavfile.write(tmp, model.sample_rate, pcm)
            tmp.rename(out / f"{it['key']}.wav")
            row = {
                "key": it["key"],
                "text": it["text"],
                "frames": int(np.ceil(len(pcm) / frame)),
                "eos": counter.no_eos == 0,
                "chunks": counter.chunks,
                "gen_sec": round(seconds, 3),
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="model config (.yaml) of the packaged model")
    parser.add_argument("eval_dir", type=Path, help="evaluation set (items.jsonl)")
    parser.add_argument("--name", required=True, help="result directory under eval_dir")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snake", default=None, help="snake.safetensors of the package (W2 preview)")
    args = parser.parse_args()

    (args.eval_dir / args.name).mkdir(exist_ok=True)
    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=work, args=(args.config, str(args.eval_dir), args.name, i, args.workers, args.threads, args.seed, args.snake))
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    rows = [json.loads(line) for line in open(args.eval_dir / f"{args.name}.jsonl", encoding="utf-8")]
    frame_rate = yaml.safe_load(Path(args.config).read_text())["mimi"]["frame_rate"]
    audio_sec = sum(r["frames"] for r in rows) / frame_rate
    gen_sec = sum(r["gen_sec"] for r in rows)
    print(
        f"{len(rows)} items, {audio_sec:.0f} s of audio, {gen_sec / max(audio_sec, 1e-9):.3f} RTF per worker "
        f"({args.threads} threads), no EOS in {sum(not r['eos'] for r in rows)}, wall {time.monotonic() - start:.0f} s"
    )


if __name__ == "__main__":
    main()
