"""Generate an eval set with the stock pocket-tts wheel on CPU (plan D3: the drop-in check).

Runs in the published wheel's environment, not in this project's, and uses only its public API
(TTSModel.load_model, get_state_for_audio_prompt, generate_audio) with its defaults:

    uv run python scripts/eval_tts.py runs/D1_6l --snapshot 200000 --set jkyb_mini --cfg 1.0 \
        --tag-suffix wheel --export-items
    uvx --from pocket-tts==3.3.0 python scripts/release/wheel_generate.py release/japanese.yaml \
        runs/D1_6l/eval/step00200000/jkyb_mini_rotation_cfg1.0_wheel --workers 6
    uv run python scripts/eval_tts.py runs/D1_6l --snapshot 200000 --set jkyb_mini --cfg 1.0 \
        --tag-suffix wheel --from-audio

Reads <dir>/items.json (key, text, prompt wav), writes <dir>/audio/<key>.wav (16-bit) and
appends one row per item to <dir>/generation.jsonl: tokens and frames of the whole text, the
chunks the stock code split it into, how many of them hit the length budget without EOS, and
the generation time. Items that already have a wav are skipped.
"""

import argparse
import json
import logging
import multiprocessing
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

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


def work(config: str, out_dir: str, shard: int, shards: int, threads: int, seed: int) -> None:
    from pocket_tts import TTSModel

    torch.set_num_threads(threads)
    torch.manual_seed(seed + shard)
    counter = Counter()
    wheel_log = logging.getLogger("pocket_tts")
    wheel_log.setLevel(logging.INFO)
    wheel_log.addHandler(counter)
    wheel_log.propagate = False

    out = Path(out_dir)
    items = json.loads((out / "items.json").read_text())[shard::shards]
    items = [it for it in items if not (out / "audio" / f"{it['key']}.wav").exists()]
    model = TTSModel.load_model(config=config)
    frame = round(model.sample_rate / model.config.mimi.frame_rate)
    states: dict[str, dict] = {}
    with open(out / "generation.jsonl", "a", encoding="utf-8") as log:
        for it in items:
            if it["prompt"] not in states:
                states[it["prompt"]] = model.get_state_for_audio_prompt(Path(it["prompt"]))
            counter.no_eos = counter.chunks = 0
            start = time.monotonic()
            audio = model.generate_audio(states[it["prompt"]], it["text"])
            seconds = time.monotonic() - start
            pcm = (audio.clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
            tmp = out / "audio" / f".{it['key']}.tmp"
            scipy.io.wavfile.write(tmp, model.sample_rate, pcm)
            tmp.rename(out / "audio" / f"{it['key']}.wav")
            row = {
                "key": it["key"],
                "tokens": int(model.flow_lm.conditioner.prepare(it["text"]).shape[-1]),
                "frames": int(np.ceil(len(pcm) / frame)),
                "chunks": counter.chunks,
                "no_eos": counter.no_eos > 0,
                "gen_sec": round(seconds, 3),
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="model config (.yaml) of the packaged model")
    parser.add_argument("out_dir", type=Path, help="result directory holding items.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    (args.out_dir / "audio").mkdir(exist_ok=True)
    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=work, args=(args.config, str(args.out_dir), i, args.workers, args.threads, args.seed))
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    rows = [json.loads(line) for line in open(args.out_dir / "generation.jsonl")]
    audio_sec = sum(r["frames"] for r in rows) * 0.08
    gen_sec = sum(r["gen_sec"] for r in rows)
    print(
        f"{len(rows)} items, {audio_sec:.0f} s of audio, {gen_sec / max(audio_sec, 1e-9):.3f} RTF per worker "
        f"({args.threads} threads), no EOS in {sum(r['no_eos'] for r in rows)}, wall {time.monotonic() - start:.0f} s"
    )


if __name__ == "__main__":
    main()
