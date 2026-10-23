"""W1g: a student's teacher-forced latents of the decoder targets' clips, as decoder inputs (GTA).

The wheel decoder learns on the teacher's latents (decode_targets.py), but at run time it decodes
the student's, each frame drawn from the flow head with its own noise. The 2.5-4 kHz comb of the
student's speech is near the teacher's at temperature 0 and falls as the temperature rises, with
the past right (student_tf.py) or not, so the frame noise is what costs the harmonics. As the
ground-truth-aligned inputs of a TTS vocoder (Tacotron 2, FastSpeech), the student here sees the
teacher's latents as its past and draws every frame (temperature 0.7 and one LSD step, as in
generation); the decoder then learns to turn such frames into the teacher's audio, the targets'
wav of the same clip. Rows are those of the student's training jsonl whose id the targets have.

    uv run python scripts/wheel/gta_latents.py data/irodori_distill/w1_gta \
        --targets data/irodori_distill/w1_targets --rows data/irodori_distill/mix/P3h_all_train.jsonl \
        --weights runs/irodori_distill/P3m_6l/snapshots/step_00020000.safetensors

Writes out/lat/<id>.npy (float16: the student's frames up to the row's speech end, the targets'
own frames after it) and out/wav/<id>.npy, a link to the target's wav, last (train_decoder.py
takes a clip once its wav is there).
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from pocket_tts_ja.dv_loader import _Corpus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from student_synth import load_student  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--targets", type=Path, required=True, help="decode_targets.py output")
    parser.add_argument("--rows", type=Path, required=True, help="the student's training jsonl")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/dv_6l_4k.yaml"))
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="the first rows in the jsonl's order")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    (args.out / "lat").mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in (args.targets / "wav").glob("*.npy")}
    done = {p.stem for p in (args.out / "wav").glob("*.npy")}
    ids = [json.loads(line)["id"] for line in open(args.rows, encoding="utf-8")]
    corpus = _Corpus(str(args.rows))
    todo = [i for i, key in enumerate(ids) if key in have and key not in done][: args.limit]
    todo.sort(key=lambda i: int(corpus.speech_end[i]))  # less padding per batch
    print(f"{len(todo)} clips ({len(done)} done)", flush=True)

    device = torch.device("cuda")
    model = load_student(args.config, args.weights, device)
    fl = model.flow_lm
    tokenizer = fl.conditioner.tokenizer
    torch.manual_seed(args.seed)
    t0 = time.time()
    for b in range(0, len(todo), args.batch_size):
        idx = todo[b : b + args.batch_size]
        lats = [torch.from_numpy(np.asarray(corpus.latents(i, int(corpus.speech_end[i])), dtype=np.float32)) for i in idx]
        prompts = [torch.from_numpy(np.asarray(corpus.prompt(i), dtype=np.float32)) for i in idx]
        T, Tv = max(z.shape[0] for z in lats), max(p.shape[0] for p in prompts)
        x = torch.zeros(len(idx), T, fl.ldim)
        v = torch.zeros(len(idx), Tv, fl.ldim)
        for k, (z, p) in enumerate(zip(lats, prompts, strict=True)):
            x[k, : z.shape[0]] = z
            v[k, : p.shape[0]] = p
        x, v = x.to(device), v.to(device)
        n_voice = torch.tensor([p.shape[0] for p in prompts], device=device)
        tokens = [torch.tensor(tokenizer.encode(corpus.texts[i]), dtype=torch.long, device=device) for i in idx]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.backbone_z(fl, (x - fl.emb_mean) / fl.emb_std, tokens, v, n_voice, cfg_dropout=False)
        with torch.no_grad():
            noise = args.temp**0.5 * torch.randn(*z.shape[:2], fl.ldim, device=device)
            drawn = model.flow.decode(partial(fl.flow_net, z.float().reshape(-1, z.shape[-1])), noise.reshape(-1, fl.ldim), args.flow_steps)
            drawn = (drawn.reshape(*z.shape[:2], fl.ldim) * fl.emb_std + fl.emb_mean).cpu().numpy()
        for k, i in enumerate(idx):
            key, n = ids[i], lats[k].shape[0]
            target = np.load(args.targets / "lat" / f"{key}.npy")
            assert np.abs(target[:n].astype(np.float32) - lats[k].numpy()).max() < 1e-2, f"{key}: not the targets' latents"
            np.save(args.out / "lat" / f"{key}.npy", np.concatenate([drawn[k, :n], target[n:]]).astype(np.float16))
            (args.out / "wav" / f"{key}.npy").symlink_to((args.targets / "wav" / f"{key}.npy").resolve())
        if (b // args.batch_size) % 200 == 0:
            done_n = b + len(idx)
            print(f"{done_n}/{len(todo)} clips, {done_n / (time.time() - t0):.0f} clips/s", flush=True)
    print(f"{args.out}: {len(todo)} clips in {time.time() - t0:.0f} s; GTA_DONE", flush=True)


if __name__ == "__main__":
    main()
