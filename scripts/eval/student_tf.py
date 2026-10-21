"""P2: the teacher-forced sample decode of a student on held-out voices (plan P2, evaluation 1).

For validation rows (voices never trained on) the student sees the teacher's latents as its past,
as in training, and draws every frame from its flow head (temperature 0.3, one LSD step); the
drawn frames, put side by side and decoded with DACVAE, show what the student can say when its
past is right. Next to them the teacher's own latents, decoded the same way. Free generation that
is much worse than this points at the error build-up of autoregression (P3c); this being bad
points at capacity, data or the representation.

Writes an evaluation directory (items.jsonl; gender from the median F0 of the teacher's clip,
F above 165 Hz) that the usual scripts score: teacher/, <name>/ (24 kHz), orig/ = teacher/ (so
similarity.py compares with the teacher's clip).

    uv run python scripts/eval/student_tf.py data/irodori_distill/mix/P2_s3_valid.jsonl \
        runs/irodori_distill/P2_6l/snapshots/step_00020000.safetensors --name P2_20k
    -> exp/irodori_distill/eval/tf_val/
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

from pocket_tts_ja.dacvae import DACVAE, SAMPLE_RATE as DV_SR
from pocket_tts_ja.dv_loader import _Corpus

from hf_structure import voiced_f0
from student_synth import OUT_SR, load_student


def median_f0(y: np.ndarray, sr: int) -> float:
    f0 = [f for _, f in voiced_f0(y, sr)]
    return float(np.median(f0)) if f0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("valid_jsonl", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/dv_6l_4k.yaml"))
    parser.add_argument("--out", type=Path, default=Path("exp/irodori_distill/eval/tf_val"))
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = load_student(args.config, args.weights, device)
    fl = model.flow_lm
    tokenizer = fl.conditioner.tokenizer
    codec = DACVAE.load(str(device))
    corpus = _Corpus(str(args.valid_jsonl))
    rows = [json.loads(line) for line in open(args.valid_jsonl, encoding="utf-8")][: args.rows]
    (args.out / "teacher").mkdir(parents=True, exist_ok=True)
    (args.out / args.name).mkdir(exist_ok=True)
    if not (args.out / "orig").exists():
        (args.out / "orig").symlink_to("teacher")

    def write(path: Path, z: torch.Tensor) -> np.ndarray:
        y = soxr.resample(codec.decode(z[None].float())[0].cpu().numpy(), DV_SR, OUT_SR, quality="HQ")
        sf.write(str(path), np.clip(y, -1, 1), OUT_SR, subtype="PCM_16")
        return y

    items = []
    torch.manual_seed(args.seed)
    for b in range(0, len(rows), args.batch_size):
        idx = list(range(b, min(b + args.batch_size, len(rows))))
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
        tokens = [torch.tensor(tokenizer.encode(rows[i]["transcript"]), dtype=torch.long, device=device) for i in idx]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = (x - fl.emb_mean) / fl.emb_std
            z = model.backbone_z(fl, normalized, tokens, v, n_voice, cfg_dropout=False)
        with torch.no_grad():
            noise = args.temp**0.5 * torch.randn(*z.shape[:2], fl.ldim, device=device)
            drawn = model.flow.decode(partial(fl.flow_net, z.float().reshape(-1, z.shape[-1])), noise.reshape(-1, fl.ldim), 1)
            drawn = drawn.reshape(*z.shape[:2], fl.ldim) * fl.emb_std + fl.emb_mean
        for k, i in enumerate(idx):
            n = lats[k].shape[0]
            key = rows[i]["id"]
            y = write(args.out / "teacher" / f"{key}.wav", x[k, :n])
            write(args.out / args.name / f"{key}.wav", drawn[k, :n])
            gender = "F" if median_f0(y, OUT_SR) > 165 else "M"
            items.append({"key": key, "text": rows[i]["text"], "speaker": rows[i]["ref_speaker"], "gender": gender})
    with open(args.out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(items)} rows, {sum(it['gender'] == 'F' for it in items)} F")


if __name__ == "__main__":
    main()
