"""Generate an evaluation set from a training snapshot and score it (plan §6).

Sets:
  jkyb_mini  data/eval/jkyb_mini.jsonl (1,500 JKYB-Parakeet rows), scored by `jkyb-eval tts`
  jkyb       the full 13,536 rows
  jvs        zero-shot voice cloning on JVS: --jvs-speakers speakers x 10 parallel100
  jvs_bal    the same with 5 male + 5 female speakers (jvs is 8 M / 2 F)
             sentences, prompted by the speaker's first nonpara30 utterance; Kana-CER
             against the phoneme-label reading, Text CER, ECAPA similarity to the speaker's
             real recording of the same sentence, EOS failures

Generation follows the stock inference defaults: temp 0.3, 1 LSD step, eos threshold -4,
5 frames after EOS, and the per-row length budget tokens / 3 + 2 s (a row that reaches it
without EOS counts as no_eos). JKYB voices rotate over 4 JVS speakers (2 male, 2 female)
by key hash; `--voice jvs001` uses the Irodori comparison prompt instead. Scalars go to the
run's wandb run at the snapshot's step (no audio is uploaded).

    uv run python scripts/eval_tts.py runs/A5_8k --set jkyb_mini --snapshot latest
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import soundfile as sf
import torch

from pocket_tts_ja.jvs import list_items, read_genders
from pocket_tts_ja.text import normalize
from training.args import load_args
from training.modules.builders import build_models

REPO = Path(__file__).resolve().parents[1]
JKYB_PROJECT = REPO / "third-party" / "Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition"
JVS = REPO / "data" / "jvs_ver1"
ROTATION = ("jvs006", "jvs041", "jvs007", "jvs066")  # M low/high F0, F low/high F0
JVS001 = JVS / "jvs001" / "parallel100" / "wav24kHz16bit" / "VOICEACTRESS100_001.wav"
FRAME_RATE = 12.5
MIN_FRAMES = 8
# non-interactive ssh sessions do not have ~/.local/bin on PATH. The benchmark env is pinned to
# Python 3.13: under 3.14 its ASR DataLoader cannot pickle its workers.
UV = os.environ.get("UV") or shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


def snapshot_path(run_dir: Path, which: str) -> tuple[Path, int]:
    snaps = sorted((run_dir / "snapshots").glob("step_*.safetensors"))
    if which == "latest":
        p = snaps[-1]
    elif which.isdigit():
        p = run_dir / "snapshots" / f"step_{int(which):08d}.safetensors"
    else:
        p = Path(which)
    step = int(re.search(r"(\d+)", p.stem).group(1))  # type: ignore[union-attr]
    return p, step


def load_model(run_dir: Path, snapshot: Path, device: torch.device) -> tuple[Any, Any]:
    args = load_args(run_dir / "args.yaml")
    args.start_from_pretrained = False
    args.distill_cfg_coef = 0.0
    model, mimi, _ = build_models(args)
    state = safetensors.torch.load_file(str(snapshot))
    model.flow_lm.load_state_dict(
        {k.removeprefix("flow_lm."): v for k, v in state.items()}, strict=True
    )
    model.to(device).eval()
    mimi.to(device).eval()
    from training.modules.utils import stamp_state_names

    stamp_state_names(mimi)
    return model, mimi


def first_nonpara(spk: str) -> Path:
    return sorted((JVS / spk / "nonpara30" / "wav24kHz16bit").glob("*.wav"))[0]


JKYB_SCHEMA_SETS = {
    "jkyb_mini": REPO / "data" / "eval" / "jkyb_mini.jsonl",
    # held-out Aozora works with ruby-verified readings (Irodori experiments, JKYB schema);
    # excluded from the synthetic corpus, so it measures generalization beyond JKYB
    "aozora": REPO / "data" / "eval" / "aozora_eval_1k.jsonl",
}


def build_items(which: str, voice: str, jvs_speakers: int) -> list[dict[str, Any]]:
    if which == "jsut":
        # JSUT BASIC5000 sentences with verified kana (jsut-label): general reading, Kana-CER
        items = []
        for line in open(REPO / "data" / "eval" / "jsut_eval_1k.jsonl"):
            r = json.loads(line)
            h = int(hashlib.md5(r["key"].encode()).hexdigest(), 16)
            prompt = JVS001 if voice == "jvs001" else first_nonpara(ROTATION[h % len(ROTATION)])
            items.append({"key": r["key"], "text": r["text"], "yomi": r["kana"], "prompt": str(prompt)})
        return items, None  # type: ignore[return-value]
    if which in ("jkyb_mini", "jkyb", "aozora"):
        if which in JKYB_SCHEMA_SETS:
            path = JKYB_SCHEMA_SETS[which]
        else:
            path = Path(
                next(
                    (Path.home() / ".cache/huggingface/hub").glob(
                        "datasets--Parakeet-Inc--joyo-kanji-yomi-benchmark-parakeet/snapshots/*/data/common_kanji_source.jsonl"
                    )
                )
            )
        items = []
        for line in open(path):
            r = json.loads(line)
            if voice == "jvs001":
                prompt = JVS001
            else:
                h = int(hashlib.md5(r["key"].encode()).hexdigest(), 16)
                prompt = first_nonpara(ROTATION[h % len(ROTATION)])
            items.append({"key": r["key"], "text": r["text"], "prompt": str(prompt)})
        return items, path  # type: ignore[return-value]
    # JVS zero-shot, 10 fixed sentences per speaker. "jvs": speakers spread over the id range
    # (8 M / 2 F for 10). "jvs_bal": 5 M + 5 F spread over each gender's ids, excluding the
    # JKYB rotation voices.
    rows = {i.key: i for i in list_items(JVS, ("parallel100",))}
    if which == "jvs_bal":
        genders = read_genders(JVS)
        spks = []
        for g in ("M", "F"):
            pool = sorted(k for k, v in genders.items() if v[0] == g and k not in ROTATION)
            spks += [pool[i] for i in np.linspace(0, len(pool) - 1, jvs_speakers // 2).round().astype(int)]
    else:
        spks = [f"jvs{n:03d}" for n in np.linspace(1, 100, jvs_speakers).round().astype(int)]
    sentence_ids = [f"VOICEACTRESS100_{n:03d}" for n in range(5, 100, 10)]
    items = []
    for spk in spks:
        for sid in sentence_ids:
            row = rows.get(f"{spk}_parallel100_{sid}")
            if row is None:
                continue
            items.append(
                {
                    "key": row.key,
                    "text": row.text,
                    "yomi": row.yomi,
                    "prompt": str(first_nonpara(spk)),
                    "reference": row.wav,
                    "gender": row.gender,
                }
            )
    return items, None  # type: ignore[return-value]


@torch.no_grad()
def generate(
    model: Any, mimi: Any, items: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace
) -> dict[str, int]:
    device = next(model.parameters()).device
    tokenize = model.flow_lm.conditioner.tokenizer.encode
    voice_cache: dict[str, torch.Tensor] = {}

    def voice(path: str) -> torch.Tensor:
        if path not in voice_cache:
            wav, sr = sf.read(path, dtype="float32")
            assert sr == mimi.sample_rate
            voice_cache[path] = mimi.encode_to_latent(torch.from_numpy(wav)[None, None].to(device))[0]
        return voice_cache[path]

    todo = [it for it in items if not (out_dir / f"{it['key']}.wav").exists()]
    for it in todo:
        it["tokens"] = tokenize(normalize(it["text"]))
    todo.sort(key=lambda it: len(it["tokens"]))
    counts = {"no_eos": 0, "silent": 0}
    gen_log = open(out_dir.parent / "generation.jsonl", "a", encoding="utf-8")
    torch.manual_seed(args.seed)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    from pocket_tts.modules.stateful_module import init_states

    for b in range(0, len(todo), args.batch_size):
        chunk = todo[b : b + args.batch_size]
        budgets = [math.ceil((len(c["tokens"]) / 3.0 + 2.0) * FRAME_RATE) for c in chunk]
        outs = model.generate(
            [torch.tensor(c["tokens"], dtype=torch.long) for c in chunk],
            [voice(c["prompt"]) for c in chunk],
            max_frames=max(budgets),
            temp=args.temp,
            n_steps=1,
            cfg_coef=args.cfg,
            eos_threshold=args.eos_threshold,
            eos_countdown=args.frames_after_eos + 1,
        )
        for c, lat, budget in zip(chunk, outs, budgets, strict=True):
            no_eos = lat.shape[0] >= budget
            gen_log.write(json.dumps({"key": c["key"], "tokens": len(c["tokens"]), "frames": int(lat.shape[0]),
                                      "budget": budget, "no_eos": no_eos}, ensure_ascii=False) + "\n")
            if no_eos:
                counts["no_eos"] += 1
                lat = lat[:budget]
            if lat.shape[0] < MIN_FRAMES:
                counts["silent"] += 1
                wav = np.zeros(int(0.1 * mimi.sample_rate), dtype=np.float32)
            else:
                state = init_states(mimi, 1, (lat.shape[0] + MIN_FRAMES) * ratio)
                wav = mimi.decode_from_latent(lat[None].to(device), state)[0, 0].float().cpu().numpy()
            sf.write(str(out_dir / f"{c['key']}.wav"), np.clip(wav, -1, 1), mimi.sample_rate)
    gen_log.close()
    return counts


def run_jkyb_eval(audio_dir: Path, dataset: Path, out_dir: Path) -> dict[str, Any]:
    cmd = [
        UV, "run", "--python", "3.13", "--project", str(JKYB_PROJECT), "--extra", "asr",
        "jkyb-eval", "tts", str(audio_dir), "--dataset", str(dataset),
        "--device", "cuda", "--output-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads((out_dir / "summary.json").read_text())


def run_jvs_scoring(items: list[dict[str, Any]], audio_dir: Path, out_dir: Path) -> dict[str, Any]:
    subprocess.run(
        [UV, "run", "--python", "3.13", "--project", str(JKYB_PROJECT), "--extra", "asr", "python",
         str(REPO / "scripts" / "asr_transcribe.py"), str(audio_dir)],
        check=True, cwd=REPO,
    )
    asr_dir = audio_dir.parent / f"asr_{audio_dir.name}"
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    subprocess.run(
        [UV, "run", "--python", "3.13", "--project", str(JKYB_PROJECT), "--extra", "asr", "python",
         str(REPO / "scripts" / "score_jvs_cer.py"), str(out_dir / "items.json"), str(asr_dir),
         str(out_dir / "cer.json")],
        check=True, cwd=REPO,
    )
    cer = json.loads((out_dir / "cer.json").read_text())
    if all("reference" in it for it in items):
        cer["sim_ecapa"] = speaker_similarity(items, audio_dir)
    return cer


def speaker_similarity(items: list[dict[str, Any]], audio_dir: Path) -> float:
    import torchaudio.functional as AF
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
        savedir=str(REPO / "data/models/spkrec-ecapa-voxceleb"),
    )

    def emb(path: Path | str) -> torch.Tensor:
        wav, sr = sf.read(str(path), dtype="float32")
        x = AF.resample(torch.from_numpy(wav)[None].cuda(), sr, 16000)
        return torch.nn.functional.normalize(enc.encode_batch(x)[0, 0], dim=-1)

    sims = [float(emb(audio_dir / f"{it['key']}.wav") @ emb(it["reference"])) for it in items]
    return float(np.mean(sims))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--set", choices=("jkyb_mini", "jkyb", "aozora", "jvs", "jvs_bal", "jsut"), default="jkyb_mini")
    parser.add_argument("--snapshot", default="latest")
    parser.add_argument("--voice", choices=("rotation", "jvs001"), default="rotation")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--eos-threshold", type=float, default=-4.0)
    parser.add_argument("--frames-after-eos", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--jvs-speakers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N items only (smoke tests)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    snap, step = snapshot_path(args.run_dir, args.snapshot)
    tag = f"{args.set}_{args.voice}_cfg{args.cfg}" + (f"_n{args.jvs_speakers}" if args.set == "jvs" else "")
    if args.temp != 0.3:
        tag += f"_t{args.temp}"
    if args.limit:
        tag += f"_limit{args.limit}"
    out_dir = args.run_dir / "eval" / f"step{step:08d}" / tag
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items, dataset = build_items(args.set, args.voice, args.jvs_speakers)
    if args.limit:
        items = items[: args.limit]
        if dataset is not None:
            keys = {it["key"] for it in items}
            subset = out_dir / "dataset_subset.jsonl"
            with open(dataset) as src, open(subset, "w") as dst:
                for line in src:
                    if json.loads(line)["key"] in keys:
                        dst.write(line)
            dataset = subset

    device = torch.device("cuda")
    model, mimi = load_model(args.run_dir, snap, device)
    counts = generate(model, mimi, items, audio_dir, args)
    del model
    torch.cuda.empty_cache()

    metrics: dict[str, float] = {f"{k}_count": float(v) for k, v in counts.items()}
    if args.set in ("jkyb_mini", "jkyb", "aozora"):
        summary = run_jkyb_eval(audio_dir, dataset, out_dir / "results")
        metrics.update(_jkyb_scalars(summary))
    else:
        metrics.update(run_jvs_scoring(items, audio_dir, out_dir))
    (out_dir / "metrics.json").write_text(json.dumps({"step": step, **metrics}, indent=2))
    print(json.dumps({"step": step, "set": tag, **metrics}, indent=2, ensure_ascii=False))
    if not args.no_wandb:
        from pocket_tts_ja.wandb_log import init_run, log_scalars

        run = init_run(args.run_dir, job_type="eval")
        log_scalars(f"eval/{tag}", step, metrics)
        run.finish()


def _jkyb_scalars(summary: dict[str, Any]) -> dict[str, float]:
    """Headline numbers from jkyb-eval's summary.json, plus per-category accuracy."""
    m = summary["metrics"]
    out = {
        "accuracy": m["accuracy"]["rate"],
        "relaxed_accuracy": m["relaxed_accuracy"]["rate"],
        "target_kana_cer@1": m["target_kana_cer"]["cer_at_1"],
        "sentence_kana_cer@1": m["sentence_kana_cer"]["cer_at_1"],
    }
    if "text_cer" in m:
        out["text_cer@1"] = m["text_cer"]["cer_at_1"]
    for cat, node in summary.get("breakdowns", {}).get("reading_category", {}).items():
        out[f"accuracy_{cat}"] = node["metrics"]["accuracy"]["rate"]
    return out


if __name__ == "__main__":
    main()
