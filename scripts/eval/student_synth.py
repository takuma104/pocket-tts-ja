"""P2: a DACVAE student on an evaluation set (scripts/eval/make_eval_dir.py layout).

Each item's voice prompt is encoded as the teacher's references were (DACVAE at -16 dB,
scripts/teacher/encode_refs.py) and its text normalised as main's data was
(pocket_tts_ja.text.normalize, then a final period when the sentence has no final punctuation,
like the stock wheel). Generation is upstream's batched `TrainableTTS.generate`: temperature 0.3,
one LSD step, no CFG, EOS at logit > -4 followed by --eos-countdown frames (the trained tail is
12). The latents are decoded with DACVAE and written at 24 kHz (soxr HQ), like the teacher's
outputs of P0c, so both are scored by the same scripts.

--prompt-encoder FILE (W1, scripts/wheel/train_encoder.py's encoder_step_*.safetensors): the
prompt goes the wheel's way instead, through that stock-wheel Mimi encoder: 24 kHz, the level
as recorded, end_on_pause, the encoder's latents as they come (with --delay K the first K
frames are its silence and the last K of the speech are missing).

    uv run python scripts/eval/student_synth.py exp/irodori_distill/eval/jsut \
        runs/irodori_distill/P2_6l/snapshots/step_00020000.safetensors --name P2_20k
    -> <eval dir>/<name>/<key>.wav, <name>_latents/<key>.npy, <name>.jsonl (frames, eos)

--noise-rho R (a probe, not the wheel's sampling): the flow head's noise of each frame is
R x the previous frame's + sqrt(1 - R^2) x a fresh draw, so every frame's noise is still
N(0, temp) but it moves slowly over the utterance instead of jumping at every frame.
"""

import argparse
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import soxr
import torch

from pocket_tts.data.audio_utils import end_on_pause
from pocket_tts.models.mimi import MimiModel
from pocket_tts.models.tts_model import TTSModel
from pocket_tts_ja.dacvae import DACVAE, REF_LOUDNESS_DB, SAMPLE_RATE as DV_SR, load_wav
from pocket_tts_ja.eval_sets import item_seed
from pocket_tts_ja.text import normalize
from pocket_tts_ja.wheel_codec import build_dv_mimi
from training.args import TrainArgs
from training.modules.builders import load_model_config
from training.modules.model import TrainableTTS
from training.modules.samplers import build_flow

OUT_SR = 24000


def load_student(config_path: Path, weights: Path, device: torch.device) -> TrainableTTS:
    config = load_model_config(str(config_path), {})
    flow_lm = TTSModel._from_pydantic_config(
        config, temp=0.3, sampler_decode_steps=1, noise_clamp=None, eos_threshold=-4.0, origin=None
    ).flow_lm
    flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros((config.flow_lm.transformer.d_model, config.mimi.quantizer.dimension))
    )
    state = safetensors.torch.load_file(str(weights))
    flow_lm.load_state_dict({k.removeprefix("flow_lm."): v for k, v in state.items()}, strict=True)
    model = TrainableTTS(flow_lm, build_flow("lsd"), TrainArgs())
    return model.to(device).eval()


def load_wheel_encoder(weights: Path, device: torch.device) -> MimiModel:
    """The stock-wheel Mimi with a W1 encoder (`mimi.*` names of train_encoder.py)."""
    mimi = build_dv_mimi(init_stock=False)
    state = {k.removeprefix("mimi."): v for k, v in safetensors.torch.load_file(str(weights)).items()}
    missing = [k for k in state if k not in mimi.state_dict()]
    assert not missing, f"not in the Mimi: {missing[:5]}"
    mimi.load_state_dict(state, strict=False)
    return mimi.to(device).eval()


def wheel_prompt(mimi: MimiModel, path: Path, device: torch.device) -> torch.Tensor:
    """[frames, 32]: the prompt as the wheel encodes it (TTSModel.get_state_for_audio_prompt)."""
    x = end_on_pause(torch.from_numpy(load_wav(path, OUT_SR))[None], OUT_SR)
    with torch.no_grad():
        return mimi.encode_to_latent(x[None].to(device))[0].float()


def model_text(text: str) -> str:
    t = normalize(text)
    return t if t.endswith((".", "!", "?", "…")) else t + "."


@contextmanager
def correlated_noise(rho: float, ldim: int) -> Iterator[None]:
    """AR(1) frame noise for one generate() call: it draws torch.randn(B, ldim) once per frame."""
    randn: Callable[..., torch.Tensor] = torch.randn
    prev: list[torch.Tensor] = []

    def ar1(*size: int, **kwargs: object) -> torch.Tensor:
        x = randn(*size, **kwargs)
        if len(size) == 2 and size[1] == ldim:
            if prev and prev[0].shape == x.shape:
                x = rho * prev[0] + (1 - rho**2) ** 0.5 * x
            prev[:] = [x]
        return x

    torch.randn = ar1
    try:
        yield
    finally:
        torch.randn = randn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("weights", type=Path, help="flow_lm.* safetensors (a run's snapshots/)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/dv_6l_4k.yaml"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=1, help="LSD decode steps of the flow head per frame")
    parser.add_argument("--noise-rho", type=float, default=0.0, help="AR(1) correlation of the frame noise (probe)")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--eos-threshold", type=float, default=-4.0)
    parser.add_argument("--eos-countdown", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (smoke tests)")
    parser.add_argument("--prompt-encoder", type=Path, default=None, help="W1 encoder: prompts the wheel's way")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_student(args.config, args.weights, device)
    tokenizer = model.flow_lm.conditioner.tokenizer
    codec = DACVAE.load(str(device))
    wheel_enc = load_wheel_encoder(args.prompt_encoder, device) if args.prompt_encoder else None
    items = [json.loads(line) for line in open(args.eval_dir / "items.jsonl", encoding="utf-8")][: args.limit]
    out = args.eval_dir / args.name
    lat_dir = args.eval_dir / f"{args.name}_latents"
    out.mkdir(exist_ok=True)
    lat_dir.mkdir(exist_ok=True)
    todo = [it for it in items if not (out / f"{it['key']}.wav").exists()]
    texts = {it["key"]: model_text(it["text"]) for it in todo}
    todo.sort(key=lambda it: len(texts[it["key"]]))
    prompts: dict[str, torch.Tensor] = {}
    log = open(args.eval_dir / f"{args.name}.jsonl", "a", encoding="utf-8")
    t0, audio_s = time.time(), 0.0
    for b in range(0, len(todo), args.batch_size):
        batch = todo[b : b + args.batch_size]
        for it in batch:
            if it["prompt"] not in prompts and wheel_enc is not None:
                prompts[it["prompt"]] = wheel_prompt(wheel_enc, Path(it["prompt"]), device)
            elif it["prompt"] not in prompts:
                prompts[it["prompt"]] = codec.encode_file(Path(it["prompt"]), loudness_db=REF_LOUDNESS_DB).to(device)
        torch.manual_seed(item_seed(batch[0]["key"]))
        tokens = [torch.tensor(tokenizer.encode(texts[it["key"]]), dtype=torch.long) for it in batch]
        with torch.no_grad(), correlated_noise(args.noise_rho, model.ldim):
            lats = model.generate(
                tokens, [prompts[it["prompt"]] for it in batch], max_frames=args.max_frames, temp=args.temp, n_steps=args.steps,
                cfg_coef=args.cfg, eos_threshold=args.eos_threshold, eos_countdown=args.eos_countdown,
            )
        for it, z in zip(batch, lats, strict=True):
            n = int(z.shape[0])
            np.save(lat_dir / f"{it['key']}.npy", z.cpu().numpy().astype(np.float16))
            y = codec.decode(z[None].float())[0].cpu().numpy() if n > 0 else np.zeros(DV_SR // 10, dtype=np.float32)
            y = soxr.resample(y, DV_SR, OUT_SR, quality="HQ")
            sf.write(str(out / f"{it['key']}.wav"), np.clip(y, -1, 1), OUT_SR, subtype="PCM_16")
            audio_s += len(y) / OUT_SR
            log.write(json.dumps({"key": it["key"], "text": texts[it["key"]], "frames": n,
                                  "eos": n < args.max_frames}, ensure_ascii=False) + "\n")
        log.flush()
    print(f"{args.eval_dir.name}/{args.name}: {len(todo)} items, {audio_s:.0f} s of audio in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
