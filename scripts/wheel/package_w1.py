"""W1: build the files of a DACVAE student for the stock pocket-tts wheel (>= 4.0; plan P4).

The model config's mimi section is the stock-wheel Mimi at 25 Hz x 32 dims
(pocket_tts_ja.wheel_codec); its decode side comes from a W1 decoder (scripts/wheel/
train_decoder.py, `mimi.quantizer / upsample / decoder_transformer / decoder`), its encode side
from a W1 encoder (scripts/wheel/train_encoder.py, `mimi.encoder / encoder_transformer /
downsample`), and the FlowLM from a student snapshot (EMA, `flow_lm.*`, CFG baked in: the wheel
runs without CFG). The decoder and the encoder must share the delay K they were trained with:
the wheel's audio then starts K frames late, and the frames after EOS (12, as the students were
trained and evaluated) cover the end.

Writes into --out:
  <name>.yaml               model config for `pocket-tts generate --config`
  <name>/model.safetensors  FlowLM + Mimi, as the stock loader reads it (--dtype of every tensor)
  tokenizer.model           the SentencePiece model

With --repo the yaml points at hf://<repo>/... (add --revision once the weights are committed);
without it, at the local files, to test with the wheel first:

    uv run python scripts/wheel/package_w1.py configs/model/dv_6l_4k.yaml \
        runs/irodori_distill/P3d_6l/snapshots/step_00020000.safetensors \
        --decoder runs/irodori_distill/W1_main_d2/decoder_step_00150000.safetensors \
        --encoder runs/irodori_distill/W1e_d2/encoder_step_00020000.safetensors \
        --name japanese_dv --out exp/irodori_distill/package/p3d_20k
    uvx --from pocket-tts==4.0.1 pocket-tts generate --config exp/irodori_distill/package/p3d_20k/japanese_dv.yaml \
        --voice prompt.wav --text "こんにちは。"

The result is loaded once with the stock loader (strict) as a check.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import safetensors.torch
import torch
import yaml

from pocket_tts.models.tts_model import TTSModel

DECODE_SIDE = ("quantizer", "upsample", "decoder_transformer", "decoder")
ENCODE_SIDE = ("encoder", "encoder_transformer", "downsample")
HEADER = """# Japanese Pocket TTS ({name}): {layers}-layer FlowLM on DACVAE latents (25 Hz), {params:.1f}M parameters with
# its Mimi. Distilled from Irodori-TTS; classifier-free guidance is baked in. Use it with the stock
# pocket-tts (>= 4.0):
#   pocket-tts generate --config {config_ref} --voice prompt.wav --text "こんにちは。"
# Weights: {license}.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def side(path: Path, modules: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """The `mimi.<module>.*` tensors of a W1 file; every one must belong to `modules`."""
    state = safetensors.torch.load_file(str(path))
    assert state, path
    stray = sorted({k.split(".")[1] for k in state} - set(modules))
    assert all(k.startswith("mimi.") for k in state) and not stray, f"{path}: {stray}"
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot", type=Path, help="student EMA snapshot (flow_lm.*)")
    parser.add_argument("--decoder", type=Path, required=True, help="W1 decoder (decoder_step_*.safetensors)")
    parser.add_argument("--encoder", type=Path, required=True, help="W1 encoder (encoder_step_*.safetensors)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=None, help="Hugging Face repo id the files will live in")
    parser.add_argument("--revision", default=None, help="commit of --repo to pin the files to")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--temp", type=float, default=0.7, help="default_temperature")
    parser.add_argument("--frames-after-eos", type=int, default=12, help="model_recommended_frames_after_eos")
    parser.add_argument("--license", default="not for distribution (test package)", help="license line of the yaml header")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.model_config.read_text())
    flow = {k: v for k, v in safetensors.torch.load_file(str(args.snapshot)).items()}
    assert flow and all(k.startswith("flow_lm.") for k in flow), args.snapshot
    state = {**flow, **side(args.decoder, DECODE_SIDE), **side(args.encoder, ENCODE_SIDE)}
    dtype = getattr(torch, args.dtype)
    state = {k: v.to(dtype).contiguous() for k, v in state.items()}
    weights_dir = args.out / args.name
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = weights_dir / "model.safetensors"
    safetensors.torch.save_file(state, str(weights))
    tokenizer = args.out / "tokenizer.model"
    shutil.copyfile(cfg["flow_lm"]["lookup_table"]["tokenizer_path"], tokenizer)

    def ref(path: Path, pin: bool = True) -> str:
        if args.repo is None:
            return str(path.resolve())
        rev = f"@{args.revision}" if args.revision and pin else ""
        return f"hf://{args.repo}/{path.relative_to(args.out)}{rev}"

    cfg["weights_path"] = ref(weights)
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = ref(tokenizer)
    cfg["flow_lm"]["dtype"] = cfg["mimi"]["dtype"] = args.dtype
    cfg["default_temperature"] = args.temp
    cfg["model_recommended_frames_after_eos"] = args.frames_after_eos
    config_path = args.out / f"{args.name}.yaml"
    header = HEADER.format(
        name=args.name,
        layers=cfg["flow_lm"]["transformer"]["num_layers"],
        params=sum(v.numel() for v in state.values()) / 1e6,
        config_ref=ref(config_path, pin=False) if args.repo else f"{args.name}.yaml",
        license=args.license,
    )
    config_path.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    if args.repo is None:
        model = TTSModel.load_model(config=str(config_path))
        loaded = model.state_dict()
        assert set(loaded) == set(state), sorted(set(loaded) ^ set(state))[:10]
        print(f"stock loader: {len(loaded)} tensors, temp {model.temp}, voice cloning {model.has_voice_cloning}")
    print(f"{config_path}")
    print(f"{weights}  {weights.stat().st_size / 1e6:.1f} MB  sha256 {sha256(weights)[:16]}")
    print(f"{tokenizer}  sha256 {sha256(tokenizer)[:16]}")


if __name__ == "__main__":
    main()
