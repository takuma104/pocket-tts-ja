"""Build the release files of one model for the stock pocket-tts wheel (plan D3).

Writes into --out:
  <name>.yaml               model config for `pocket-tts generate --config`
  <name>/model.safetensors  FlowLM (the EMA snapshot) + Mimi, as the stock loader reads it. Mimi
                            is the base file's (bfloat16, like every official file); --dtype
                            bfloat16 also stores the FlowLM that way (the official 6-layer
                            files do, halving the download; the 24-layer ones are float32)
  tokenizer.model           the SentencePiece model (shared by the 6-layer and 24-layer models)

--mimi-decoder replaces the decode side of Mimi with a fine-tuned one (plan M2); the encoder, and
with it the latents and voice prompts, stay the base file's.

With --repo the yaml points at hf://<repo>/... (add --revision once the weights are committed, to
pin them to that commit; the yaml goes into a later one); without it, at the local files, to test
with the wheel before the upload:

    uv run python scripts/package_release.py configs/model/ja_6l_4k.yaml \
        runs/D1_6l/snapshots/step_00200000.safetensors --name japanese --out release
    uvx pocket-tts generate --config release/japanese.yaml --voice prompt.wav --text "こんにちは。"

The result is loaded once with the stock loader (strict) as a check.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import safetensors.torch
import torch
import yaml
from merge_snapshot_weights import merged_state
from pocket_tts.models.tts_model import TTSModel

HEADER = """# Japanese Pocket TTS ({name}): {layers}-layer FlowLM, {params:.1f}M parameters with Mimi.
# Classifier-free guidance is baked in; use it with the stock pocket-tts (>= 3.3.0):
#   pocket-tts generate --config {config_ref} --voice prompt.wav --text "こんにちは。"
# Weights: {license}.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--name", required=True, help="japanese / japanese_24l")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=None, help="Hugging Face repo id the files will live in")
    parser.add_argument("--revision", default=None, help="commit of --repo to pin the files to")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="of the FlowLM")
    parser.add_argument("--temp", type=float, default=None, help="default_temperature (config value if omitted)")
    parser.add_argument("--frames-after-eos", type=int, default=None, help="model_recommended_frames_after_eos")
    parser.add_argument("--license", default="CC BY-NC 4.0 (trained on Emilia)",
                        help="license line of the yaml header (the CC BY line: its own)")
    parser.add_argument("--mimi-decoder", type=Path, default=None,
                        help="fine-tuned decode side of Mimi (scripts/m2/train_mimi_decoder.py) stored in place "
                             "of the base file's")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.model_config.read_text())
    weights_dir = args.out / args.name
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = weights_dir / "model.safetensors"
    state, _ = merged_state(args.model_config, args.snapshot)
    assert all(v.dtype == torch.float32 for k, v in state.items() if k.startswith("flow_lm."))
    if args.mimi_decoder is not None:
        decoder = safetensors.torch.load_file(str(args.mimi_decoder))
        assert decoder and set(decoder) <= set(state), f"{args.mimi_decoder}: names not in the model"
        for name, tensor in decoder.items():
            assert tensor.shape == state[name].shape, name
            state[name] = tensor.to(state[name].dtype)  # bfloat16, like the rest of Mimi in the file
    if args.dtype == "bfloat16":
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
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
    if args.temp is not None:
        cfg["default_temperature"] = args.temp
    if args.frames_after_eos is not None:
        cfg["model_recommended_frames_after_eos"] = args.frames_after_eos
    config_path = args.out / f"{args.name}.yaml"
    header = HEADER.format(
        name=args.name,
        layers=cfg["flow_lm"]["transformer"]["num_layers"],
        params=sum(v.numel() for v in state.values()) / 1e6,
        config_ref=ref(config_path, pin=False) if args.repo else f"{args.name}.yaml",  # not in that commit
        license=args.license,
    )
    config_path.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    if args.repo is None:
        model = TTSModel.load_model(config=str(config_path))
        loaded = model.state_dict()
        assert set(loaded) == set(state), set(loaded) ^ set(state)
        print(f"stock loader: {len(loaded)} tensors, temp {model.temp}, voice cloning {model.has_voice_cloning}")
    print(f"{config_path}")
    print(f"{weights}  {weights.stat().st_size / 1e6:.1f} MB  sha256 {sha256(weights)[:16]}")
    print(f"{tokenizer}  sha256 {sha256(tokenizer)[:16]}")


if __name__ == "__main__":
    main()
