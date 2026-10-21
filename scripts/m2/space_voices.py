"""Space voices check: the preset voices of the HF Space through the released students and decoders.

The Space (takuma104/pocket-tts-ja) clones ten preset voices (voices/, Irodori-TTS voice design). To
find where a voice goes wrong with the CC release, every student (a published config, hf:// or local
yaml, read by the stock TTSModel exactly as the Space does) generates the same sentences for every
voice with the stock inference path (end_on_pause, chunking, temperature from the config, seeded),
the latents are kept, and the same latents are decoded by every decoder given. A difference between
two decoder columns of one student is the decoder alone; a difference between two students under
the same decoder is the language model. The voice prompts themselves also go through the Mimi
encoder and every decoder (the reconstruction), which leaves the language model out entirely.

    uv run python scripts/m2/space_voices.py --out exp/m2/space_voices --voices data/space_voices \
        --student nc6=hf://takuma104/pocket-tts-ja/japanese.yaml \
        --student cc6=hf://takuma104/pocket-tts-ja-cc/japanese.yaml \
        --decoder mimi=stock --decoder keep4k=runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors \
        --decoder M2CH2_soup4=runs/M2H_soups/M2CH2_soup4.safetensors --sentences 20

Output:
    <out>/items.jsonl                      key, voice, gender, text, yomi (the JSUT kana)
    <out>/tts/<student>__<decoder>/<key>.wav
    <out>/recon/orig/<voice>.wav, <out>/recon/<decoder>/<voice>.wav
    <out>/stats.jsonl                      per wav: frames, no_eos, peak before clipping, clipped share, RMS

A decoder spec is NAME=stock (the original Mimi), NAME=<decoder safetensors> (scripts/m2/train_mimi_decoder.py
or a soup), or NAME=<full model safetensors / hf:// path> (the mimi.* decoder tensors of a released model, so
the bf16 copy a release ships can be told from the fp32 decoder it was made from). Existing wavs are kept.
"""

import argparse
import copy
import json
import queue
import threading
from pathlib import Path

import numpy as np
import numpy.typing as npt
import safetensors.torch
import soundfile as sf
import torch
from pocket_tts import TTSModel
from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts.models.tts_model import MAX_TOKEN_PER_CHUNK
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.utils import download_if_necessary

from pocket_tts_ja.codec_ft import DECODER_MODULES, load_decoder
from pocket_tts_ja.mimi_utils import decode, encode, load_mimi, load_pocket_config

REPO = Path(__file__).resolve().parents[2]


def read_voices(voices: Path) -> list[dict[str, str]]:
    spec = json.loads((voices / "voices.json").read_text(encoding="utf-8"))
    return [{"id": v["id"], "gender": "F" if v["id"].startswith("female") else "M", "wav": str(voices / f"{v['id']}.wav")}
            for v in spec["voices"]]


def read_sentences(path: Path, n: int) -> list[dict[str, str]]:
    """key, text, kana per row; a row without kana gets pyopenjtalk's reading (a looser Kana-CER reference)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    out = []
    for r in rows[:n]:
        yomi = r.get("kana")
        if not yomi:
            from pocket_tts_ja.kana import reading

            yomi = reading(r["text"])
        out.append({"key": r["key"], "text": r["text"], "yomi": yomi})
    return out


def build_decoder(spec: str, device: torch.device):  # noqa: ANN201 (MimiModel)
    """A stock Mimi whose decode side is replaced by the spec's (see the module docstring)."""
    mimi = load_mimi(load_pocket_config(), device)
    if spec == "stock":
        return mimi
    path = Path(download_if_necessary(spec)) if spec.startswith("hf://") else Path(spec)
    state = safetensors.torch.load_file(str(path))
    if any(k.startswith("flow_lm.") for k in state):  # a released model: keep its decode-side tensors
        wanted = {f"mimi.{m}.{n}" for m in DECODER_MODULES for n in getattr(mimi, m).state_dict()}
        state = {k: v for k, v in state.items() if k in wanted}
        assert set(state) == wanted, f"{spec}: {len(wanted ^ set(state))} decoder tensors missing"
        tmp = path.parent / f".decoder_{path.stem}.safetensors"
        safetensors.torch.save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
        path = tmp
    load_decoder(mimi, path)
    return mimi


@torch.no_grad()
def generate_latents(model: TTSModel, state, text: str, seed: int) -> list[tuple[torch.Tensor, bool]]:  # noqa: ANN001
    """The stock generate_audio_stream without its decoder thread: (normalized latents [1, T, 32], no_eos) per chunk."""
    chunks = split_into_best_sentences(
        model.flow_lm.conditioner.tokenizer, text, MAX_TOKEN_PER_CHUNK, model.pad_with_spaces_for_short_inputs,
        remove_semicolons=model.remove_semicolons, append_terminal_punctuation=model.append_terminal_punctuation,
        capitalize_first_letter=model.capitalize_first_letter, replace_characters=model.replace_characters,
    )
    torch.manual_seed(seed)
    out: list[tuple[torch.Tensor, bool]] = []
    for chunk in chunks:
        prepared_text, guess = prepare_text_prompt(
            chunk, model.pad_with_spaces_for_short_inputs, model.remove_semicolons, model.append_terminal_punctuation,
            model.capitalize_first_letter, model.replace_characters,
        )
        frames_after_eos = model.model_recommended_frames_after_eos
        if frames_after_eos is None:
            frames_after_eos = guess + 2
        chunk_state = copy.deepcopy(state)
        prepared = model.flow_lm.conditioner.prepare(prepared_text)
        max_gen_len = model._estimate_max_gen_len(prepared.shape[1])
        latents_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        model._generate(chunk_state, prepared, max_gen_len, frames_after_eos, latents_queue, result_queue, threading.Event())
        latents: list[torch.Tensor] = []
        while True:
            latent = latents_queue.get()
            if latent is None:
                break
            latents.append(latent)
        if not result_queue.empty():
            raise result_queue.get()[1]
        out.append((torch.cat(latents, dim=1), len(latents) >= max_gen_len))
    return out


@torch.no_grad()
def decode_chunks(mimi, emb_std: torch.Tensor, emb_mean: torch.Tensor, chunks: list[tuple[torch.Tensor, bool]]) -> npt.NDArray[np.float32]:  # noqa: ANN001
    """What the stock decoder thread makes of the latents: every chunk from a fresh state, 5 ms fade-in."""
    ratio = int(mimi.encoder_frame_rate / mimi.frame_rate)
    fade = torch.linspace(0, 1, mimi.sample_rate // 200)
    wavs = []
    device = next(mimi.parameters()).device
    for latents, _ in chunks:
        x = latents.to(device) * emb_std.to(device) + emb_mean.to(device)
        state = init_states(mimi, batch_size=1, sequence_length=latents.shape[1] * ratio)
        y = mimi.decode_from_latent(x, state)[0, 0].float().cpu()
        n = min(fade.numel(), y.shape[-1])
        y[:n] *= fade[:n]
        wavs.append(y)
    return torch.cat(wavs).numpy()


def stats(wav: npt.NDArray[np.float32]) -> dict[str, float]:
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    return {"peak": peak, "clipped": float(np.mean(np.abs(wav) >= 1.0)) if len(wav) else 0.0,
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-9)) if len(wav) else -180.0}


def write(path: Path, wav: npt.NDArray[np.float32], sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("exp/m2/space_voices"))
    parser.add_argument("--voices", type=Path, default=Path("data/space_voices"))
    parser.add_argument("--student", action="append", default=[], help="NAME=config (hf:// or yaml)")
    parser.add_argument("--decoder", action="append", default=[], help="NAME=stock | decoder file | released model file")
    parser.add_argument("--set", type=Path, default=REPO / "data/eval/jsut_eval_1k.jsonl")
    parser.add_argument("--sentences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-recon", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    voices = read_voices(args.voices)
    sentences = read_sentences(args.set, args.sentences)
    args.out.mkdir(parents=True, exist_ok=True)
    items = [{"key": f"{v['id']}__{s['key']}", "voice": v["id"], "gender": v["gender"], "text": s["text"], "yomi": s["yomi"]}
             for v in voices for s in sentences]
    (args.out / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    decoders = {name: build_decoder(spec, device) for name, spec in (d.split("=", 1) for d in args.decoder)}
    stats_log = open(args.out / "stats.jsonl", "a", encoding="utf-8")

    if not args.no_recon:
        stock = decoders.get("mimi") or load_mimi(load_pocket_config(), device)
        for v in voices:
            wav, sr = sf.read(v["wav"], dtype="float32")
            assert sr == stock.sample_rate and wav.ndim == 1
            write(args.out / "recon" / "orig" / f"{v['id']}.wav", wav, sr)
            latents = encode(stock, wav, device)
            for name, mimi in decoders.items():
                dst = args.out / "recon" / name / f"{v['id']}.wav"
                if dst.exists():
                    continue
                y = decode(mimi, latents, len(wav)).float().cpu().numpy()
                write(dst, y, sr)
                stats_log.write(json.dumps({"dir": f"recon/{name}", "key": v["id"], "frames": int(latents.shape[0]), **stats(y)}) + "\n")
        stats_log.flush()

    for student, config in (s.split("=", 1) for s in args.student):
        pending = [it for it in items
                   if not all((args.out / "tts" / f"{student}__{name}" / f"{it['key']}.wav").exists() for name in decoders)]
        if not pending:
            print(f"{student}: done already")
            continue
        model = TTSModel.load_model(config=config)
        model.to(device)
        emb_std, emb_mean = model.flow_lm.emb_std.cpu(), model.flow_lm.emb_mean.cpu()
        states = {v["id"]: model.get_state_for_audio_prompt(Path(v["wav"])) for v in voices}
        print(f"{student}: {config}, temp {model.temp}, {len(pending)} items", flush=True)
        for i, it in enumerate(pending):
            chunks = generate_latents(model, states[it["voice"]], it["text"], args.seed + i)
            frames = sum(int(c[0].shape[1]) for c in chunks)
            no_eos = any(c[1] for c in chunks)
            for name, mimi in decoders.items():
                y = decode_chunks(mimi, emb_std, emb_mean, chunks)
                write(args.out / "tts" / f"{student}__{name}" / f"{it['key']}.wav", y, mimi.sample_rate)
                stats_log.write(json.dumps({"dir": f"tts/{student}__{name}", "key": it["key"], "frames": frames,
                                            "no_eos": no_eos, "chunks": len(chunks), **stats(y)}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(pending)}", flush=True)
                stats_log.flush()
        del model, states
        torch.cuda.empty_cache()
    stats_log.close()


if __name__ == "__main__":
    main()
