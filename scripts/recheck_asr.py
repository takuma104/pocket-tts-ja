"""Re-transcribe training clips from their Mimi latents with kana-whisper (JKYB's kana ASR).

Short Emilia clips often contain speech their transcript lacks: the phrase said twice in
language-lesson videos, or the start of the next utterance. Trained on them, the model keeps
talking after the text ends. This decodes each clip exactly as the loader feeds it (latents up
to speech_end + 3 frames) and writes kana transcripts for `filter_asr_recheck.py`.

    uv run python scripts/recheck_asr.py data/emilia_ja/train.jsonl --out data/emilia_ja/asr_recheck.jsonl

Resumable: ids already in --out are skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soxr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocket_tts.modules.stateful_module import init_states  # noqa: E402

from pocket_tts_ja.mimi_utils import DECODE_CONTEXT_FRAMES, load_mimi, load_pocket_config  # noqa: E402

TRAIL_FRAMES = 3  # loader keeps speech_end + 3 frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="sbintuitions/kana-whisper")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--select", default=None, help="only ids listed in this file (one per line)")
    args = parser.parse_args()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = torch.device("cuda")
    rows = [json.loads(line) for line in open(args.manifest)]
    if args.select:
        keep = set(Path(args.select).read_text().split())
        rows = [r for r in rows if r["id"] in keep]
    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in open(args.out)}
    rows = [r for r in rows if r["id"] not in done]
    if args.limit is not None:
        rows = rows[: args.limit]
    for r in rows:
        r["_frames"] = min(int(r.get("speech_end", r["lat_len"])) + TRAIL_FRAMES, int(r["lat_len"]))
    rows.sort(key=lambda r: r["_frames"])
    print(f"{len(rows)} clips to transcribe ({len(done)} already done)", flush=True)

    mimi = load_mimi(load_pocket_config(), device)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    hop = int(round(mimi.sample_rate / mimi.frame_rate))  # samples per latent frame
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, dtype=torch.float16, low_cpu_mem_usage=True,
                                                      use_safetensors=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, dtype=torch.float16, device=device,
                   batch_size=args.batch_size,
                   generate_kwargs={"language": "ja", "task": "transcribe", "max_new_tokens": args.max_new_tokens})
    sr16 = int(processor.feature_extractor.sampling_rate)

    base = args.manifest.parent
    mm: dict[str, np.ndarray] = {}
    out = open(args.out, "a", encoding="utf-8")
    t0, n = time.time(), 0
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b : b + args.batch_size]
        T = max(r["_frames"] for r in batch)
        lat = np.zeros((len(batch), T, 32), dtype=np.float32)
        for k, r in enumerate(batch):
            arr = mm.setdefault(r["lat_file"], np.load(base / r["lat_file"], mmap_mode="r"))
            lat[k, : r["_frames"]] = arr[r["lat_start"] : r["lat_start"] + r["_frames"]]
        with torch.no_grad():
            state = init_states(mimi, len(batch), (T + DECODE_CONTEXT_FRAMES) * ratio)
            wav = mimi.decode_from_latent(torch.from_numpy(lat).to(device), state)[:, 0].float().cpu().numpy()
        inputs = [{"raw": soxr.resample(wav[k, : r["_frames"] * hop], mimi.sample_rate, sr16), "sampling_rate": sr16}
                  for k, r in enumerate(batch)]
        for r, res in zip(batch, asr(inputs), strict=True):
            out.write(json.dumps({"id": r["id"], "kana": res["text"].strip()}, ensure_ascii=False) + "\n")
        n += len(batch)
        if (b // args.batch_size) % 100 == 0:
            out.flush()
            el = time.time() - t0
            print(f"{n}/{len(rows)} {n / el:.1f} clips/s eta {(len(rows) - n) / max(n / el, 1e-9) / 3600:.2f}h", flush=True)
    out.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
