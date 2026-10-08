"""S3 (post): QC the synthesized corpus and turn the accepted clips into training rows.

Run on the machine holding the audio (galleria2), from the repo root, after
scripts/synth/irodori_synth.py has written data/synth/s3/audio/<id>.wav:

  1. kana-whisper transcripts (scripts/asr_transcribe.py --skip-text, JKYB environment)
  2. QC, in the JKYB environment (this script with --qc): Sentence Kana-CER of each clip
     against the row's yomi, and target-reading exactness via `jkyb-eval g2p` on the tagged
     rows (extra targets of merged sentences are re-keyed `<id>__t<k>`). Accept: Sentence
     Kana-CER <= --max-sentence-cer and every target read correctly.
  3. Mimi latents of the accepted clips + speech end (this script with --encode, root env)
     -> data/synth/s3/latents/part_<k>.npy and data/synth/s3/manifest.jsonl with
     speaker = "syn:<reference speaker>" and the reference's real Emilia utterance id.

    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/asr_transcribe.py data/synth/s3/audio --skip-text
    uv run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
        python scripts/synth/finalize_s3.py --qc
    uv run python scripts/synth/finalize_s3.py --encode
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("data/synth")
S3 = ROOT / "s3"
JKYB = Path("third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def qc(max_sentence_cer: float):
    from jkyb_eval.metrics import character_error_rate
    from jkyb_eval.normalization import canonicalize_yomi

    kana = {r["key"]: r["yomi"] for r in read_jsonl(S3 / "asr_audio" / "kana.jsonl")}
    texts = {r["id"]: r for r in read_jsonl(ROOT / "texts.jsonl")}
    # target exactness through the benchmark's own aligner
    preds = S3 / "qc_g2p_predictions.jsonl"
    with open(preds, "w", encoding="utf-8") as f:
        for r in read_jsonl(ROOT / "texts_tagged_jkyb.jsonl"):
            if r["key"] in kana:
                f.write(json.dumps({"key": r["key"], "yomi": kana[r["key"]]}, ensure_ascii=False) + "\n")
        for r in read_jsonl(ROOT / "texts_tagged_jkyb_extra.jsonl"):
            base = r["key"].split("__t")[0]
            if base in kana:
                f.write(json.dumps({"key": r["key"], "yomi": kana[base]}, ensure_ascii=False) + "\n")
    tagged = S3 / "qc_tagged_all.jsonl"
    with open(tagged, "w", encoding="utf-8") as f:
        for name in ("texts_tagged_jkyb.jsonl", "texts_tagged_jkyb_extra.jsonl"):
            for r in read_jsonl(ROOT / name):
                base = r["key"].split("__t")[0]
                if base in kana:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subprocess.run(
        [str(Path(sys.executable).parent / "jkyb-eval"), "g2p", str(preds), "--dataset", str(tagged),
         "--output-dir", str(S3 / "qc_g2p")],
        check=True,
    )
    exact: dict[str, bool] = {}
    for r in read_jsonl(S3 / "qc_g2p" / "details" / "all.jsonl"):
        base = r["key"].split("__t")[0]
        exact[base] = exact.get(base, True) and bool(r["target_exact"])
    stats = {"clips": 0, "accepted": 0, "bad_sentence_cer": 0, "bad_target": 0, "no_yomi": 0}
    with open(S3 / "qc.jsonl", "w", encoding="utf-8") as f:
        for key, hyp in kana.items():
            t = texts.get(key)
            if t is None:
                continue
            stats["clips"] += 1
            if t.get("yomi"):
                cer = character_error_rate(canonicalize_yomi(t["yomi"]), canonicalize_yomi(hyp))
            else:
                cer = None
                stats["no_yomi"] += 1
            ok = cer is not None and cer <= max_sentence_cer
            if cer is not None and not ok:
                stats["bad_sentence_cer"] += 1
            if key in exact and not exact[key]:
                stats["bad_target"] += 1
                ok = False
            stats["accepted"] += int(ok)
            f.write(json.dumps({"id": key, "sentence_kana_cer": cer, "target_exact": exact.get(key),
                                "accepted": ok}, ensure_ascii=False) + "\n")
    (S3 / "qc_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def encode(part_size: int):
    import numpy as np
    import soundfile as sf
    import torch

    from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config
    from pocket_tts_ja.text import normalize

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from precompute_emilia_ja import FRAME, speech_end_frames

    accepted = [r["id"] for r in read_jsonl(S3 / "qc.jsonl") if r["accepted"]]
    inputs = {r["id"]: r for r in read_jsonl(ROOT / "s3_input.jsonl")}
    texts = {r["id"]: r for r in read_jsonl(ROOT / "texts.jsonl")}
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    mimi = load_mimi(load_pocket_config(), device)
    lat_dir = S3 / "latents"
    lat_dir.mkdir(exist_ok=True)
    out = open(S3 / "manifest.jsonl", "w", encoding="utf-8")
    for p in range(0, len(accepted), part_size):
        ids = accepted[p : p + part_size]
        lats, rows, n = [], [], 0
        order = sorted(ids, key=lambda i: Path(S3 / "audio" / f"{i}.wav").stat().st_size)
        for b in range(0, len(order), 32):
            chunk = order[b : b + 32]
            wavs = [sf.read(str(S3 / "audio" / f"{i}.wav"), dtype="float32")[0] for i in chunk]
            T = max(len(w) for w in wavs)
            T += (-T) % FRAME
            x = torch.zeros(len(wavs), 1, T)
            for k, w in enumerate(wavs):
                x[k, 0, : len(w)] = torch.from_numpy(w)
            with torch.no_grad():
                z = mimi.encode_to_latent(x.to(device))
            for k, (i, w) in enumerate(zip(chunk, wavs, strict=True)):
                lat = z[k, : int(np.ceil(len(w) / FRAME))].half().cpu().numpy()
                lats.append(lat)
                inp = inputs[i]
                rows.append({
                    "id": i, "transcript": normalize(texts[i]["text"]),
                    "speaker": f"syn:{inp['ref_speaker']}", "ref_id": inp["ref_id"],
                    "source_type": "irodori", "duration": round(len(w) / 24000, 3),
                    "lat_file": f"synth/s3/latents/part_{p // part_size:03d}.npy",
                    "lat_start": n, "lat_len": int(lat.shape[0]),
                    "speech_end": min(speech_end_frames(w), int(lat.shape[0])),
                })
                n += int(lat.shape[0])
        np.save(lat_dir / f"part_{p // part_size:03d}.npy", np.concatenate(lats))
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"part {p // part_size}: {len(rows)} clips", flush=True)
    out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--max-sentence-cer", type=float, default=0.05)
    parser.add_argument("--part-size", type=int, default=20000)
    args = parser.parse_args()
    if args.qc:
        qc(args.max_sentence_cer)
    if args.encode:
        encode(args.part_size)


if __name__ == "__main__":
    main()
