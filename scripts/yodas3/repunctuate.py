"""Y2b: add the punctuation that most YODAS3 labels lack.

The labels are Whisper transcripts made without a prompt, and 72% of them carry no punctuation
at all (Emilia: 2-7%; commas per 100 characters 0.9 against 4-5). The model is told where to
pause and where a sentence ends by "," and ".", so two thirds of the natural speech would teach
it pauses without them.

No audio is read again: every kept utterance is decoded from its Mimi latents, transcribed
once more by whisper-large-v3-turbo, now with a punctuated prompt, and only the punctuation of
that second transcript is carried over, at the places where its characters match the label.
The words stay those heard in the original audio. Labels that already have a comma or a
sentence end inside are kept; a missing final mark is added when the second transcript ends
with one.

Writes <root>/{train,valid}_punct.jsonl (same rows, `transcript` replaced, the old one under
`transcript_raw` when it changed) and resumes from <root>/repunct.jsonl (id, second transcript).

    uv run python scripts/yodas3/repunctuate.py --root data/yodas3_kept
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF
from pocket_tts.modules.stateful_module import init_states
from rapidfuzz.distance import Levenshtein
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pocket_tts_ja.mimi_utils import DECODE_CONTEXT_FRAMES, load_mimi, load_pocket_config
from pocket_tts_ja.text import normalize

MARKS = ",.?!"
PROMPT = "こんにちは。今日は、いい天気ですね。それでは、始めましょうか。"


def strip_marks(text: str) -> tuple[str, list[str]]:
    """The text without punctuation and spaces, and for each kept character the marks after it
    (index 0: marks before the first character)."""
    chars: list[str] = []
    after: list[str] = [""]
    for c in text:
        if c in MARKS:
            after[-1] += c
        elif not c.isspace():
            chars.append(c)
            after.append("")
    return "".join(chars), after


def transfer(label: str, second: str) -> tuple[str, str]:
    """`label` with the punctuation of `second`; the second value says what was done."""
    body, own = strip_marks(label)
    other, marks = strip_marks(second)
    if not body or not other:
        return label, "kept"
    inner = any(own[1:-1]) or bool(own[0])
    if inner:
        if not own[-1] and marks[-1] and body[-1] == other[-1]:
            return label + marks[-1][0], "final"
        return label, "kept"
    add = [""] * (len(body) + 1)
    for op, a0, a1, b0, _ in Levenshtein.opcodes(body, other).as_list():
        if op != "equal":
            continue
        for k in range(a1 - a0):
            if marks[b0 + k + 1]:  # marks after other[b0 + k], which equals body[a0 + k]
                add[a0 + k + 1] = marks[b0 + k + 1][0]
    add[-1] = own[-1] or add[-1]
    if not any(add):
        return label, "none"
    last = len(body) - 1  # like the Emilia transcripts: a space after a sentence end inside the text
    out = "".join(c + add[i + 1] + (" " if add[i + 1] in tuple(".?!") and i < last else "") for i, c in enumerate(body))
    return out, "added"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/yodas3_kept"))
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--batch-seconds", type=float, default=600.0, help="padded audio per batch")
    parser.add_argument("--limit", type=int, default=0, help="first N utterances only (a test; writes no manifests)")
    args = parser.parse_args()

    splits: dict[str, list[dict[str, Any]]] = {
        split: [json.loads(line) for line in open(args.root / f"{split}.jsonl", encoding="utf-8")]
        for split in ("train", "valid")
    }
    rows = [r for part in splits.values() for r in part]
    if args.limit:
        rows = rows[: args.limit]
    cache = args.root / "repunct.jsonl"
    second: dict[str, str] = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            d = json.loads(line)
            second[d["id"]] = d["text"]
    todo = sorted((r for r in rows if r["id"] not in second), key=lambda r: r["lat_len"])
    print(f"{len(rows)} utterances, {len(todo)} to transcribe", flush=True)

    if todo:
        device = torch.device("cuda")
        mimi = load_mimi(load_pocket_config(), device)
        ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
        frame = round(mimi.sample_rate / mimi.frame_rate)
        proc = WhisperProcessor.from_pretrained(args.asr_model)
        asr = WhisperForConditionalGeneration.from_pretrained(
            args.asr_model, dtype=torch.float16, attn_implementation="sdpa"
        ).to(device).eval()
        prompt_ids = proc.get_prompt_ids(PROMPT, return_tensors="pt").to(device)
        maps: dict[str, Any] = {}
        done_sec = 0.0
        batches = 0
        with open(cache, "a", encoding="utf-8") as log, torch.no_grad():
            start = 0
            while start < len(todo):
                end = start + 1
                while end < len(todo) and (end + 1 - start) * todo[end]["lat_len"] * 0.08 <= args.batch_seconds:
                    end += 1
                batch = todo[start:end]
                start = end
                frames = batch[-1]["lat_len"]
                lat = torch.zeros(len(batch), frames, 32)
                for i, r in enumerate(batch):
                    src = maps.setdefault(r["lat_file"], np.load(args.root / r["lat_file"], mmap_mode="r"))
                    piece = np.asarray(src[r["lat_start"] : r["lat_start"] + r["lat_len"]], dtype=np.float32)
                    lat[i, : r["lat_len"]] = torch.from_numpy(piece)
                state = init_states(mimi, len(batch), (frames + DECODE_CONTEXT_FRAMES) * ratio)
                wav = mimi.decode_from_latent(lat.to(device), state)[:, 0]
                wav16 = AF.resample(wav, mimi.sample_rate, 16000)
                clips = [wav16[i, : r["lat_len"] * frame * 2 // 3].cpu().numpy() for i, r in enumerate(batch)]
                feats = proc.feature_extractor(
                    clips, sampling_rate=16000, return_tensors="pt", device=str(device)
                ).input_features.to(device, torch.float16)
                ids = asr.generate(feats, language="ja", task="transcribe", max_new_tokens=220, prompt_ids=prompt_ids)
                for r, text in zip(batch, proc.batch_decode(ids, skip_special_tokens=True), strict=True):
                    text = text.strip().removeprefix(PROMPT).strip()
                    second[r["id"]] = text
                    log.write(json.dumps({"id": r["id"], "text": text}, ensure_ascii=False) + "\n")
                done_sec += sum(r["duration"] for r in batch)
                batches += 1
                if batches % 50 == 0:
                    print(f"{start}/{len(todo)} utterances, {done_sec / 3600:.1f} h", flush=True)

    counts: Counter[str] = Counter()
    chars = commas = unpunctuated = 0
    for r in rows:
        new, what = transfer(r["transcript"], normalize(second[r["id"]]))
        counts[what] += 1
        if new != r["transcript"]:
            r["transcript_raw"], r["transcript"] = r["transcript"], new
        chars += len(r["transcript"])
        commas += r["transcript"].count(",")
        unpunctuated += not any(c in r["transcript"] for c in MARKS)
    print(dict(counts))
    print(f"commas per 100 characters {100 * commas / chars:.2f}, no punctuation at all {100 * unpunctuated / len(rows):.1f}%")
    if args.limit:
        for r in rows[:12]:
            print(r.get("transcript_raw", "(unchanged)"), "->", r["transcript"], "| second:", second[r["id"]])
        return
    for split, part in splits.items():
        with open(args.root / f"{split}_punct.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
