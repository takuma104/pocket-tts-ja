# pocket-tts-ja

Japanese models for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai), fine-tuned
from the English 24-layer teacher with the official training code.

- Runs with the **stock `pocket-tts` wheel** (3.3.0 or later), on CPU, with no code changes: the
  release is a config file, a SentencePiece model and a `safetensors` file.
- Takes **plain Japanese text** (kanji and kana as written). There is no G2P front end; the
  model reads the kanji itself.
- **Zero-shot voice cloning** from a few seconds of reference audio, as in the original.
- Two models: `japanese` (6 layers, 110M parameters with the codec, 219 MB) and `japanese_24l`
  (24 layers, 336M parameters, 1.3 GB, the teacher with classifier-free guidance baked in).

Weights: [takuma104/pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja) (CC BY-NC 4.0).
The experiment plan and the full log of results are in Japanese:
[`docs/plans/ja_finetune_plan.md`](docs/plans/ja_finetune_plan.md). This README summarizes them.

## Usage

```bash
uvx pocket-tts generate --config hf://takuma104/pocket-tts-ja/japanese.yaml \
    --voice prompt.wav --text "こんにちは。今日はいい天気ですね。"
```

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(config="hf://takuma104/pocket-tts-ja/japanese.yaml")
voice = model.get_state_for_audio_prompt("prompt.wav")
audio = model.generate_audio(voice, "こんにちは。今日はいい天気ですね。")  # 24 kHz, torch.Tensor
```

- `--voice` is a recording of the voice to clone (a few seconds of clean speech). Without it the
  CLI clones its default voice (alba, an English speaker), which reads Japanese as well (JSUT
  Kana-CER 1.85%, against 2.29% with the JVS prompts). The pre-made voices cannot be selected by
  name (`--voice alba`): they are embeddings computed with the official weights.
- The config sets the sampling temperature to 0.3, the value every number below was measured
  with.
- Use `japanese_24l.yaml` for the 24-layer model.

## Results

All numbers are from generated audio transcribed by an ASR model, with JVS speakers (never
trained on) as voice prompts, temperature 0.3, one LSD step, seed 0.

**Kanji reading: [JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition)**,
all 13,536 sentences, official TTS protocol (kana-whisper). Accuracy is the share of sentences
whose target word is read correctly.

| Model | Prompts: 4 JVS speakers in rotation | Prompt: jvs001 |
|---|---:|---:|
| 24-layer teacher, step 700k (CFG 2.0, not released as is) | 92.97% | 93.68% |
| `japanese` (6 layers, CFG baked in) | **94.15%** | **94.73%** |
| `japanese_24l` (24 layers, CFG baked in) | 93.31% | 94.06% |

The goal of the project was 90%. The 6-layer model reads at least as well as the 24-layer ones,
so `japanese` is the one to use unless the rarer failure to stop (see Limitations) matters.

For reference, with the jvs001 prompt
[Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) scores 94.0% and
[Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi), which
synthesized part of the training data, 95.7%
([Irodori-TTS-Experiments](https://github.com/takuma104/Irodori-TTS-Experiments)).

**Other sets** (GPU inference from the training checkpoints):

| Set | Metric | Teacher 700k | `japanese` | `japanese_24l` |
|---|---|---:|---:|---:|
| JKYB-mini (1,500 sentences) | accuracy | 93.3% | 93.8% | 93.6% |
| Aozora Bunko ruby (1,000 sentences) | target reading accuracy | 69.4% | 70.0% | 70.4% |
| JSUT (1,000 sentences) | Kana-CER | 2.66% | 2.39% | 2.55% |
| JVS zero-shot (5 male + 5 female speakers, 10 sentences each) | Kana-CER | 3.05% | 2.58% | 2.34% |
| JVS zero-shot | similarity to the speaker's recording of the same sentence (ECAPA) | 0.538 | 0.536 | 0.538 |
| Short texts (172 texts of 2–8 tokens) | did not stop / extra speech | 1 / 2 | 0 / 1 | 0 / 1 |

**Stock wheel on CPU** (`pocket-tts==3.3.0`, the packaged models; `japanese` is stored in bfloat16,
`japanese_24l` in float32), next to the GPU evaluation of the same checkpoint:

| Set | Metric | `japanese` GPU | `japanese` wheel | `japanese_24l` GPU | `japanese_24l` wheel |
|---|---|---:|---:|---:|---:|
| JKYB-mini | accuracy | 93.8% | 94.5% | 93.6% | 93.3% |
| JSUT | Kana-CER | 2.39% | 2.29% | 2.55% | 2.43% |
| JVS zero-shot | Kana-CER / similarity | 2.58% / 0.536 | 2.07% / 0.533 | 2.34% / 0.538 | 2.10% / 0.535 |
| Short texts | did not stop / extra speech | 0 / 1 | 1 / 2 | 0 / 1 | 1 / 3 |

The two evaluations differ in the random numbers, in how long text is split and, for `japanese`,
in the weight precision, and the sets are small: the differences are of the size seen between
neighboring checkpoints.

## How it was trained

1. **Tokenizer and text front end.** A Japanese SentencePiece model with 4,000 pieces replaces the
   English one; the text embedding is re-initialized. Text is normalized by rules that the stock
   inference code can apply from the config (`src/pocket_tts_ja/text.py`).
2. **24-layer teacher** (`configs/train/B1_4k*.yaml`). Starts from the English 24-layer teacher,
   LSD objective, Mimi frozen, 700k steps on one RTX 5090. The voice prompt of each training clip
   is another utterance of the same speaker.
3. **Synthetic speech for readings.** Natural speech contains few of the rarer kanji readings, so
   sentences rich in them (from KANJIDIC2, JMdict, Wikipedia and Aozora Bunko) were synthesized
   with Irodori-TTS-v4.1-Small-Yomi in about 3,000 voices and kept only when an ASR model
   confirmed the reading. Words were chosen without the JKYB keys, and JKYB
   sentences and near duplicates were excluded.
4. **Short texts.** The stock inference splits long text into chunks of at most 50 tokens, so the
   model sometimes gets a clause of a few tokens on its own, and it used to keep talking after
   those. A last stage (660k to 700k) adds short clips to the mix.
5. **Distillation** (`configs/train/D0_24l.yaml`, `D1_6l_b.yaml`). The stock inference has no
   classifier-free guidance, so guidance at scale 2.0 is distilled into the weights: into a copy of
   the 24-layer model (`japanese_24l`) and into a 6-layer student initialized from the teacher's
   layers 0, 1, 2, 21, 22 and 23 (`japanese`).
6. **Packaging** (`scripts/package_release.py`) and a check with the stock wheel on CPU
   (`scripts/release/wheel_eval.sh`).

## Training data

Hours in the final mix (clips are repeated to reach the mixing ratios):

| Source | Hours | License |
|---|---:|---|
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) Japanese (Emilia and Emilia-YODAS), filtered and re-checked by ASR | 1,531 | CC BY-NC 4.0 / CC BY 4.0 |
| The same clips with a changed gain (against quiet sentence starts) | 227 | |
| Synthetic speech from Irodori-TTS, readings confirmed by ASR | 828 | text: CC BY-SA and public domain |

ReazonSpeech is not used. JVS and JSUT are used for evaluation only.

## Limitations

- About 5 to 6% of the JKYB sentences are still misread, mostly rare readings and words from the
  appendix of the Joyo kanji table (jukujikun, 87% correct). Numbers, counters and Latin letters were not
  evaluated separately.
- In 0.1 to 0.3% of the JKYB sentences `japanese` does not end the utterance and keeps generating
  up to the length limit (`japanese_24l`: 0.02 to 0.09%).
- Reading accuracy and speaker similarity depend on the voice prompt (the two JKYB columns differ
  by 0.6 to 0.8 points).
- The codec (Mimi) is the original one. Real Japanese recordings lose clarity when they pass
  through it (Kana-CER of JVS recordings: 3.3% as recorded, 5.4% after Mimi).
- Voice cloning must not be used to impersonate people without their consent. The prohibited uses
  of the original Pocket TTS apply.

## Repository

| Path | Content |
|---|---|
| `src/pocket_tts_ja/` | text normalization, data loader (prompt from the same speaker, mixing), kana utilities, codec fine-tuning |
| `scripts/prepare_emilia_ja.py`, `precompute_emilia_ja.py`, `recheck_asr.py` | Emilia: filtering, Mimi latents, ASR re-check |
| `scripts/synth_text/`, `scripts/synth/` | texts and synthesis of the Irodori data, QC |
| `scripts/yodas3/` | YODAS3 preprocessing (for a CC BY line, not part of these weights) |
| `scripts/train_ja.py`, `configs/` | training (wraps the upstream trainer in `third-party/pocket-tts`) |
| `scripts/eval_tts.py`, `eval_queue.py`, `summarize_evals.py` | evaluation (JKYB, JVS, JSUT, Aozora, short texts) |
| `scripts/distill/`, `package_release.py`, `scripts/release/` | distillation, packaging, check with the stock wheel |
| `scripts/m2/` | fine-tuning of the Mimi decoder |

Python environment: `uv sync` (the upstream repository and the benchmark are git submodules).

## Acknowledgements

[Kyutai](https://kyutai.org/) for Pocket TTS and its training code;
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset);
[Irodori-TTS](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (Aratako);
[JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition) and the
original Joyo Kanji Yomi Benchmark (SB Intuitions); the JVS and JSUT corpora.

## License

Code: MIT. Weights: CC BY-NC 4.0 (the training data includes Emilia).
