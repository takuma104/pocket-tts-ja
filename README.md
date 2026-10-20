# pocket-tts-ja

Japanese models for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai), fine-tuned
from the English 24-layer model with the official training code.

- Runs with the **stock `pocket-tts` wheel** (3.3.0 or later), on CPU, with no code changes: the
  release is a config file, a SentencePiece model and a `safetensors` file.
- Takes **plain Japanese text** (kanji and kana as written). There is no G2P front end; the
  model reads the kanji itself.
- **Zero-shot voice cloning** from a few seconds of reference audio, as in the original.
- Two models in each release: `japanese` (6 layers, 110M parameters with the codec, 219 MB) and
  `japanese_24l` (24 layers, 336M parameters, 1.3 GB), both with classifier-free guidance
  distilled into the weights.

Two releases, which differ in the training data:

| Weights | Training data | License | JKYB, `japanese` |
|---|---|---|---:|
| [takuma104/pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja) | Emilia Japanese (including its non-commercial part), synthetic speech | CC BY-NC 4.0 | 94.75% |
| [takuma104/pocket-tts-ja-cc](https://huggingface.co/takuma104/pocket-tts-ja-cc) | CC BY data only: YODAS3, Emilia-YODAS, synthetic speech | CC BY 4.0 (commercial use with attribution) | 94.77% |

**Technical report** (how the training data was built and how the models were trained, including
the decoder fine-tuning): [English](docs/report/pocket_tts_ja_report_en.pdf),
[日本語](docs/report/pocket_tts_ja_report_ja.pdf). The experiment plan and the full log of results
are in Japanese: [`docs/plans/ja_finetune_plan.md`](docs/plans/ja_finetune_plan.md). This README
summarizes them.

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

- For the CC BY release use `hf://takuma104/pocket-tts-ja-cc/japanese.yaml`; for the 24-layer
  models, `japanese_24l.yaml`.
- `--voice` is a recording of the voice to clone (a few seconds of clean speech). The CC BY repo has
  ten synthetic voices in `voices/` (for example
  `hf://takuma104/pocket-tts-ja-cc/voices/female_03.wav`), made with Irodori-TTS from text
  descriptions. Without `--voice` the CLI clones its default voice (alba, an English speaker),
  which reads Japanese as well (JSUT Kana-CER 1.80% / 2.00%, against 2.15% / 2.23% with the JVS
  prompts). The pre-made voices cannot be selected by name (`--voice alba`): they are embeddings
  computed with the official weights.
- The config sets the sampling temperature to 0.3, the value every number below was measured
  with.

## Results

All numbers are from generated audio transcribed by an ASR model, with JVS speakers (never
trained on) as voice prompts, temperature 0.3, one LSD step, seed 0, with each release's
fine-tuned decoder of the codec unless marked.

**Kanji reading: [JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition)**,
all 13,536 sentences, official TTS protocol (kana-whisper). Accuracy is the share of sentences
whose target word is read correctly. Prompts: 4 JVS speakers in rotation / jvs001.

| Model | `pocket-tts-ja` (CC BY-NC) | `pocket-tts-ja-cc` (CC BY) |
|---|---:|---:|
| `japanese` (6 layers) | **94.75 / 94.84%** | **94.77 / 94.49%** |
| `japanese_24l` (24 layers) | 94.32 / 94.49% | 94.36 / 94.85% |
| *Original Mimi decoder:* `japanese` | 94.15 / 94.73% | 93.93 / 93.96% |
| *Original Mimi decoder:* 24-layer teacher (CFG 2.0, not released) | 92.97 / 93.68% (700k steps) | 92.82 / 93.82% (580k steps) |

The goal of the project was 90%. The 6-layer students read as well as the 24-layer models and
better than their teachers. For reference, with the jvs001 prompt
[Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) scores 94.0% and
[Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi), which
synthesized part of the training data, 95.7%
([Irodori-TTS-Experiments](https://github.com/takuma104/Irodori-TTS-Experiments)).

**Other sets**, `japanese`: GPU inference from the training checkpoint, and the **stock wheel on
CPU** (`pocket-tts==3.3.0`, the released bfloat16 file):

| Set | Metric | CC BY-NC, GPU | CC BY-NC, wheel | CC BY, GPU | CC BY, wheel |
|---|---|---:|---:|---:|---:|
| JKYB-mini (1,500 sentences) | accuracy | 95.1% | 95.3% | 94.9% | 95.1% |
| Aozora Bunko ruby (1,000 sentences) | target reading accuracy | 71.3% | | 70.7% | |
| JSUT (1,000 sentences) | Kana-CER | 2.28% | 2.15% | 2.14% | 2.23% |
| JVS zero-shot (5 male + 5 female speakers, 10 sentences each) | Kana-CER | 2.32% | 1.70% | 2.18% | 2.09% |
| JVS zero-shot | similarity to the speaker's recording of the same sentence (ECAPA) | 0.577 | 0.575 | 0.651 | 0.650 |
| Short texts (172 texts of 2–8 tokens) | did not stop / extra speech | 0 / 1 | 1 / 1 | 0 / 2 | 0 / 1 |

`japanese_24l` is at the same level; its numbers are on the model cards. The two evaluations
differ in the random numbers, in how long text is split and in the weight precision, and the sets
are small: the differences are of the size seen between neighboring checkpoints.

## How it was trained

The [technical report](docs/report/pocket_tts_ja_report_en.pdf) has the details and what each step
changed.

1. **Tokenizer and text front end.** A Japanese SentencePiece model with 4,000 pieces replaces the
   English one; the text embedding is re-initialized. Text is normalized by rules that the stock
   inference code can apply from the config (`src/pocket_tts_ja/text.py`).
2. **24-layer teacher** (`configs/train/B1_4k*.yaml`, `BCC_4k*.yaml`). Starts from the English
   24-layer model, LSD objective, Mimi frozen, on one RTX 5090 (700k steps; CC BY line 580k). The
   voice prompt of each training clip is another utterance of the same speaker.
3. **Synthetic speech for readings.** Natural speech contains few of the rarer kanji readings, so
   sentences rich in them (from KANJIDIC2, JMdict, Wikipedia, Tatoeba and Aozora Bunko) were
   synthesized with Irodori-TTS-v4.1-Small-Yomi in about 3,000 voices and kept only when an ASR
   model confirmed the reading. Words were chosen without the JKYB keys, and JKYB sentences and
   near duplicates were excluded.
4. **The end of the utterance.** The stock inference splits long text into chunks of at most 50
   tokens and keeps generating a few frames after the model's end-of-speech decision, which
   upstream training never shows the model. A last stage of short clips (CC BY-NC line), and six
   frames of digital silence learned after every training target (CC BY line), stop the model
   from talking on after the text.
5. **Distillation** (`configs/train/D0_*.yaml`, `D1_*.yaml`). The stock inference has no
   classifier-free guidance, so guidance at scale 2.0 is distilled into the weights: into a copy of
   the 24-layer model (`japanese_24l`) and into a 6-layer student initialized from the teacher's
   layers 0, 1, 2, 21, 22 and 23 (`japanese`).
6. **Decoder of the codec** (`scripts/m2/`; the CC BY decoder on the branch `mimi_ft`). Only the
   decoder side of Mimi (10M parameters) is fine-tuned on Japanese speech, with GAN losses, so
   latents and voice prompts stay those of the original Mimi. The CC BY decoder is trained from
   the original one on CC BY audio only.
7. **Packaging** (`scripts/package_release.py`) and a check with the stock wheel on CPU
   (`scripts/release/wheel_eval.sh`).

## Training data

Hours in the final mixes (clips are repeated to reach the mixing ratios):

| Source | CC BY-NC line | CC BY line | License |
|---|---:|---:|---|
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) Japanese (Emilia and Emilia-YODAS), filtered and re-checked by ASR | 1,531 | | CC BY-NC 4.0 / CC BY 4.0 |
| Emilia-YODAS Japanese only (the CC BY part of Emilia), filtered and re-checked by ASR | | 305 | CC BY 4.0 |
| [YODAS3](https://huggingface.co/datasets/espnet/yodas3) Japanese, filtered, re-checked by ASR, punctuation restored | | 556 | CC BY 3.0 |
| The same Emilia clips with a changed gain (against quiet sentence starts) | 227 | 48 | |
| Synthetic speech from Irodori-TTS, readings confirmed by ASR | 828 | 367 | text: CC BY-SA, CC BY and public domain |

ReazonSpeech is not used. JVS and JSUT are used for evaluation only.

## Limitations

- About 5% of the JKYB sentences are still misread, mostly rare readings and words from the
  appendix of the Joyo kanji table (jukujikun, 86 to 89% correct). Numbers, counters and Latin
  letters were not evaluated separately.
- In 0.1 to 0.3% of the JKYB sentences the CC BY-NC `japanese` does not end the utterance and keeps
  generating up to the length limit (CC BY `japanese`: 0.04 to 0.07%; the 24-layer models: under
  0.1%).
- 6 to 9% of the clips of the CC BY models end with a short, quiet sound after a pause.
- Reading accuracy and speaker similarity depend on the voice prompt.
- Real Japanese recordings still lose some clarity when they pass through the codec (Kana-CER of
  JVS recordings: 3.3% as recorded, 4.7% with the fine-tuned decoders, 5.4% with the original
  Mimi).
- Voice cloning must not be used to impersonate people without their consent. The prohibited uses
  of the original Pocket TTS apply.

## Repository

| Path | Content |
|---|---|
| `src/pocket_tts_ja/` | text normalization, data loader (prompt from the same speaker, mixing), kana utilities, codec fine-tuning |
| `scripts/prepare_emilia_ja.py`, `precompute_emilia_ja.py`, `recheck_asr.py` | Emilia: filtering, Mimi latents, ASR re-check |
| `scripts/synth_text/`, `scripts/synth/` | texts and synthesis of the Irodori data, QC |
| `scripts/yodas3/` | YODAS3 preprocessing (CC BY line) |
| `scripts/train_ja.py`, `configs/` | training (wraps the upstream trainer in `third-party/pocket-tts`) |
| `scripts/eval_tts.py`, `eval_queue.py`, `summarize_evals.py` | evaluation (JKYB, JVS, JSUT, Aozora, short texts) |
| `scripts/distill/`, `package_release.py`, `scripts/release/` | distillation, packaging, check with the stock wheel |
| `scripts/m2/` | fine-tuning of the Mimi decoder |
| `docs/report/` | technical report (LaTeX sources, `build.sh`) |

Python environment: `uv sync` (the upstream repository and the benchmark are git submodules).

## Acknowledgements

[Kyutai](https://kyutai.org/) for Pocket TTS and its training code;
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset);
[YODAS v3](https://huggingface.co/datasets/espnet/yodas3);
[Irodori-TTS](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (Aratako);
[JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition) and the
original Joyo Kanji Yomi Benchmark (SB Intuitions); the JVS and JSUT corpora.

## License

Code: MIT. Weights: `pocket-tts-ja` CC BY-NC 4.0 (the training data includes Emilia),
`pocket-tts-ja-cc` CC BY 4.0.
