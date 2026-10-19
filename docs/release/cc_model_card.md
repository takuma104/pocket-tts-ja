<!--
Draft of the model card (README.md) of the CC BY line's Hugging Face repo. Not published.
Open before the release (docs/plans/ja_finetune_plan.md, B-CC "公開の準備"):
- repo id (written here as takuma104/pocket-tts-ja-cc) and the user's go-ahead;
- the decoder: the original Mimi, or a fine-tuned decoder trained on CC BY-compatible audio only
  (branch mimi_ft). Sections marked DECODER change with it;
- numbers marked TBD come from runs/run_release_cc.sh NAME [DECODER];
- the commit hashes of the pinned configs.
-->
---
license: cc-by-4.0
language:
- ja
library_name: pocket-tts
pipeline_tag: text-to-speech
base_model: kyutai/pocket-tts
datasets:
- espnet/yodas3
- amphion/Emilia-Dataset
tags:
- pocket-tts
- voice-cloning
---

# Pocket TTS Japanese (CC BY)

Japanese models for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai), fine-tuned
from the English 24-layer model with the official training code, **on CC BY data only**. The
weights are CC BY 4.0, so they can be used commercially, with attribution (see License). A
sibling trained with more data, under CC BY-NC 4.0, is
[takuma104/pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja). Code, training recipe
and the full log of results: [github.com/takuma104/pocket-tts-ja](https://github.com/takuma104/pocket-tts-ja).

- Runs with the **stock `pocket-tts` package** (3.3.0 or later), on CPU, with no code changes.
- Takes **plain Japanese text** (kanji and kana as written). There is no G2P front end; the model
  reads the kanji itself.
- **Zero-shot voice cloning** from a few seconds of reference audio, as in the original.

| Model | Config | FlowLM | Parameters (with the codec) | File |
|---|---|---|---:|---:|
| `japanese` | `japanese.yaml` | 6 layers | 110M | 219 MB |
| `japanese_24l` | `japanese_24l.yaml` | 24 layers | 336M | 1.3 GB |

日本語の概要: Kyutai の Pocket TTS を、CC BY のデータだけで日本語にファインチューニングしたモデルです。
ウェイトは CC BY 4.0 で、クレジットを表示すれば商用にも使えます（下の License を参照）。公式の
`pocket-tts`（3.3.0 以降）にそのまま読み込め、CPU で動きます。入力は漢字かな交じりの文で、読みはモデルが
決めます（G2P は使いません）。数秒の参照音声で声を指定します。Emilia（非商用）も使ったモデルは
[takuma104/pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja)（CC BY-NC 4.0）にあります。

## Usage

```bash
uvx pocket-tts generate --config hf://takuma104/pocket-tts-ja-cc/japanese.yaml \
    --voice hf://takuma104/pocket-tts-ja-cc/voices/female_03.wav \
    --text "こんにちは。今日はいい天気ですね。"
```

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(config="hf://takuma104/pocket-tts-ja-cc/japanese.yaml")
voice = model.get_state_for_audio_prompt("prompt.wav")
audio = model.generate_audio(voice, "こんにちは。今日はいい天気ですね。")  # 24 kHz, torch.Tensor
```

- `--voice` is a recording of the voice to clone (a few seconds of clean speech). `voices/` has ten
  synthetic voices (five female, five male, described in `voices/voices.json`), made with
  [Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) from text
  descriptions, so they are nobody's voice. Without `--voice` the CLI clones its default voice
  (alba, an English speaker), which reads Japanese as well (JSUT Kana-CER TBD, against TBD with
  the JVS prompts). The pre-made voices cannot be selected by name (`--voice alba`): they are
  embeddings computed with the official weights.
- The config sets the sampling temperature to 0.3, the value every number below was measured
  with. Classifier-free guidance (scale 2.0) is distilled into the weights.
- Use `japanese_24l.yaml` for the 24-layer model. Its config ends the audio 3 frames (240 ms)
  after the model decides that the speech is over, instead of the 5 the package would pick for
  Japanese text: the 24-layer model sometimes makes a short sound in those last frames.
- To keep a fixed version, add a commit hash: `hf://takuma104/pocket-tts-ja-cc/japanese.yaml@<commit>`.

## Results

All numbers are from generated audio transcribed by an ASR model, with speakers of the JVS corpus
(never trained on) as voice prompts, temperature 0.3, one LSD step. DECODER: measured with TBD.

**Kanji reading: [JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition)**,
all 13,536 sentences, official TTS protocol (kana-whisper). Accuracy is the share of sentences
whose target word is read correctly.

| Model | Prompts: 4 JVS speakers in rotation | Prompt: jvs001 |
|---|---:|---:|
| `japanese` | TBD | TBD |
| `japanese_24l` | TBD | TBD |
| *CC BY-NC sibling:* `japanese` (pocket-tts-ja) | 94.75% | 94.84% |

**Other sets**, `japanese` with the stock package on CPU (the released bfloat16 file) next to the
GPU evaluation of the same checkpoint:

| Set | Metric | GPU | Stock package, CPU |
|---|---|---:|---:|
| JKYB-mini (1,500 sentences) | accuracy | TBD | TBD |
| Aozora Bunko ruby (1,000 sentences) | target reading accuracy | TBD | |
| JSUT (1,000 sentences) | Kana-CER | TBD | TBD |
| JVS zero-shot (5 male + 5 female speakers, 10 sentences each) | Kana-CER | TBD | TBD |
| JVS zero-shot | similarity to the speaker's recording of the same sentence (ECAPA) | TBD | TBD |
| Short texts (172 texts of 2–8 tokens) | did not stop / extra speech | TBD | TBD |

`japanese_24l` on these sets: TBD. TBD: which of the two to recommend.

## Training

1. A Japanese SentencePiece model with 4,000 pieces, trained on the transcripts of the training
   data below, replaces the English tokenizer; the text embedding is re-initialized. The codec
   (Mimi) stays frozen while the language model is trained.
2. The 24-layer teacher starts from the English 24-layer model and is trained for 580k steps
   (LSD objective, the voice prompt is another utterance of the same speaker).
3. Natural speech contains few of the rarer kanji readings, so sentences rich in them were
   synthesized with [Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi)
   in 2,998 voices taken from Emilia-YODAS (CC BY) recordings, and kept only when an ASR model
   confirmed the reading. The words were chosen without the JKYB keys, and JKYB sentences and
   near duplicates were excluded.
4. Six frames of digital silence (the Mimi latents of silence) follow every complete training
   target and are learned like speech; the end-of-speech target stays where the speech ends. The
   stock package keeps generating for a few frames after the end of speech, and upstream training
   never shows the model those frames: without this, the model filled them with short fillers
   ("うん", "えー") after about 10% of the sentences.
5. Guidance at scale 2.0 is distilled into the weights: into a copy of the 24-layer model
   (`japanese_24l`, 50k steps) and into a 6-layer student initialized from the teacher's layers
   0, 1, 2, 21, 22 and 23 (`japanese`, 160k steps).
6. DECODER: the codec is the original Mimi. / The decoder of the codec is fine-tuned on TBD.

| Training data | Hours | License |
|---|---:|---|
| [YODAS3](https://huggingface.co/datasets/espnet/yodas3) Japanese, filtered, re-checked by ASR, punctuation restored | 556 | CC BY 3.0 |
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) Japanese (the CC BY part of Emilia), filtered and re-checked by ASR | 305 | CC BY 4.0 |
| The same Emilia-YODAS clips with a changed gain | 48 | |
| Synthetic speech from Irodori-TTS (voices from Emilia-YODAS), readings confirmed by ASR | 367 | model: MIT; text: CC BY-SA, CC BY and public domain |

The synthetic texts come from Japanese Wikipedia, JMdict / KANJIDIC2 (CC BY-SA 4.0), Tatoeba (CC BY
2.0 FR), Aozora Bunko (public domain), and sentences written by LLMs for words with rare readings.
The non-commercial part of Emilia and ReazonSpeech are not used. JVS and JSUT are used for
evaluation only.

## Limitations

- About TBD% of the JKYB sentences are still misread, mostly rare readings and words from the
  appendix of the Joyo kanji table (jukujikun). Numbers, counters and Latin letters were not
  evaluated separately.
- In TBD% of the JKYB sentences `japanese` does not end the utterance and keeps generating up
  to the length limit (`japanese_24l`: TBD%).
- Reading accuracy and speaker similarity depend on the voice prompt.
- DECODER: real Japanese recordings lose clarity when they pass through the codec (Kana-CER of JVS
  recordings: 3.3% as recorded, 5.4% through the original Mimi).

## Prohibited use

The prohibited uses of the original Pocket TTS apply: use of the model must comply with all
applicable laws and regulations and must not result in, involve, or facilitate any illegal,
harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without
limitation, voice impersonation or cloning without explicit and lawful consent; misinformation,
disinformation, or deception (including fake news, fraudulent calls, or presenting generated
content as genuine recordings of real people or events); and the generation of unlawful, harmful,
libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content.

## License and acknowledgements

Weights: CC BY 4.0. They derive from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
(CC BY 4.0) by Kyutai ([paper](https://arxiv.org/abs/2509.06926)). When you use them, credit
this model and the original.

Training data: [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) (CC BY 3.0; Chen et al.,
"YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech",
Interspeech 2026), Emilia-YODAS from [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) (CC BY 4.0; He
et al., "Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech
Generation", [arXiv:2501.15907](https://arxiv.org/abs/2501.15907)), and speech synthesized with Irodori-TTS (MIT) from the texts above.

```bibtex
@inproceedings{chen26d_interspeech,
  title     = {{YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech}},
  author    = {William Chen and Shinnosuke Takamichi and Sayaka Shiota and Satoru Fukayama and Samuele Cornell and Shinji Watanabe},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {5367--5372},
  doi       = {10.21437/Interspeech.2026-386},
}
```

Thanks to Kyutai for Pocket TTS and its training code, to the authors of YODAS v3 and
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset),
[Irodori-TTS](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (Aratako),
[JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition) and the
original Joyo Kanji Yomi Benchmark (SB Intuitions), and of the JVS and JSUT corpora.
