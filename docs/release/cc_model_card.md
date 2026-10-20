<!--
Model card (README.md) of takuma104/pocket-tts-ja-cc; scripts/release/stage_cc_repo.sh writes it
into the clone without this comment and the inline DECODER comments.
- Decoder: M2CH2_soup4 (branch mimi_ft, docs/plans/mimi_ft_cc_plan.md C4), chosen by the user on
  2026-10-20. The places with a DECODER comment change with the decoder.
- Numbers: runs/run_release_cc.sh M2CH2_soup4 runs/M2H_soups/M2CH2_soup4.safetensors and
  runs/run_voices_cc.sh M2CH2_soup4; the original Mimi rows from runs/run_release_cc.sh mimi,
  runs/run_voices_cc.sh and runs/run_bcc_teacher_jkyb.sh (10/20).
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
How the data was built and the models were trained: technical report ([English](https://github.com/takuma104/pocket-tts-ja/blob/main/docs/report/pocket_tts_ja_report_en.pdf),
[日本語](https://github.com/takuma104/pocket-tts-ja/blob/main/docs/report/pocket_tts_ja_report_ja.pdf)).

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
学習データの作り方と学習の方法は[テクニカルレポート](https://github.com/takuma104/pocket-tts-ja/blob/main/docs/report/pocket_tts_ja_report_ja.pdf)にまとめています。

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
  descriptions, so they are nobody's voice. With them `japanese` reads JSUT at a Kana-CER of
  2.23% (1.3 to 3.4% per voice). Without `--voice` the CLI clones its default voice
  (alba, an English speaker), which reads Japanese as well (JSUT Kana-CER 2.00%, against 2.23%
  with the JVS prompts). The pre-made voices cannot be selected by name (`--voice alba`): they
  are embeddings computed with the official weights.
- The config sets the sampling temperature to 0.3, the value every number below was measured
  with. Classifier-free guidance (scale 2.0) is distilled into the weights.
- Use `japanese_24l.yaml` for the 24-layer model. Its config ends the audio 3 frames (240 ms)
  after the model decides that the speech is over, instead of the 5 the package would pick for
  Japanese text: the 24-layer model sometimes makes a short sound in those last frames.
- To keep a fixed version, add a commit hash: `hf://takuma104/pocket-tts-ja-cc/japanese.yaml@<commit>`.

## Results

All numbers are from generated audio transcribed by an ASR model, with speakers of the JVS corpus
(never trained on) as voice prompts, temperature 0.3, one LSD step, with the fine-tuned decoder
of the codec (see Training) unless marked. <!-- DECODER -->

**Kanji reading: [JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition)**,
all 13,536 sentences, official TTS protocol (kana-whisper). Accuracy is the share of sentences
whose target word is read correctly.

| Model | Prompts: 4 JVS speakers in rotation | Prompt: jvs001 |
|---|---:|---:|
| `japanese` | 94.77% | 94.49% |
| `japanese_24l` | 94.36% | 94.85% |
| *Original Mimi decoder:* `japanese` | 93.93% | 93.96% |
| *Original Mimi decoder:* `japanese_24l` | 93.34% | 94.08% |
| *Original Mimi decoder:* 24-layer teacher before the guidance was distilled (CFG 2.0, not released) | 92.82% | 93.82% |
| *CC BY-NC sibling* ([pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja), its own fine-tuned decoder): `japanese` | 94.75% | 94.84% |
| *CC BY-NC sibling:* `japanese_24l` | 94.32% | 94.49% |

**Other sets**, `japanese` with the stock package on CPU (the released bfloat16 file) next to the
GPU evaluation of the same checkpoint:

| Set | Metric | GPU | Stock package, CPU |
|---|---|---:|---:|
| JKYB-mini (1,500 sentences) | accuracy | 94.9% | 95.1% |
| Aozora Bunko ruby (1,000 sentences) | target reading accuracy | 70.7% | |
| JSUT (1,000 sentences) | Kana-CER | 2.14% | 2.23% |
| JVS zero-shot (5 male + 5 female speakers, 10 sentences each) | Kana-CER | 2.18% | 2.09% |
| JVS zero-shot | similarity to the speaker's recording of the same sentence (ECAPA) | 0.651 | 0.650 |
| Short texts (172 texts of 2–8 tokens) | did not stop / extra speech | 0 / 2 | 0 / 1 |

With the original Mimi decoder the stock package gives JKYB-mini 94.4%, JSUT 2.49% and JVS 2.15% /
0.560: the fine-tuned decoder mostly raises the similarity to the speakers' recordings (the CC BY-NC
sibling: 0.575).

`japanese_24l` (3 frames after the end of speech, as in its config) is close on these sets
(GPU: JKYB-mini 94.9%, Aozora Bunko 70.3%, JSUT 2.13%, JVS 2.47% / 0.645, short texts 0 / 0;
stock package on CPU: JKYB-mini 94.1%, JSUT 2.17%, JVS 2.30% / 0.641, short texts 0 / 0). The two
models read equally well. `japanese_24l` fails to stop and ends with a short sound a little less
often (see Limitations); `japanese` is three times smaller and faster.

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
6. The decoder of the codec (10M parameters) is fine-tuned from the original one on CC BY audio
   only, discriminators included. <!-- DECODER --> First 70k steps (multi-scale mel loss, MPD and
   MS-STFT discriminators, and a loss on the batch-averaged spectrum against the tonal artifacts
   locked to the decoder's upsampling strides) on clips of up to 8 seconds: 164 hours of YODAS3
   clips cut again from the videos in full band (the training clips above whose video reaches
   15 kHz and whose DNSMOS scores are high) and 116 hours of the synthetic speech; in the last 10k of these steps the band above 4 kHz is held at the level of
   the original decoder. Then 10k steps in which the band above 4 kHz is also trained towards
   the recordings, on the YODAS3 clips with the highest predicted MOS (UTMOSv2) among those
   without band limits or clipping, female and male apart (7 to 14 hours); four such runs (three clip sets, two seeds) are averaged. Two videos
   whose voices were heard to be synthetic are left out of the decoder's data. The encoder is
   untouched, so latents and voice prompts are those of the original Mimi.

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

- About 5% of the JKYB sentences are still misread, mostly rare readings and words from the
  appendix of the Joyo kanji table (jukujikun, 86 to 88% correct). Numbers, counters and Latin letters
  were not evaluated separately.
- In 0.04 to 0.07% of the JKYB sentences `japanese` does not end the utterance and keeps
  generating up to the length limit (`japanese_24l`: 0 to 0.015%).
- 6 to 9% of the clips end with a short, quiet sound (a breath or a small voice) after a pause
  (`japanese` 8 to 9%, `japanese_24l` 6 to 7%, on JKYB-mini).
- Reading accuracy and speaker similarity depend on the voice prompt.
- Real Japanese recordings still lose clarity when they pass through the codec (Kana-CER of JVS
  recordings: 3.3% as recorded, 4.7% through Mimi with the fine-tuned decoder; 5.4% with the
  original decoder). <!-- DECODER -->

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
Interspeech 2026), Emilia-YODAS from [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset)
(CC BY 4.0; He et al., "Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for
Speech Generation", [arXiv:2501.15907](https://arxiv.org/abs/2501.15907)), and speech synthesized
with Irodori-TTS (MIT) from the texts above.

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
