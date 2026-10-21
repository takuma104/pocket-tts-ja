# Irodori-TTS → PocketTTS 型 FlowLM の latent 直接蒸留: 実験計画

元のアイディア: [irodori_tts_distill_idea.md](irodori_tts_distill_idea.md)(ChatGPT 案)。本計画は、その案を main / mimi_ft の記録と、両モデルの実装確認(2026-10-22)で裏付けて具体化したもの。

## 0. 目的とゴール

- **目的**: Irodori-TTS(教師)が生成した **DACVAE latent** を、PocketTTS 型の AR FlowLM(生徒)に直接学習させる。生徒の出力は Irodori と同じ DACVAE decoder で復号し、Mimi 経由の「DACVAE で復号 → Mimi で再符号化」という二重 codec を取り除く。主な狙いは、女性音声(特に高 F0)の品質改善。
- **小さく始める**: 最初は 6L・25 Hz・DACVAE 凍結の PoC で、「品質が上がるか」「AR 生徒が教師の分布を追えるか」だけを確かめる。
- **最終ゴール(うまくいった場合)**: PocketTTS 公式 wheel で動くウエイトとして公開する。wheel 側の制約(§1.3)が設計に効くため、早めにリスク確認を入れる(P1-side)。

## 1. 前提として分かっていること

### 1.1 これまでの記録(main / mimi_ft)

- **Mimi による劣化**
  - JVS の再構成で Kana-CER が 3.16% → 5.45% に悪化した。女性は +2.94pt、男性は +1.60pt で、悪化の大きい話者ワースト 10 はすべて女性(main M0)。
  - Irodori の出力をそのまま評価すると JKYB 95.06%、Mimi を通すと 93.28% に落ちる(main)。
- **高域の倍音が出ない(mimi_ft)**
  - Mimi 系 decoder はどれも、4–4.5 kHz から上の倍音が出ない。女性 JVS の 4–6 kHz comb は録音 4.7 dB に対し、元 Mimi 1.2、keep4k 1.3、gs2_80k 0.6 dB だった。
  - その帯域に出るのは、ピッチパルスに同期したノイズだった。
  - decoder の容量(層数・幅)や F0 誤差は原因から除外済み。
- **Space ボイス female_02 / female_05(mimi_ft C6)**
  - F0 が 350 / 406 Hz と高く、4 kHz より上に強い倍音を持つのはこの 2 声だけ。CC decoder(M2CH2_soup4)はこの帯域をノイズで埋めてしまう。
  - データで直すのも難しい。YODAS3 の女性 28k clips のうち、F0 400 Hz を超えるものは 0 本だった。
  - → **Mimi 経路を外す動機ははっきりしている。** 一方で、DACVAE が録音の高域倍音を再現できるかは**まだ誰も直接確かめていない**。間接的な証拠はある: Irodori の voice-design 出力 female_02/05 は 4 kHz より上に強い倍音を持っている。
- **評価指標の盲点(mimi_ft)**
  - UTMOSv2 / UTMOS22 は 16 kHz 入力なので、8 kHz より上は見えない。高域のノイズ化に対しては、UTMOSv2 はむしろ点数を上げた。
  - ViSQOL / SCOREQ は試聴の判定と逆順になった。Kana-CER も mel loss もこの劣化を見ない。
  - **品質判定は試聴と高域構造の指標(comb / level)で行う**必要がある。
- **読みの精度**
  - 読みの精度は、その読みが学習データに出てくる回数でほぼ決まる。どの停滞も、新しい合成セットを入れたときにだけ解けた(main)。
  - → PoC の JKYB でアーキテクチャの良し悪しを判断してはいけない。

### 1.2 Irodori-TTS v4.1-Small(実装確認済み)

| 項目 | 内容 |
|---|---|
| 教師 | Irodori-TTS-v4.1-Small-Yomi v3(`~/co/Irodori-TTS-v4.1-Small-Yomi`, MIT, JKYB 95.7%)。main の合成で使ったもの。ソースは `~/co/Irodori-TTS-Experiments/Irodori-TTS`(89f9d8f) |
| 構成 | ModernBERT-ja-310m の text encoder と、DiT 12 層(dim 1280)。全体で 766M。発話全体を双方向で見る非 AR |
| 話者条件 | 参照音声を -16 dB に正規化して DACVAE で符号化(mean)。4 フレームずつ patch にし、8 層の ReferenceLatentEncoder を通す。専用の話者埋め込みネットワークはない |
| latent | 25 Hz × 32 次元、patch 1(1 フレームが 1 token)。**正規化定数はなく**、raw の mean latent をそのまま扱う。次元ごとの std は 0.6–1.8 |
| 長さ | 長さはサンプリング前に duration predictor(または明示の `seconds`)で決める。発話の後ろには平坦な tail が残り、`find_flattening_point` で音声の終端を判定する |
| サンプリング | RF Euler 40 step。CFG は text 3.0 / speaker 5.0 を t∈[0.5,1] で適用し、1 サンプルあたり 80 DiT pass。seed は item ごと |
| latent の取り出し | `inference_runtime.py:1415-1423` の `z = z[:, :latent_steps]`、つまり decode の直前。watermark は音声に後からかけるので、**latent には入っていない** |
| codec | Aratako/Semantic-DACVAE-Japanese-32dim。48 kHz、hop 1920、MIT。encoder 27M / decoder 71M、Snake 活性化。**非因果**で、先読みは約 7 フレーム(約 280 ms)。5090 での decode は約 350 倍速 |
| 生成速度 | 5090 1 枚で 6.5–11 文/s、音声換算で約 44 秒/s。**最大で約 1,000 h/GPU-day**。VRAM は batch 32 で約 16 GB |
| 既存 latent | `Irodori-TTS-Experiments/outputs/yomi_*/…/latents` に約 465k 件(11 GB)。JVS 参照か参照なしの短文で、配布不可の JVS 由来を含む → **パイプライン検証にだけ使う** |

### 1.3 PocketTTS 側(submodule a922b91 = v4.0.1、実装確認済み)

- **FlowLM は codec に依存しない**
  - `emb_mean` / `emb_std` で正規化した latent を `input_linear` に入れ、LSD flow head で次フレームを生成する。
  - teacher forcing は `cat[bos_emb, normalized[:, :-1]]`(`training/modules/conditioner.py:37-39`)。
  - 過去の latent にノイズを加えるような augmentation はない。
- **6L の規模**: FlowLM 89.4M(d1024 / 16 heads)と Mimi 20M(うち decode 側 10.3M)。24L は FlowLM 316M。
- **wheel の YAML だけで設定できるもの**
  - 48 kHz / 25 Hz / ldim 32 と、48 kHz / 12.5 Hz / ldim 64(2 フレーム連結)。どちらもランダム重みで、stock の `TTSModel` によるロードと生成を確認済み(`exp/irodori_distill/phase0_probes/dacvae_{25hz,stack64}.yaml`)。
- **wheel で実現できないもの**
  - codec クラスは Mimi で固定されている。中身は SEANet(**ELU で固定**、因果)と 2 層 transformer。
  - **DACVAE の decoder(Snake、非因果)はそのままでは載らない。**
  - voice prompt は同梱の encoder で符号化される。そのため、DACVAE の encoder を模倣する encoder も別に必要になる。
- **バージョン**: v4.0 で追加された config キー(`decoder_seanet`、`mimi.weights_path` など)は、3.3.0 では読めない。main のリリースは 3.3.0 で検証している。
- **CPU 速度(1 スレッド、推定)**
  - 6L・25 Hz: FlowLM が律速になり約 2.4 倍速。
  - 64 次元・12.5 Hz: 約 4.7 倍速。
  - 現行の Japanese 6L は 4.4–4.8 倍速。

## 2. 基本方針

**「DACVAE latent 空間を教師と生徒で共有した、系列レベルの蒸留」**とする。教師の RF の内部状態やベクトル場は真似しない。教師が最後まで積分して得た latent 列を学習データとして、生徒の FlowLM が `p(z_t | z_<t, text, voice)` を LSD で学習する。

| 設計項目 | PoC の既定 | 後で比較する案 |
|---|---|---|
| 生徒サイズ | 6L(d1024、stock の english.yaml と同じ形) | 24L で学習してから 6L に depth distill(main と同じ手順)、upstream の small 系 |
| latent | 32 次元・25 Hz のまま | 2 フレーム連結して 64 次元・12.5 Hz(P3a) |
| 初期値 | main の CC 6L 生徒 D1-CC(`runs/D1_cc_6l/snapshots/step_00160000.safetensors`)の transformer と text table。tokenizer は `ja_cc_4k` | English 6L から、またはランダム初期化から |
| 作り直すもの | `input_linear`、`bos_emb`、`speaker_proj_weight`、`flow_net`、`out_eos`、`emb_mean` / `emb_std` | — |
| latent 統計 | 教師 latent から次元ごとの mean / std を事前に計算して固定する(EMA にはしない) | — |
| 話者 prompt | 教師に与えた参照 wav の DACVAE latent(-16 dB 正規化、mean)。教師の条件と完全に一致させる | 同じ声の別の合成発話を prompt にする(P3g) |
| 損失 | upstream の LSD(p_equal 0.75、distill_prob 0.25、flow_batch_multiplier 4)と EOS BCE(重み 0.1) | 過去 latent へのノイズ augmentation(P3c) |
| 終端 | target は教師 latent の `speech_end` までに、教師自身の平坦な tail を 12 フレーム(0.48 s)足したもの。tail にも flow loss をかけ、EOS は speech_end の次のフレーム | — |
| CFG | text / voice の dropout 0.2 は残す。ただし評価の主条件は CFG 1.0 | 教師データには CFG(3.0 / 5.0)が既に効いているので、生徒には不要かもしれない。要るなら後で CFG distill |
| codec | DACVAE を凍結し、全文をまとめて decode する | wheel 用 decoder(P1-side / P4) |

教師データが CFG 済みの合成音声だけなので、**生徒は CFG なしでも足りる可能性が高い**(推論)。これは stock wheel(CFG なし)にとって有利。

## 3. フェーズ計画

GPU は本機の RTX 5090 1 枚だけを前提にする(galleria2 は使わない)。教師データの生成と学習は同じ GPU を取り合う。

### P0: 準備と codec の検証(1–2 日 + 試聴)

**P0a 環境・コードの移植**(詳細は §5)

- `third-party/Irodori-TTS` を submodule として追加する(Aratako/Irodori-TTS @ 89f9d8f)。依存は `uv add` で入れる。ASR 系は Python 3.13 で動かす(3.14 では forkserver で止まる)。
- main と mimi_ft から必要なものだけを持ってくる。
- v4.0.1 への対応も入れる:
  - `post_eos.py` が `z_proj` を飛ばしている問題
  - loader のフレーム定数を frame_rate から引数で決めるようにする

**P0b codec の比較(ChatGPT 案の 1)**

- **素材**
  - JVS 女性 5 話者・男性 5 話者 × 100 clips(mimi_ft の `recon_jvs.py` と同じ構成)
  - リリース voices 10 声(female_02 / 05 を含む)の prompt 自体
  - Irodori 教師の 48 kHz 出力(高 F0 の女性声)
- **経路**
  - 録音
  - DACVAE 再構成(48 kHz)
  - DACVAE 再構成を 24 kHz にリサンプルしたもの(Mimi と公平に比べるため)
  - 元 Mimi
  - M2CH2_soup4(CC)
  - keep4k(NC)
- **指標**
  - Kana-CER(kana-whisper)
  - ECAPA 類似度
  - UTMOSv2 hp40 と UTMOS22(`mos_pair` で同じ文どうしの差を取る)
  - 4–6 / 6–8 / 8–10 kHz の comb と level(`hf_structure.py` は 48 kHz 用に直す)
- **試聴ページ**: ラウドネスを揃える。難しいケースとして `jvs010_VOICEACTRESS100_081`、`jvs002_…_059`、female_02/05 を入れる。
- **判定**
  - DACVAE 再構成の女性 4–6 kHz comb が録音に近い(目安 ≥3.5 dB。録音 4.7 dB、元 Mimi 1.2 dB)。
  - Kana-CER が Mimi 系より良い。
  - **試聴で女性声、特に female_02/05 がはっきり良い。**
  - ここで DACVAE の利点が見えなければ、この計画の主な動機がなくなる。その場合は進め方を見直す。

**P0c 基準値**

- 教師(Yomi v3)を、生徒と同じ評価セット・同じ prompt で測る(§4)。これが上限の目安になる。
- 現行の CC リリース(D1-CC + soup4)の数値と並べる。

### P1: 教師 latent の生成(PoC 分で約 0.5 GPU 日、全量で約 1 GPU 日)

- **スクリプト** `scripts/teacher/gen_latents.py`
  - main の `scripts/synth/irodori_synth.py` と、Irodori-TTS-Experiments の `generate_teacher_latents.py` を元にする。
  - 保存するもの:
    - decode 前の z(切り詰めない、fp16 `[T,32]` を shard にまとめる)
    - `speech_end`(flattening point)
    - text(元の文と正規化後の両方)
    - ref_id、seed、サンプリング設定、教師の版
- **設定**: main の合成と同じにする。
  - 教師は Yomi v3、40 step、CFG text 3.0 / speaker 5.0(t∈[0.5,1])
  - DiT は bf16、text encoder と codec は fp32
  - caption なし
  - 短文は S9 と同じく `seconds` を明示する(0.13 s/mora)
- **入力にする組(推奨: 最初から CC BY のみ)**
  - main の CC ラインで QC を通った (text, ref_id, seed) を使う。`data/synth/*_cc/manifest.jsonl` にあり、309k 文 / 553 h。
  - 同じ seed で生成し直すので、出力は以前の音声とほぼ同じになるはず。ただし batch 構成による bf16 の数値差はありうる。
  - 元の入力文と seed をどう復元するかは、P1 の最初に確認する。
- **参照 wav**
  - `refs.jsonl` には galleria2 上のパスしか書かれていない。`shard` / `key` を使って、`/mnt/ext11/Emilia-Dataset` の tar から本機に取り出し直す。
  - CC 用の Emilia-YODAS の声は、manifest の `ref_id` から作り直す。
  - 参照は DACVAE で符号化し、prompt latent として一緒に保存する。
- **QC**
  - decode した音声を kana-whisper にかけ、main と同じ基準(target 完全一致、文 Kana-CER ≤0.08)で判定する。
  - 不合格のものも flag を付けて残す。
  - 音声は試聴用のサンプル(約 500 clips、48 kHz)だけ残す。それ以外は latent からいつでも作り直せる。
- **量**
  - まず s3_cc から約 120k 文(約 220 h)を作って PoC に回す。
  - そのあと CC 全量 553 h を作る。
  - latent は 25 Hz × 32 × fp16 で 1.6 KB/s。553 h でも約 3.2 GB。
- **分割**: 声とテキストの両方が学習と重ならない val を作る(例: 50 声)。

**P1-side: wheel 用 decoder の可否を早めに確認(推奨、P1 と並行)**

- **ねらい**: 最終ゴールの最大のリスクは「stock wheel の decoder(ELU の SEANet、因果)で、DACVAE latent から高域の倍音を出せるか」。
- **やり方**
  - stock の構成の decoder(まず 24 kHz 出力)を、DACVAE latent を入力にして学習する。
  - target は CC BY 音声の DACVAE decode 出力と録音。
  - レシピは main の `train_mimi_decoder.py` の mel + MS-STFT GAN。
  - 数時間(20–40k step)回し、11 GB 程度なので生成と並行できる。
- **測るもの**: 女性 4–6 kHz の comb を、DACVAE decoder と M2CH2_soup4 と比べる。
- **読み方**
  - 倍音が出る → 案 W1(§P4)が成り立つ。あわせて、M2 の天井は「Mimi latent の側にあった」と分かる。
  - 出ない → ELU と因果構造が天井。案 W2(upstream への追加)が必要になる。

### P2: PoC 生徒(6L・25 Hz、学習 4–6 h + 評価)

- **データ**: P1 の約 220 h。
- **学習設定**
  - AdamW(lr 2e-4、wd 0.1)、batch 64、EMA 0.999、bf16、per-layer compile
  - text / voice dropout 0.2
- **学習量**: まず 60k step。最初に steps/s と VRAM を測り、25 Hz で系列が倍になる影響を確かめる。
- **評価のタイミング**: 5k / 10k / 20k / 40k / 60k step で評価する。自由生成は**早い段階から聴く**。
- **評価(ChatGPT 案の 3: teacher-forced と自由生成を分ける)**
  1. **teacher-forced**
     - val の LSD loss。
     - 「TF サンプル decode」: 正しい過去を与えて各フレームを 1 つずつサンプルし、並べて decode した音声。
  2. **自由生成**(temperature 0.3、LSD 1 step、CFG 1.0(参考で 2.0)、EOS 閾値 -4)
     - JKYB-mini、JSUT 1k、jvs_bal、short_ja、10 voices × JSUT 20 文
  3. **教師との比較**: 同じ text と voice で、長さの比、ECAPA(生徒 vs 教師)、F0 の中央値の差。
- **判定(目安、PoC 用)**
  - **(a) 品質**
    - 10 voices の試聴で、女性声(特に female_02/05)が D1-CC + soup4 より良く、教師に近い。
    - 女性 4–6 kHz comb が soup4 経路以上である。
  - **(b) 明瞭さ**: JSUT Kana-CER ≤3%(D1-CC は 2.14%)、EOS 失敗 ≤1%。
  - **(c) JKYB-mini**: 学習データ量で決まる指標なので、基準は設けず記録だけする。
- **結果の読み方**
  - TF は良いのに自由生成で崩れる → AR の誤差蓄積が問題 → P3c へ。
  - TF でも悪い → 容量、データ、または表現の問題。

### P3: 改善と比較(数日〜1 週間)

**一度に 1 つだけ変える。** 品質が下がったときに原因を切り分けられるようにするため。

- **P3a 2 フレーム連結**: 64 次元・12.5 Hz と 25 Hz を、同じデータ・同じ step 数で比べる。見るのは品質、CER、CPU の実時間倍率(推定 4.7 倍 vs 2.4 倍)。
- **P3b データの拡大**
  - CC 全量 553 h に広げる。
  - さらに、JKYB の出現回数を狙ったテキスト(main の S6/S7 のリスト)と短文(S9)を足し、1,000–1,500 h まで増やす。
  - QC で落ちた難しい読みは、main と同じく target をカタカナにして合成し直す。
- **P3c 誤差蓄積への対策**(自由生成が TF より明らかに悪い場合だけ): 過去 latent に小さいノイズを足す、または、生徒が生成した過去を一部混ぜる。
- **P3d CFG**: CFG 1.0 で足りるか確かめる。足りなければ、CFG を焼き込む D0 型の蒸留を wheel 化の前に入れる。
- **P3e 容量**: 6L を直接学習する場合と、24L で学習してから 6L に depth distill する場合を比べる。CPU 速度を優先するなら upstream の small 系(drifting head)も候補にする。
- **P3f 実録音を混ぜる**: YODAS3 の 559 h(CC BY)を DACVAE で符号化して混ぜる。韻律の多様性や自然さと、教師だけで学習した場合を比べる。
- **P3g prompt の多様化**: 同じ声の別の合成発話や、任意の録音を prompt にする。wheel では利用者が好きな音声を渡すため。

### P4: 公式 wheel 化(P2/P3 で品質を確認できてから)

| 案 | 中身 | 長所 | 短所・リスク |
|---|---|---|---|
| **W1** stock の構成に蒸留 | DACVAE decoder を模倣する stock 構成の因果 decoder を作る(P1-side の延長)。同様に、DACVAE encoder を模倣する SEANet encoder を作って latent を回帰させる。最後に、その encoder で FlowLM の `speaker_proj` を短く FT する | wheel ≥4.0 にそのまま載り、ストリーミングもできる | ELU / 因果構造の天井(P1-side で判定)。encoder の模倣誤差 |
| **W2** upstream に追加 | pocket-tts に DACVAE 型(Snake、先読みチャンク)の decoder を選べるオプションを PR する | DACVAE の品質をそのまま出せる | Kyutai に採用されるかどうか次第。先読み約 280 ms ぶん初回遅延が増える |
| **W3** voice を固定 | prompt は使わず、`export-voice` で作った声の状態だけを配る | encoder が要らない | voice cloning ができない |

どの案でも次は共通:

- CFG は焼き込む。
- `model_recommended_frames_after_eos` を 25 Hz 用に設定し直す。
- `package_release.py` を流用する。
- 必要な wheel が ≥4.0 になる。PyPI に出ているかは要確認。
- モデルカードには、CC BY の表記と Irodori の倫理的利用制限を書く。

## 4. 評価プロトコル

- **評価セット**(`data/eval/` を流用)
  - JKYB-mini(1,500)、JSUT 1k、short_ja(172)
  - jvs_bal(5F + 5M × 10)。ファイルとしては置いていないので、`data/jvs_ver1` から main と同じ選び方で作る
  - リリース voices 10 声 × JSUT 20 文(`exp/release_cc/voices/`)
- **難しいケース**
  - jvs026 / 055 / 096 の「一方で、漁業と商業で、リャネス港は繁栄していた。」
  - female_02 / female_05
  - 男性の対照: jvs009 / 031
- **生成設定**: temperature 0.3、LSD 1 step、EOS 閾値 -4、長さの上限は tokens/3+2 s、seed 0。CFG は 1.0 を主にし、2.0 は参考。
- **指標**
  - Kana-CER(kana-whisper)、Text CER(whisper-large-v3-turbo)
  - ECAPA 類似度、LUFS の差
  - UTMOSv2 hp40 / UTMOS22。50 GB 級の RAM を使うので、`flock runs/.mos.lock` で 1 本ずつ回す。
  - 高域の comb / level(48 kHz 対応版)
  - EOS 失敗数、テキストの後に余計な発話が出る率
- **比較の条件**: 48 kHz のままと、24 kHz にリサンプルしたもの(Mimi 系と並べる用)の両方で出す。
- **判定の主役は試聴**
  - 同じ文・ラウドネスを揃えたページで比べる。
  - 6 文程度の試聴ページでは約 7% の確率で順位が入れ替わる。数値は 1,000 clips 規模で取る。
- **誤差の大きさ**: JKYB-mini は評価ごとに ±0.5–1pt 揺れる。

## 5. コードとデータの構成

- **持ってくるもの**(型ヒント付きで整理し直し、必要なものだけ)
  - main:
    - `src/pocket_tts_ja/{text,kana,loader,post_eos}.py`
    - `scripts/train_ja.py`
    - `scripts/{eval_tts,eval_queue,asr_transcribe,score_jvs_cer,summarize_evals,make_listening_page}.py`
    - `scripts/synth/{irodori_synth,select_refs,finalize_s3}.py`
    - `scripts/{make_model_config,package_release}.py`(P4)
  - mimi_ft:
    - `scripts/m2/{mos_eval,mos_pair,hf_structure,hf_coherence,make_recon_page,recon_jvs,space_voices,space_voices_report,make_space_page}.py`
    - 依存: utmosv2、pyworld
  - Irodori-TTS-Experiments: `scripts/{generate_teacher_latents,batch_synth}.py` の latent 保存部分。
- **新しく書くもの**
  - `src/pocket_tts_ja/dacvae.py`: encode / decode と -16 dB 正規化のラッパー。
  - DACVAE 用の生成ループ: pocket_tts の FlowLM で latent を作り、DACVAE で decode する。stock の `TTSModel` は Mimi 前提なので、研究用の評価では使わない。
  - loader の 25 Hz 対応。
- **出力の置き場所**: 新しい生成物はすべて `data/irodori_distill/`、`runs/irodori_distill/`、`exp/irodori_distill/` の下に置き、main の生成物と混ざらないようにする。

## 6. 既存ローカル生成物の扱い(提案)

- **当面はそのまま置いておく**
  - /home には 389 GB の空きがある。本計画で増えるのは 50 GB 未満の見込み(latent 数 GB、生徒の snapshot など)。
  - 移動すると、main の記録に書かれたパスが切れる。
- **流用するもの**
  - `data/eval/`、`data/tokenizer/{ja_4k,ja_cc_4k}`、`data/models`(ECAPA、DNSMOS)、`data/jvs_ver1`
  - `data/synth/` のテキストリストと manifest(Mimi latent は使わない)
  - `exp/release_cc/voices/`
  - 基準ウエイト: `runs/D1_cc_6l`、`runs/D1_6l_b`、`runs/M2H_soups/M2CH2_soup4.safetensors`、`runs/M2_gs2gan_keep4k`
- **空きが要るときの移動候補**: `runs/BCC_4k`(130 GB)などの中間 run、合わせて約 300 GB。NAS は 97% 埋まっているので、移すなら事前に相談する。

## 7. リスクと対策

| リスク | 対策・判定 |
|---|---|
| DACVAE も女性の高域を再現できない | P0b で判定し、だめなら計画を見直す |
| 非 AR の教師が持つ大域的な韻律を、AR 生徒が追えない / 長文で崩れる | TF と自由生成を分けて評価する(P2)。P3c、P3e |
| 25 Hz で AR の step 数が倍になる(CPU 速度、学習時の系列長) | P3a の 2 フレーム連結 |
| 教師の誤読をそのまま覚える(教師の上限は JKYB 95.7%) | QC と、カタカナでの再合成 |
| QC が難しい読みだけを落とす(main の教訓) | カタカナ target での再合成 |
| 短文が引き伸ばされる(duration predictor) | `seconds` を明示する(S9 方式) |
| wheel の decoder の天井(ELU、因果) | P1-side で早めに確認し、W1 / W2 を判断する |
| ストリーミング: DACVAE が非因果 | 研究評価は全文 decode。wheel では W1 が因果、W2 は先読みチャンク |
| ライセンス: DACVAE の源流 facebook/dacvae-watermarked のカードが、YAML では apache-2.0、本文では SAM License と食い違っている | 公開前に確認する。Irodori 本体と codec は MIT(倫理的利用制限あり) |
| watermark: 教師の音声には SilentCipher が入るが、生徒の出力には入らない | 公開時の方針を決める(§8) |
| GPU が 1 枚で、生成と学習が競合する | PoC は 220 h に絞る。生成中の空き VRAM で P1-side を回す |

## 8. 決定事項(ユーザー、10/22)

1. **ライセンスのライン**: 最初から CC BY だけで揃える。参照声は Emilia-YODAS、テキストは main の CC ラインで監査済みのもの。
2. **P1-side**: P1 と並行して早めに回す。
3. **wheel 化の方式**: P1-side と P2 の結果を見てから決める。希望は W1。ただし W2 / W3 も場合によってはありうる。
4. **既存ローカル生成物**: 現状のまま置いておく。
5. **watermark**: 基本的には入れない方向。
6. **galleria2**: GPU は別の実験(mimi_ft)で使用中。ディスクの読み取りだけなら `ssh galleria2-direct` でアクセスしてよい。これまでのデータは `/home/takuma/co/pocket-tts-ja`、`-git`、`-synth-env` のどこかにある。GPU が空いたらユーザーが知らせる。
7. **指標**: mimi_ft で試聴に近いと分かった NISQA を、UTMOSv2 と並べて使う。使うのは NISQA-TTS の自然さと、NISQA v2 の MOS / coloration(mimi_ft `mimi_ft_cc_plan.md` C6)。どちらも 48 kHz の全帯域を見るので、UTMOS 系の 16 kHz の盲点を補える。

## 9. 進捗ログ

(ここに追記していく)
