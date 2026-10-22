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
  - main の CC ラインで QC を通った (text, ref_id, seed) を使う。`data/synth/*_cc/manifest.jsonl` にあり、326,640 文 / 553 h(3,000 声)。
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

### 10/22 P0a: 環境とツール(27b5cc0, dbf6444)

- **環境**
  - 本体: torch 2.14.1、`dacvae` は 414c207 に固定、protobuf は override。
  - 教師: `third-party/Irodori-TTS/.venv`(`uv sync --extra cu128`、torch 2.10)。
  - ASR: JKYB の Python 3.13 環境。
  - NISQA: `third-party/NISQA`。重みは CC BY-NC-SA なので評価専用。
- **DACVAE の符号化**(`src/pocket_tts_ja/dacvae.py`)
  - Irodori と同じく torchaudio で resample する。soxr だと latent が平均 0.056 ずれた。
  - irodori_tts.codec との差は encode が最大 0.0075、decode が 1e-4。
- **評価セット**(`eval_sets.py`): voices / jsut / short / jvs_bal / jkyb_mini。jvs_bal は ROTATION の 4 話者を除いた男女各 5 名。
- **NISQA**(`nisqa_eval.py`): mimi_ft と同じ数値が出ることを確認した(female_02: 4.62 / 4.33 / 4.20 / 3.55)。

### 10/22 P0b: codec の比較 → DACVAE で進める(数値と試聴の両方で確定)

- **素材**(`exp/irodori_distill/p0/recon_*`)
  - Space: リリース 10 声の prompt(Irodori 生成)。
  - Anime: Irodori voice design の高い声、150 clips。
  - y3hi: YODAS3 の自然音声。F0 ≥ 300 Hz の女声 300 clips(1 動画 4 clips まで)。
  - JVS: 男女各 5 名 × 100 clips。
- **経路**
  - 24 kHz で比べた(DACVAE は 48 kHz decode を soxr で 24 kHz に落とした)。
  - Mimi 系: 元 Mimi、keep4k(NC)、soup4(現行 CC リリース)、base2(M2C_base2)。
- **4–6 kHz comb と類似度**(女声。JVS は男女別、y3hi / Anime は女声のみ)

| | 録音 | DACVAE | 元 Mimi | keep4k | soup4 | base2 |
|---|---:|---:|---:|---:|---:|---:|
| comb 4–6k dB: JVS F | 4.8 | 5.1 | 1.2 | 1.4 | 1.5 | 1.5 |
| 〃 JVS M | 3.2 | 2.3 | 0.3 | 0.4 | 0.4 | 0.4 |
| 〃 y3hi | 5.9 | 6.1 | 2.0 | 2.2 | 2.0 | 2.2 |
| 〃 Anime | 6.9 | 7.1 | 1.7 | 1.8 | 1.3 | 1.8 |
| level 4–6k dB vs 録音: JVS F | 0 | −1.2 | −6.8 | −6.7 | −1.3 | −6.5 |
| F0 誤差の中央値(cents): JVS F | 0 | 1.3 | 3.8 | 3.5 | 3.5 | 3.6 |
| 〃 y3hi | 0 | 1.7 | 4.3 | 4.3 | 4.3 | 4.3 |
| 〃 Anime | 0 | 1.0 | 4.7 | 4.5 | 4.8 | 4.6 |
| 1 周期相関 4–6k(波形): JVS F | 0.15 | 0.16 | 0.03 | 0.04 | 0.05 | 0.05 |
| 〃 y3hi | 0.19 | 0.19 | 0.05 | 0.07 | 0.06 | 0.07 |
| 〃 Anime | 0.21 | 0.22 | 0.06 | 0.07 | 0.05 | 0.07 |
| ECAPA 類似度 vs 録音: JVS | – | 0.891 | 0.698 | 0.779 | 0.805 | 0.766 |
| 〃 y3hi | – | 0.887 | 0.699 | 0.767 | 0.802 | 0.763 |
| 〃 Anime | – | 0.975 | 0.767 | 0.840 | 0.836 | 0.835 |
| 〃 Space | – | 0.983 | 0.833 | 0.885 | 0.881 | 0.879 |

- **自然さ・内容**

| | 録音 | DACVAE | 元 Mimi | keep4k | soup4 | base2 |
|---|---:|---:|---:|---:|---:|---:|
| NISQA-TTS: JVS | 4.616 | 4.542 | 4.351 | 4.287 | 4.380 | 4.213 |
| 〃 y3hi | 4.136 | 4.313 | 4.183 | 4.020 | 4.131 | 3.952 |
| 〃 Anime | 4.038 | 4.005 | 3.667 | 3.426 | 3.327 | 3.402 |
| 〃 Space(10 clips) | 3.716 | 3.652 | 3.791 | 3.634 | 3.557 | 3.584 |
| NISQA v2 col: JVS | 4.397 | 4.404 | 4.264 | 4.217 | 4.281 | 4.126 |
| 〃 y3hi | 4.024 | 4.039 | 4.077 | 3.908 | 3.963 | 3.801 |
| 〃 Anime | 3.390 | 3.240 | 2.854 | 2.944 | 2.867 | 2.887 |
| UTMOSv2 hp40: JVS | 3.413 | 3.400 | 2.973 | 3.007 | 3.394 | 3.179 |
| 〃 y3hi | 2.941 | 2.898 | 2.551 | 2.583 | 2.909 | 2.677 |
| 〃 Anime | 2.621 | 2.542 | 2.050 | 2.103 | 2.362 | 2.125 |
| Kana-CER: JVS(読み基準) | 3.20% | 3.52% | 5.32% | 4.56% | 4.56% | 4.59% |
| 〃 Anime(読み基準) | 7.93% | 8.05% | 9.48% | 9.07% | 8.92% | 9.15% |
| 〃 y3hi(録音の書き起こし基準) | – | 2.05% | 5.16% | 4.17% | 3.91% | 4.19% |

(全数値は各 `exp/irodori_distill/p0/recon_*/report_*.txt`)

- **読み**
  - **DACVAE は 4 セットとも、どの指標でも録音にほぼ並ぶ。** 女声の 4–6 kHz comb は録音並み(JVS F 5.1 vs 4.8、y3hi 6.1 vs 5.9)。F0 の揺れは Mimi 系の 1/3 以下。類似度は Mimi 系より 0.09–0.21 高い。
  - Mimi 系の decoder はどれも高域の倍音を作れない(comb 1–2 dB、1 周期相関 ≈0)。soup4 は高域の level だけを戻している(JVS 4–6k level −1.3 dB、keep4k / base2 は −6.5 dB)。M2 の天井はここ。
  - UTMOSv2 は soup4 の高域ノイズを本物の倍音と区別できない(JVS 3.394 vs DACVAE 3.400、y3hi 2.909 vs 2.898)。NISQA-TTS は分けられる(Anime 3.33 vs 4.01、Space female_02 3.55 vs 4.55)。§8-7 のとおり NISQA を主に見る。
  - DACVAE の弱点は男声の 4–6 kHz comb(2.3 vs 録音 3.2)と、8–10 kHz の level(−3.3 dB)。どちらも Mimi 系より良い。
  - Space(10 clips)だけは NISQA-TTS で元 Mimi が録音より上に出る。clip が少なく、prompt 自体が Irodori 生成なので、重くは見ない。
  - 48 kHz の DACVAE 出力(dacvae48)は、24 kHz の素材に対して NISQA mos / col がやや下がる(JVS col 4.327 vs 24 kHz 4.404)。DACVAE が 12 kHz より上に何かを足している。24 kHz の wheel では問題にならない。
- **試聴ページ**: `exp/listening/p0_codec/index.html`(ローカル専用、JVS を含む)。4 セクション × 7 列。各声から有声の多い clip を 2 つ選んだ。
- **判定**: 数値の上では、§P0b の基準をすべて満たした。
- **試聴(ユーザー、10/22)**: DACVAE は 48 kHz・24 kHz とも問題なし。録音との差も、48 kHz と 24 kHz の差も聞き分けられなかった。Irodori の生成音声だけでなく、YODAS3 と JVS でも同じ。**DACVAE で確定**。

### 10/22 P0c: 教師(Yomi v3)の基準値

- watermark なし、評価セットごとに `exp/irodori_distill/eval/<set>/`。teacher latent も `teacher_latents/` に保存した(P1-side の val に使う)。

| set | Kana-CER | 類似度 | NISQA-TTS | UTMOSv2 hp40 |
|---|---:|---:|---:|---:|
| voices | 1.18% | 0.779 | 3.609 | 3.097 |
| jvs_bal | 2.05%(録音 1.69%) | 0.712 | 4.375 | 3.550 |
| short | 3.71% | 0.507 | – | – |
| jsut | 1.77% | 0.708 | 4.422 | 3.520 |

- 現行リリース(D1-CC + soup4)は JSUT 2.14%、JVS 2.18% / 0.651。教師はこれより Kana-CER で少し、類似度で 0.06 良い。生徒が蒸留で狙えるのは、主にこの差と音質(高域)になる。
- 速度は 41.8 audio-s/s(jsut)。553 h なら 13–14 h。

### 10/22 P1: 教師 latent の生成を開始

- **入力**(`scripts/teacher/make_rows.py`)
  - main の CC ラインの manifest から (text, ref, seed) を復元した。7 セットで 326,640 文 / 553.4 h / 3,000 声。
  - 内訳: s3 148,815 / 275 h、s4 16,118 / 15 h、s5 34,956 / 65.2 h、s6a 26,755 / 52.1 h、s6b 64,418 / 124.7 h、s7 10,851 / 12.3 h、s9 24,727 / 9.1 h。
  - 参照 wav は galleria2 から読み取りだけで持ってきた(refs_cc 2,311 + refs_yodas 689)。
  - prompt latent(−16 dB に正規化してから符号化)は `data/irodori_distill/teacher/refs/` に置いた。
- **QC を計画から変えた**: 「target 完全一致」ではなく、以前の合格音声との書き起こしの一致で判定する(`scripts/teacher/qc.py`、Kana-CER ≤0.02)。同じ seed・text・参照で生成し直すので、以前の書き起こしが手本になる。
  - 最初の 1,000 clips で確かめた。以前の書き起こしと完全一致が 95.4%、≤0.08 が 99.0%、平均 0.0032。
  - 外れた 4.6% は、batch 構成による bf16 の差で、別のテイクが出たもの(例: 「テンキモラスベカラズ」→「テンキモラスレカラアズ」)。計画どおり落とす。
- **speech_end**
  - Irodori の flattening point は、末尾の短い無音(−100 dB)を見逃す。
  - そこで、16 kHz の QC 音声のフレーム RMS(ピークから 45 dB 以内)で決める(`scripts/teacher/pack.py`)。
  - latent は切らずに丸ごと保存する。
- **生成**: `runs/irodori_distill/p1_generate.sh s3`(detached)。latent は fp16 で、QC 用に 16 kHz の音声も書く。
  - 速度は単独で 6.8 clips/s。P1-side と並行すると 4.8 clips/s(s3 は約 8 h)。
- **QC**: `runs/irodori_distill/p1_pipeline.sh` が行う。
  - 生成中も 10 分ごとに、新しい clip を kana-whisper にかける。P1-side と並行する間は batch 4(約 7 clips/s)。
  - セットの生成と書き起こしが揃ったら、qc → pack → 音声の削除。
  - s3 が終わったら、残りの 6 セットの生成に進む。

### 10/22 P1-side: wheel 用 decoder の probe を開始

- `runs/irodori_distill/W1_probe`(`p1side_probe.sh`、detached)。生成中の s3 latent で学習する(検証ごとに読み直す)。
- batch 8 は生成と並ぶと OOM になるので、batch 4 にした(10.4 GB、GAN 込み)。
- 速度は mel だけで 12 step/s、GAN 込みで 2.8 step/s。30k step で約 2.7 h。
- step 2500(mel のみ): comb46 1.2 dB(target 5.5)、4–6 kHz の level +8.4 dB。倍音はまだなく、ノイズ状の高域が出ている。GAN 前なので想定どおり。
- step 5000 / 7500(GAN 1k / 3.5k step): comb46 1.07 → 1.02 dB(target 5.5)、level46 +4.3 → +4.1 dB。高域はまだノイズのまま。
- step 10000 / 12500 / 15000: comb46 1.03 / 0.84 / 0.79 dB、level46 +4.2 / +4.1 / +3.9 dB。
- **倍音の読みは、まだ意味がない。** 原因は 200 Hz(SEANet の内部フレームレート)と 25 Hz の倍数に立つスペクトル線、つまりバズ。
  - 線の強さ(長時間平均スペクトル、2–10 kHz)

| | 200 Hz の線 |
|---|---:|
| probe step 2500(mel のみ) | 12.9 dB |
| probe step 7500 | 5.7 dB |
| probe step 15000 | 4.7 dB |
| 録音 | 0.3 dB |
| DACVAE | 0.65 dB |
| 元 Mimi | 1.6 dB |
| soup4 | 0.4 dB |

  - この線は自前 F0 の comb にも混ざる。step 2500 で comb 4–6k が 8.0 dB(target 4.5)と出たのは、この線のため。
  - 新しく作った部分(`output_proj` と stride 8 の upsample)を乱数で始めたことで、stock の decoder に分布外の入力が入り、バズが出ていると読む。
- **線形の橋渡し初期化は失敗した**(`scripts/wheel/bridge_init.py`)。
  - やり方: upsample を線形補間で始め、`output_proj` を最小二乗で決める。目標は「教師 latent → DACVAE decode → stock Mimi encode」で得た、stock decoder transformer の入力。
  - 結果: R² 0.146(最適な lag 25 ms)。橋渡しした decoder は 200 Hz の線が 44.7 dB になった。
  - DACVAE と Mimi の latent は線形には結びつかない。stock の transformer と SEANet は、学習で適応させるしかない。
- probe は step 15000 から再開した(10/22 07:25)。
  - 学習 clip の DACVAE decode を一度だけ行い、メモリに保持するようにした(20k clips まで)。速度はほぼ変わらない(2.7 step/s)。
  - 生成は probe の GAN 区間と ASR に挟まれ、1.3–1.8 clips/s に落ちる(単独の 1/4 以下)。

### 10/22 P2 の準備(58df703, a7dc740)

- **終端の設計を修正**: 教師の latent には平坦な tail がない。
  - Irodori の duration predictor は窓を発話にぴったり合わせる。最後のモーラは、保存された最後の 2–3 フレームで減衰して終わる。
  - エネルギーで測った speech end のあとのフレーム数は、s3 で中央値 0。
  - そこで、§2 の「教師自身の tail 12 フレーム」は、ほぼすべて DACVAE の無音 latent(`src/pocket_tts_ja/dv_silence_latent.npy`、デジタル無音の encode)になる。
  - mask は uint8(1 発話 / 2 tail)。EOS は tail の最初のフレーム。flow loss は tail にもかける(`pocket_tts_ja.post_eos`、main 版の `z_proj` 抜けを直した)。
- **学習**(`scripts/train/train_student.py`、`configs/train/P2_6l.yaml`)
  - upstream の学習ループに、main の `train_ja.py` と同じ差し替えを入れた。
  - 差し替えるもの: loader、encode_batch、forward、build_models、write_samples(DACVAE で decode)。
  - D1-CC から warm start する。作り直すのは `input_linear` / `bos_emb` / `speaker_proj_weight` / `flow_net` / `out_eos`。
  - latent 統計は s3 の発話フレームから計算して固定する。DACVAE latent はもともと mean ≈0.13、std ≈1.0。
  - CPU の smoke test で、warm start、統計、mask、sample、checkpoint、snapshot まで通ることを確かめた。
- **評価**
  - `scripts/eval/student_synth.py`: upstream の batched `generate` を使う。設定は temp 0.3、LSD 1 step、CFG 1.0、EOS −4 + 12 フレーム。出力は 24 kHz。
  - `runs/irodori_distill/p2_eval.sh`: 生成と採点をまとめて行う。
  - 現行 CC リリース(D1-CC + soup4)を、同じ項目で並べる。
    - jsut / jvs_bal / short は main の出力を使う(key・prompt・text が完全に一致する)。
    - voices は stock wheel 3.3.0 で作った(`scripts/eval/wheel_generate.py`)。
- **基準値**: 生成音声の自前 F0 の comb(`scripts/eval/tts_comb.py`、voices 女声の 4–6 kHz)

| | 録音(prompt) | 教師 | リリース |
|---|---:|---:|---:|
| 女声全体 | 4.4 | 4.2 | 1.6 |
| female_02 | 5.6 | 5.6 | 1.7 |
| female_05 | 8.0 | 6.8 | 0.7 |

- NISQA-TTS の教師 − リリース: jsut +0.29、jvs_bal +0.17、short +0.43、voices −0.09。
  - voices でリリースが上回るのは、男声(3.58 vs 3.28)のため。男声の prompt 自体が合成音声で、NISQA-TTS は 3.31。

### 10/22 P1-side: 30k step の結果 → 小規模の W1 は使えない。probe はいったん止める

- probe(batch 4、30k step、GAN は 5k step から、約 5 h)は 08:46 に終わった。
- 学習中の検証(voices の female_02 / 05 × JSUT)
  - mel は 0.323 → 0.297(step 17.5k → 30k)。まだ少しずつ下がっている。
  - comb46 は 0.74 → 0.70 dB(target 5.5)。4–6 kHz の level は +3.7 → +4.4 dB。
- **P0b の再構成セットでの採点**(`p1side_eval.sh`、DACVAE encoder → probe decoder、24 kHz)

| | 録音 | DACVAE | 元 Mimi | soup4 | probe 30k |
|---|---:|---:|---:|---:|---:|
| 200 Hz の線 dB: Anime | 0.48 | 0.42 | 5.32 | 4.17 | 3.03 |
| 〃 y3hi | 0.29 | 0.48 | 1.89 | 0.89 | 1.66 |
| 〃 JVS | 0.26 | 0.54 | 1.46 | 0.25 | 0.97 |
| 25 Hz の線 dB: Anime | −0.06 | −0.27 | 0.37 | 0.37 | **1.50** |
| comb 1–2.5k(録音の F0): Anime | 23.3 | 22.2 | 18.4 | 16.8 | **8.7** |
| 〃 y3hi | 20.2 | 19.1 | 18.4 | 16.8 | **10.4** |
| 〃 JVS F | 19.3 | 18.0 | 16.8 | 15.9 | **10.8** |
| comb 2.5–4k(自前 F0): Anime | 9.9 | 10.3 | 8.6 | 7.0 | **1.2** |
| 〃 y3hi | 8.2 | 9.2 | 9.1 | 7.9 | **1.0** |
| 〃 JVS F | 8.0 | 9.0 | 8.4 | 7.7 | **1.0** |
| comb 4–6k(自前 F0): JVS F | 3.7 | 4.4 | 1.0 | 1.3 | 0.3 |
| F0 誤差の中央値 / p90(cents): Anime | 0 | 1.0 / 4.5 | 4.7 / 20.2 | 4.8 / 20.1 | **12.4 / 94.8** |
| 〃 y3hi | 0 | 1.7 / 6.9 | 4.3 / 17.0 | 4.3 / 16.8 | **10.2 / 44.2** |
| 〃 JVS F | 0 | 1.3 / 5.2 | 3.8 / 14.5 | 3.5 / 13.0 | **8.4 / 31.8** |
| 有声フレームの一致: Anime | 100% | 98.9% | 96.7% | 96.1% | **79.0%** |
| ECAPA 類似度 vs 録音: Anime | – | 0.975 | 0.767 | 0.836 | 0.745 |
| 〃 y3hi | – | 0.887 | 0.699 | 0.802 | 0.730 |
| 〃 JVS | – | 0.891 | 0.698 | 0.805 | 0.751 |
| NISQA-TTS: Anime | 4.038 | 4.005 | 3.667 | 3.327 | **2.073** |
| 〃 y3hi | 4.136 | 4.313 | 4.183 | 4.131 | **2.538** |
| 〃 JVS | 4.616 | 4.542 | 4.351 | 4.380 | **3.133** |
| NISQA v2 col: JVS | 4.397 | 4.404 | 4.264 | 4.281 | **3.247** |

(全数値は `exp/irodori_distill/p0/recon_*/report_W1_30k_*.txt`)

- **読み**
  - バズは減った。200 Hz の線は元 Mimi 並みかそれ以下になった。ただし 25 Hz(latent のフレームレート)の線が Anime で 1.5 dB 残る。
  - 問題は倍音ではなく、**周期性そのもの**。ピッチが揺れ(F0 誤差の中央値は元 Mimi の 2–3 倍、p90 は 2–5 倍)、有声フレームの 1–2 割が無声になる。そのため 1–2.5 kHz の comb が録音の半分になり、2.5 kHz より上には倍音の構造がほとんど残らない。元 Mimi は 2.5–4 kHz までは録音並み。
  - NISQA は soup4 より 1.0–1.6 低い。リリースに使える水準から遠い。
  - **ELU / 因果の天井については、まだ何も言えない**。この decoder は元 Mimi の水準(倍音は 2.5–4 kHz まで)にも届いていない。原因は学習量の不足と読む。新しい入力空間に、stock の transformer と SEANet を batch 4・30k step で合わせ直すのは小さすぎる。mimi_ft の decoder は、合った latent 空間の stock から始めて、もっと大きな予算で学習した。
  - mimi_ft の C6 / HF 計画の経験: Mimi latent からは、未学習の clip に 4 kHz 以上の倍音を出せなかった(学習した clip には出た)。DACVAE latent は倍音の情報を持っている(DACVAE の decode は録音並み)。stock 構成がそれを汎化して使えるかは、まだ開いた問い。
- **当面の扱い**
  - probe はここで止める。GPU は P1 の生成と P2 に回す。
  - W1 を本気でやるなら、decoder の学習を作り直すことになる。目安は batch 16 以上で 100k step 以上(本機を占有して 1 日以上)。P2 で生徒の質が確かめられてから、galleria2 の GPU が空くのを待って回すのがよい。
  - wheel 化の判断(§8-3)は予定どおり P2 のあと。その際の選択肢に、次の **W0** を足す。
    - **W0**: 教師の音声を stock Mimi で encode し直し、その latent(12.5 Hz)で生徒を学習する。D1-CC をそのまま(flow head も)warm start でき、decoder は現行の soup4、voice cloning も stock の encoder のまま。wheel 化の作業はほぼない。
    - 得られるのは言語モデル側の改善(読み・韻律・教師の安定さ)だけ。高域はリリースの天井(倍音なし)のまま。
    - P0b の数字で見ると、NISQA-TTS の codec による差(JVS: DACVAE 4.542 vs soup4 4.380)は、教師 − リリースの差(jsut +0.29)の半分ほど。残りの半分は言語モデル側の差と読める。
    - latent は保存してあるので、DACVAE decode → 24 kHz → Mimi encode で作れる。

### 10/22 P2a: 生成済みの s3 の一部で、P2 の経路を先に通す

- s3 の生成は残り約 12 万 clip(単独で約 5 h)。P2 本番を待つあいだに、経路のバグを早く見つけるための短い学習を回す。
- **s3a**: 09:22 時点で書き起こし済みの 29,568 clip(`rows_s3a.jsonl`、`teacher/s3a/` は s3 への symlink)。
  - QC: 一致 89.8%、CER ≤ 0.02 が 93.1%(27,531 clip が通過)、≤ 0.05 が 98.6%。最初の 1,000 clip(95.4% 一致)より少し低い。
  - pack: speech end 後のフレーム数は中央値 1、0 が 47.6%(s3 の最初の分と同じ)。
- **split**: `make_split.py --sets s3a --voices-from s3 --name P2a` で、train 27,065 行 / 49.7 h / 2,257 声、valid 466 行 / 39 声。
  - held-out の声を、全体の s3 と同じにした(`--voices-from` を足した。声の候補を rows_<set>.jsonl からとる)。P2 本番と同じ 50 声のうち、39 声が s3a に入っている。
  - 長さ: clip は中央値 5.9 s、p99 15.4 s、最大 23.1 s。参照声は中央値 7.8 s、最大 12.0 s。
- **メモリ**: 25 Hz では系列が main の 2 倍になる。batch 16 で、最長の bucket(23 s + 参照 12 s ≈ 900 フレーム)が 14.5 GB を超えた。
  - batch 64 を一度に載せるのは、GPU を占有しても無理(推定 50 GB 以上)。P2 本番も accumulation を使う(単独なら batch 16–32 × 4–2 になる見込み)。
- `configs/train/P2a_6l.yaml`: P2_6l と同じモデル・初期値・損失。batch 8 × accumulation 8(実効 64)、5k step、warmup 500。
  - 生成の横で 10 GB、1.3 step/s。約 65 分で終わる。
  - その間、パイプラインの ASR は batch 4 で回す(`p1side_probe.pid` に P2a の pid を書いた)。
- **結果(5k、10:11)**: 経路は最後まで通った。学習は進んでいるが、5k では TF でもまだ内容が出ない。
  - データの確認: pack 後の latent は clip ごとの latent と一致した(200 行)。DACVAE で decode して書き起こすと、テキストどおりに読んでいる(4 行)。text token も batch に正しく入っている。
  - 検証の flow_loss: 0.925(1k)→ 0.848 → 0.815 → 0.801 → 0.797(5k)。EOS loss は 0.009。
  - **TF の Kana-CER**(held-out 200 行。教師の latent は 3.29%): 1k 238%(ASR の暴走)→ 2k 132% → 3k 80% → 4k 66% → **5k 57%**。cosine で LR が 0 に向かうなかでも下がり続けている。
  - LSD の decode を 8 step にしても 56%(1 step は 57%)。one-step の写像ではなく、backbone と flow head がまだ弱い。
  - TF の有声フレームは教師の 55–57%。NISQA-TTS は教師 −0.97。
  - 自由生成(voices 200 文): Kana-CER 128%、EOS なし 48 文、類似度 0.283(教師 0.779、リリース 0.689)、NISQA-TTS はリリース −0.78。
  - ただし女声の有声フレームの comb は、2.5–4k 8.1 / 4–6k 5.2 dB(教師 8.0 / 4.2、リリース 5.9 / 1.6)。DACVAE 経路の高域の倍音は、生徒の初期でも出ている。
  - 読み: 学習量が足りない段階。fresh の input / flow head で 5k step(32 万サンプル)は短すぎる。main の生徒は、head の蒸留つきでも 20k で JKYB-mini 74%、60k で 92%。
- **P2 本番の段取り**
  - P2 と生成を同時に回すと、どちらも約半分の速さになる(生成は 6.8 → 3.1 clips/s)。並べても得はないので、順番に回す。
  - s3 の生成を単独で終える(残り約 10.8 万 clip、14 時台)。続けて QC と pack。
  - `runs/irodori_distill/p2_start.sh`(detached): s3 の QC_PACKED を待ち、`make_split.py --sets s3 --name P2_s3` を作り、P2_6l を単独で学習する。落ちたら checkpoint から再開する。終わったら残りのセット(s4 s5 s6a s6b s7 s9)の生成を始める。
    - パイプラインが s3 の後に残りを自動で始める部分は止めた。
  - P2_6l は batch 16 × accumulation 4(実効 64)、valid 60 batch。
  - `runs/irodori_distill/p2_eval_queue.sh`(detached): 5k / 10k / 20k / 40k / 60k の snapshot ごとに TF(`tf_val/`)と自由生成(voices・jsut。40k 以降は jvs_bal・short も)を測る。

### 10/22 galleria2 の GPU が使えるようになった(ユーザー、10:30 ごろ)

- galleria2 に `~/co/pocket-tts-ja-distill` を作った(このブランチの clone。`third-party/Irodori-TTS` は既存の `~/co/Irodori-TTS` への symlink で、同じ commit 89f9d8f)。教師の checkpoint、DACVAE、kana-whisper は galleria2 に既にある。参照 wav(`synth_src/refs_cc`、846 MB)と rows をコピーした。
- **s3 を 2 台で分けた**: 本機は rows 0–95k(`rows_s3.jsonl` を 95k 行に切り、全体は `rows_s3_full.jsonl` に残した)。galleria2 は残りの 53,815 行を s3b として生成する(`rows_s3b.jsonl`)。それぞれ約 2 h。
  - 本機の生成は 10:39 に止めて、切った rows で再開した(既存の clip は飛ばす。止める直前の出力は全部読めることを確かめた)。
- galleria2 は s3b のあと、s4 s5 s6a s6b s7 s9 を生成する(`p1_generate.sh`)。QC と pack も galleria2 で行う(`runs/irodori_distill/p1_qc_sets.sh`、p1_pipeline.sh と同じ手順)。
- `p2_start.sh` を書き直した: 本機の s3 と galleria2 の s3b の QC_PACKED を待ち、s3b の pack を取ってきて、`make_split.py --sets s3 s3b --voices-from s3 s3b --name P2_s3` を作り、P2_6l を本機で単独に学習する。held-out の声は P2a と同じ(声の候補は s3 全体)。
- **s3b の失敗と s3c**: s3 の参照は `refs_cc`(110,885 行)と `refs_yodas`(37,930 行)の 2 つのフォルダにまたがる。galleria2 には refs_cc だけをコピーしていたため、s3b の 37,930 行が参照の読み込みで失敗した。
  - 止める前に、s3b は生成できた 15,885 clip で QC と pack まで進んだ(15,000 clip が通過。audio は削除済み)。これはそのまま使う。
  - refs_yodas(689 声、257 MB)をコピーした。失敗した行を `rows_s3c.jsonl` として生成し直している(11:45 ごろから、約 1.6 h)。
  - s4 も refs_yodas の行を含む。途中で止めたので、再開時に失敗した行も生成される。最後に書かれた 1,600 ファイルは、すべて読めることを確かめた。
  - `p2_start.sh` は s3 / s3b / s3c の pack を待つ(`make_split.py --sets s3 s3b s3c --voices-from s3 s3b`)。P2 の開始は 13 時台後半の見込み。

### 10/22 P2: 学習開始(13:36)と 5k の結果

- **データ**: s3(本機 89,155 clip)+ s3b(15,000)+ s3c(35,705)。QC の通過率は s3 93.8%、s3b 94.4%、s3c 94.1%。
  - `make_split.py --sets s3 s3b s3c --voices-from s3 s3b --name P2_s3`: train 137,469 行 / 252.8 h / 2,950 声、valid 2,000 行 / 50 声(P2a と同じ held-out の声)。
  - latent の統計(330 万の音声フレーム): 平均の絶対値 0.135(最大 0.675)、std 1.005(0.768–1.619)。
- **速度**: batch 16 × accumulation 4 で 4.7 step/s(単独)。評価が横で走ると約 3.7 step/s。GPU メモリは学習と TF decode を合わせて 20 GB。
  - 60k step は約 4 h(18 時ごろに終わる)。s3 だけで約 28 epoch になる。
- **検証の flow_loss**: 0.828(2.5k)→ 0.759(5k)→ 0.714 → 0.685(10k)→ 0.666 → 0.652(15k)。EOS loss 0.0081 → 0.0054。まだ下がり続けている。
- **5k の評価**(LR はまだ最大付近。P2a の 5k は cosine で LR 0 まで下げきった値なので、同じ step でも比べられない)
  - **TF**(held-out 200 行):
    - Kana-CER 46.9%(教師の latent は 2.46%)。P2a の 5k は 57%。
    - 類似度 0.676。NISQA-TTS は教師 −0.94。
    - 有声フレームは教師の 51–57%。
  - **自由生成**: まだ崩れている。
    - voices: Kana-CER 118%、EOS なし 106/200、類似度 0.122、NISQA-TTS はリリース −0.75。
    - jsut: Kana-CER 134%、EOS なし 363/1000、類似度 0.068、NISQA-TTS はリリース −0.62。
    - 女声の comb は 2.5–4k 9.2 / 4–6k 6.0 dB(教師 8.0 / 4.2、リリース 5.9 / 1.6)。ただし有声フレームは教師の 37% しかない。
  - 読み: TF では声も内容も出はじめている(類似度 0.68)。自由生成で崩れるのは、まだ学習の初期だから。10k / 20k を見て判断する。
- **galleria2**: s3c(13:28)、s4(13:57 生成、14:08 pack、15,462 clip)が済んだ。s5 を生成中(約 6 clip/s)。残り s6a s6b s7 s9 で、全部終わるのは 21 時台の見込み。

### 10/22 P2: 10k の結果。TF は大きく良くなった。自由生成は voices では音質がリリース並みだが、JVS の声はまだ真似できない

- **TF**(held-out 200 行): Kana-CER 46.9% → **14.9%**、類似度 0.676 → 0.767、NISQA-TTS は教師 −0.74。
- **voices**(10 声 × JSUT 20 文):
  - Kana-CER 82.9%、EOS なし 38/200、類似度 0.538(EOS が出た 162 文は 0.605)。
  - NISQA-TTS はリリース −0.05(差なし)。mos −0.35、col −0.19。UTMOSv2 は −0.84。
  - 女声の comb は 2.5–4k 6.7 / 4–6k 3.5 dB。有声フレームは教師並みになった(5,793 / 6,614)。リリースは 5.9 / 1.6 dB。
- **jsut**(JVS 4 声 × 1,000 文):
  - Kana-CER 95.1%、**EOS なし 601/1000**、**類似度 0.063**(教師 0.708、リリース 0.606)。NISQA-TTS はリリース −0.36。
- **JVS の prompt を調べた**(最初の 200 文、10k)
  - 4 つの prompt は約 5 s で、前後に 0.5 s ずつ無音がある。音量は −27.6 dB。学習の参照(refs_cc / refs_yodas)は中央値 7.7 s(最短 5.0 s)、前 0.05 s・後 0.10 s まで詰めてあり、約 −18 dB。音量は encode の前に −16 dB に揃えている。
  - 無音を詰めた場合(約 4–5 s): 類似度 0.057 → 0.098、Kana-CER 91% → 103%、EOS なし 100/200。
  - 無音を詰め、同じ話者の 2 文目をつないだ場合(7.6–10.4 s): 類似度 0.132、Kana-CER 89%、EOS なし 73/200。
  - DACVAE latent の統計(学習の統計で正規化した平均・std)は、JVS も学習の参照と同じ範囲だった。
  - 崩れる前の最初の 3 s でも、prompt との類似度は 0.25–0.31 しかない(voices は 0.57)。出力は話者ごとに違う(平均 embedding の cosine は 0.3–0.7)ので、prompt を無視しているのではない。
  - 読み: 無音と長さは少しだけ効く。主な原因は、短いスタジオ録音の声にまだ汎化できていないこと。学習の参照は YODAS / CC 由来の 2,950 声だけ。20k 以降でも残るなら、P3g(prompt の多様化)の優先度を上げる。
- 残りの評価は 20k / 40k / 60k。40k からは jvs_bal と short も測る。
