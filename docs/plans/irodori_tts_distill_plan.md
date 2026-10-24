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

### 10/22 P2: 20k の結果。voices の自由生成で音質・声とも リリース並みになった。JVS は prompt の長さで大きく変わる

- **検証の flow_loss**: 0.634(20k)→ 0.624(25k)→ 0.617(30k)。
- **TF**(held-out 200 行): Kana-CER **6.69%**(教師の latent 2.46%)、類似度 0.806、NISQA-TTS は教師 −0.62。
- **voices**(10 声 × 20 文):
  - Kana-CER 32.3%(10k 83%)、**EOS なし 2/200**(10k 38)。
  - 類似度 **0.702**(教師 0.779、リリース 0.689)。
  - リリースとの差: NISQA-TTS +0.02、mos +0.05、col +0.10(女声 +0.23)。UTMOSv2 は −0.46。
  - 女声の comb は 2.5–4k 6.8 / 4–6k 3.6 dB(教師 8.0 / 4.2、リリース 5.9 / 1.6)。有声フレームは教師並み。
- **jsut**(JVS 4 声 × 1,000 文、元の prompt): Kana-CER 81.0%、**EOS なし 729/1000**(10k の 601 より増えた)、類似度 0.111、NISQA-TTS はリリース −0.22。
- **JVS の prompt の長さ**(最初の 200 文、20k):

  | prompt | 長さ | EOS なし | Kana-CER | 類似度 |
  |---|---:|---:|---:|---:|
  | 元のまま(前後に無音 0.5 s) | 4.8–6.1 s | 約 7 割 | 77.9% | 0.114 |
  | 無音を詰めた | 3.9–5.3 s | 98/200 | 66.2% | 0.211 |
  | 詰めて 2 文 | 7.6–10.4 s | 44/200 | 59.3% | 0.302 |
  | 詰めて 3–4 文 | 12.8–17.0 s | **6/200** | 58.3% | 0.336 |

  - 10k では長さはほとんど効かなかった(2 文で 0.132)。20k では、EOS の失敗は短い prompt でほぼ説明できる。
  - 13 s 以上にしても、類似度 0.34 と Kana-CER 58% は voices(0.70 / 32%)に遠い。長さ以外の差(スタジオ録音の読み上げの声)も残っている。
  - 対策の候補(P3g): 学習の参照の長さを 3–15 s に広げる(短い crop を混ぜる)。無音つきや別の発話も混ぜる。JVS は学習に使えない(評価専用)ので、YODAS3 などの CC BY の声で行う。
- **試聴ページ**: `exp/listening/p2_20k/index.html`(ローカル専用、JVS を含む)。voices 10 声 × 3 文、jsut 4 声 × 3 文(元の prompt / 13–17 s の prompt)。列は教師、リリース、10k、20k。音量は −23 LUFS に揃えた(`scripts/eval/make_student_page.py`)。

### 10/22 P2: 40k の結果と CFG。CFG 1.5 で内容の誤りが半分になり、長い prompt なら JVS の声も出る

- **検証の flow_loss**: 0.607(40k)。42.5k から 50k までは 0.6074–0.6079 で横ばいになった(s3 だけで約 20 epoch)。
- **TF**(held-out 200 行): Kana-CER 5.55%、類似度 0.820、NISQA-TTS は教師 −0.59。
- **自由生成(CFG 1.0)**

  | セット | Kana-CER | EOS なし | 類似度 | NISQA-TTS | NISQA mos | col | UTMOSv2 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | voices | 21.0% | 1/200 | 0.728(リリース 0.689) | +0.10 | +0.20(F +0.40) | +0.19(F +0.34) | −0.24 |
  | jsut | 61.1% | 470/1000 | 0.256(0.606) | −0.04 | −1.06 | −0.49 | −0.52 |
  | jvs_bal | 36.2% | 26/100 | 0.419(0.650) | −0.03 | −0.61 | −0.18 | −0.28 |
  | short | 92.0% | 66/172 | 0.179(0.440) | +0.01 | −0.64 | −0.28 | −0.34 |

  - 差はすべてリリースとの差(括弧内はリリースの値)。
  - 女声の comb(voices)は 2.5–4k 6.8 / 4–6k 3.6 dB。jvs_bal は 10.0 / 4.7 dB(教師 7.6 / 3.4、リリース 7.4 / 1.6)。
  - voices の誤りの内訳: 置換 8.4%、脱落 9.3%、挿入 3.3%。多いのは**母音が伸びたまま止まる**崩れで(「…ヨオオオオ」)、その後の文が落ちる。1 文字あたり 6 フレームを超える文が 16%。
- **CFG**(40k、voices 200 文)

  | CFG | Kana-CER | EOS なし | 類似度 | NISQA-TTS | mos | col |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1.0 | 21.0% | 1 | 0.728 | +0.10 | +0.20 | +0.19 |
  | **1.5** | **9.85%** | 0 | **0.762** | −0.05 | +0.01 | +0.07 |
  | 2.0 | 10.8% | 0 | 0.757 | −0.19 | −0.25 | −0.10 |
  | 3.0 | 40.2% | 0 | 0.635 | −0.88 | −1.34 | −0.96 |

  - CFG 1.5 で、伸びたまま止まる崩れはほぼ消えた(1 文字 6 フレーム超は 2.5%)。残りは EOS が早く出て文の後半が落ちる誤り(脱落 4.4%、置換 4.8%)。
  - 短文(JVS の元の prompt)は CFG 1.5 で 92.0% → 60.3%、EOS なし 66 → 1、類似度 0.179 → 0.388。NISQA-TTS はリリース +0.17。
  - **JVS の 13–17 s の prompt(jsut_trim3)+ CFG 1.5**: Kana-CER 13.1%、類似度 **0.589**(元の prompt・CFG 1.0 の 20k では 77.9% / 0.114。jsut 全体のリリースは 0.606)。JVS の声が出ないのは、短い prompt と CFG なしの組み合わせだった。
- 60k の snapshot は、キューの評価(CFG 1.0)のあと、全セットを CFG 1.5 でも測る(`runs/irodori_distill/p2_60k_cfg.sh`)。
- **試聴ページ**: `exp/listening/p2_40k/index.html`(ローカル専用)。voices、JVS の 13–17 s prompt、JVS の元の prompt、短文の 4 セクション。列は教師、リリース、20k、40k、40k の CFG 1.5。

### 10/22 P2: 60k(学習終了、18:16)の結果と P2 の判定

- 学習は 4 h 40 分(評価を横で回しながら)。peak の GPU メモリは 15.8 GiB。検証の flow_loss は 0.607 で、40k 以降は横ばい。
- **TF**(held-out 200 行): Kana-CER 5.28%、類似度 0.819。
- **自由生成**(差はすべてリリースとの差。括弧内はリリースの値)

  | セット | CFG | Kana-CER | EOS なし | 類似度 | NISQA-TTS | mos | col |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | voices | 1.0 | 20.2% | 0/200 | 0.738 | +0.03 | +0.20 | +0.18 |
  | voices | 1.5 | **8.56%**(1.28%) | 0 | **0.767**(0.689) | −0.09 | +0.01 | +0.07 |
  | jsut | 1.0 | 52.1% | 287/1000 | 0.373 | +0.04 | −0.70 | −0.29 |
  | jsut | 1.5 | 26.1%(2.05%) | 20 | 0.571(0.606) | +0.09 | −0.29 | −0.19 |
  | jsut_trim3(13–17 s の prompt) | 1.5 | 11.9% | 0/200 | 0.590 | – | – | – |
  | jvs_bal | 1.0 | 27.6% | 8/100 | 0.508 | +0.01 | −0.22 | −0.05 |
  | jvs_bal | 1.5 | 14.8%(2.01%) | 2 | 0.592(0.650) | −0.01 | −0.07 | +0.03 |
  | short | 1.0 | 132.7% | 36/172 | 0.218 | +0.06 | −0.44 | −0.13 |
  | short | 1.5 | 64.5%(4.18%) | 3 | 0.383(0.440) | +0.21 | +0.14 | −0.03 |

  - 女声の comb(CFG 1.0、2.5–4k / 4–6k): voices 6.3 / 3.6 dB、jvs_bal 10.2 / 4.8、jsut 9.2 / 4.5。教師は 8.0 / 4.2、7.6 / 3.4、6.9 / 3.5。リリースは 5.9 / 1.6、7.4 / 1.6、5.8 / 1.1。
  - UTMOSv2(CFG 1.0): voices −0.22、jsut −0.37、jvs_bal −0.17。
- **判定**(§3 P2 の目安)
  - (a) 品質: 女声の 4–6 kHz comb はリリースの 2 倍以上で、教師並みかそれ以上。NISQA はリリース並みか、それより良い。**試聴で確認する**。
  - (b) 明瞭さ: **未達**。JSUT は CFG 1.5 で 26.1%(目安 ≤3%)。voices でも 8.56%。TF(5.28%)との差は、自由生成での誤差の蓄積と、短い prompt への弱さ。
  - (c) JKYB-mini: 採点の移植がまだ。
- **P3 で詰めること**: データの全量化(s4–s9。galleria2 で 21 時台に生成が終わる)、学習時の参照の長さを広げる、CFG の焼き込み(CFG 1.5 の相当)、12.5 Hz の 2 フレーム連結。
- **試聴ページ**: `exp/listening/p2_60k/index.html`(ローカル専用)。voices、jvs_bal、JVS の 13–17 s prompt、元の prompt、短文。列は教師、リリース、40k の CFG 1.5、60k、60k の CFG 1.5。

### 10/22 wheel 化の方式は W1 に決定(galleria2 を 1〜2 日)、P3b を開始

- **決定**(§8-3): W1。galleria2 を 1〜2 日使ってよい、との回答(試聴はまだ)。W1 の decoder が元 Mimi の水準にも届かなければ、W0 か W2 に切り替える。
- **W1 の作り直し**(`scripts/wheel/train_decoder.py`、436647b)
  - **遅延つきの因果 decoder**(`--delay K`): フレーム t で、フレーム t − K の音声を出す。
    - DACVAE の encoder は非因果なので、フレーム t の終わりの情報は、その後の latent にも載っている。遅延なしの因果 decoder はそれを推測するしかない。probe(K = 0)のピッチの揺れと有声の欠落は、これが一因と読む。
    - wheel は EOS のあとのフレーム(`model_recommended_frames_after_eos`)も decode するので、wheel の変更は要らない。代わりに、冒頭に K フレームの無音が入り、遅延が K × 40 ms 増える。
  - **target は先に一度だけ decode する**(`scripts/wheel/decode_targets.py`)。
    - DACVAE の decode は 5090 で実時間の約 270 倍。batch にしても速くならない(padding の分だけ遅くなる)。
    - 各 clip の後ろに、DACVAE の無音 latent を 12 フレーム付ける。生徒が EOS のあとに出すものと同じで、遅延 decoder はここで発話を出し切ることを学ぶ。
  - batch 16、cosine で LR を 0.1 倍まで下げる、discriminator には各 window の 1 s を切り出して見せる(mimi_ft と同じ)。
  - 学習中の検証は、教師の latent と、生徒(P2 60k、CFG 1.5)の自由生成の latent の 2 つ。どちらも、同じ latent を DACVAE で decode したものと比べる。
    - 測るのは mel、F0 誤差(cents の中央値と p90)、有声フレームが有声のまま残る割合、女声の 4–6 kHz の comb。
  - 速度(本機): batch 16 の GAN 込みで 4.0 step/s、17.4 GB。mel だけなら 20 step/s。
- **W1a: 遅延の比較**(本機、`runs/irodori_distill/w1_delay_sweep.sh`)
  - mel だけで K = 0 / 1 / 2 / 4 を各 10k step 学習する(s3 の 20k clips)。
  - 検証の mel と F0 誤差で K を決め、galleria2 の本番(150k step、約 12 h)に使う。
- **galleria2**: P1 の生成は 21:24 にすべて終わった(s9: 24,727 clips)。QC と pack は 21:31 に終わった。
  - W1 の target の decode を、s3b〜s9 の 231k clips について始めた。8 プロセスで約 41 clips/s、約 1.5 h かかる。
  - 1 プロセスは CPU の 1 コアで頭打ちになる(kernel の起動待ち)ので、8 プロセスにした。
- **P3b: データの全量化**(本機、`configs/train/P3b_6l.yaml`、`runs/irodori_distill/p3b_start.sh`、21:37 開始)
  - `make_split.py --sets s3 s3b s3c s4 s5 s6a s6b s7 s9 --voices-from s3 s3b --name P3b_all`: train 303,035 行 / 508.7 h / 2,950 声。
  - 変えるのは学習データだけにした。warm start、step 数(60k)、schedule、latent 統計、検証の行は P2 と同じなので、検証の loss は P2 と直接比べられる。
  - W1a と GPU を分け合う間は 2.1 it/s。単独なら 4.7 it/s。

### 10/22 W1a: 遅延の比較 → K = 2 に決定。W1 の本番を galleria2 で開始

- mel だけで、K = 0 / 2 / 4 を各 5k step 学習した。s3 の 20k clips、batch 16、P3b と GPU を分け合って 11 step/s。
- 検証(EMA)は、教師の latent と生徒(P2 60k、CFG 1.5)の latent の 2 つ。どちらも、同じ latent の DACVAE decode と比べた mel L1。

| K(遅延) | 学習 mel @5k | 検証 mel @2.5k: 教師 | 〃 生徒 | 検証 mel @5k: 教師 | 〃 生徒 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.248 | 0.509 | 0.513 | 0.368 | 0.350 |
| **2**(80 ms) | **0.214** | 0.408 | 0.417 | **0.311** | **0.313** |
| 4(160 ms) | 0.220 | 0.416 | 0.408 | 0.313 | 0.316 |

- **読み**
  - 2 フレームの先読みで mel が 15% 下がる。DACVAE の latent は、フレーム t の音を後ろのフレームにも載せている。
  - 4 にしても、それ以上は良くならない。
  - よって、probe(K = 0)の誤差の一部は、因果 decoder が未来を推測していたことから来ていた。
- mel だけの学習では波形の位相が揃わないので、F0 誤差と有声率はまだ意味がない(中央値 400 cents 前後で、K による差はない)。これらは GAN が入ってから見る。
- **W1 本番**(`runs/irodori_distill/w1_main.sh 2` → `W1_main_d2`、galleria2、22:15 開始)
  - K = 2、batch 16、150k step。
  - 開始時の target は 87,534 clips(169 h)。検証のたびに読み直す。
  - 最初の起動は、target の decode(8 プロセス、26 GB)と GPU メモリが衝突して OOM になった。decode を 3 プロセスに減らし、自動の再試行で始まった。
- **W1 本番の経過**(検証は教師の latent)

| step | mel | 200 Hz の線 | 25 Hz の線 | F0 誤差の中央値 / p90(cents) | 有声のまま残る割合 |
|---:|---:|---:|---:|---:|---:|
| 2.5k(mel のみ) | 0.414 | 13.8 | 5.6 | – | – |
| 7.5k(GAN 3.5k step) | 0.304 | 7.7 | 8.4 | 389 / 1239 | 58% |
| 10k | 0.296 | 6.0 | 6.9 | 108 / 1118 | 57% |
| 12.5k | 0.289 | 4.6 | 4.3 | 46 / 236 | 62% |
| 15k | 0.283 | 4.8 | 2.5 | 16 / 85 | 80% |

  - 線の強さは保存された検証音声から測った(`scripts/eval/spectral_lines.py`。DACVAE decode は 200 Hz 0.57 / 25 Hz −0.15 dB)。de3f723 からは、学習中の検証にも入れた。
  - 初めのうちの F0 誤差が大きいのは、200 Hz のバズに自己相関の F0 が引っ張られるため。実際は 400 Hz の声を 200 Hz と測る。時間のずれはない(lag 0)。
  - 25 Hz の線が 7.5k まで上がったのは、probe(K = 0)と同じ経過。probe は 5k の 7.1 dB が最大で、30k では 0.8 dB まで下がった。
  - 4–6 kHz の level は、7.5k 以降 target の +0.4〜0.8 dB に収まっている。probe は +4 dB だった。
- target の decode は 22:32 に終わった。GPU メモリを空けるため 5 プロセスを止めたので、133,744 clips になった。
- **W1 encoder**(`runs/irodori_distill/w1_encoder.sh`、galleria2、W1 本番と並行)
  - データは、教師の参照(refs_cc + refs_yodas、3,000 声)と Emilia-YODAS JA の 3 本(JA-B000000〜2)。
  - 遅延 0 と 2 を、各 20k step 学習する。
- **JKYB-mini**(`runs/irodori_distill/jkyb_eval.sh`): main と同じ 1,500 行と JVS 4 声の割り当てで採点する。
  - P2 60k(CFG 1.5)は 69.3%(リリースは 95.07%)、EOS なしは 11 文。
  - prompt の声ごとでは 63.7%(jvs006)から 74.3%(jvs007)。短い JVS の prompt に弱いことの影響が混ざる。

### 10/22 P2 60k の試聴(ユーザー)と、その切り分け

- **ユーザーの所見**
  - 5 h 弱の学習としては良い。読みとイントネーションに、まだおかしいところがある。
  - female_02 / 05 は、P2 のどれも声が二重に聞こえる(ケロケロの一種)。ほかの女声でも文の一部で、男声でもまれに同じ状態になる。
  - jvs031 の「一方で、漁業と商業で、…」は、句読点で長い間が空く(教師 4 s、P2 60k 8 s)。
- **間延びは temperature 0.3 が原因**(`scripts/eval/pauses.py`。150 ms 以上の無音を間と数える)
  - 教師の間は、prompt の背景音に近い −65 dB 前後で、0.4〜0.5 s。生徒の長い間は −77 dB と静かすぎ、1〜2 s 続く。
  - EOS 後の tail(デジタル無音の latent)とは別のもの(latent の距離 4 前後。tail は 0)。
  - jvs_bal で temperature と CFG を振った結果(P2 60k)

| サンプリング | 1 s 以上の間がある clip | 発話の長さ(教師比) | Kana-CER | 類似度 | NISQA-TTS / mos(教師との差) |
|---|---:|---:|---:|---:|---:|
| 教師 | 1% | 1.00 | 2.05% | 0.712 | – |
| temp 0.3 | 54% | 1.85 | 27.6% | 0.508 | −0.17 / −0.31 |
| temp 0.3、CFG 1.5 | 34% | 1.40 | 14.8% | 0.592 | −0.18 / −0.16 |
| temp 0.7 | 11% | 1.08 | 14.8% | 0.635 | −0.21 / +0.04 |
| **temp 0.7、CFG 1.5** | **3%** | 1.04 | **9.3%** | **0.639** | −0.18 / +0.02 |
| temp 1.0 | 0% | 1.00 | 19.7% | 0.624 | −0.35 / −0.08 |
| temp 1.0、CFG 1.5 | 0% | 0.97 | 10.6% | 0.632 | −0.27 / −0.07 |

  - temp 0.3 では、無音区間の雑音まで平均に縮み、学習データにない静かな間になる。そこから再開できなくなると読む。
  - P2 の自由生成の評価はすべて temp 0.3 だったので、明瞭さの差のかなりの部分はこのサンプリングの問題だった。
  - 以降の評価(P2 の測り直し、P3b、P3g)では、temp 0.7 の CFG 1.0 / 1.5 も測る(`cfg_sweep.sh` の `TEMP=0.7`)。
- **声の二重は 2 つが重なっている**
  - **(1) female_02 / 05 は OOD**: 声の高さの中央値は 351 / 374 Hz で、学習の参照 3,000 声(99 パーセンタイルは 289 Hz)のどれよりも高い。female_03(261 Hz)は 97 パーセンタイル、female_01(234 Hz)は 92 パーセンタイル。
  - **(2) フレームごとのサンプリングの雑音**: 2.5〜4 kHz の comb は、温度を下げるほど教師に近づく。teacher-forced(held-out の声)でも、女声は教師より 2.0 dB 低い。
  - female_02 / 05 の 40 文で比べた結果(P2 60k)

| サンプリング | Kana-CER | 2.5–4k comb | NISQA-TTS / mos(教師との差) |
|---|---:|---:|---:|
| 教師 | 1.2% | 10.9 | – |
| temp 0.3 | 26.4% | 6.7 | −0.06 / +0.27 |
| temp 0 | 49.8% | 9.8 | −0.27 / +0.32 |
| temp 0.7 | 29.8% | 5.8 | −0.05 / −0.04 |
| temp 0.3、LSD 4 step | 28.8% | 7.8 | +0.06 / +0.46 |
| temp 0、LSD 4 step | 83.6% | 13.8 | −0.20 / −0.31 |
| temp 0.3、CFG 1.5 | 14.0% | 5.7 | −0.11 / +0.20 |

  - 温度を下げると倍音は戻る。ただし生成が止まり、読みが崩れる。
  - 温度は、間(上げたい)と倍音(下げたい)で逆を向く。
  - 対策の候補
    - 高い声のデータを足す。Emilia-YODAS の 300 Hz 以上の話者、または Irodori v4.1-Small の caption で作る新しい声(評価の 10 声は使わない)。
    - 1 step の head の質を上げる。4 step の出力への head 蒸留、または drifting head。
    - 温度を下げて、CFG で止まりを防ぐ。いま比べている。
  - wheel では、温度は yaml の `default_temperature` で既定を変えられる。LSD の step 数は既定を変えられず、利用者がフラグ `--sampler-decode-steps` で渡す必要がある。

### 10/23 未明: 温度 0.7 での測り直し、短文の止まらなさ、W1 30k、高い声のデータ(hi1)

- **温度を下げて CFG で止まりを防ぐ案は、倍音を戻さない**(female_02 / 05 の 40 文、P2 60k)

| サンプリング | Kana-CER | 2.5–4k comb | NISQA-TTS / mos(教師との差) |
|---|---:|---:|---:|
| 教師 | 1.2% | 10.9 | – |
| temp 0.3 | 26.4% | 6.7 | −0.06 / +0.27 |
| temp 0.1、CFG 1.5 | 15.0% | 6.2 | −0.21 / +0.28 |
| temp 0.2、CFG 1.5 | 14.2% | 6.0 | −0.23 / +0.21 |
| temp 0.1、LSD 4 step、CFG 1.5 | 18.1% | 7.3 | −0.06 / +0.26 |
| temp 0.3、LSD 4 step、CFG 1.5 | 15.8% | 6.8 | −0.02 / +0.31 |

  - CFG 1.5 をかけると、温度を 0.1 まで下げても comb は 6.0–6.2 dB にとどまる(CFG なしの temp 0 は 9.8)。LSD 4 step でも 7.3 dB。
  - よって、female_02 / 05 の二重はサンプリングでは直らない。モデル側(OOD のデータと、1 step の head の質)で直す。
  - temp 0.5 と、temp 0.5 / 0.7 での LSD 4 step を、jvs_bal(間)と female_02 / 05(倍音)で測っている(`runs/irodori_distill/sampling_check3.sh`)。
- **P2 60k を temp 0.7 で測り直した**(`p2_t07.sh`。CFG 1.5 どうしの比較。括弧内はリリース)

| セット | Kana-CER: temp 0.3 → 0.7 | 1 s 以上の間がある clip: 0.3 → 0.7 | 類似度(temp 0.7) |
|---|---:|---:|---:|
| voices | 8.56% → 11.0%(1.28%) | 15.5% → 7.5% | 0.754(0.689) |
| jsut | 26.1% → **10.6%**(2.05%) | 52.1% → 3.9% | 0.623(0.606) |
| jvs_bal | 14.8% → **9.3%**(2.01%) | 34% → 3% | 0.639(0.650) |
| jsut_trim3(13–17 s の prompt) | 11.9% → 9.95% | 19.0% → 0% | 0.600 |
| short | 64.5% → 51.0%(4.18%) | 32.6% → 5.8% | 0.443(0.440) |

  - EOS なしは、どのセットも 0。
  - **JKYB-mini は 69.3% → 70.1%** で、ほぼ変わらない。単語の読みは知識の問題で、サンプリングでは動かない。
  - voices だけは temp 0.3 のほうが良い。これは prompt が長く、きれいな声のため。
- **短文(short)は、文の後も話し続ける**
  - 「はい。」が 13 s の「アーアー…」になる。「なるほど。」は「ナルホド、ソレ、ツラミ」、「すごい！」は「ヒスゴー、ソーマー」。
  - 原因は、P2 の学習データ(s3)に短い文がほぼないこと。P3b のデータには s9(単語の短文、5 文字以下が 6,497 行)が入る。P3b 60k の short で確かめる。
- **W1 30k**(W1_main_d2、遅延 2、batch 16。`w1_eval.sh 30000`)。probe(遅延 0、batch 4)の 30k、soup4 と比べた。

| | 録音 | DACVAE | soup4 | probe 30k | **W1 30k** |
|---|---:|---:|---:|---:|---:|
| comb 2.5–4k(自前 F0): Anime | 9.9 | 10.3 | 7.0 | 1.2 | 2.8 |
| 〃 y3hi | 8.2 | 9.2 | 7.9 | 1.0 | 2.8 |
| 〃 JVS F | 8.0 | 9.0 | 7.7 | 1.0 | 2.7 |
| F0 誤差の中央値 / p90(cents): Anime | 0 | 1.0 / 4.5 | 4.8 / 20.1 | 12.4 / 94.8 | 6.9 / 107 |
| 〃 y3hi | 0 | 1.7 / 6.9 | 4.3 / 16.8 | 10.2 / 44.2 | 5.1 / 22.3 |
| 〃 JVS F | 0 | 1.3 / 5.2 | 3.5 / 13.0 | 8.4 / 31.8 | **3.6 / 13.2** |
| 有声フレームの一致: Anime | 100% | 98.9% | 96.1% | 79.0% | 89.2% |
| 〃 JVS F | 100% | 99.4% | 98.6% | 94.5% | 98.7% |
| ECAPA 類似度 vs 録音: Anime | – | 0.975 | 0.836 | 0.745 | 0.818 |
| 〃 y3hi | – | 0.887 | 0.802 | 0.730 | **0.809** |
| 〃 JVS | – | 0.891 | 0.805 | 0.751 | **0.835** |
| NISQA-TTS: Anime | 4.038 | 4.005 | 3.327 | 2.073 | 2.846 |
| 〃 y3hi | 4.136 | 4.313 | 4.131 | 2.538 | 3.533 |
| 〃 JVS | 4.616 | 4.542 | 4.380 | 3.133 | 3.986 |
| 200 Hz の線 dB: Anime | 0.48 | 0.42 | 4.17 | 3.03 | 4.81 |
| 〃 JVS | 0.26 | 0.54 | 0.25 | 0.97 | 1.77 |

  - **P2 60k の自由生成(voices、CFG 1.5)を W1 30k で decode**: Kana-CER 8.60%(DACVAE decode 8.56%)、類似度 0.730(0.767、リリース 0.689)、NISQA-TTS は DACVAE decode −0.23(女声 −0.37、男声 −0.09)。女声の comb 2.5–4k は 5.2 → 1.6 dB(リリース 5.9)。
  - **読み**: 声の高さと有声の一致は、JVS で soup4 並みになった。類似度は soup4 を上回る。足りないのは **2.5 kHz より上の倍音**で、soup4 の 1/3 程度。200 Hz の線も soup4 より強い。NISQA は soup4 より 0.4–0.6 低い。
  - 学習中の検証では、4–6 kHz の comb は 2.5k から 35k までずっと 0.9–1.0 dB(target 5.5)。mel は下がり続けている(0.414 → 0.249)。倍音が出てくるかどうかが、残り 115k step の焦点になる。
  - 150k で倍音が soup4 に届かなければ、W0 か W2 に切り替える(§10/22 の決定どおり)。
- **高い声のデータ hi1**(`scripts/teacher/high_f0_refs.py`)
  - まだ参照に使っていない Emilia-YODAS(CC BY)の話者から選んだ。条件は main の select_refs.py と同じ(held-out の bucket を除く、56 kbps 以上、5–12 s、日本語)。ただし DNSMOS 3.0 以上、3 発話以上に緩めた。候補は 5,598 声。
  - 声の高さの中央値: 50 / 90 / 95 / 99 パーセンタイルが 176 / 265 / 283 / 329 Hz。280 Hz 以上が 270 声、300 Hz 以上が 139 声、350 Hz 以上が 19 声。
  - **280 Hz 以上の 270 声**(280–412 Hz、中央値 301 Hz)を参照にした。prompt latent は `teacher/refs/` に足した(計 3,270 声)。
  - rows_hi1: 1 声 80 文、計 21,600 行(元の長さで 36.4 h)。文は pack 済みのセット全体から無作為に選んだ。短い s9 の単語も含む。P2 の検証の文と声は除いた。
  - QC は、同じ文に main が受け入れた書き起こしとの一致(別の声なので `--max-cer 0.04`)。
  - 生成は galleria2 で、W1 と並行(`runs/irodori_distill/hi1_pipeline.sh`、batch 8)。本機では P3b とその評価の横で GPU が 32 GB に達したため、こちらに移した。
- **temp 0.5 と LSD 4 step**(`sampling_check3.sh`、すべて CFG 1.5)

| サンプリング | jvs_bal: 1 s 以上の間 | jvs_bal: Kana-CER | jvs_bal: 女声 comb 2.5–4k | female_02 / 05: comb | female_02 / 05: NISQA-TTS / mos |
|---|---:|---:|---:|---:|---:|
| 教師 | 1% | 2.05% | 7.6 | 10.9 | – |
| temp 0.7 | **3%** | 9.26% | 6.5 | 5.1 | −0.29 / −0.16 |
| temp 0.5 | 13% | **8.05%** | 7.5 | 5.8 | −0.16 / −0.06 |
| temp 0.5、4 step | 24% | 10.3% | 7.7 | 6.3 | +0.00 / +0.12 |
| temp 0.7、4 step | 8% | 9.88% | 7.0 | 5.8 | −0.02 / −0.11 |

  - 学習データ内の声(jvs_bal)では、temp 0.5 で女声の倍音が教師並みになる。ただし長い間が 13% 残る。既定の温度は 0.5〜0.7 の間で選ぶ。
  - 4 step は倍音と NISQA を少し上げるが、間を増やす。
- **P3h の決定(ユーザー)**: P3g の後に P3h を入れる。
  - P3h(`configs/train/P3h_6l.yaml`): P3b と P3g の 60k のうち良いほうから、P3b のデータ + hi1 で 20k step を追加学習する。学習率は P3b の半分(1e-4)。
  - 「良いほう」は、40k の snapshot の temp 0.7 / CFG 1.5 での Kana-CER(voices と jsut の平均)で決める(`runs/irodori_distill/p3h_start.sh`)。
  - 学習済みの生徒をそのまま全部読み込む `ja_init_full` を `train_student.py` に足した(smoke で step 1 の flow_loss 0.67。P2 の検証は 0.607)。
  - P3d(CFG 1.5 の焼き込み)の教師は、P3h の 20k にした。データも P3h のもの(`p3d_start.sh` が P3h の終了を待つ)。
  - 評価(`p3_eval_queue.sh`)は 10k と 20k。20k で全セットと JKYB-mini を測る。P3d の JKYB-mini は CFG なし。voices では、声ごとの comb も出すようにした(`cfg_sweep.sh`)。
- **見込み**: hi1 の生成は 1.5 clips/s で、05:30 ごろに終わる。P3g の学習は 07:30〜08:00 ごろに終わり、P3h(約 1.5 h)、P3d(約 1.5 h)と続く。W1 は hi1 と並行の間 2.3 step/s で、11 時ごろに 150k に達する見込み。

### 10/23 朝: P3b 60k(学習終了 02:37)の結果、hi1 の QC

- **P3b 60k の自由生成**(temp 0.7、CFG 1.5。P2 60k は同じ条件。`runs/irodori_distill/p3b_t07_queue.sh`)

| | 教師 | P2 60k | P3b 60k |
|---|---:|---:|---:|
| Kana-CER: voices | 1.18% | 11.02% | **5.28%** |
| 〃 jsut(JVS の声、元の prompt) | 1.77% | 10.61% | **6.95%** |
| 〃 jvs_bal | 2.05% | 9.26% | **7.10%** |
| 〃 jsut_trim3(13–17 s の prompt) | – | 9.95% | **7.34%** |
| 〃 短文(5 文字以下) | 3.71% | 50.95% | **7.41%** |
| JKYB-mini 正答率 | – | 70.1% | **81.4%** |
| 1 s 以上の間: voices / jsut / 短文 | 0.5 / 0.4 / 0.0% | 7.5 / 3.9 / 5.8% | 1.0 / 8.7 / 1.2% |
| 類似度: voices / jsut | – | 0.754 / 0.623 | 0.757 / 0.619 |
| NISQA-TTS: voices / jsut | – | 3.567 / 4.167 | 3.579 / 4.160 |
| 女声 comb 2.5–4k(voices、自身の F0) | 8.0 | 4.6 | 4.5 |
| 〃 female_02 / 05 | 10.0 / 11.7 | 4.9 / 5.3 | 4.5 / 5.4 |

  - **読み**: データを増やしただけで、Kana-CER はどのセットでも P2 のおよそ半分になった。JKYB-mini は 81.4%。
  - **短文の止まらなさは直った**(Kana-CER 51% → 7.4%、長さの比 1.31 → 0.79)。s9 の短文が効いた。
  - jvs031 の「一方で、漁業と商業で、リャネス港は繁栄していた。」(ユーザーの指摘): 教師 4.92 s、P2 60k の temp 0.3 / CFG 1.5 で 7.52 s、P3b 60k の temp 0.7 / CFG 1.5 で 4.32 s。
  - jsut の長い間は 3.9% → 8.7% と増えた(40k では 13.7%)。voices と短文では減った。
  - 声と音質(類似度、NISQA)は P2 と同じ。**倍音(female_02 / 05 を含む)も変わらない**。声の数を増やしても(2,950 声)、高い声の倍音は戻らない。hi1(P3h)でここを見る。
  - 試聴ページ: `exp/listening/p3b_60k/`(JVS を含むためローカルのみ)。
- **hi1 の QC**: 21,600 行のうち 17,025 行が通った(`--max-cer 0.04`、79%)。
  - main の書き起こしとの一致は、完全一致 52.2%、0.02 以下 61.3%、0.05 以下 84.6%。
  - 声ごとの通過率は中央値 81%、5 パーセンタイル 66%。30% を下回るのは 3 声だけ。
  - P3h_all: 学習 320,060 行 / 538.0 h / 3,220 声。
- **W1**: 101.5k step(07:00)。hi1 が終わり、4.0 step/s に戻った。150k は 10:30 ごろ。
  - 学習中の検証では、4–6 kHz の comb は 100k でも 1.19 dB(target 5.5)。100k の snapshot を本機で採点中(`w1_eval.sh 100000`)。
  - encoder の学習(W1e)は、hi1 と W1 の横で最初の batch で GPU メモリが足りなくなった。W1 の終了後に GPU を占有して回す(`w1_encoder_train.sh`)。encoder 用のペアは作成済み(Emilia-YODAS 3 shard と参照、学習 52,212 clips / 120.5 h)。

### 10/23 07:35: P3h は P3b から開始。P3g の 40k、W1 の 100k

- **P3h の初期値は P3b 60k**(`p3h_start.sh` の自動判定。40k の Kana-CER は voices と jsut の平均で P3b 6.14%、P3g 6.32%)。07:35 に学習開始。
- **P3g 40k と P3b 40k**(temp 0.7、CFG 1.5)

| | P3b 40k | P3g 40k |
|---|---:|---:|
| Kana-CER: voices / jsut | 4.99 / **7.30%** | **4.92** / 7.72% |
| 類似度: voices / jsut | **0.758** / 0.621 | 0.727 / **0.639** |
| NISQA-TTS: voices / jsut | 3.583 / 4.183 | **3.750** / 4.196 |
| NISQA mos: voices | 4.195 | **4.430** |
| 女声 comb 2.5–4k(voices、自身の F0) | 4.5 | **5.7** |
| 〃 female_02 / 05 | 4.4 / 5.4 | **5.8 / 6.7** |
| 1 s 以上の間: voices / jsut | **0.0 / 13.7%** | 11.0 / 21.9% |

  - prompt の多様化(P3g)で、**女声の倍音が 1.2〜1.4 dB 戻り**、NISQA も上がった。JVS の声(jsut)の類似度も上がった。
  - その代わり、voices の類似度が下がり、長い間が増えた(voices 0 → 11%)。
  - 読みはほぼ同じなので、P3h は判定どおり P3b から(hi1 の効果を P3b 60k と比べて見るため)。P3g の 60k は評価中。倍音の差が 60k でも残るなら、最終モデルにも prompt の多様化を入れるかを検討する。
- **W1 100k**(`w1_eval.sh 100000`。P0b の往復と、P2 60k の自由生成の decode)

| | DACVAE | soup4 | W1 30k | W1 100k |
|---|---:|---:|---:|---:|
| comb 2.5–4k(自身の F0): JVS F | 9.0 | 7.7 | 2.7 | **6.0** |
| 〃 Anime | 10.3 | 7.0 | 2.8 | **5.1** |
| 〃 y3hi | 9.2 | 7.9 | 2.8 | **6.1** |
| comb 4–6k: JVS F | 4.4 | 1.3 | – | 0.3 |
| F0 誤差 median / p90 cents: JVS F | 1.3 / 5.2 | 3.5 / 13.0 | 3.6 / 13.2 | **2.6 / 9.5** |
| 類似度: JVS / y3hi / Anime | 0.891 / 0.887 / 0.975 | 0.805 / 0.802 / 0.836 | 0.835 / 0.809 / 0.818 | **0.858 / 0.842 / 0.868** |
| NISQA-TTS: JVS / y3hi / Anime | 4.542 / 4.313 / 4.005 | 4.380 / 4.131 / 3.327 | 3.986 / 3.533 / 2.846 | 4.297 / 3.992 / 3.288 |
| 200 Hz の線 dB: Anime / JVS | 0.42 / 0.54 | 4.17 / 0.25 | 4.81 / 1.77 | 3.24 / 0.86 |
| 25 Hz の線 dB: Anime / JVS | −0.27 / 0.44 | 0.37 / −0.02 | – | 0.94 / 0.58 |
| 有声の一致: Anime | 98.9% | 96.1% | – | 89.6% |

  - **2.5–4 kHz の倍音は 30k の倍以上になり、soup4 の 8 割に届いた**。声の高さと類似度は soup4 を上回る。NISQA は soup4 との差が −0.4〜−0.6 から −0.04〜−0.14 に縮んだ。
  - 学習中の検証は 4–6 kHz の comb しか見ていなかった。そこはずっと平らなままで(0.3–0.9 dB、soup4 1.2–1.3)、2.5–4 kHz の伸びが見えていなかった。
  - 残る差: 4–6 kHz、Anime の有声の一致(89.6%)、25 Hz の線(フレーム周期、0.6–0.9 dB)。
  - P2 60k の自由生成の decode: 女声 comb 2.5–4k は 1.6 → 3.0 dB(DACVAE decode 5.2、リリース 5.9)。NISQA-TTS は DACVAE decode との差が −0.23 → −0.13。Kana-CER は 8.45%(DACVAE decode 8.56%)。
  - W1 は 150k まで続ける。150k の採点は自動で走る(`w1_eval.sh 150000`)。

### 10/23 10:00: P3g 60k(prompt の多様化)と P3h 10k(+ hi1)

すべて temp 0.7、CFG 1.5。P3h は 09:21 に学習を終え、P3d(CFG の焼き込み)が続いている(10:45 ごろ終了の見込み)。

| | P3b 60k | P3g 60k | P3h 10k |
|---|---:|---:|---:|
| Kana-CER: voices / jsut | 5.28 / 6.95% | 4.57 / 6.85% | **3.33 / 5.71%** |
| 〃 jvs_bal / 短文 / jsut_trim3 | 7.10 / **7.41** / **7.34%** | 6.98 / 9.70 / 8.26% | – |
| JKYB-mini | **81.4%** | 80.4% | (20k で測る) |
| 類似度: voices / jsut | 0.757 / 0.619 | 0.719 / **0.633** | **0.767** / 0.625 |
| NISQA-TTS: voices / jsut | 3.579 / 4.160 | **3.738 / 4.208** | 3.623 / 4.177 |
| 女声 comb 2.5–4k(voices、教師 8.0) | 4.5 | **5.8** | 5.0 |
| 〃 female_02 / 05(教師 10.0 / 11.7) | 4.5 / 5.4 | **5.9 / 7.2** | 5.0 / 6.3 |
| 1 s 以上の間: voices / jsut | **1.0 / 8.7%** | 9.5 / 15.5% | 2.0 / 19.9% |
| 〃 jvs_bal / jsut_trim3 | **2.0 / 1.0%** | 9.0 / 7.0% | – |

- **P3g**: 60k でも、女声の倍音は P3b より 1.3〜1.8 dB 高い。NISQA も高い。ただし長い間が増え、voices の類似度は下がる。読みはほぼ同じ。
- **P3h 10k**: 読みは P3b 60k よりさらに良く、類似度も上がった。女声の倍音は +0.5〜0.9 dB で、P3g の伸びより小さい。jsut の長い間は 19.9% と増えた。10k は学習率の高い時期にあたる(P3b でも 40k の 13.7% から 60k で 8.7% に減った)。20k の結果を待つ。

### 10/23 11:00: W1 150k(学習終了 10:26)→ W1 で進める。encoder の学習中

| | DACVAE | soup4 | W1 100k | W1 150k |
|---|---:|---:|---:|---:|
| comb 2.5–4k(自身の F0): JVS F / Anime / y3hi | 9.0 / 10.3 / 9.2 | 7.7 / 7.0 / 7.9 | 6.0 / 5.1 / 6.1 | **6.2 / 5.1 / 6.4** |
| F0 誤差 median / p90: JVS F | 1.3 / 5.2 | 3.5 / 13.0 | 2.6 / 9.5 | **2.5 / 9.2** |
| 類似度: JVS / y3hi / Anime | 0.891 / 0.887 / 0.975 | 0.805 / 0.802 / 0.836 | 0.858 / 0.842 / 0.868 | **0.858 / 0.844 / 0.873** |
| NISQA-TTS: JVS / y3hi / Anime | 4.542 / 4.313 / 4.005 | 4.380 / 4.131 / 3.327 | 4.297 / 3.992 / 3.288 | **4.326 / 4.045 / 3.369** |
| 有声の一致: Anime | 98.9% | 96.1% | 89.6% | 89.0% |

- 100k から 150k の伸びは小さい(学習率の減衰の終わり)。
- **判定: W1 で進める。** NISQA は soup4 の −0.09〜+0.04 で、JVS の女声では −0.015。声の高さと類似度は soup4 を上回る。
  - 足りないのは 2.5 kHz より上の倍音(soup4 の 7〜8 割、4–6 kHz はほぼ出ない)と、Anime の有声の一致。§10/22 の基準(「元 Mimi の水準にも届かなければ W0 / W2」)には当たらない。
- **P2 60k の自由生成の decode**: 100k とほぼ同じ。女声 comb 2.5–4k 3.0 dB(DACVAE decode 5.2、リリース 5.9)。NISQA-TTS は DACVAE decode の −0.145、リリースの −0.23。類似度 0.742(リリース 0.689)。
  - wheel の音質は、生徒の倍音(DACVAE decode で 5.0–5.8 dB)と W1 の損失の両方で決まる。生徒側を上げる(P3g の prompt の多様化、P3h の hi1)ことが、wheel でもそのまま効く。
- 試聴ページ: `exp/listening/w1_150k/`(JVS / Anime / y3hi の往復。orig、DACVAE、soup4、stock Mimi、W1 100k / 150k。ローカルのみ)。
- **encoder(W1e)**: galleria2 で遅延 0 を学習中(15k で held-out の R² 0.901、リリースの声 0.868)。遅延 2 が続き、12:20 ごろに終わる。
  - `student_synth.py --prompt-encoder`: prompt を wheel と同じ経路(24 kHz、録音のままの音量、end_on_pause、W1e の latent)で作る(7f60f8a)。
  - `runs/irodori_distill/w1e_eval.sh`: 両 encoder の 20k で、P3d 20k(temp 0.7、CFG 1.0)の voices と jsut を生成し、DACVAE の prompt の場合と比べる。遅延と、speaker_proj の FT が要るかどうかをこれで決める。

### 10/23 12:20: P3h 20k、P3d 10k、長い間の切り分け、encoder、P3c を開始

- **P3h 20k**(temp 0.7、CFG 1.5。P3b 60k との比較)

| | 教師 | P3b 60k | P3h 20k |
|---|---:|---:|---:|
| Kana-CER: voices / jsut / jvs_bal | 1.18 / 1.77 / 2.05% | 5.28 / 6.95 / 7.10% | **3.70 / 5.63 / 5.92%** |
| 〃 短文 / jsut_trim3 | 3.71% / – | 7.41 / 7.34% | 7.41 / **6.56%** |
| JKYB-mini | – | 81.4% | **85.2%** |
| 類似度: voices / jsut / jvs_bal | – | 0.757 / 0.619 / 0.629 | **0.767** / 0.620 / **0.639** |
| NISQA-TTS: voices / jsut | – | 3.579 / 4.160 | **3.642 / 4.183** |
| 女声 comb 2.5–4k(voices) | 8.0 | 4.5 | **5.1** |
| 〃 female_02 / 05 | 10.0 / 11.7 | 4.5 / 5.4 | **5.2 / 6.7** |
| 1 s 以上の間: voices / jsut / jvs_bal | 0.5 / 0.4 / 1% | **1.0 / 8.7** / 2.0% | 2.0 / 13.5 / 2.0% |

  - hi1 を足して 20k step 続けると、読み(JKYB +3.8pt)、類似度、倍音(female_05 +1.3 dB)がそろって良くなった。jsut の長い間だけが増えた。
- **P3d 10k**(CFG 1.5 を焼き込み、temp 0.7、CFG 1.0)は、P3h 20k の CFG 1.5 とほぼ同じ。Kana-CER は voices 4.44% / jsut 5.47%、類似度 0.765 / 0.618、女声 comb 5.3(female_02 / 05: 6.0 / 7.2)。1 s 以上の間は voices 4.0% / jsut 17.5%。20k を評価中。
- **jsut の長い間の切り分け**(P3h 20k)
  - 声で大きく違う: jvs006 80/263、jvs007 29/252、jvs041 11/258、jvs066 15/227。教師は全部で 4 clip。
  - jvs006 の prompt(5.08 s)は中に 0.5 / 0.6 s の間を含むが、それを 0.1 s に縮めても減らない(94/263)。末尾の無音を wheel と同じく end_on_pause で切っても(115)、0.5 s に延ばしても(106)、減らない。jvs006 だけを同じ条件で生成し直すと 89、batch 1 で 96。
  - 教師の学習データの 1 s 以上の間は 0–5%(s4 5.0%、hi1 0.5%、s9 0%。各セット 400 clips を decode して測った)。
  - → データや prompt の間を写しているのではない。**自由生成で生徒自身が陥る**(温度を下げるほど増える)。P3c(誤差蓄積への対策)の条件に当たる。
- **P3c**(`configs/train/P3c_6l.yaml`、galleria2 で 12:18 から): P3h と同じ設定で、学習時に半分の行の過去 latent にノイズを足す(行ごとの std は 0–0.5、flow の target はきれいなまま。`pocket_tts_ja.post_eos.ContextNoise`、1323798)。13:30 ごろ終わる見込み。snapshot は本機に取ってきて、P3h と同じ評価にかける(`p3c_fetch.sh`)。
- **encoder(W1e)の 20k**: 遅延 0 は held-out の R² 0.902 / リリースの声 0.871。**遅延 2 は 0.957 / 0.921**。DACVAE の encoder が非因果なので、2 フレームの先読みが効く。生徒での比較は P3d 20k の評価のあと(`w1e_eval.sh`)。

### 10/23 13:30: P3d 20k、wheel 用の encoder の影響、公式 wheel(4.0.1)での初の通し

- **P3d 20k**(CFG 1.5 を焼き込み、temp 0.7、CFG 1.0): P3h 20k の CFG 1.5 と同等。CFG なしで済む(生成の計算が半分)。

| | P3h 20k(CFG 1.5) | P3d 20k(CFG 1.0) |
|---|---:|---:|
| Kana-CER: voices / jsut / jvs_bal | 3.70 / 5.63 / 5.92% | 3.08 / 5.63 / 5.31% |
| 〃 短文 / jsut_trim3 | 7.41 / 6.56% | 7.26 / 6.84% |
| JKYB-mini | 85.2% | 85.1% |
| 類似度: voices / jsut / jvs_bal | 0.767 / 0.620 / 0.639 | 0.771 / 0.619 / 0.633 |
| NISQA-TTS: voices / jsut | 3.642 / 4.183 | 3.621 / 4.173 |
| 女声 comb 2.5–4k / female_02 / 05 | 5.1 / 5.2 / 6.7 | 5.2 / 5.9 / 7.1 |
| 1 s 以上の間: voices / jsut | 2.0 / 13.5% | 3.5 / 14.9% |

- **wheel の prompt 経路**(`w1e_eval.sh`: P3d 20k、prompt を W1e の encoder で作る。DACVAE の prompt との比較)
  - voices: Kana-CER 3.08 → 3.17%(遅延 2)/ 3.48%(遅延 0)、類似度 0.771 → 0.764 / 0.764、NISQA-TTS +0.08 / +0.15。
  - jsut: Kana-CER 5.63 → 5.39 / 5.07%、類似度 0.619 → 0.619 / 0.611、NISQA-TTS −0.04 / −0.02。
  - → encoder による損失はほぼない。**遅延 2 を使い、speaker_proj の FT は省く**。
- **package と公式 wheel**
  - `scripts/wheel/package_w1.py`: 生徒の snapshot、W1 の decoder、W1e の encoder を、model config と model.safetensors にまとめる。stock loader で strict に読めることを確かめる(214 tensors、voice cloning あり)。既定は temperature 0.7、frames_after_eos 12。
  - PyPI の pocket-tts 4.0.1 は third-party/pocket-tts と同一(diff なし)。
  - `scripts/wheel/wheel_generate.py`(main の版を評価セットの形式に合わせた): 公式 wheel の公開 API だけで CPU 生成する。
- **公式 wheel での voices**(P3d 20k + W1 150k + W1e 遅延 2。`uvx --from pocket-tts==4.0.1`、6 worker × 2 thread で 200 clips を 315 s、EOS なしは 0)

| | リリース | P3d 20k(研究用の経路、DACVAE) | **P3d 20k(公式 wheel)** |
|---|---:|---:|---:|
| Kana-CER | **1.28%** | 3.08% | 3.17% |
| 類似度 | 0.689 | **0.771** | 0.738 |
| NISQA-TTS / mos / col | **3.699** / 4.186 / 4.058 | 3.621 / 4.249 / 4.149 | 3.609 / **4.237** / 4.049 |
| 1 s 以上の間 | **0.0%** | 3.5% | 3.0% |
| 女声 comb 2.5–4k | **5.9** | 5.2 | 3.2 |
| 〃 female_02 / 05 | **6.8 / 7.3** | 5.9 / 7.1 | 4.2 / 4.8 |

  - **通しで動いた**。研究用の経路との差は、ほぼ W1 の decoder によるもの(類似度 −0.03、女声の倍音 −2 dB)。
  - リリースとの比較: 類似度は +0.05。NISQA-TTS は −0.09(mos は +0.05)。女声の倍音は 3.2 dB で、リリースの 5.9 dB を下回る。読み(Kana-CER、JKYB-mini 85% vs 95%)もまだ及ばない。
  - 試聴ページ: `exp/listening/wheel_p3d_20k/`(教師、リリース、研究用の経路、公式 wheel)。jsut / jvs_bal / 短文も公式 wheel で生成している(`runs/irodori_distill/wheel_sets.sh`)。

### 10/23 15:20: P3c 10k、公式 wheel での jsut / jvs_bal / 短文、wheel の文の区切り方が生徒の弱点だった → P3e を開始

- **P3c 10k**(文脈ノイズ。temp 0.7、CFG 1.5、P3h 10k との比較)
  - 1 s 以上の間: jsut 19.9 → 11.6% と減ったが、voices は 2.0 → 7.5%(4 → 15 clips)と増えた。
  - Kana-CER: voices 3.33 → 2.97%、jsut 5.71 → 5.70%。類似度と NISQA はほぼ同じ。
  - → 判定は 20k の結果を待つ。
- **公式 wheel での P3d 20k**(`wheel_sets.sh`。Kana-CER、括弧内は研究用の経路)
  - jsut 6.50%(5.63%)、短文 5.36%(7.26%)、jvs_bal **13.64%**(5.31%)。
  - jvs_bal の差は、wheel が 2 チャンクに分けた文だけで生じている(34.2% vs 6.1%)。1 チャンクの文は同等(5.8 vs 5.0%)。
- **原因は wheel の区切り方**(`split_into_best_sentences`、50 tokens)
  - 50 tokens を超える文は読点で分け、断片を `", "` でつなぐ。生徒は読点の後の空白を見たことがない(P3h の学習テキストで 0.00%。リリースの YODAS データでは空白のある行が 54%)。
  - 研究用の経路でも再現する(wheel のチャンクをそのまま入力、P3d 20k): 空白ありで 18.3%、空白を除くと 6.7%。decoder ではなくテキストの問題。
  - 複数の文を 1 チャンクにまとめる(`"A. B."`)のも生徒には未知だった(学習テキストで 0.09%)。jsut の 2 文を 1 チャンクにした `pair_probe`(150 件)で、P3d 20k は 19.6%(P3h 20k の CFG 1.5 も 19.4%)、1 s 以上の間が 35%。1 文目の後で EOS を出して 2 文目を落とすことが多い。リリースは 1.47%。段落を入力すれば必ず起きるので、実用上の大きな欠点。
- **学習なしの対処**: package の `replace_characters` に `" " → ""` を足す(wheel はチャンクごとに置換をかけるので、つないだ空白が消える)

| 公式 wheel、P3d 20k の Kana-CER | 標準の config | 空白を削除 | 研究用の経路 | リリース |
|---|---:|---:|---:|---:|
| jvs_bal | 13.64% | **5.96%** | 5.31% | 2.01% |
| jsut の 2 チャンクの 85 件 | 15.80% | **6.10%** | 9.02% | 2.68% |
| 2 文で 1 チャンク(pair_probe) | 21.60% | **9.42%** | 19.64% | 1.47% |

  - jvs_bal の 1 s 以上の間は 9.0 → 1.0%。NISQA は変わらない(+0.02)。
  - `package_w1.py` は既定で空白を消すようにした(`--keep-spaces` で無効)。評価セットは `scripts/eval/make_chunk_sets.py` で作る(pair_probe、chunk_probe/chunk_sp・chunk_nosp)。
  - 2 文の 9.4% はまだリリースに遠く、1 s 以上の間も 31% 残る。
- **jvs_bal の NISQA**: 公式 wheel は研究用の経路より −0.34(jsut は −0.16、voices は −0.01)。空白の削除では変わらないので、W1 の decoder によるもの。JVS の声で W1 の損失が大きい(A の再学習の候補)。
- **P3e を galleria2 で開始**(15:08、`configs/train/P3e_6l.yaml`、P3h 20k から 20k step。`pocket_tts_ja.dv_loader.TextAug`)
  - 行の 30% を、同じ参照話者の 2〜3 行をつないだ 1 行にする。間に 0.25–0.6 s の無音 latent を入れ、50 tokens 以内に収める。テキストは半分を `"A. B."`、半分を `"A.B."` でつなぐ。
  - 行の 10% で、読点の後に空白を入れる。
  - それ以外は P3h と同じ(CFG dropout あり)。CFG の焼き込み(P3d の方法)はこの後に行う。P3d の焼き込みは EOS head を凍結するので、1 文目での EOS は直せない。
  - 評価(`p3e_fetch.sh`): 10k / 20k を pair_probe、chunk_sp、voices、jsut で、20k は jvs_bal、短文、JKYB-mini も。

### 10/23 16:10: P3c 20k は P3h 20k より良い。P3e 10k で 2 文・読点の断片が直った → 両方を合わせた P3i を開始

- **P3c 20k**(文脈ノイズ。temp 0.7、CFG 1.5)

| | P3h 20k | P3c 20k |
|---|---:|---:|
| Kana-CER: voices / jsut / jvs_bal / 短文 / jsut_trim3 | 3.70 / 5.63 / 5.92 / 7.41 / 6.56% | **2.99 / 4.87 / 5.55 / 6.70** / 6.67% |
| JKYB-mini | 85.2% | **86.5%** |
| 類似度: voices / jsut | 0.767 / 0.620 | 0.769 / 0.621 |
| NISQA-TTS: voices / jsut / jvs_bal | 3.642 / 4.183 / 4.179 | 3.661 / 4.200 / 4.188 |
| 1 s 以上の間: voices / jsut / jvs_bal / jsut_trim3 | **2.0** / 13.5 / 2.0 / **3.5%** | 5.5 / **7.7 / 1.0** / 5.5% |
| 女声 comb 2.5–4k | 5.1 | 5.0 |

  - → 文脈ノイズは残す。jsut の長い間は半減した。voices では増えた(4 → 11 clips / 200)。
- **P3e 10k**(temp 0.7、CFG 1.5、P3h 20k との比較)
  - 2 文で 1 チャンク(pair_probe): Kana-CER 19.37 → **7.72%**。1 s 以上の間は 34.7 → 29.3%。
  - wheel の読点の断片(chunk_sp): 19.75 → **7.44%**。
  - voices: Kana-CER 3.22%、類似度 0.769、NISQA-TTS 3.615、1 s 以上の間 4.5%。単文の性能は保たれている。
  - pair_probe の長い間の位置(モーラの比で 2 文の境目かどうかを判定。150 clips 中): P3h 20k は境目 35・それ以外 35、P3e 10k は境目 9・それ以外 41。境目の間はほぼ直った。残りは、文中の長い間(P3c が減らしたもの)。
- **P3i を galleria2 で P3e の後に開始**(`configs/train/P3i_6l.yaml`): P3c 20k から、P3e のテキストの形と文脈ノイズで 20k step。評価は `p3i_fetch.sh`(P3e と同じセット、基準は P3c 20k)。CFG の焼き込みは P3e / P3i の良い方に行う。

### 10/23 16:50: P3e 20k は単文でも P3h 20k より良い。CFG の焼き込み P3j を開始

- **P3e 20k**(temp 0.7、CFG 1.5)

| | P3h 20k | P3c 20k | **P3e 20k** |
|---|---:|---:|---:|
| Kana-CER: 2 文(pair_probe)/ 読点の断片(chunk_sp) | 19.37 / 19.75% | | **6.71 / 7.13%** |
| Kana-CER: voices / jsut / jvs_bal / 短文 | 3.70 / 5.63 / 5.92 / 7.41% | **2.99** / 4.87 / 5.55 / 6.70% | 3.39 / **4.33 / 4.91 / 6.47%** |
| JKYB-mini | 85.2% | **86.5%** | 86.4% |
| 類似度: voices / jsut | 0.767 / 0.620 | 0.769 / 0.621 | 0.768 / **0.625** |
| NISQA-TTS: voices / jsut / jvs_bal | 3.642 / 4.183 / 4.179 | **3.661 / 4.200** / 4.188 | 3.645 / 4.171 / **4.215** |
| 1 s 以上の間: 2 文 / jsut / voices / jvs_bal | 34.7 / 13.5 / **2.0** / 2.0% | — / **7.7** / 5.5 / 1.0% | **11.3** / 8.3 / 3.5 / 1.0% |

  - 複数の文を学習したことで、単文の読みも良くなった。長い間も減った(jsut 13.5 → 8.3%)。
  - pair_probe の 1 s 以上の間(150 clips 中): 境目 2、それ以外 15(10k では 9 / 41)。
- **P3j**(`configs/train/P3j_6l.yaml`、この GPU): P3e 20k に CFG 1.5 を焼き込む(P3d の方法)。行には P3e と同じテキストの形を使う。焼き込み後、研究用の経路で評価し、package にして公式 wheel を通す(`p3j_start.sh`)。
- P3i は galleria2 で 16:44 に開始した。P3e の起動スクリプトの出力を /dev/null に向けてしまい、P3e の終了の印が残らなかったため、20 分遅れた。

### 10/23 18:30: P3i 20k は P3e 20k よりおおむね良い → P3i に焼き込む P3k を開始。P3j(P3e の焼き込み)は CFG 1.5 をほぼ再現

- **P3i 20k**(temp 0.7、CFG 1.5)と **P3j 20k**(P3e 20k に CFG 1.5 を焼き込み、CFG なし)

| | P3e 20k | **P3i 20k** | P3j 20k(焼き込み) |
|---|---:|---:|---:|
| Kana-CER: 2 文(pair_probe)/ 読点の断片(chunk_sp) | **6.71** / 7.13% | 7.85 / **5.89%** | 7.82 / 7.39% |
| Kana-CER: voices / jsut / jvs_bal / 短文 | 3.39 / 4.33 / 4.91 / 6.47% | **2.73 / 4.05** / 4.93 / **5.84%** | 3.30 / 4.29 / **4.61** / 6.07% |
| JKYB-mini | 86.4% | **87.5%** | 87.2% |
| 類似度: voices / jsut | 0.768 / 0.625 | **0.772 / 0.626** | 0.767 / 0.622 |
| NISQA-TTS: voices / jsut / jvs_bal | 3.645 / 4.171 / 4.215 | **3.676 / 4.191 / 4.238** | 3.631 / 4.159 / 4.211 |
| 1 s 以上の間: 2 文 / jsut / voices / jvs_bal | **11.3 / 8.3** / 3.5 / **1.0%** | 12.7 / 9.2 / **2.5 / 1.0%** | 15.3 / 10.8 / 2.5 / 3.0% |

  - P3i は 10k の時点で全セットで P3e 10k を上回った(2 文 6.12 vs 7.72%、読点の断片 7.09 vs 7.44%、jsut 4.24 vs 4.54%)。20k では 2 文だけ P3e に負けた(150 文なので誤差の範囲かもしれない)。pair_probe の長い間は境目 2、それ以外 19(P3e 20k は 2 / 15)。
  - P3j は P3e 20k の CFG 1.5 をほぼ再現した(2 文だけ +1.1 pt、長い間はやや増えた)。
- **P3j 20k を公式 wheel で**(voices、package `exp/irodori_distill/package/p3j_20k`、空白を削除): Kana-CER 2.75%(研究用の経路では 3.30%)、類似度 0.738(0.767)、NISQA-TTS 3.610(3.631。女声 −0.18、男声 +0.13)、1 s 以上の間 1.5%。類似度は W1 の encoder、女声の NISQA は W1 の decoder の分だけ下がる(P3d のときと同じ傾向)。
- **P3k**(`configs/train/P3k_6l.yaml`、galleria2 で 17:59 に開始): P3i 20k に CFG 1.5 を焼き込む。P3j の方法に、P3i の文脈ノイズを加えた(生徒と固定した教師が同じノイズ入りの過去を見る。ずれた文脈でも CFG の効き方を焼き込むため)。焼き込み後、P3j と同じ評価と公式 wheel(`p3k_fetch.sh`)。

### 10/23 20:10: 2 文の失敗は早すぎる EOS だった → package の EOS 閾値を −1 に。P3k は P3j より良い。P3l を開始

- **pair_probe の失敗はすべて「1 文目で EOS」**(`scripts/eval/pair_drops.py`。150 文中、文が 1 つ抜けた数): リリース 0、P3d 20k 49、P3e 20k 10、P3i 20k 12、P3j 20k 14、P3j 20k を公式 wheel で 20。抜けるのは常に 2 文目。
- **EOS 閾値**(wheel の `default_eos_threshold`。標準は −4 で、EOS の logit がこれを超えると止まる)。P3j 20k、研究用の経路、temp 0.7、Kana-CER:

| 閾値 | −4(標準) | −3 | −2 | −1 | 0 | +2 |
|---|---:|---:|---:|---:|---:|---:|
| 2 文(pair_probe) | 7.82%(14 抜け) | 5.09%(5) | 3.91%(1) | | **3.71%(0)** | 3.92%(0) |
| voices | 3.30% | | 2.82% | | **2.82%** | 2.84%(2 clips が止まらない) |
| jsut / jvs_bal / 短文 | 4.29 / 4.61 / 6.07% | | | 4.20 / 4.41 / 6.07% | 4.20 / 4.41 / 6.07% | |
| 読点で終わる断片を含む chunk_nosp | **6.33%** | 6.80% | 6.83% | | 7.08% | |

  - 閾値を上げると、2 文目を落とさなくなる。代わりに、読点で終わる断片(長い文を wheel が読点で分けたもの)では、テキストの後に最後の句を繰り返す(1 断片あたり +0.6 s。文末で終わる断片は変化なし)。
  - 公式 wheel を通した結果(P3j 20k、閾値 0 と −4): 2 文 9.63 → **3.91%**(20 → 0 抜け)、voices 2.75 → 2.79%、wheel が 2 つに分ける jsut の 85 文(jsut_chunk2)4.86 → **4.74%**(1 s 以上の間 21 → 12%)、jvs_bal 4.81 → 5.01%。文全体で見ると、読点の断片の損は小さい。
  - → **−1** にする(P3k では −1 と 0 で 2 文の抜けは 0、読点の断片は −1 の方が少し良い。+2 で止まらない clip が出るので、そこから余裕を取る)。`package_w1.py --eos-threshold`。読点で終わる断片を学習させる(教師の潜在を読点の間で切る)のは、潜在だけでは無音を検出できないので保留。
- **P3k 20k**(P3i 20k に CFG 1.5 を焼き込み、文脈ノイズあり。CFG なし、閾値 −4)

| | P3j 20k | **P3k 20k** |
|---|---:|---:|
| Kana-CER: 2 文 / 読点の断片(chunk_sp) | 7.82 / 7.39% | 7.88 / **6.12%** |
| Kana-CER: voices / jsut / jvs_bal / 短文 | 3.30 / 4.29 / 4.61 / 6.07% | **2.44 / 3.87 / 4.23** / 6.15% |
| JKYB-mini | 87.2% | **87.7%** |
| 類似度: voices / jsut | 0.767 / 0.622 | **0.771 / 0.625** |
| NISQA-TTS: voices / jsut / jvs_bal | 3.631 / 4.159 / 4.211 | **3.681 / 4.198 / 4.219** |
| 1 s 以上の間: 2 文 / jsut / voices / jvs_bal | 15.3 / 10.8 / **2.5** / 3.0% | **10.0 / 9.8** / 5.5 / **1.0%** |

  - 焼き込みは P3i 20k の CFG 1.5 より良い所もある(voices 2.73 → 2.44%、jvs_bal 4.93 → 4.23%)。閾値 −1 で 2 文は 3.68%(抜け 0)、jsut 3.83%。
  - package `exp/irodori_distill/package/p3k_20k`(閾値 −1、空白を削除)で公式 wheel を通す(`p3k_wheel.sh`)。
- **P3l**(`configs/train/P3l_6l.yaml`、galleria2 で 19:22 に開始): P3i 20k から同じ recipe で 40k step。P3i の 10k → 20k でも Kana-CER が下がり続けていたため(voices 3.17 → 2.73%、jsut 4.24 → 4.05%)。評価は `p3l_fetch.sh`。

### 10/23 21:20: P3k を公式 wheel で(閾値 −1)。2 文の抜けは 1/150 に。P3l 20k は P3i 20k より読みが良い → P3l 40k に焼き込む P3m を準備

- **P3k 20k を公式 wheel で**(package `exp/irodori_distill/package/p3k_20k`: W1 150k、W1e delay 2、閾値 −1、空白を削除。CPU、温度 0.7)。括弧内は同じ生徒を研究用の経路(GPU、DACVAE でデコード、閾値 −1)で生成したもの:

| | リリース(wheel) | P3j 20k(wheel、閾値 −4 / 0) | **P3k 20k(wheel、−1)** |
|---|---:|---:|---:|
| Kana-CER: voices | **1.28%** | 2.75 / 2.79% | 2.59%(2.44%) |
| Kana-CER: 2 文(pair_probe)、抜け | **1.47%、0** | 9.63%、20 / 3.91%、0 | 3.82%、1(部分的)(3.68%、0) |
| Kana-CER: wheel が 2 つに分ける jsut 85 文(jsut_chunk2) | **2.68%** | 4.86 / 4.74% | 4.04%(5.26%) |
| Kana-CER: jvs_bal / 短文 / jsut | **2.01 / 4.18 / 2.05%** | 4.81 / 5.36 / 4.03%(−4) | 4.67 / 5.91 / 3.63%(4.21 / 5.68 / 3.83%) |
| 類似度: voices / jvs_bal / jsut | 0.689 / **0.650** / 0.606 | 0.738 / – / – | **0.744** / 0.637 / **0.622** |
| NISQA-TTS: voices / jvs_bal / jsut | **3.699 / 4.201 / 4.127** | 3.610 / – / – | 3.664 / 3.919 / 4.018 |
| 1 s 以上の間: 2 文 / jsut_chunk2 / jsut | **0.7 / 0.0 / 0.1%** | – / 21 → 12% / – | 10.7 / 20.0 / 7.9% |

  - 読み(Kana-CER)はすべてのセットで P3j の wheel より良い。リリースとの差は voices 1.3 pt、jsut 1.6 pt まで縮んだ。類似度は voices と jsut でリリースより高い。
  - wheel が長い文を読点で 2 つに分けると、P3k はかえって良くなる(jsut_chunk2 5.26 → 4.04%、jsut 3.83 → 3.63%)。閾値 −1 の読点の断片の損は、文全体では見えない。
  - NISQA-TTS は W1 の decoder の分だけ下がる(研究用の経路との差: jvs_bal −0.30、jsut −0.18、2 文 −0.18、voices −0.02。女声で大きい)。P3d / P3j と同じ傾向で、リリースとの差の大半はここ(jvs_bal −0.28、jsut −0.11)。
  - 長い間は研究用の経路と同じ程度に残る(jsut_chunk2 20%、jsut 7.9%)。リリースにはほとんどない。
  - 試聴ページ: `exp/listening/wheel_p3k_20k/index.html`(ローカル専用、JVS を含む)。voices、2 文、jsut_chunk2、jvs_bal、jsut。列は教師、リリース、P3k(研究用の経路)、P3k(wheel)。
- **P3l 20k**(P3i 20k からさらに 20k step、つまり 40k。温度 0.7、CFG 1.5、閾値 −4)

| | P3i 20k | **P3l 20k** |
|---|---:|---:|
| Kana-CER: 2 文、抜け / 読点の断片(chunk_sp) | 7.85%、12 / 5.89% | **6.15%、10 / 5.41%** |
| Kana-CER: voices / jsut | **2.73** / 4.05% | 2.75 / **3.55%** |
| 類似度: voices / jsut | 0.772 / 0.626 | 0.773 / 0.625 |
| NISQA-TTS: voices / jsut | 3.676 / 4.191 | 3.682 / 4.194 |
| 1 s 以上の間: 2 文 / jsut / voices | **12.7 / 9.2 / 2.5%** | 15.3 / 13.9 / 3.5% |

  - jsut の読みが 0.5 pt 良くなった(cosine の学習率がまだ半分の時点)。長い間はやや増えた。40k は学習終了(21:50 ごろ)後に評価する(`p3l_fetch.sh`。jvs_bal、短文、JKYB-mini も)。
- **P3m**(`configs/train/P3m_6l.yaml`): P3l 40k に CFG 1.5 を焼き込む(P3k の方法のまま、教師だけ替える)。galleria2 で P3l の学習が終わり次第始まる(`p3m_start.sh`)。P3l 40k が 20k より悪ければ止める。評価は閾値 −1 で、研究用の経路と公式 wheel の両方を P3k と並べる(`p3m_fetch.sh`。`cfg_sweep.sh` / `jkyb_eval.sh` に `EOS=-1` と比較用の `CMP` を足した)。

### 10/23 22:30: P3l 40k は 20k よりさらに良い(JKYB-mini 90.5%)→ P3m(その焼き込み)を続ける。案 A(W1s)を準備

- **P3l 40k**(P3i 20k + 40k step、温度 0.7、CFG 1.5、閾値 −4)

| | P3i 20k | P3l 20k | **P3l 40k** |
|---|---:|---:|---:|
| Kana-CER: voices / jsut | 2.73 / 4.05% | 2.75 / 3.55% | **2.26 / 3.28%** |
| Kana-CER: jvs_bal / 短文 | 4.93 / 5.84% | | **3.76 / 4.57%** |
| Kana-CER: 2 文、抜け / 読点の断片 | 7.85%、12 / 5.89% | 6.15%、10 / 5.41% | **5.93%、7 / 5.21%** |
| JKYB-mini | 87.5% | | **90.5%**(リリース 95.1%) |
| 1 s 以上の間: 2 文 / jsut / voices / 読点の断片 | **12.7 / 9.2 / 2.5** / 13.8% | 15.3 / 13.9 / 3.5 / 13.8% | 14.0 / 12.2 / 4.5 / 19.5% |

  - 読みはすべてのセットで最良。長い間は P3i より少し多い。
- **案 A(W1s、ユーザー了承 10/23 22:20)**: W1 の decoder の NISQA の損(P3k の wheel で jvs_bal −0.30、jsut −0.18、主に女声)を、main の soup4 と同じ方法で減らす。
  - W1 150k(`W1_main_d2/checkpoint.pt`: decoder、EMA、識別器)から 10k step の fine-tune を 4 本(`train_decoder.py --init`、lr は本番の半分、cosine)。レシピ: 同じデータ(seed 1)/ hi1 の高い声 21.6k clips を足す(`w1_targets_hi1`)/ 識別器に 1.5 s(本番 1 s。2 s は batch 16 でメモリ不足)/ mel の重みを半分。それを平均する(`scripts/wheel/soup_decoders.py`)。
  - galleria2 で P3m の後に順に回す(`w1s_start.sh`、約 3.5 h)。評価は P3m 20k の同じ潜在をそれぞれの decoder で decode して並べ(voices、jvs_bal、jsut)、soup で package して公式 wheel も通す(`w1s_eval.sh`)。

### 10/24 00:40: P3m 20k(P3l 40k の焼き込み)は P3k より読みが良い。公式 wheel で jsut 3.03%、短文はリリース並み

- **P3m 20k を公式 wheel で**(package `exp/irodori_distill/package/p3m_20k`: W1 150k、閾値 −1、空白を削除)。括弧内は研究用の経路(閾値 −1)

| | リリース(wheel) | P3k 20k(wheel) | **P3m 20k(wheel)** |
|---|---:|---:|---:|
| Kana-CER: voices / 2 文 / jsut_chunk2 | **1.28 / 1.47 / 2.68%** | 2.59 / 3.82 / 4.04% | 2.24 / 3.37 / 3.80%(2.20 / 2.91 / 4.33%) |
| Kana-CER: jvs_bal / 短文 / jsut | **2.01 / 4.18 / 2.05%** | 4.67 / 5.91 / 3.63% | 4.04 / 4.26 / 3.03%(4.12 / 4.18 / 3.29%) |
| 2 文の抜け(150 中) | 0 | 1 | 1 |
| 類似度: voices / jvs_bal / jsut | 0.689 / 0.650 / 0.606 | **0.744** / 0.637 / **0.622** | 0.742 / **0.653** / 0.620 |
| NISQA-TTS: voices / jvs_bal / jsut | **3.699 / 4.201 / 4.127** | 3.664 / 3.919 / 4.018 | 3.682 / 3.908 / 3.989 |
| 1 s 以上の間: voices / 2 文 / jsut_chunk2 / jsut | **0.0 / 0.7 / 0.0 / 0.1%** | 0.5 / 10.7 / 20.0 / 7.9% | 2.0 / 10.7 / 20.0 / 8.1% |

  - 読みはすべてのセットで P3k より良い(jsut 3.63 → 3.03%、jvs_bal 4.67 → 4.04%、短文 5.91 → 4.26%)。短文はリリースと並んだ。JKYB-mini は研究用の経路で 89.9%(P3k 87.7%、P3l 40k の CFG 1.5 で 90.5%)。
  - NISQA と類似度は P3k とほぼ同じ。NISQA のリリースとの差(jvs_bal −0.29、jsut −0.14)は decoder の分で、W1s で詰める。
  - 長い間は変わらない(研究用の経路の jsut_chunk2 は 40% に増えたが、wheel で分けると 20% で P3k と同じ)。
  - 試聴ページ: `exp/listening/wheel_p3m_20k/index.html`(ローカル専用)。列は教師、リリース、P3k(wheel)、P3m(研究用の経路)、P3m(wheel)。文は P3k のページと同じ。短文も追加。
- **W1s**: galleria2 で 23:10 に開始。same(23:50 終了)、hi は終了間近、残り 2 本で 02:15 ごろに soup。評価は `w1s_eval.sh`。

### 10/24 02:20: 試聴の結果(高い声が残った)→ 段階的な自己蒸留に切り替える。その関門として S1(24L)を開始

- **試聴(ユーザー、10/24 02:00)**
  - W1 150k: DACVAE と soup4 がどちらも問題なく聞こえる clip では、W1 も DACVAE とほぼ同じ。decoder の蒸留はそこでは成功している。問題は F0 の高い所で、声が裏返る(dev_c1_090 の「手入れ」、dev_c2_090、dev_c3_097 がはっきりした例)。soup4 もノイジーだが、派手な裏返りはない。
  - P3m 20k: female_02 / 05 以外は、研究用の経路と wheel がほぼ同じ(wheel がわずかに劣る)。female_02 / 05 は研究用の経路(DACVAE の decode)でも崩れが目立つ。P2 60k からはかなり良くなったが、崩れ方の本質は変わらない。
  - ユーザーの評価: アーキテクチャをまたぐ直接蒸留はできている。ただし主な狙い(女声、特に高 F0)はほぼ進んでいないか、悪くなった方向。
- **判断: 段階的な自己蒸留に進む(ユーザー了承 10/24 02:10、galleria2 を 1–2 日)**
  - 通常の fine-tune(実録音など)は後回しにする。教師の出力には高い声の倍音がある(female_02 / 05 の comb 2.5–4k で 10.0 / 11.7 dB)ので、問題は target ではなく、生徒がそれを再現できないこと。データ(hi1)、prompt、温度、LSD の step 数、CFG はどれも差を埋めなかった。teacher-forced でも女声は 2 dB 足りない。
  - Kyutai の small 系(2026-10-20 の blog「The tiny TTS was actually 10x too big」)は、同じアーキテクチャの教師(24L、300M、8 step の head)から段階的に小さくしている。生徒の backbone は、教師の head への入力(PCA で 64 次元に絞ったもの)を、凍結した教師の head と共に学ぶ。1 step の head は、教師の head から 1 frame あたり 64 サンプルを取る drifting で蒸留する(点ごとの二乗誤差では「平均にぼやける」)。最終サイズから冷えた状態で始めると UTMOS は 3.7 前後にとどまり、段階的に切って再学習すると 4.36。
  - Irodori は AR ではないので、1 frame ごとの分布を出せない。今の 6L は 1 frame に 1 サンプルから学んでいる。同じアーキテクチャの大きな FlowLM を間に入れると、その分布が手に入る。
  - 必要な道具は upstream(third-party/pocket-tts 4.0.1)の学習コードにそろっている(`distill_teacher_*`、`head_distill`(mse / drifting)、`training/scripts/shrink_checkpoint.py`、`refl.py`)。公式 wheel 4.0 も `z_dim`、`num_kv_heads`、`decoder_seanet` / `decoder_transformer` を読める。
  - 6L は d1024 のままなので、24L → 6L の depth distillation は z の幅をそろえる必要がない(head をそのまま共有できる)。
- **S1**(`configs/train/S1_24l.yaml`、`configs/model/dv_24l_4k.yaml`): 24L d1024 を main の CC 24L(D1-CC の教師、`BCC_4k_peos/weights_580000`)から warm start する(潜在に結びついた部分は P2 と同じく新規)。データと文・文脈のレシピは P3l のもの(P3h のデータ、P3e の文の形、P3c の文脈ノイズ)を最初から使う。スケジュールは P3b のもの(lr 2e-4、cosine、60k step)。
  - smoke(本機): 1.6–1.7 step/s(6L の約半分)、ピーク 28 GB。60k step で約 10 h。
  - galleria2 で W1s の後に始まる(`s1_start.sh`)。
- **関門の判定**(`s1_eval.sh`、20k と 40k の snapshot を自動で取ってくる)
  - P3l 40k(焼き込み前の 6L)と同じサンプリング(温度 0.7、CFG 1.5、閾値 −4)で比べる。24L は LSD 1 / 4 / 8 step。セットは voices、jvs_bal、jsut(1 step のみ)、teacher-forced。
  - 見るもの: female_02 / 05 の comb が教師(10.0 / 11.7 dB)に近づくか。それと試聴での二重(試聴ページ `exp/listening/s1_24l_<k>/`)。
  - 近づけば S2(24L → 6L、CFG の焼き込み、必要なら head の drifting 蒸留)に進む。近づかなければ、段階的な経路でも高い声は直らない。その場合は別の手を考える。
- **decoder の高 F0**: W1s(案 A)の結果を待つ。hi1 を足した fine-tune を含む。ユーザーの挙げた 3 clip と、オクターブの跳びを数えて比べる。W1 はすでに stock の重みから始めている。残る手は二つ: decoder を大きくする(wheel 4.0 は `decoder_seanet` で別サイズを読める)、生徒の生成した潜在でも学習する(Kyutai の decoder 蒸留)。

### 10/24 03:50: W1s(案 A)はほぼ効かない。効いたのは高い声を足した 1 本(hi)だけで、平均で薄まる

- **S1**: galleria2 で 02:15 に開始。1.7 step/s、29.8 GB。20k は 05:30 ごろ、40k は 08:50 ごろ。
- **W1s の soup**(4 本の平均。soup からの相対距離は最大 0.056): P3m 20k の同じ潜在を decode して W1 150k と比べた NISQA-TTS は、voices +0.014、jvs_bal +0.009、jsut +0.012。公式 wheel でも同じ(P3m_20k_wheel_s4: voices +0.020、jvs_bal +0.009)。
- **4 本のうち hi(hi1 の高い声を足した)だけが、高い声で効く**(ほかの 3 本は ±0.01)

| | DACVAE | W1 150k | **W1s_hi** | W1s_soup4 |
|---|---:|---:|---:|---:|
| voices の女声 NISQA-TTS(研究用の経路 3.976) | – | 3.776 | **3.814** | 3.803 |
| female_02 / 05 の comb 2.5–4k(生徒の DACVAE decode 6.5 / 8.1) | – | 4.4 / 5.6 | **5.0 / 6.5** | 4.6 / 5.8 |
| Anime 往復: comb 2.5–4k / F0 の p90(cents)/ 有声の残り | 10.3 / 4.5 / 98.9% | 5.1 / 21.3 / 89.0% | **5.9 / 17.4 / 91.5%** | 5.4 / 18.0 / 88.3% |
| Anime 往復: NISQA-TTS / col | 4.005 / – | 3.369 / 2.915 | **3.453 / 2.991** | 3.416 / 2.936 |
| y3hi 往復: comb 2.5–4k / NISQA-TTS | 9.2 / 4.313 | 6.4 / 4.045 | **7.1 / 4.112** | 6.6 / 4.070 |

  - jvs_bal と jsut(JVS の声)では、hi も差がない。
  - decoder の差(研究用の経路との差は NISQA-TTS で jvs_bal −0.22、jsut −0.09)は、10k の fine-tune では動かない。高い声のデータを足すと高い声だけが少し良くなる。それでも DACVAE には遠い(Anime の有声 91.5% vs 98.9%)。
  - 試聴ページ: `exp/listening/w1s_recon/`(Anime と y3hi。列は元音声、DACVAE、soup4、W1 150k、W1s_hi、W1s_soup4。ユーザーの挙げた dev_c1_090 / dev_c2_090 / dev_c3_097 を含む)。
- **decoder の次の候補**
  - (a) hi のレシピを長く回す。10k で効いたのはこれだけ。本機で 40k step 約 3 h。ただし、S1 の評価(24L の生成でピーク 19 GB)とは GPU に同時に載らない(W1 は 17.4 GB)。
  - (b) decoder を大きくする。wheel 4.0 は `decoder_seanet` / `decoder_transformer` で別のサイズを読める。CPU 速度とのトレードオフになる。
  - (c) 生徒の生成した潜在でも学習する(Kyutai の decoder 蒸留)。

### 10/24 04:45: S1 の関門の基準(結果を見る前に決めておく)と S2 の準備

- **ユーザー(10/24 04:00)**: w1s_recon は数値どおりほぼ変わらない(dev_c2_090、dev_c3_097 も変化を感じない)。試聴は昼過ぎになるので、**定量評価で判断して進める**。
- **6L の基準値**(P3l 40k、温度 0.7、CFG 1.5、研究用の経路): comb 2.5–4k は voices の女声 5.5 dB(教師 8.0)、female_02 / 05 は 6.6 / 7.7(教師 10.0 / 11.7)、jvs_bal の女声は 6.3(教師 7.6)。
- **S1 の判定**(24L の LSD 1 / 4 / 8 step のうち最良。温度 0.7、CFG 1.5)
  - **通過**: female_02 / 05 の comb が教師との差の半分以上を詰める(8.3 / 9.7 dB 以上)。voices の女声 comb が 6.8 dB 以上。そのうえで、Kana-CER(voices、jsut)が 6L の +1 pt 以内、voices の女声 NISQA-TTS が 6L 以上。
  - **途中**: 改善はあるが半分に届かない → 40k / 60k で見直す。
  - **不通過**: 40k で +0.5 dB 未満 → 段階的な経路をやめる。
  - 改善が 4 / 8 step にしか出ない場合、S2 の 6L は wheel の既定の 1 step で動くので、head の蒸留も要る(upstream の `head_distill`、`drifting`、1 frame あたり 64 サンプル)。wheel 4.0 の yaml は `flow.type: drifting` を読める。
- **S2 の準備**(`configs/train/S2_6l.yaml`)
  - `train_student.py` に depth distillation を足した: `distill_teacher_config` が別の model のとき、教師はその model(重みは strict に読む)。生徒はその下 3 層と上 3 層から始め([0, 1, 2, 21, 22, 23]、upstream の `shrink`)、head は教師のもの(凍結)。
  - レシピは main の D1(lr 4e-4、cosine)。CFG 1.5 を焼き込む。データ、文の形、文脈ノイズは P3m と同じ。60k step。
  - smoke(本機、教師は S1 の smoke): 2.7 step/s(60k で約 6 h)、distill_mse は 60 step で 0.046 → 0.007。
- **W1r(decoder の容量)の準備**: SEANet decoder の各段の残差ブロックを 1 → 3 に増やす(`train_decoder.py --res-layers 3`。追加のブロックは dilation 2 / 4)。
  - `wheel_codec.deepen_decoder` は学習済みの decoder をそのまま深くする。追加のブロックの最後の 1x1 conv を 0 にするので、最初は元と同じ出力になる(W1 150k で最大差 0.0 を確認)。
  - パラメータは 4.0M → 4.3M。計算量は TTS 全体で約 +20% の見込み。
  - 公式 wheel 4.0.1 は yaml の `mimi.decoder_seanet` でそのまま読み、生成できた(strict な loader の確認も通った)。
  - 実験: W1s_hi と同じ 10k step(W1 150k から、w1_targets + hi1、seed 2)。違いは深さだけ。本機の速度は 3.6 step/s、20 GB(W1 は 17.4 GB)。
  - S1 20k の評価の後に本機で回す(`w1r_start.sh`)。比較は W1s_hi との差: Anime / y3hi の往復(有声の残り、F0、comb、NISQA)と、P3m 20k の潜在の decode(voices、jvs_bal)。

### 10/24 06:20: S1 20k は関門を通らない。深さで読みは大きく良くなるが、倍音は変わらない(同じ step 数どうしで +0.1–0.3 dB)

- **S1 20k**(24L、温度 0.7、CFG 1.5、閾値 −4、研究用の経路。P3l 40k は同じサンプリングの 6L。`s1_eval.sh 20000`)

| | comb 2.5–4k: voices 女声 / female_02 / female_05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal / jsut | NISQA-TTS 女声: voices / jvs_bal | 1 s 以上の間(voices) |
|---|---:|---:|---:|---:|
| 教師 | 8.0 / 10.0 / 11.7 / 7.6 | 1.18 / 2.05 / 1.77% | 3.939 / 4.604 | 0.5% |
| P3l 40k、1 step | 5.5 / 6.6 / 7.7 / 6.3 | 2.26 / 3.76 / 3.28% | 3.950 / 4.373 | 4.5% |
| P3l 40k、4 step | 6.0 / 7.1 / 8.7 / 6.7 | 2.20 / 4.16 / – | 4.005 / 4.372 | 3.5% |
| S1 20k、1 step | 5.7 / 6.5 / 7.8 / 6.4 | 1.37 / 2.49 / 2.04% | 3.978 / 4.365 | 18.5% |
| S1 20k、4 step | 6.3 / 7.4 / 8.8 / 6.9 | 1.29 / 2.31 / – | 4.022 / 4.420 | 25.0% |
| S1 20k、8 step | 6.4 / 7.8 / 8.9 / 6.9 | 1.38 / 2.29 / – | 4.035 / 4.408 | 21.0% |

- **判定(事前の基準): 通過しない**。最良(8 step)でも female_02 / 05 は 7.8 / 8.9 dB(基準 8.3 / 9.7)、voices の女声は 6.4 dB(基準 6.8)。
  - 字面では「途中」に当たる。6L の 1 step と比べると +1.2 dB ある。
  - ただし、その差のほとんどは LSD の step 数による。6L も 4 step で同じだけ上がる。
  - 同じ step 数どうしの深さの効果は +0.1–0.3 dB しかない。1 step では −0.1 / +0.1(female_02 / 05)、4 step では +0.3 / +0.1。
  - teacher-forced(温度 0.3、1 step)でも同じで、女声の comb は 4.4 dB(6L は 4.3、教師は 6.1)。
  - 不通過の線(40k で +0.5 dB 未満)は、同じ step 数どうしで比べると、20k の時点ですでに下回っている。
- **深さは読みを大きく良くする**
  - Kana-CER は 6L → 24L で voices 2.26 → 1.37%、jvs_bal 3.76 → 2.49%、jsut 3.28 → 2.04%(教師 1.77%)。この branch で一番大きな読みの改善。
  - 発話の長さ(教師比)は 0.90 → 0.96。類似度も jvs_bal 0.654 → 0.668、jsut 0.625 → 0.642 と上がる。
  - NISQA は、1 step では 6L とほぼ同じ(jsut 4.180 → 4.193)。
- **退行: 長い間が増える**。1 s 以上の間がある clip は、voices で 4.5 → 18.5%、jvs_bal で 1 → 5%、jsut で 12.2 → 13.5%。40k で見直す。
- **まとめ**: 倍音の不足は、深さ(S1)、データ(hi1)、温度、CFG、LSD の step 数のどれでも埋まらなかった。
  - 一方で、温度 0 では comb は教師並みになる(female_02 / 05 の 40 文、P2 60k: 9.8 dB、4 step で 13.8 dB。教師 10.9)。
  - よって、倍音を消しているのは、フレームごとに引く雑音そのもの。その雑音はモデルの大きさでは減らない。
  - 実際、温度 0.7 で teacher-forced に引いた潜在は教師の潜在と平均 0.50 ずれる(潜在の平均の大きさは 0.80)。
  - decoder が学んだのは教師のきれいな潜在だけで、こういうフレームは見ていない。
- **次**
  - **S1 は 60k まで続ける**(読みのため)。40k / 60k の評価では、同じ step 数どうしで比べる。長い間が残るかも見る。
  - **S2 は読みのために回す**。基準の「通過」は倍音で決めたが、S2 の価値は読みにもある。教師(S1 の 40k か 60k)は、40k の評価で長い間を見てから決める。
  - **W1g(新規): 生徒自身のフレームで wheel decoder を fine-tune する(GTA)**
    - TTS の vocoder の定番の直し方(Tacotron 2 / FastSpeech の ground-truth-aligned fine-tune)。Kyutai も、decoder をモデル自身の生成した潜在で蒸留している。
    - 入力: P3m 20k の teacher-forced の潜在(温度 0.7、1 step。生成と同じ)。P3h_all_train のうち w1_targets にある 123,905 clip(`scripts/wheel/gta_latents.py` → `data/irodori_distill/w1_gta`)。
    - target: 同じ clip の教師の音声(w1_targets の wav。教師の潜在の DACVAE decode)。
    - 学習は W1s_same と同じ(W1 150k から 10k step、seed 1)。違いは、clip の半分が GTA の入力になることだけ。
    - 変わるのは decoder の重みだけなので、wheel はそのまま。
    - 評価: P3m 20k の自由生成の decode(voices、jvs_bal。female_02 / 05 の comb を先に見る)と、往復(anime、y3hi)できれいな潜在で失うもの。
    - 本機の GPU では W1r の後に回す(`w1g_start.sh`)。S1 40k / 60k の評価はその後(`s1_eval_chain.sh`)。

### 10/24 07:20: W1r(decoder の残差ブロック 1 → 3)は NISQA を少し上げるが、倍音は変えない

- W1s_hi と同じ 10k step で、違いは深さだけ(`w1r_start.sh`。括弧内は W1s_hi)

| | comb 2.5–4k(女声) | F0 p90 / 有声 | NISQA-TTS |
|---|---:|---:|---:|
| 往復 anime | 6.0(5.9) | 17.0 / 92.0%(17.4 / 91.5%) | 3.559(3.453。DACVAE 4.005) |
| 往復 y3hi | 7.2(7.1) | 13.0 / 98.0%(13.0 / 97.9%) | 4.145(4.112。DACVAE 4.313) |
| P3m 20k の decode、voices(female_02 / 05) | 3.8(3.8)、5.2 / 6.6(5.0 / 6.5) | – | 3.640(3.604。DACVAE 3.718) |
| P3m 20k の decode、jvs_bal | 5.4(5.4) | – | 4.050(4.017。DACVAE 4.232) |

- 容量は NISQA-TTS を +0.03〜0.11 上げる(anime の往復が一番大きい)。Kana-CER と類似度は変わらない。comb、F0、有声の割合は W1s_hi と同じ。
- 10k step だけの比較なので、長く回せばもう少し伸びる余地はある。ただし、高い声の倍音は容量の問題ではない。DACVAE との差(voices で NISQA −0.08、comb −1.7 dB)の大部分は残る。
- 次は W1g(GTA)。W1r の深さは、W1g の結果と合わせて最後に決める(wheel の計算量は約 +20%)。

### 10/24 08:20: W1g(生徒のフレームでの GTA fine-tune)は小さいが確かに効く。wheel の decoder が、女声の NISQA の損失の全部になっている

- **W1g**: W1s_same と同じ 10k step で、clip の半分の入力を P3m 20k の teacher-forced の潜在にしたもの(`w1g_start.sh`。括弧内は W1s_same)

| | NISQA-TTS(女声) | comb 2.5–4k: 女声 / female_02 / 05 | Kana-CER | 類似度 |
|---|---:|---:|---:|---:|
| P3m 20k の decode、voices | 3.625 / 3.855(3.582 / 3.778) | 3.5 / 4.9 / 5.7(3.3 / 4.5 / 5.5) | 2.06%(2.11%) | 0.736(0.745) |
| P3m 20k の decode、jvs_bal | 4.035(4.011) | 5.5(5.2) | 4.27%(4.45%) | 0.633(0.636) |
| 往復 anime | 3.506(3.365) | 5.3(5.1) | – | – |
| 往復 y3hi | 4.072(4.050) | 6.5(6.3) | – | – |

  - 生徒の潜在でも、きれいな潜在(往復)でも、NISQA は上がる。往復の F0 は少し悪い(p90 +1 cent)。類似度は −0.003〜−0.009。
  - 大きさは W1s_hi、W1r と同じくらい(それぞれ +0.02〜0.1)。
- **損失の内訳(voices の女声)**

| | NISQA-TTS | comb 2.5–4k |
|---|---:|---:|
| 教師(DACVAE) | 3.939 | 8.0 |
| P3m 20k(DACVAE) | 3.976 | 5.5 |
| P3m 20k(W1 150k、wheel) | 3.776 | 3.3 |
| P3m 20k(W1r / W1g、wheel) | 3.865 / 3.855 | 3.8 / 3.5 |

  - DACVAE で decode すれば、P3m の女声の NISQA は教師並み。wheel の decoder が −0.2 を失う。comb は、生徒と decoder がそれぞれ約 2 dB 失う。
  - 往復 anime では、公式 Mimi(自分の encoder と decoder)は comb 8.6 dB、有声 96.7%。W1 は 5.1 dB、89.0%。同じ構造でも、decoder にはまだ伸びる余地がある。
- **次: W1x**。3 つの改善を合わせる(`w1x_start.sh`)。
  - W1r_k3(3 ブロック、w1_targets + hi1)から、さらに 20k step。
  - 入力の約半分を GTA にする(w1_gta と w1_gta_hi1。hi1 の 15,749 clip も P3m 20k で teacher-forced に引いた)。
  - S1 40k の温度の確認の後に回す。S1 60k の評価は W1x の後に移した。

### 10/24 09:30: S1 40k: 倍音は同じ step 数どうしで +0.3〜0.7 dB(不通過の線 +0.5 dB のちょうど上下)。読みは教師並み、長い間はほぼ戻った → S2 は S1 60k から予定どおり

- **S1 40k**(条件は 20k と同じ。`s1_eval_40k.out`)

| | comb 2.5–4k: voices 女声 / female_02 / female_05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal / jsut | NISQA-TTS 女声: voices / jvs_bal / jsut | 1 s 以上の間: voices / jvs_bal / jsut |
|---|---:|---:|---:|---:|
| 教師 | 8.0 / 10.0 / 11.7 / 7.6 | 1.18 / 2.05 / 1.77% | 3.939 / 4.604 / 4.486 | 0.5 / 1.0 / 0.4% |
| P3l 40k、1 step | 5.5 / 6.6 / 7.7 / 6.3 | 2.26 / 3.76 / 3.28% | 3.950 / 4.373 / 4.150 | 4.5 / 1.0 / 12.2% |
| P3l 40k、4 step | 6.0 / 7.1 / 8.7 / 6.7 | 2.20 / 4.16 / – | 4.005 / 4.372 / – | 3.5 / 3.0 / – |
| S1 40k、1 step | 6.0 / 7.3 / 8.1 / 6.6 | 1.60 / 2.27 / **1.78%** | 4.049 / 4.421 / 4.191 | 8.0 / 6.0 / 8.3% |
| S1 40k、4 step | 6.5 / 7.6 / 9.1 / 7.0 | 1.68 / 2.01 / – | 4.028 / 4.413 / – | 9.0 / 4.0 / – |
| S1 40k、8 step | 6.5 / 7.6 / 9.0 / 7.1 | 1.64 / 1.85 / – | 4.067 / 4.411 / – | 8.5 / 3.0 / – |

- **倍音**: 20k より伸びた。
  - 同じ step 数どうしの深さの効果は、1 step で +0.5 / +0.7 / +0.4 / +0.3 dB、4 step で +0.5 / +0.5 / +0.4 / +0.3 dB(20k では +0.1〜0.3)。teacher-forced の女声も 4.7 dB(6L 4.3、教師 6.1)。
  - 不通過の線(+0.5 dB)のちょうど上下にある。最良でも、通過の基準(8.3 / 9.7 / 6.8)には届かない(7.6 / 9.1 / 6.5)。よって「途中」。
- **読みは教師並み**: jsut は 1.78%(教師 1.77%)。jvs_bal は 8 step で 1.85%(教師 2.05%)。類似度も jvs_bal 0.670(6L 0.654)、jsut 0.640(6L 0.625)。
- **NISQA-TTS の女声は 6L より +0.04〜0.10**。voices の 1 step は 4.049(教師 3.939)。
- **長い間はほぼ戻った**: voices は 18.5% → 8.0%(6L 4.5%)、jsut は 8.3% で 6L(12.2%)より少ない。jvs_bal は 6.0%(6L 1.0%)。
- 検証の flow_loss は 0.549(42.5k)。6L の P3l 40k は 0.573。
- **判断**
  - 倍音だけで見ると、深さの効果は小さく、まだ伸びている途中。読みと NISQA は、はっきり 6L より良い。
  - **S2 は S1 60k から予定どおり回す**(galleria2、S1 の終了後、12:15 頃から約 6 h)。
  - 60k の評価は W1x の後(本機)。温度の確認(0.85 / 1.0)は回している。
- **S1 40k の温度**(CFG 1.5、1 step。`s1_temp.sh`): 温度を上げると長い間は減るが、倍音と NISQA も下がる。6L と同じ trade-off。

| 温度 | 1 s 以上の間: voices / jvs_bal | comb 2.5–4k: voices 女声 / female_02 / 05 | NISQA-TTS 女声: voices / jvs_bal | Kana-CER: voices / jvs_bal |
|---|---:|---:|---:|---:|
| 0.7 | 8.0 / 6.0% | 6.0 / 7.3 / 8.1 | 4.049 / 4.421 | 1.60 / 2.27% |
| 0.85 | 5.5 / 3.0% | 5.5 / 6.5 / 7.7 | 3.997 / 4.368 | 1.64 / 2.33% |
| 1.0 | 3.5 / 1.0% | 5.3 / 6.4 / 7.9 | 3.956 / 4.356 | 1.77 / 2.45% |

  - 温度 0.7 のままにする。24L の長い間は、温度だけでも 6L 並みに戻せる。

### 10/24 11:35: W1x(深さ + hi1 + GTA)は 3 つの改善がほぼ足し算で効く。P3m 20k の voices では、女声の NISQA-TTS が DACVAE の経路にほぼ並ぶ(3.955、DACVAE 3.976)

- **W1x**: W1r_k3(3 ブロック、w1_targets + hi1)から 20k step。入力の約半分は GTA(w1_gta、w1_gta_hi1)。seed 3(`w1x_start.sh`)。

| | NISQA-TTS 女声: voices / jvs_bal | comb 2.5–4k: voices 女声 / female_02 / 05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal | 類似度: voices / jvs_bal |
|---|---:|---:|---:|---:|
| P3m 20k(DACVAE) | 3.976 / 4.375 | 5.5 / 6.5 / 8.1 / 6.4 | 2.20 / 4.12% | 0.770 / 0.648 |
| W1 150k | 3.776 / 4.145 | 3.3 / 4.4 / 5.6 / 5.2 | 2.08 / 4.39% | 0.746 / 0.636 |
| W1s_hi | 3.814 / 4.144 | 3.8 / 5.0 / 6.5 / 5.4 | 2.15 / 4.37% | 0.748 / 0.637 |
| W1r_k3 | 3.865 / 4.188 | 3.8 / 5.2 / 6.6 / 5.4 | 2.11 / 4.29% | 0.750 / 0.636 |
| W1g | 3.855 / 4.132 | 3.5 / 4.9 / 5.7 / 5.5 | 2.06 / 4.27% | 0.736 / 0.633 |
| **W1x** | **3.955 / 4.201** | **4.3 / 5.9 / 7.2 / 6.1** | 2.11 / 4.19% | 0.746 / 0.638 |

  - voices では、W1 150k と DACVAE の差(女声 NISQA-TTS −0.200)の 9 割を詰めた。NISQA の mos / col も +0.08〜0.09(男声も +0.04〜0.07)。
  - jvs_bal の女声はまだ −0.17(4.201、DACVAE 4.375)。comb は 6.1 dB(DACVAE 6.4)まで来た。
  - 読みと類似度は変わらない。
- **往復(教師の潜在、女声)**

| | anime: comb / F0 p90 / 有声 / NISQA-TTS | y3hi: comb / F0 p90 / 有声 / NISQA-TTS |
|---|---:|---:|
| DACVAE | 10.3 / 4.5 / 98.9% / 4.005 | 9.2 / 6.9 / 98.9% / 4.313 |
| W1 150k | 5.1 / 21.3 / 89.0% / 3.369 | 6.4 / 13.7 / 97.4% / 4.045 |
| W1r_k3 | 6.0 / 17.0 / 92.0% / 3.559 | 7.2 / 13.0 / 98.0% / 4.145 |
| W1g | 5.3 / 21.8 / 89.0% / 3.506 | 6.5 / 14.4 / 97.5% / 4.072 |
| **W1x** | **6.8** / 17.1 / **93.9%** / **3.698** | **7.8** / 13.4 / **98.3%** / **4.218** |

  - GTA は生徒の潜在のためのものだが、きれいな潜在でも W1x がいちばん良い。F0 の p90 は W1r_k3 と同じ(GTA だけの W1g は悪かった)。
  - anime の NISQA はまだ DACVAE より −0.31。
- **step 数**(`w1x_curve.sh`、P3m 20k の voices の女声): NISQA-TTS は 5k 3.920、10k 3.962、15k 3.962、20k 3.955。comb は 4.3 / 4.3 / 4.4 / 4.3。jvs_bal も 10k で止まる。
  - 10k で頭打ちなので、同じ設定で長く回しても伸びない。
- **判断**
  - wheel の decoder は W1x 20k にする。GTA の入力は P3m 20k から引いたものなので、S2 を採るなら S2 の teacher-forced の潜在で引き直す(124k clip で約 11 分)。
  - 公式 wheel で P3m 20k + W1x を確かめる(`p3m_w1x_wheel.sh`: voices、jvs_bal、jsut。CPU の RTF も、k=1 の 1.75〜1.83 と比べる)。
- **公式 wheel 4.0.1 で P3m 20k + W1x**(`p3m_w1x_wheel.sh`。本機の CPU。exp/irodori_distill/package/p3m_20k_w1x。`decoder_seanet` は strict な loader で読めた)

| | NISQA-TTS 女声: voices / jvs_bal / jsut | comb 2.5–4k: voices 女声 / female_02 / 05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal / jsut | 1 s 以上の間: voices / jsut |
|---|---:|---:|---:|---:|
| release(CC BY 版、D1-CC + soup4) | 3.819 / 4.281 / 4.082 | 5.9 / 6.8 / 7.3 / 7.4 | 1.28 / 2.01 / 2.05% | 0.0 / 0.1% |
| P3m 20k(GPU、DACVAE) | 3.976 / 4.375 / 4.175 | 5.5 / 6.5 / 8.1 / 6.4 | 2.20 / 4.12 / 3.29% | 4.5 / 12.9% |
| P3m 20k wheel(W1 150k) | 3.862 / 3.940 / 3.905 | 3.5 / 5.6 / 4.9 / 5.3 | 2.24 / 4.04 / 3.03% | 2.0 / 8.1% |
| P3m 20k wheel(soup4) | 3.893 / 3.947 / 3.925 | 3.8 / 5.8 / 5.5 / 5.5 | 2.24 / 4.08 / 3.03% | 2.0 / 8.1% |
| **P3m 20k wheel(W1x)** | **4.020** / 4.005 / 3.957 | **4.6 / 6.5 / 6.8 / 6.4** | **1.88 / 3.70** / 3.02% | 2.0 / 8.0% |

  - wheel でも W1x は全部の set で W1 150k より良い(NISQA-TTS 女声 +0.05〜0.16、mos / col も +0.03〜0.15)。voices の女声は release(CC BY 版)も GPU の経路も超えた。
  - jvs_bal の wheel は、decoder によらず GPU の経路より −0.37 低い(W1 150k でも同じ)。wheel 側の生成(文の分け方など)の差で、decoder の問題ではない。
  - CPU の RTF(6 worker × 2 thread、1 worker あたり): voices 1.727、jvs_bal 1.599、jsut 1.783。k=1 のとき(1.750 / 1.829 / 1.784)と差が見えない。この測り方では、深い decoder の費用は見えないくらい小さい。単独の RTF は、リリースの前に測る。
  - 読み(jsut 3.0%)と長い間(jsut 8%)は release(2.05%、0.1%)にまだ負ける。これは生徒の側の問題で、S2 で見る(S1 40k は jsut 1.78%)。

### 10/24 12:50: S1 60k: 倍音は 40k から伸びない(同じ step 数どうしで +0.3〜0.7 dB のまま)。読みは教師並みのまま → 関門は最終的に通らない。S2 は読みのために続ける

- **S1 60k**(条件は 20k / 40k と同じ。`s1_eval_60k.out`。S1 は 12:15 に終わり、S2 は galleria2 で 12:16 から回っている)

| | comb 2.5–4k: voices 女声 / female_02 / female_05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal / jsut | NISQA-TTS 女声: voices / jvs_bal / jsut | 1 s 以上の間: voices / jvs_bal / jsut |
|---|---:|---:|---:|---:|
| 教師 | 8.0 / 10.0 / 11.7 / 7.6 | 1.18 / 2.05 / 1.77% | 3.939 / 4.604 / 4.486 | 0.5 / 1.0 / 0.4% |
| P3l 40k、1 step | 5.5 / 6.6 / 7.7 / 6.3 | 2.26 / 3.76 / 3.28% | 3.950 / 4.373 / 4.150 | 4.5 / 1.0 / 12.2% |
| P3l 40k、4 step | 6.0 / 7.1 / 8.7 / 6.7 | 2.20 / 4.16 / – | 4.005 / 4.372 / – | 3.5 / 3.0 / – |
| S1 40k、1 step | 6.0 / 7.3 / 8.1 / 6.6 | 1.60 / 2.27 / 1.78% | 4.049 / 4.421 / 4.191 | 8.0 / 6.0 / 8.3% |
| S1 60k、1 step | 6.0 / 7.3 / 8.0 / 6.8 | 1.58 / 2.11 / 1.82% | 4.042 / 4.355 / 4.205 | 10.5 / 3.0 / 8.6% |
| S1 60k、4 step | 6.3 / 7.8 / 8.7 / 7.0 | 1.48 / 2.01 / – | 4.106 / 4.403 / – | 10.5 / 3.0 / – |
| S1 60k、8 step | 6.3 / 7.5 / 9.0 / 6.9 | 1.53 / 2.03 / – | 4.103 / 4.399 / – | 8.5 / 3.0 / – |

- **倍音**: 同じ step 数どうしの深さの効果は、1 step で +0.5 / +0.7 / +0.3 / +0.5 dB、4 step で +0.3 / +0.7 / 0.0 / +0.3 dB。40k と同じ。teacher-forced の女声も 4.7 dB(40k と同じ。6L 4.3、教師 6.1)。
  - 事前の基準(8.3 / 9.7 / 6.8)には最後まで届かない。**関門は通らない**。深さ(容量)は、倍音の損失の主な原因ではない。
- **読みは教師並み**: jsut 1.82%、jvs_bal 2.01〜2.11%(教師 1.77 / 2.05%)。類似度は jvs_bal 0.672(6L 0.654、教師 0.712)。
- **長い間**: voices 10.5%(40k 8.0%)、jsut 8.6%(6L 12.2%)。
- **判断**
  - S2(S1 60k → 6L)は続ける。狙いは倍音ではなく読み(wheel の P3m 20k は jsut 3.0%、release 2.05%)と NISQA。
  - 倍音の損失は温度で決まる(温度 0 では教師並み)。容量でないなら、残る候補は frame ごとの noise の引き方。flow head は 40 ms の frame ごとに新しい noise を引くので、教師の滑らかな軌跡のまわりで frame が揺れる、という仮説を試す(`noise_probe.sh`: frame の noise を AR(1) で相関させる。各 frame の noise の分布は N(0, temp) のまま)。
- **noise の確認 1**(`noise_probe.sh`。P3m 20k、CFG 1.0、閾値 −1、研究用の経路。ρ は frame の noise の AR(1) の相関)

| | comb 2.5–4k: voices 女声 / female_02 / 05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal | 1 s 以上の間: voices / jvs_bal | NISQA-TTS 女声: voices / jvs_bal | 類似度 jvs_bal |
|---|---:|---:|---:|---:|---:|
| 教師 | 8.0 / 10.0 / 11.7 / 7.6 | 1.18 / 2.05% | 0.5 / 1.0% | 3.939 / 4.604 | 0.712 |
| 温度 0.7(P3m 20k) | 5.5 / 6.5 / 8.1 / 6.4 | 2.20 / 4.12% | 4.5 / 4.0% | 3.976 / 4.375 | 0.648 |
| 温度 0.5 | 5.9 / 7.1 / 8.2 / 7.1 | 2.09 / 4.45% | 7.5 / 12.0% | 4.003 / 4.400 | 0.644 |
| 温度 0.35 | 6.1 / 7.1 / 7.7 / 7.5 | 2.73 / 6.12% | 8.0 / 22.0% | 4.010 / 4.367 | 0.629 |
| 0.7、ρ 0.5 | 5.8 / 7.0 / 8.0 / 6.8 | 2.17 / 4.12% | 5.5 / 8.0% | 3.965 / 4.352 | 0.644 |
| 0.7、ρ 0.8 | 6.2 / 7.2 / 8.9 / 7.3 | 2.37 / 4.53% | 7.5 / 16.0% | 4.011 / 4.344 | 0.636 |
| 0.7、ρ 0.9 | 6.3 / 7.6 / 8.5 / 7.6 | 2.73 / 6.40% | 13.0 / 37.0% | 3.996 / 4.348 | 0.623 |
| 0.7、ρ 0.95 | 6.9 / 8.2 / 9.5 / 7.6 | 4.90 / 12.19% | 20.0 / 55.0% | 4.036 / 4.284 | 0.602 |

  - frame の noise がゆっくり動くほど、倍音は教師に近づく(ρ 0.95 で female_02 / 05 は 8.2 / 9.5 dB、jvs_bal の女声は教師並み)。frame ごとの noise の揺れが、倍音の損失の大きな部分。
  - ただし、間と読みも同じだけ悪くなる(ρ 0.95 で jvs_bal の 55% に 1 s 以上の間)。温度を下げたときと同じ trade-off で、ただの得にはならない。
  - 次: 間を決める方向と、倍音を決める方向を分けられるかを見る(`noise_probe2.sh`)。frame の大きさの軸(`latent_pca.py --loudness-from`: log RMS の R² 0.74)だけ温度 0.7 にして、残り 31 軸の温度を下げる。逆の組み合わせも対照として試す。
- **noise の確認 2**(`noise_probe2.sh`。温度を軸ごとに変える。L は大きさの軸、r は残り 31 軸)

| | comb 2.5–4k: voices 女声 / female_02 / 05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal | 1 s 以上の間: voices / jvs_bal | 発話の長さ(教師比): voices / jvs_bal |
|---|---:|---:|---:|---:|
| 温度 0.7(P3m 20k) | 5.5 / 6.5 / 8.1 / 6.4 | 2.20 / 4.12% | 4.5 / 4.0% | 0.90 / 0.98 |
| 温度 0.35 | 6.1 / 7.1 / 7.7 / 7.5 | 2.73 / 6.12% | 8.0 / 22.0% | 0.93 / 1.12 |
| L 0.7、r 0.35 | 6.1 / 6.9 / 8.1 / 7.4 | 3.32 / 6.02% | 8.5 / 26.0% | 0.97 / 1.18 |
| L 1.0、r 0.35 | 6.1 / 7.0 / 8.6 / 7.5 | 3.02 / 6.50% | 8.5 / 27.0% | 0.95 / 1.17 |
| L 0.7、r 0.2 | 6.5 / 7.1 / 8.6 / 8.1 | 5.19 / 10.95% | 13.0 / 44.0% | 1.08 / 1.35 |
| L 0.7、r 0 | 崩れる(Kana-CER 192%) | | | 2.84 / 3.17 |
| L 0.2、r 0.7 | 5.6 / 6.8 / 8.1 / 6.5 | 2.11 / 4.02% | 5.5 / 2.0% | 0.90 / 0.97 |

  - **仮説は外れ**: 大きさの軸の温度を保っても、残りの軸の温度を下げれば、全体の温度を下げたときと同じだけ間が延び、読みも悪くなる。大きさの軸の温度は、どちらにもほとんど効かない。線形な方向では、間と倍音を分けられない。
  - noise が小さいほど、生徒はゆっくり話す(r 0 では教師の 3 倍の長さ)。自己回帰の低温での停滞と同じ形。
  - 解釈: 各 frame で「今の音を続ける / 次の音へ進む」の選択がある。flow head(9.8M params)がそれらを 1 つの frame に混ぜて出すので、倍音がぼける。noise を減らすと混ぜ方は鋭くなるが、「続ける」側へ寄って進みが止まる。S1(24L)は backbone を 4 倍にしたが head は同じで、読みは教師並み、倍音は +0.5 dB だった。これも head が詰まりどころだという見方と合う。
- **次: head の容量**(`h_start.sh`。`train_student.py` に `ja_head_only` を足した: `ja_init_from` の backbone を固定し、flow head だけを新しく学習する。head の大きさは model_config の `flow.dim` / `flow.depth`。公式 wheel も yaml のこの 2 つを読む)
  - H1: dim 1024 × depth 6(37.3M params、P3l の head の 3.8 倍)。H0: 512 × 6 を最初から(対照: 20k step の新しい head が P3l の head に届くか)。H2: 512 × 12。
  - どれも P3l 40k の backbone(CFG を焼く前)に、P3l のデータで 20k step(lr 4e-4、本機で約 85 分ずつ)。評価は P3l 40k と同じ(温度 0.7、CFG 1.5、閾値 −4、1 / 4 step)。
  - CPU の費用: head は frame ごとに 1 回動く。dim 1024 で 1 frame の計算は約 +50%、depth 12 で約 +13%。

### 10/24 15:00: S1 の試聴(8 step で female_03 / 05 が大きく良くなる)、S2 20k、LSD の step 数と decoder の大きさの CPU 費用

- **ユーザー(10/24 14:40)**
  - s1_24l_60k: S1_60k_t07s8_cfg1.5 は female_03 / 05 で大きく良くなった(あと少し)。ほかの声は教師とほぼ区別がつかない。イントネーションも問題ない。方向性はよい。
  - w1x_recon: dev_c1_090 / dev_c2_090 / dev_c3_097 の裏返りは、ほんのわずか良くなったかも、という程度。深さで難しいなら、次は幅か、との問い。
- **comb は聴感と合わない**: female_03 の comb 2.5–4k は、S1 の 1 / 4 / 8 step で 4.6 / 4.9 / 4.6 dB(教師 6.3、P3l 40k 4.7、P3m 20k 4.2)。female_05 だけが 8.0 / 8.7 / 9.0 と step で上がる。comb は、聞こえている改善の一部しか測っていない。
- **S2 20k**(S1 60k → 6L、CFG 1.0、閾値 −1、温度 0.7、研究用の経路。4 / 8 step は `s2_steps.sh`)

| | Kana-CER: voices / jsut / jvs_bal | NISQA-TTS 女声: voices / jsut / jvs_bal | comb 2.5–4k: voices 女声 / female_02 / 03 / 05 / jvs_bal 女声 | 1 s 以上の間: voices / jsut / jvs_bal |
|---|---:|---:|---:|---:|
| 教師 | 1.18 / 1.77 / 2.05% | 3.939 / 4.486 / 4.604 | 8.0 / 10.0 / 6.3 / 11.7 / 7.6 | 0.5 / 0.4 / 1.0% |
| P3m 20k | 2.20 / 3.29 / 4.12% | 3.976 / 4.175 / 4.375 | 5.5 / 6.5 / 4.2 / 8.1 / 6.4 | 4.5 / 12.9 / 4.0% |
| S1 60k、8 step(CFG 1.5、閾値 −4) | 1.53 / – / 2.03% | 4.103 / – / 4.399 | 6.3 / 7.5 / 4.6 / 9.0 / 6.9 | 8.5 / – / 3.0% |
| S2 20k、1 step | 2.06 / 2.90 / 3.58% | 4.021 / 4.261 / 4.331 | 6.1 / 6.6 / 4.8 / 8.1 / 7.1 | 8.0 / 14.2 / 4.0% |
| S2 20k、4 step | 1.89 / … / … | 4.080 / … / … | 6.6 / 7.7 / 5.0 / 8.8 / … | 12.5 / … / … |
| S2 20k、8 step | 1.71 / … / … | 4.078 / … / … | 6.6 / 6.9 / 4.7 / 9.4 / … | 12.0 / … / … |

  - 20k(60k のうち)で、読みと倍音は P3m 20k より少し良い(jsut 2.90%、jvs_bal 3.58%、jvs_bal の女声の comb +0.7 dB)。
  - S2 は S1 の flow head をそのまま使うので、step 数の効果も引き継ぐ。8 step で読み 1.71%、female_05 の comb 9.4 dB(S1 の 8 step は 9.0)。
  - 長い間は増える(voices 1 step 8.0%、8 step 12.0%。P3m 4.5%、教師 0.5%)。S1 と同じ傾向。
  - W1x を通すと(`s2_w1x.sh`)、女声の NISQA-TTS は voices 3.950 / jvs_bal 4.198 で、P3m 20k の W1x(3.955 / 4.201)と同じ。comb は 4.8 dB(P3m 4.3)。DACVAE との差(voices −0.07、4–6 kHz の comb 3.3 → 1.3 dB)は、W1x の GTA が P3m の frame だったことにもよる。最終の生徒で GTA を引き直して fine-tune する予定(変わらず)。
- **公式 wheel での step 数**: `TTSModel.load_model(sampler_decode_steps=N)` / CLI `--sampler-decode-steps N`。読み込み時の引数で、既定は 1。yaml には書けない。
- **CPU の費用**(`scripts/wheel/cpu_cost.py`。公式 wheel 4.0.1、galleria2 の空いたコア 2 つ、1 process 2 threads。P3m 20k + W1x の package、voices の 20 文、計 100 s の音声)

| | 音声 1 s あたりの計算時間 |
|---|---:|
| 生成(FlowLM と decoder)、1 / 4 / 8 step | 0.460 / 0.637 / 0.819 s |
| decoder のみ、n_filters × block 数: 64 × 1(stock の大きさ)/ 64 × 3(W1x) | 0.148 / 0.197 s |
| 〃 96 × 1 / 96 × 3 | 0.190 / 0.264 s |
| 〃 128 × 1 / 128 × 3 | 0.266 / 0.361 s |

  - 8 step は生成全体で +78%。それでも 2 threads で実時間の 1.2 倍の速さ。
  - decoder を 96 × 3 にすると、decoder は +34%(音声 1 s あたり +0.07 s)。128 × 3 は +83%。
  - なお、wheel 4.0.1 の Kyutai の小型版(`english_2m_2026-10.yaml`)は、decoder を逆に細くしている(n_filters 24、decoder transformer d128 × 4)。
- **decoder の幅**: 深さ(1 → 3 block)は NISQA を +0.03〜0.11 上げたが、F0、有声の残り、comb は変えなかった。幅も同じ結果になるおそれはある。ただ、費用は大きくない(96 × 3 で +0.07 s/s)ので試す。
  - W1w: W1x を関数を変えずに幅 96 に広げる(net2wider。チャネルを複製し、出側の重みを不均等に分けて、複製どうしが別れて学習できるようにする)。そこから 20k step、データと GTA は W1x と同じ。
  - 比べるもの: Anime / y3hi の往復(有声の残り、F0 の p90、comb、NISQA)、ユーザーの挙げた 3 clip、生徒の潜在の decode(4–6 kHz の comb)。
  - GPU は、本機の H の probe(〜18:15)か galleria2 の S2(〜18:30)の後。
- **次**
  - S2 40k(16:25 ごろ)と 60k(18:30 ごろ)を、1 / 4 / 8 step で自動で評価する。
  - H1(head 1024 × 6)の結果(15:25 ごろ)。大きい head が 1 step で 8 step と同じ効果を出せるかを見る。出せれば、wheel の既定(1 step)のまま使える。

### 10/24 15:35: 1 step と 8 step の聴き比べ(ユーザー)、H1 の voices、head の step 蒸留(HS)を開始

- **ユーザー(10/24 15:20)**: female_02 / 05 では、S1_60k_t07s8_cfg1.5 のほうが S1_60k_t07_cfg1.5(1 step)より明らかに裏返りが少なく、良い。ほかの声はほぼ差がわからない。最終的には 1 step にしたいが、現時点では 8 step でもよい。
  - 生徒の高い声の裏返りの一部は、head の 1 step のサンプリングから来ている。
- **H1(head 1024 × 6、P3l 40k の backbone、20k step)の voices**(温度 0.7、CFG 1.5、閾値 −4)

| | comb 2.5–4k: 女声 / female_02 / 05 | NISQA-TTS 女声 | Kana-CER | 1 s 以上の間 |
|---|---:|---:|---:|---:|
| P3l 40k、1 step | 5.5 / 6.6 / 7.7 | 3.950 | 2.26% | 4.5% |
| P3l 40k、4 step | 6.0 / 7.1 / 8.7 | 4.005 | 2.20% | 3.5% |
| H1、1 step | 5.9 / 6.8 / 8.6 | 3.969 | 2.09% | 3.0% |
| H1、4 step | 6.2 / 7.6 / 9.1 | 4.027 | 2.15% | 4.0% |

  - 大きい head の 1 step は、元の head の 4 step にほぼ並ぶ(comb)。NISQA はその半分も届かない。間は増えない。
  - jvs_bal / jsut と H0(対照)は続きで出る。
- **HS: head の step 蒸留**(`train_student.py` の `ja_head_steps`、`configs/train/HS_s2_20k.yaml`、`runs/irodori_distill/hs_start.sh`)
  - upstream の head distillation を使う。凍結した head のコピーを 8 step で回し、同じ noise(温度 0.7)から 1 step で同じ出力を出すよう、head だけを学習する。backbone(S2 20k)は固定。
  - 1 step の写像を 8 step の写像そのものに合わせるので、データへの回帰と違って平均化が起きない。head は S2 20k のもの(S1 60k の head)から始める。lr 1e-4、10k step。
  - GPU 1.7 GB、7.7 it/s(H の probe の横で回る)。最初の head_loss(1 step と 8 step の差の二乗平均、正規化した潜在)は 0.010、500 step で 0.006。
  - 評価: voices、jvs_bal、jsut で 1 step。S2 20k の 1 / 8 step と比べる。効けば、S2 60k でも同じことをする(H1 の大きさの head と組み合わせる案もある)。
- **H1 の jvs_bal / jsut**: jvs_bal の女声の comb 6.6 dB(P3l 40k の 1 / 4 step は 6.3 / 6.7)、女声の NISQA-TTS と読みは変わらない(jsut 3.30%、P3l 3.28%)。head を 3.8 倍にしても効果は小さい。
  - H0 / H2 は止めた(15:40)。HS と decoder の幅(W1w)に GPU を回す。HS が効かなければ再開する(H0 は step 264 の checkpoint から)。
- **W1w を開始**(15:50、本機、`runs/irodori_distill/w1w_start.sh`)
  - `wheel_codec.widen_decoder`(net2wider): 広げたチャネルは元のチャネルのコピー。読む側の重みをコピーの数で割るので、noise なしなら出力は同じ(相対 RMS 差 4.5e-7)。
  - Adam はコピーどうしに同じ更新を与え続けるので、追加したチャネルへの重みに、その層の重みの std の 0.02 倍の noise を足す。W1x との出力の差は 0.6%。
  - W1x(64 × 3)→ 96 × 3(decoder 8.3M params)。学習の開始時の mel は 0.204(W1x の開始時 0.208)。
  - HS と GPU を分け合って 1.9 step/s、24.7 GB。20k step は 19:00 ごろ。

### 10/24 16:40: HS(head の 8 → 1 step 蒸留)は、8 step の改善の大部分を 1 step で出す

- **HS_s2_20k**(S2 20k の backbone を固定、head を 10k step。15:31〜16:05、W1w と GPU を分け合って後半は 1.3〜3.4 it/s)
  - head_loss(1 step と 8 step の差の二乗平均): 学習 0.010 → 0.0030、検証 0.0026。最後まで下がり続けた。
- **結果**(温度 0.7、CFG 1.0、閾値 −1、研究用の経路)

| | NISQA-TTS 女声: voices / jvs_bal / jsut | comb 2.5–4k: voices 女声 / female_02 / 03 / 05 / jvs_bal 女声 | Kana-CER: voices / jvs_bal / jsut | 1 s 以上の間: voices / jvs_bal / jsut |
|---|---:|---:|---:|---:|
| 教師 | 3.939 / 4.604 / 4.486 | 8.0 / 10.0 / 6.3 / 11.7 / 7.6 | 1.18 / 2.05 / 1.77% | 0.5 / 1.0 / 0.4% |
| P3m 20k | 3.976 / 4.375 / 4.175 | 5.5 / 6.5 / 4.2 / 8.1 / 6.4 | 2.20 / 4.12 / 3.29% | 4.5 / 4.0 / 12.9% |
| S2 20k、1 step | 4.021 / 4.331 / 4.261 | 6.1 / 6.6 / 4.8 / 8.1 / 7.1 | 2.06 / 3.58 / 2.90% | 8.0 / 4.0 / 14.2% |
| S2 20k、8 step | 4.078 / 4.400 / 4.288 | 6.6 / 6.9 / 4.7 / 9.4 / 7.4 | 1.71 / 3.58 / 2.94% | 12.0 / 8.0 / 12.9% |
| **HS、1 step** | **4.072 / 4.363 / 4.272** | **6.5 / 7.3 / 4.9 / 8.9 / 7.3** | 1.93 / 3.40 / 2.98% | 10.5 / 7.0 / 11.7% |

  - 1 step から 8 step への差のうち、女声の NISQA-TTS は voices 89%、jvs_bal 46%、jsut 41% が戻る。comb は女声全体でほぼ全部(female_05 は 62%、female_02 は 8 step を超える)。
  - 読みと間は、1 step と 8 step の間。間は 1 step(S2)より増える(voices 10.5%、8 step と同じ傾向)。
  - 試聴ページ: `exp/listening/hs_s2_20k/`(ローカルのみ。JVS を含む。列は教師、P3m 20k、S2 20k の 1 step、8 step、HS)。
- **次**: S2 60k にも同じことを 20k step で行う(`configs/train/HS_s2_60k.yaml`。`hs_start.sh` が S2 60k の snapshot を待って自動で始める)。cpu の費用は 1 step のまま(head の大きさは変わらない)。

### 10/24 17:45: 残るのは female_02 / 05 の裏返り・二重だけ(ユーザー)。noise の切り詰め(noise_clamp)の確認

- **ユーザー(10/24 17:40、hs_s2_20k)**: HS の 1 step は 8 step とほぼ同じ。S2 20k の 8 step を超える感じはない。教師との距離は **female_02 / 05 に限った話で、高い声の裏返りや二重の感じ**。ほかの声(主に HS を聴いた)はほぼ問題ない。間は少しだけ気になるものがある(例: jvs055 の「遺灰」)。
  - jvs055「遺灰」の間(`pauses.py` の区切り、35 dB): 教師 0.36 / 0.50 / 0.12 / 0.40 …、HS 0.40 / 0.84 / 0.74 / 0.10 / 0.62 s。1 s は超えないが、読点の間が教師より 0.3〜0.6 s 長い。jvs_bal 全体でも間の合計は教師の 1.5 倍(1.49 / 1.00 s)。release(main 系、Mimi 12.5 Hz)は 0.88 s で、1 s 以上の間は 0%。
- **S2 40k の評価の遅れ**: `s2_fetch.sh` は W1x のときの「`train_decoder.py` が動いている間は待つ」条件を持っていて、W1w の学習中ずっと止まっていた。17:22 に条件を外した `s2_fetch2.sh`(40k と 60k)に置き換えた(出力は同じ `s2_fetch.out` に追記)。
- **noise の切り詰め**(`clamp_probe.sh`。`student_synth.py --noise-clamp`: 公式 wheel の `noise_clamp` と同じ truncated normal。読み込み時の引数で、既定は無し)
  - 裏返りが、noise の裾から引いたまれな frame から来るなら、温度をあまり下げずに切り詰めで消せる。
  - HS_s2_20k(1 step、温度 0.7)で、±2.5 / 2.0 / 1.5(3.0 / 2.4 / 1.8 σ。noise の std は 0.84 → 0.83 / 0.80 / 0.70)。対照に、切り詰めなしの温度 0.5(std 0.71)。voices と jvs_bal。batch 8(1.2 GB)で W1w の横で回す。
