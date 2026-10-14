# Mimi decoder FT: 高域改善の追加実験案

作成: 2026-10-15（JST）

目的: 採用済みの日本語 Pocket TTS の明瞭度と自然さを維持しながら、特に女性の有声区間で、4kHz 以上の高域を自然な倍音として復元する。

本書は会話で提案した内容を、別マシン・別インスタンスへの引き継ぎ用にまとめたもの。以下の追加実験は未実施で、効果は未確認。既存実験の事実は [ja_finetune_plan.md](ja_finetune_plan.md)、以前の文献調査は [m2_idea.md](m2_idea.md) を参照する。

## 1. 出発点と、既存実験から分かっていること

### 現在の採用版

- デコーダーは **`M2_gs2gan_keep4k`（90k step）**。ユーザーが 2026-10-14 に試聴して採用し、公開版の `japanese` / `japanese_24l` に反映済み。
- FlowLM と encoder は変更していない。decoder 側の追加学習だけで改善している。
- この版では、4kHz 以下を原音に近づけ、4kHz 以上は元の Mimi の出力に対する mel loss でレベルを保つ。識別器に渡す音声も 4kHz で low-pass し、共有スペクトル loss は 300–4000Hz に限定している。
- JVS 再構成の Kana-CER は、元の Mimi 5.44% → 採用版 4.70%。原音との話者類似度は 0.698 → 0.779。この値は decoder 比較時の評価で、公開モデル全体の評価とは区別する。

### 残った問題

高域の問題は、少なくとも次の 2 種類に分かれる。

1. **固定周波数の線スペクトル**: アップサンプリングのストライドに対応する周期成分。共有スペクトル loss を 12kHz まで適用した `M2_gan_spec2`（`gs2_80k`）では、2dB を超える線は 0 本まで抑えられた。
2. **有声区間の高域の構造**: 女性の声で、原音にはある 4kHz 以上の倍音の縞が薄れ、帯域のエネルギーが雑音状の成分で補われる。固定周波数の線を消しても聴感上の人工音が残った。

`gs2_80k` は高域のレベルを原音に近づけたが、試聴では不採用だった。高域を元の Mimi 程度に抑えた採用版は聴感も改善したが、高域のこもりが残る。

女性の有声フレームにおける代表値（倍音の立ち方 / 原音に対する帯域レベル、dB）は次のとおり。詳細な測定条件は既存計画を参照する。

| 帯域 | 原音の倍音の立ち方 | 元の Mimi | `gs2_80k` | 採用版 `M2_gs2gan_keep4k` |
|---|---:|---:|---:|---:|
| 4–6kHz | 4.7 | 1.2 / −6.9 | 0.6 / −0.9 | 1.3 / −6.8 |
| 6–8kHz | 2.6 | 0.0 / −7.3 | 0.0 / −1.8 | 0.0 / −7.7 |

すでに試した以下の方法を、条件を変えずに繰り返す優先度は低い。

- 固定 EQ で元の Mimi の高域を持ち上げる。
- 現構成のまま高域の細かい STFT magnitude loss だけを追加する（`M2_fine4k` は試聴で悪化）。
- 無声区間だけ高域を上げる（`M2_gs2keepV4k`）。
- 元の Mimi の高域の波形をそのまま保つ（`M2_keepW4k`）。

### まだ分かっていないこと

「今回の学習条件では高域の倍音が復元できなかった」と「この構造では原理的に復元できない」は区別する。原因候補は decoder の容量、倍音を生成する仕組み、latent の情報量、学習データ・損失設計で、容量不足はまだ確定していない。

M1 プローブで確認したのは **子音情報が latent に残っていること**。高域の倍音・振幅・位相の情報が十分残っていることまでは確認していない。

Pocket 版 Mimi の `quantizer` は実際には `DummyQuantizer` の出力射影で、RVQ による量子化は行わない。容量を議論する対象は、32 次元・12.5Hz の連続 latent と encoder / decoder。RVQ のコードブック数を増やす案は、この実装には直接適用できない。

## 2. 推奨する方針と優先順位

**最初は既存 encoder と latent 空間を維持して、decoder の構造を変える。** これなら、学習済み FlowLM と既存 latent を活用できる。

1. 小さな容量実験として、SEANet decoder の残差層を 1 → 3 に増やす。
2. 高域の倍音を生成する仕組みを変える研究案として、Wavehax 系または Vocos 系の decoder を検証する。
3. 自然録音を中心にしたデータ条件、帯域別 STFT discriminator を別の比較実験として追加する。
4. 旧 latent での改善が難しい場合に、encoder / latent 次元 / frame rate の変更へ進む。

容量・データ・識別器を同時に変えると原因が分からなくなるため、比較条件をそろえて一つずつ変更する。

## 3. 案 A: SEANet decoder の容量を増やす

### 最初の変更

- `mimi.seanet.n_residual_layers: 1 → 3`
- `n_filters: 64`、latent 32 次元、frame rate 12.5Hz、sample rate 24kHz、アップサンプリング比 `[6, 5, 4]` は最初は維持する。
- Transformer の層数を増やすより、アップサンプリング後の波形生成側を深くする変更を先に試す。
- 出発点は採用済み `M2_gs2gan_keep4k`。既存ブロックの重みを移植し、追加した残差ブロックの最後の畳み込みの重み・bias をゼロ初期化する。追加ブロックを恒等写像にして、変更前の出力から学習を始める。
- 次の容量候補は `n_filters: 64 → 96`。幅の変更は形状の異なる重みの移植が必要で、残差層追加とは別の実験にする。

### 学習条件

採用版の「4kHz 以上を stock に合わせる制約」をそのまま残すと、高域の改善を要求できない。高域を原音に近づける学習条件を用意し、4kHz 以下の品質維持を併せて評価する。必要なら採用版出力との低域の距離を補助損失として使う。

構造の効果を判定するには、**残差層 1 の対照も、残差層 3 と同じデータ・損失・学習予算で回す**。採用版の評価に加え、この対照を置くことで、高域の制約を外した効果と容量変更の効果を区別する。

最初は 5k–10k step 程度の試験を候補とする。ただし、その時点で改善しないことは容量の限界を証明しない。学習の進み方と試聴を見て継続を判断する。高域の音量だけが増え、倍音構造や自然さが改善しない場合は、長時間学習への移行を急がない。

### 実装上の注意

現行の `build_mimi()` は encoder / decoder に同じ `seanet` 設定を渡す。YAML で残差層を増やすと両方が変わるため、次のいずれかが必要。

- 研究用に encoder / decoder の設定を分け、旧 encoder と拡張 decoder を組み立てる。
- 共通設定で両方に残差層を追加し、encoder の追加ブロックを恒等写像のまま凍結する。旧 encoder と encode 結果が一致することを確認する。この方式なら、既存 builder で表現できる構造を保てる。

`SEANet` は `ModuleList` を使うので、途中にブロックを追加すると後続の state dict の番号が変わる。単なる `strict=False` の読み込みではなく、既存層を対応づけて移植する処理が必要。

追加残差層の encoder / decoder 出力、ストリーミング状態、生成サンプル数を変更前と比較してから学習する。高い時間解像度で層が増えるため、パラメータ数だけでなく CPU 推論速度も測る。

## 4. 案 B: 倍音を作れる decoder へ置き換える

### Wavehax 系（症状に合いそうな研究案）

Wavehax は F0 から倍音を持つ信号を作り、それを手がかりに 2D 畳み込みで複素スペクトログラムを生成する。論文は、高い F0 に対する頑健性を報告している。女性話者の高域の倍音欠落への適用は、本リポジトリでは未検証の仮説。

Pocket 向けの構成案は次のとおり。

1. 既存 encoder の 32 次元・12.5Hz latent を入力にする。
2. 小さなネットで、より細かい時間刻みの F0 と有声 / 無声を予測する。学習時の目標は原音から抽出する。
3. 予測 F0 から、Nyquist を超える倍音を含まない励振信号を作り、decoder の手がかりにする。
4. 複素スペクトログラムを生成し、iSTFT で波形へ戻す。

まずは **原音 F0 を与える診断版**と **latent から F0 を予測する版**を比較する。原音 F0 の版は診断専用で、TTS 推論に使える構成ではない。両者の差は F0 復元の影響を調べる手がかりになるが、この比較だけで高域の全情報が latent にあるかを証明することはできない。

F0 の誤り、倍音の位相連続性、無声区間の扱いも確認する。原音 F0 でも改善しない場合は、decoder・学習条件・高域の振幅情報などに課題が残る。

### Vocos 系（比較候補）

Vocos は Fourier 係数を生成して iSTFT で波形を作る方式。既存 EnCodec の表現から復元する decoder として、元の decoder より聴感品質が改善した報告がある。旧表現を維持して decoder を置き換える方向の先例になる。

ただし、EnCodec の結果が Pocket の 32 次元 latent にそのまま当てはまるとは限らない。倍音生成の手がかりを持つ Wavehax 系と比較する候補として扱う。

### 検証の順序と互換性

最初はオフラインの再構成で可能性を確認する。改善した場合に、因果性、iSTFT の overlap-add 状態、チャンク境界、遅延、CPU 速度を検証する。学習時に未来の latent を使う構成なら、ストリーミング推論へそのまま移せない点も記録する。

Wavehax / Vocos 系への置換は、現行 wheel の decoder tensor を差し替えるだけでは動かない。推論コードの変更を伴う研究案であり、既存計画の「公式 wheel への互換」を、この提案だけで撤回したものではない。研究結果と配布可能な構成を区別する。

## 5. 案 C: データと識別器を見直す

### 高域の自然録音

これまでの GAN 学習は Irodori S3 と Emilia HQ を半々に使用した。Irodori は別の codec を通した合成音声なので、自然な高域の構造を学ぶ目標として十分かは確認が必要。

高域改善用には、**4–8kHz に実際の倍音構造が残っている、クリーンな女性の自然録音**を中心にした条件を置く。ビットレートや DNSMOS だけでは、この条件を保証できない。高域の帯域制限、雑音、残響、スペクトログラムの構造も確認する。

- 自然録音中心と既存の半々の mix を、同じ構造・損失で比較する。
- 男性、無声子音、低い音高でも退行しないことを確認する。
- JVS は引き続き評価専用にし、学習に入れない。
- 11kHz 付近の学習データの帯域端の問題と、主目標である 4–8kHz の倍音の問題を区別する。

### 帯域別 STFT discriminator

現在の全帯域 MS-STFT discriminator に対し、DAC 系の **multi-band multi-scale STFT discriminator** を比較候補にする。高域を独立して評価し、全帯域の平均では小さく見える不自然な成分を検知する狙い。

DAC では、高域の aliasing の改善が一般的な客観指標には大きく出ないことも報告されている。今回の「CER / 類似度は改善したが試聴で不採用」という結果と関係する可能性がある。ただし、高域の倍音欠落を解決する保証ではない。

固定周波数の線の再発は `periodicity.py` で別途追う。GAN の重みだけを増やすより、識別器が高域の何を検知するかを変更する比較を優先する。

## 6. YAML で encoder / decoder を作り直す場合

| 変更 | 狙い | 既存 FlowLM との関係 |
|---|---|---|
| decoder の残差層・幅を増やす | 同じ latent から波形をより良く復元する | 旧 encoder / latent を維持すれば再学習不要 |
| Wavehax / Vocos 系 decoder | 波形生成の仕組みを変える | 旧 latent を入力にすれば再学習不要。推論コードは変更 |
| latent を 32 → 64 次元にする | 保持する情報を増やす | FlowLM の入出力変更と再学習、latent の再計算が必要 |
| frame rate を 12.5 → 25Hz にする | 時間方向の情報を増やす | latent の再計算と FlowLM の再学習。系列長も増える |
| encoder を再学習する | 音声から取り出す情報を変える | 同じ 32 次元でも latent の意味・分布が変わり、FlowLM の再適応が必要になり得る |

YAML の形状変更だけで、既存の重みが対応するわけではない。整合する重みの移植または学習が必要。

latent 次元を変える場合は `inner_dim` / `quantizer.dimension` と FlowLM の latent 入出力を、frame rate を変える場合は音声のフレームサイズ、リサンプリング倍率、推論・学習の時間換算も確認する。最初の decoder 実験ではこれらを維持する。

## 7. 評価と、次の実験へ進む判断

主目標は「高域の音量」ではなく「高域を上げても自然に聞こえること」。CER / 話者類似度だけで採用を判断しない。

### 小さな試験で見るもの

- 原音 / 元の Mimi / 採用版 M2 / 新 decoder を、同じ latent・同じ文で比較する。
- 有声区間の 4–6kHz、6–8kHz の倍音の立ち方と帯域レベル。
- 無声子音の高域のレベル・構造。
- 固定周波数の線スペクトルと周期性。
- 同じ dB スケールの線形周波数スペクトログラムと、音量をそろえた試聴。
- 4kHz 以下の自然さ、文末、無音、チャンク境界の退行。
- CPU 推論速度、メモリ、ストリーミング遅延。

`hf_structure.py` は **原音 F0 を基準**に倍音の立ち方を測る。高次の倍音では小さな F0 のずれでも指標が下がるため、出力自身の F0 を使った値も併記し、倍音の消失とピッチのずれを区別する。出力自身の F0 で値が良くても、誤ったピッチを許容する指標にはしない。

試聴では、既存実験で問題が目立った jvs007 / jvs010 の再構成、jvs026 / 055 / 096 の女性 TTS、「一方で、漁業と商業で、リャネス港は繁栄していた。」を含める。男性の対照も含める。「すごい！」の「ス」は既存記録では FlowLM 側の問題の可能性があり、高域 decoder の単独の判定例にはしない。

再構成で改善した候補は、公開済み 6L の **同じ生成 latent**を復号して TTS にも効くか確認する。良い候補だけで、JVS 再構成 1,000 件、jvs_bal / JSUT / short / JKYB-mini などの既存評価を行う。

### 判断の目安

- 高域のレベルだけが上がり、倍音構造が改善せず人工音が増える: これまでの失敗と同じ方向。長い学習への移行を急がない。
- 倍音構造と試聴が改善し、低域や TTS に退行がない: 学習の拡大とストリーミング・CPU 評価へ進む。
- 旧 latent を使う decoder の複数の構造で改善が難しい: encoder / latent 容量の変更を次の候補にする。ただし、損失・データ・最適化の影響も残るため、latent の限界と断定しない。

## 8. 引き継ぎ時に参照する実装と成果物

| ファイル | 用途 |
|---|---|
| [ja_finetune_plan.md](ja_finetune_plan.md) | M0 / M1 / M2 の結果、試聴、採否、公開版の記録 |
| [m2_idea.md](m2_idea.md) | 過去の文献調査。一般的な提案と本リポジトリの実測は区別する |
| [codec_ft.py](../../src/pocket_tts_ja/codec_ft.py) | 学習対象の decoder パラメータ、ストリーミング窓、損失、識別器 |
| [mimi_utils.py](../../src/pocket_tts_ja/mimi_utils.py) | 公式 Mimi のロード、encode / decode |
| [train_mimi_decoder.py](../../scripts/m2/train_mimi_decoder.py) | M2 trainer。現在は default config で Mimi をロードする |
| [run_gan_keep.sh](../../scripts/m2/run_gan_keep.sh) | 採用版の学習条件。既存 run の checkpoint から継続するスクリプト |
| [recon_jvs.py](../../scripts/m2/recon_jvs.py) | JVS 再構成 |
| [hf_structure.py](../../scripts/m2/hf_structure.py) | 倍音構造と帯域レベル |
| [periodicity.py](../../scripts/m2/periodicity.py) | 固定周波数の線・周期性 |
| [make_recon_page.py](../../scripts/m2/make_recon_page.py) | 再構成の試聴ページ |
| [ja_24l_4k.yaml](../../configs/model/ja_24l_4k.yaml) | 現在の Mimi の構成 |
| [mimi.py](../../third-party/pocket-tts/pocket_tts/models/mimi.py) | encoder / decoder を組み立てる `build_mimi()` |
| [seanet.py](../../third-party/pocket-tts/pocket_tts/modules/seanet.py) | 残差ブロックとアップサンプリング構造 |

採用版の decoder は、既存マシンでは `runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors`。このパスは既存スクリプトと実験記録に基づくもので、引き継ぎ先での存在は未確認。`runs/`、音声、manifest、評価用 latent は別マシンにそろえる必要がある。

decoder の構造を変える場合、既存の `load_decoder()` は tensor 名・形状が同じであることを前提にしているため、移植処理と対応した model config のロードを追加する。構造が変わった run は新しいディレクトリで開始し、旧 optimizer state をそのまま resume しない。

引き継ぎ先では、まず採用済み decoder のロード・encode / decode・既存の比較音声の再現を確認し、その後に案 A の移植と対照実験に進む。新しい研究構成の評価は、旧 latent と既存の実験条件が維持されているかを記録する。

## 9. 参考文献

- [Wavehax: Aliasing-Free Neural Waveform Synthesis Based on 2D Convolution and Harmonic Prior](https://arxiv.org/abs/2411.06807): 倍音を持つ手がかりを使った複素スペクトログラム生成。高い F0 に対する頑健性。
- [Wavehax 公式実装](https://github.com/chomeyama/wavehax)
- [Vocos: Closing the gap between time-domain and Fourier-based neural vocoders](https://arxiv.org/html/2306.00814v3): Fourier 係数による生成と、既存 EnCodec 表現からの decoder 置換の実験。
- [High-Fidelity Audio Compression with Improved RVQGAN（DAC）](https://arxiv.org/html/2306.06546v2): 帯域別 STFT discriminator と高域 aliasing の評価。
- [Aliasing-Free Neural Audio Synthesis](https://arxiv.org/abs/2512.20211): 非線形活性化・アップサンプリング由来の aliasing の整理と対策。追加の構造変更を考える際の参考。
- [Continuous Audio Language Models](https://arxiv.org/html/2509.06926v2): 32 次元・12.5Hz の連続 VAE 表現の背景。一般的な品質指標の結果と、今回の高域の課題は区別する。
