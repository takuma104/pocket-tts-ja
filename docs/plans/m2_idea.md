# Lossアーティファクト対策

**User:** Anonymous  
**Created:** 10/13/2026 22:28:13  
**Updated:** 10/13/2026 22:32:50  
**Exported:** 10/14/2026 0:06:34  
**Link:** [https://chatgpt.com/c/6ace31ec-f444-83e8-8e35-8f86e971ed7b](https://chatgpt.com/c/6ace31ec-f444-83e8-8e35-8f86e971ed7b)  

## Prompt:
2026/10/13 22:28:53

Neural Codecの開発の時の話でencode/decodeを通してGANなどで学習させますが、この時のLossの取り方で、発生しやすいアーティファクトやその対策といったものの研究はありますか？特に数キロHz以上のコンポーネントが人工的な金属的響きになってしまい、同様の事象は何が原因か、対策法があるのかを知りたいです

## Response:
2026/10/13 22:28:56 · gpt-6-thinking

関連研究を調べ、特に「数kHz以上に出る金属的な響き」を、GANの損失設計・STFT系損失・量子化・デコーダ構造のどれが引き起こしやすいかに分けて整理します。論文で確認されている現象と、実装上の切り分け方法を区別してまとめます。

> **仕上げる技術提案**
>
> 実装可能な診断実験として、重み付きマルチスケールSTFT損失、帯域別評価、過渡応答の確認を整理している。位相誤差や高域ノイズを過大評価しないよう、複素STFT損失は慎重に扱い、関連研究の引用も正確に確認する。
>
> **分類と検証を整理**
>
> 損失の種類ごとに、金属的な音、過度な平滑化、高域ノイズ、周期的なリンギングとの対応を整理している。単純波、チャープ、雑音、無音を使い、帯域別誤差や各損失の勾配を比較する設計にしている。
>
> **技術提案を仕上げ分類と検証を整理**
>
> 
>
> **24件のウェブサイトを検索しました**
>
> 
>
> 2m 39s考えました

あります。しかも、今回の「**数kHz以上の高域に、人工的な金属音・共鳴・周期的な響きが混ざる**」という症状に非常に近い現象を、Neural CodecのLossとDiscriminatorの設計に結び付けて解析した研究が見つかりました。

特に興味深いのは、**GANのDiscriminatorに使用するSTFTのFFTサイズの組み合わせが、再構成音声に周期的なアーティファクトを発生させる**という報告です。

## 1\. 最も関連性が高い研究

### entity["other","Scaling Transformers for Low-Bitrate High-Quality Speech Coding","Stability AI 2024 neural audio codec research paper"]

Stability AI, Parker et al. (2024)

url論文https://arxiv.org/abs/2411.19842 — 特に Appendix B.5「Systematic bias in loss functions」

この論文では、TransformerベースのCodecを学習する過程で、高周波領域に周期的なアーティファクトが現れる問題を発見しています。

研究チームは、それぞれのLossが波形のどの時間・周波数成分に強い勾配を与えるかを解析しました。

結果は次のとおりです。

- **Waveform L1/L2 Loss**：系統的な時間・周波数バイアスは観測されなかった。
- **STFT Reconstruction Loss**：適切なWindowとOverlapを使用した場合、同様に系統的なバイアスは観測されなかった。
- **GAN Adversarial Loss / Feature Matching Loss**：特定の時間位置や周波数に勾配が集中する、明確な周期的バイアスが現れた。

特に、EnCodec、DAC、BigVGANのDiscriminator構造でも類似の傾向を確認しています。citeturn493203view1turn489494search4

#### 対策：STFTのFFTサイズを非整数比にする

一般的なSTFT Discriminatorでは、例えば256、512、1024、2048という2の累乗のFFTサイズを使います。

論文では、これらが互いに規則的な倍数関係にあることに注目し、代わりに黄金比に近い間隔で配置したFFTサイズを採用しました。

STFT DiscriminatorのFFTサイズ

従来の例：2の累乗

周期性を抑える設計

論文で採用された構成

この変更によりDiscriminatorの勾配に現れていた鋭い周期的なパターンが弱まり、再構成スペクトルの周期的アーティファクトも解消したと報告しています。citeturn185081view0turn185081view2

重要なのは、**GeneratorやCodecの容量を増やさなくても、Discriminator側の周波数分解能の設計を変えるだけで改善した**ことです。

今回の症状がGANを有効にした時点で強くなるのであれば、この論文の手法は優先して試す価値があります。

### 2\. Aliasing-Free Neural Audio Synthesis — 2025

Gu et al. (2025)

url論文https://arxiv.org/abs/2512.20211 / url実装（Pupu-Codec）https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis

こちらはLossよりもDecoderの構造に注目した研究ですが、金属的な響きの原因を特定する上で非常に重要です。

論文では人工的な高周波成分を3種類に分類しています。

| 種類 | 発生メカニズム | 音・スペクトル上の特徴 |
|--------|--------|--------|
| Tonal artifact | ConvTransposeの周期性、DCバイアス | 一定周波数の不要なピーク、リンギング |
| Mirrored aliasing | Upsamplingによるスペクトルの鏡像 | 高域に人工的な倍音・折り返しが出る |
| Folded-back aliasing | 非線形活性化で生じたNyquist超過成分 | 音高に連動して動く人工的な周波数成分 |

提案されたPupu-Codecは、ConvTransposeをリサンプリング系のUpsamplerに変更し、さらに非線形活性化をAnti-aliasedなものに置き換えています。

この手法はGANの重み調整だけでは取り除けない、**Decoder自身が生成してしまう不要な高周波成分**に対する対策です。citeturn501751academia24turn501751view1

### 3\. Upsampling Artifacts in Neural Audio Synthesis — 2021

Pons et al., ICASSP 2021

url論文https://arxiv.org/abs/2010.14356 / url検証用Notebookhttps://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio

Dolby Laboratoriesによる、Upsampling由来のTonal artifactを体系的に分析した研究です。

特に重要なのは、ConvTransposeではKernel sizeがStrideの整数倍で、Overlap数が一定であっても、重みの初期値などに由来する周期成分が発生し得ることです。

さらに、DiscriminatorのStrided Convolutionを逆伝播するときにも、周期的な高周波勾配が発生し得ると指摘しています。

つまり、**GeneratorのUpsamplingと、Discriminatorを通したLossの勾配の両方に周期構造が入り得る**わけです。citeturn489494search1turn489494search2

この研究では、無音部分や学習分布外の音声でTonal artifactが残りやすいことも観測されています。

## 4\. Lossの種類とアーティファクトの関係

この点は、urlHigh-Fidelity Audio Compression with Improved RVQGAN（DAC）https://arxiv.org/abs/2306.06546 のAblation Studyが非常に参考になります。

DACの研究では、GANを除いたReconstruction Lossのみの学習で、スペクトル上の誤差はそれほど悪化しない一方、位相の再構成が不十分となり、Buzzing（ブーンという人工音）が発生したと報告されています。citeturn785862view0turn684078view0

つまり、GANの比重が大きすぎても小さすぎても、異なる種類のアーティファクトが発生する可能性があります。

以下は論文の知見と、信号処理上の推論を合わせた整理です。各Lossが必ずその症状を起こすという意味ではありません。

| Loss・設計 | 起こり得る問題 | 対策の方向 |
|--------|--------|--------|
| Mel Loss中心 | 高域の細かい倍音構造の欠落、ぼやけ | Linear-frequency STFT Lossを併用 |
| Magnitude STFTのみ | 位相の不一致によるBuzzing、時間構造の劣化 | GAN、Feature Matching、必要に応じて位相Loss |
| Adversarial Lossが強すぎる | 不自然な倍音、周期性、学習不安定 | GAN比重を変更、Dの構造を見直す |
| STFT Discriminator | 周期的勾配バイアス、Tonal artifact | 非整数比のFFTサイズ |
| Multi-period Discriminator | 周期構造への偏り、不自然な倍音 | STFT系Dとの比較・併用 |
| Waveform L1/L2中心 | 時間・位相のずれに敏感、曖昧な成分の平滑化 | Spectral Lossと併用 |
| RVQ容量不足 | 高域の情報喪失、音色の不自然さ | Quantizerを外した比較実験 |

### 高域専用のLossを追加する方法

個人的に試す価値があると思うのは、従来のMulti-resolution Mel Lossに、**高域のLinear-frequency Log-STFT Lossを別項として追加する**方法です。

例えば24kHz音声であれば、4〜12kHzのスペクトルだけを抽出し、

$$
L_{\mathrm{HF}} =
\frac{1}{|\Omega|}
\sum_{(t,f)\in\Omega}
\left|
\log(|X_{t,f}|+\epsilon)
-
\log(|\hat X_{t,f}|+\epsilon)
\right|
$$

とする設計です。ここで $\Omega$ は高域の時間・周波数Binの集合です。

Mel Lossでは周波数Binが高域で粗くなるため、細い人工的なピークが平均化されて見逃される可能性があります。Linear-frequency STFTではそれらを直接比較できます。

ただし、Log Lossで無音に近いBinを強く評価しすぎると、微小なノイズ成分まで学習しようとするため、$\epsilon$ や有効Binの重み付けには注意が必要です。

Stability AIの論文でも、学習後半のDiscriminatorが可聴性の低い微小スペクトル成分に過度に注目する現象を観測しています。その対策として、複素STFTの振幅を非線形スケーリングし、小さいBinを相対的に弱めています。citeturn291559view2

### GANの比重を調整する方法

EnCodecの urlLoss Balancerhttps://arxiv.org/abs/2210.13438 も参考になります。

通常の加重Lossでは、例えば

$$
L_G =
\lambda_{\mathrm{mel}}L_{\mathrm{mel}}
+\lambda_{\mathrm{adv}}L_{\mathrm{adv}}
+\lambda_{\mathrm{fm}}L_{\mathrm{fm}}
$$

の係数が、そのまま勾配の寄与割合にはなりません。

EnCodecではLossごとの勾配Normを計測し、目標の寄与割合になるよう動的に調整しています。citeturn593601view0turn593601search0

ただしこれはLossの強さの調整であって、Discriminatorが持つ周波数的なバイアス自体を解決するものではありません。

## 5\. 追加で参考になるDiscriminator設計

image_group{"query":["Avocodo GAN vocoder architecture collaborative multiband subband discriminator paper"],"aspect_ratio":"4:3","layout":"carousel","num_per_query":1}

**urlAvocodo: Artifact-Free Vocoderhttps://arxiv.org/abs/2206.13404**

低域に偏るMulti-scale Discriminatorを改善し、高域も評価するSub-band Discriminatorを導入した研究です。さらにPQMFによるAnti-aliasingを使っています。citeturn501751search0

image_group{"query":["Descript DAC improved RVQGAN multi band multi scale STFT discriminator architecture figure"],"aspect_ratio":"4:3","layout":"carousel","num_per_query":1}

**urlDAC: Improved RVQGANhttps://arxiv.org/abs/2306.06546**

Multi-band Complex STFT Discriminatorで高域のAliasingを抑制。著者らは、客観評価指標への影響は小さくても、スペクトログラム上では明確に改善することを確認しています。citeturn684078view0

この2研究は、単純にGANの係数を変更するよりも、**Discriminatorが高周波のアーティファクトを適切に検知できるようにする**という方向です。

## 6\. 今回の症状を切り分ける実験

ここはかなり明確に切り分けられると思います。

### まずスペクトログラムで確認

| 観測される特徴 | 疑うべき原因 |
|--------|--------|
| 入力音の音高に関係なく、同じ周波数に横線が出る | ConvTranspose、Discriminatorの周期バイアス |
| 入力音の音高に合わせて人工的なピークも動く | Aliasing、非線形活性化による折り返し |
| 4kHz以上に広くザラザラした成分が出る | 位相再構成、GANによるノイズ生成 |
| 音の立ち上がり後に金属的な余韻が残る | リンギング、時間方向の表現不足 |
| Quantizerを外すと改善する | RVQ容量・量子化誤差 |

上の表は診断の目安であり、確定診断ではありません。

実験には、純音、Chirp、無音、ホワイトノイズ、実際の音声を使い、入力と再構成結果のLinear-frequency STFTを同じdBスケールで比較するとよいです。

### LossのAblationをするなら

実際に学習し直す場合、次の比較を推奨します。

| 実験 | 変更点 | 確認したいこと |
|--------|--------|--------|
| A | 現在の構成 | Baseline |
| B | GAN・FMをともに無効化して学習 | 人工音がGAN由来か |
| C | STFT Discriminatorだけ変更 | FFTサイズ由来の周期バイアスか |
| D | 高域Linear STFT Lossを追加 | 高域の再構成誤差を抑えられるか |
| E | MPDを無効化して学習 | Periodic Discriminatorの影響か |
| F | Quantizerをバイパスして推論 | 量子化容量が原因か |

特にBとFは原因を大きく分類するための実験として有効です。ただしBでは位相品質も落ちる可能性があるため、「人工音が減った」だけで総合的な改善とは判断できません。

また、Lossの数値を比較するだけでなく、Decoder出力の波形に対する各Lossの勾配を取り、

$$
g_i=\frac{\partial L_i}{\partial \hat{x}}
$$

これを周波数帯域別に分解してNormを測ると、**高域を実際にどのLossが支配しているか**が可視化できます。

これはStability AIのAppendix B.5の解析とも近いアプローチです。

---

## 結論

今回の金属的な響きについて、優先的に疑うべきなのは以下の3系統だと思います。

**Discriminatorの周波数バイアス**

金属的な固定周波数の響きが出ているなら、Stability AIのFFTサイズ変更の研究が最も直接的です。

---

**DecoderのUpsampling / Aliasing**

音高に連動する高域の人工音であれば、Pupu-CodecやBigVGANのAnti-aliasing手法が有力です。

---

**Mel LossとGAN Lossのバランス**

音声全体がシャリシャリ、金属的になるのであれば、Linear STFT Lossの追加やGAN勾配の調整を検討します。

一つ確認したいのですが、現在のCodecのLoss構成は **DAC/BigVGAN系のMel + Adversarial + Feature Matching** に近いでしょうか？ それとも **EnCodec系のMulti-resolution STFT + GAN** に近いでしょうか？

この違いが分かると、特に今回の高域アーティファクトに対して有効そうな変更を、かなり絞り込めると思います。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)