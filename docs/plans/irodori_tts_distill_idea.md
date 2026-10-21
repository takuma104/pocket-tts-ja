このブランチでは、IrodoriTTSの直接蒸留の実験を行いたいと考えています。

mainブランチおよびmimi_ftブランチでは、PocketTTSの日本語FTを行っていました。かなりコードがごちゃごちゃしてきたので、一旦あえて古いbaseからブランチを作成しました。必要そうなものは両ブランチから適宜コードを持ってきてください。

またこのローカルの.gitignoreされているフォルダにはmainブランチで行っていた実験中の生成物がそのままになっています。もう使わなそうですが、このままにするか、あるいは移動して整理してもらってもOKです。

下はChatGPTとの会話で、おおむねChatGPT案でいきたいと考えています。最終的には、(うまくいった場合) PocketTTSの公式wheelで動くウエイトにまで持っていき、公開したいと考えていますが、まずはChatGPT案のように小さく始めたいです。

---

# 蒸留実現性検討

**User:** Anonymous  
**Created:** 10/22/2026 2:02:33  
**Updated:** 10/22/2026 2:05:34  
**Exported:** 10/22/2026 2:15:36  
**Link:** [https://chatgpt.com/c/6ad8f025-13a0-83e8-a57c-d45ea8ed8f82](https://chatgpt.com/c/6ad8f025-13a0-83e8-a57c-d45ea8ed8f82)  

## Prompt:
2026/10/22 2:02:40

https://github.com/takuma104/pocket-tts-ja でPocketTTSの日本語FTを行っていました。とりあえず一旦完了したのですが、特に女性音声の品質がうまくいっておらず、作りなおしを色々考えています。

pocket-tts-ja でも合成音声の生成に IrodoriTTS https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small を使っていました。あるいみ広義の蒸留を行っていた訳ですが、IrodoriTTSでのlatentレベルでPocketTTSのアーキテクチャに直接蒸留できないかな? というのを考えています。

・Encoder/Decoderはボトルネックが連続表現
・IrodoriTTSのTextEncoder + RF-Ditでの生成物のlatentを、PocketTTS側のFlowLMの生成LatentへTeacher Force

とかでいけないかな?と思うのですが、実現性ありそうか検討してもらえますか?

## Response:
2026/10/22 2:02:40 · gpt-6-astra-wm

両モデルの実装で latent の定義と学習経路を確認し、直接使える部分と変換・再学習が必要な部分を整理します。女性音声の品質についても、生成モデルと音声 codec のどちらが制約になりうるかを踏まえて検討します。

両方とも32次元の連続 latent ですが、座標系は別なので、そのまま比較する損失は使えません。一方、PocketTTS の FlowLM に Irodori の latent を生成させ、出力を Irodori 側の DACVAE で復号する構成は有望です。リポジトリの記録にも元の Mimi で女性話者の劣化が大きかった結果があるため、この codec の置き換えには検証する理由があります。

> **Worked**
>
> 
>
> **Searched the web、⁨used⁩ ⁨GitHub⁩**
>
> 
>
> **⁨Used⁩ ⁨GitHub⁩、searched the web**
>
> 
>
> 1m 22s考えました

**実現性は十分あります。今回の目的なら「PocketTTS の FlowLM 構造を使って、Irodori の DACVAE latent を直接生成する」構成が、かなり筋のよい候補です。**

ポイントは、**Mimi の latent に Irodori の latent を合わせるのではなく、生徒側も Irodori と同じ latent 空間・decoder を使うこと**です。この構成なら、現在の合成音声経由の学習にある「DACVAE で復号 → Mimi で再符号化」という経路をなくせます。

実装を確認した範囲で、可能な構成、学習方法、注意点を整理します。以下は設計上の検討で、実際に学習して品質を確認した結果ではありません。

**1. 両モデルの latent は、同じ32次元でも別の表現**

| 項目 | Irodori-TTS v4.1-Small | PocketTTS |
|---|---|---|
| 音声 codec | Semantic-DACVAE-Japanese-32dim | PocketTTS 用の連続 latent Mimi |
| latent 次元 | 32 | 32 |
| フレームレート | 25 Hz | 12.5 Hz |
| 1フレームの時間 | 40 ms | 80 ms |
| 復号音声 | 48 kHz | 24 kHz |
| 生成方法 | 発話全体を RF-DiT で生成 | AR Transformer＋フレーム単位の flow head |
| 音声の過去参照 | 全体の noisy latent を参照 | 生成済み latent を参照 |

根拠は Irodori の [モデル設定](https://github.com/Aratako/Irodori-TTS/blob/main/configs/train_v4_small.yaml)、[codec 実装](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/codec.py)、PocketTTS の [設定](https://github.com/kyutai-labs/pocket-tts/blob/41cbc84af539ea78a804ffca5f9c6edc1a22ce44/pocket_tts/config/english_2026-04_24l.yaml)です。Irodori の時間換算は公式ドキュメントでも25 fpsとされています。([github.com](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md?utm_source=chatgpt.com))

したがって、既存の出力同士で

$$
\|z_{\mathrm{Mimi}}-z_{\mathrm{DACVAE}}\|^2
$$

を取ることには意味がありません。次元ごとの意味も、時間方向の対応も異なるためです。

ただし、**FlowLM 自体は Mimi 固有の生成器ではありません**。実装上も、codec の連続 latent を正規化して入力し、その次の latent の条件付き分布を flow head で学習しています。生成対象を DACVAE latent に置き換えることは可能です。

**2. 提案されている Teacher Forcing は、その形で成立します**

教師から、テキスト $x$ と話者条件 $r$ に対する、**RF の積分が完了した最終 latent 列**を取得します。

$$
z^T_{1:T}=\mathrm{Irodori}(x,r,\epsilon_T)
$$

生徒側では、これを教師データとして、

$$
h_t=\mathrm{ARTransformer}_\theta(x,r,z^T_{<t})
$$

から次のフレーム $z^T_t$ の分布を学習します。

学習時の入力は、概念的には次の並びになります。

```text
話者条件、テキスト、BOS、z₁、z₂、…、zₜ₋₁
                         ↓   ↓          ↓
学習対象                 z₁  z₂         zₜ
```

これは PocketTTS の既存の学習経路にかなり近いです。[conditioner.py](https://github.com/kyutai-labs/pocket-tts/blob/41cbc84af539ea78a804ffca5f9c6edc1a22ce44/training/modules/conditioner.py) では実際に `BOS + normalized_latents[:, :-1]` を組み立て、[model.py](https://github.com/kyutai-labs/pocket-tts/blob/41cbc84af539ea78a804ffca5f9c6edc1a22ce44/training/modules/model.py) で対応する latent を flow loss の目標にしています。

ここで重要なのは、**「teacher forcing」と「生成 latent に直接 MSE」を区別すること**です。

通常の conditional flow matching で書くと、正規化済み教師 latent $y_t$ に対して、

$$
u_\tau=(1-\tau)\epsilon+\tau y_t
$$

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E\left[
\left\|
v_\theta(u_\tau,\tau;h_t)-(y_t-\epsilon)
\right\|^2
\right]
$$

を学習します。過去入力に教師 latent を使い、次フレームは確率的に生成できるようにする形です。PocketTTS の LSD を使う場合も、教師データの差し替えという基本は同じです。

一方、独立した乱数から生成した生徒 latent と教師 latent の MSE を単純に最小化すると、複数の妥当な発声を平均する方向に働きやすくなります。**最初の実験は、既存の flow/LSD 学習＋教師 latent の teacher forcing がよいと思います。**

また、Irodori が非ARであることは、この学習の原理的な障害ではありません。教師が生成する系列の分布を、生徒が

$$
p_S(z_{1:T}\mid x,r)
=
\prod_t p_S(z_t\mid z_{<t},x,r)
$$

として学ぶことはできます。ただし、長い発話の一貫性や、自由生成時の誤差蓄積まで同等にできるかは別の検証項目です。

**3. 最も有望なのは「DACVAE を共有する小型 AR-TTS」**

想定構成は次です。

| 段階 | 処理 |
|---|---|
| 教師データ生成 | Irodori の TextEncoder＋RF-DiT → 最終 DACVAE latent を保存 |
| 生徒の学習 | テキスト・話者条件・過去の教師 latent → 次 latent の flow/LSD 学習 |
| 生徒の推論 | PocketTTS 型 FlowLM → DACVAE latent → 凍結した DACVAE decoder |
| 実録音の追加 | 同じ DACVAE encoder で符号化し、合成 latent と混ぜる |

Irodori の [inference_runtime.py](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/inference_runtime.py) では、`sample_euler` の結果を `unpatchify_latent` で戻し、`z = z[:, :latent_steps]` とした後に `codec.decode_latent(z)` を呼びます。**この復号直前の `z` が、そのまま保存したい教師データです。**

この構成の利点は、教師が実際に生成した表現をそのまま学べることです。既存の wav を DACVAE で再符号化する方法も使えますが、

$$
E_D(D_D(z))\neq z
$$

なので、教師の元の latent と完全に同じではありません。新規合成分では直接保存する価値があります。

なお、この方式は「RF-DiT の内部表現やベクトル場をコピーする蒸留」ではなく、**latent 空間での系列レベルの蒸留**です。教師の生成過程を再現する必要がないため、アーキテクチャが違っていても取り組みやすいです。

**4. 変更が必要なところ**

**フレームレートと生成コスト**

まずは **32次元・25 Hz のまま**で試すのが素直です。ただし、PocketTTS と同じ長さの音声に対する AR ステップ数が2倍になります。モデル全体の速度がちょうど半分になるとは限りませんが、CPU での軽さには影響します。

速度を重視する次の案としては、DACVAE の隣接2フレームを連結できます。

$$
\tilde z_k=[z_{2k},z_{2k+1}]\in\mathbb R^{64}
$$

これなら **64次元・12.5 Hz** となり、復号前に32次元・25 Hzへ戻せます。平均や間引きではなく単純な連結なので、教師 latent の情報は失いません。ただし、flow head が64次元の同時分布を学ぶ必要があり、品質面の難易度は別途比較が必要です。

**既存重みの再利用**

日本語 tokenizer、text embedding、Transformer 本体は初期値として試せます。しかし、音響表現が変わるため、次は再初期化・再推定の候補です。

- 音声入力の `input_linear`
- latent の `emb_mean` / `emb_std`
- `bos_emb`
- `flow_net`
- DACVAE 参照 latent を入れる場合の `speaker_proj_weight`

**Transformer 本体まで凍結して、入出力だけ合わせれば済むとは考えない方がよいです。** 既存重みからの転移と、同じ構造の新規学習を比較する価値があります。

**話者条件**

出力だけでなく、参照音声をどう入力するかも決める必要があります。

最初は、参照音声も DACVAE で符号化し、新しい speaker projection を通して prefix に入れる方式が分かりやすいです。教師と生徒に同じ参照音声を与え、合成時の caption を使うなら、その条件を生徒にも渡すか、固定して曖昧さを減らします。

Irodori の TextEncoder を生徒にも残すことは可能ですが、**latent 蒸留に必須ではありません**。小型化が目的なら、まず既存の日本語テキスト入力系で試す方が構成を評価しやすいと思います。

**ストリーミング**

FlowLM が AR でも、DACVAE の復号まで自動的にストリーミング対応になるわけではありません。[DACVAE の decoder](https://github.com/facebookresearch/dacvae/blob/main/dacvae/model/dacvae.py) には非因果的な畳み込みが含まれます。

まず全文を復号して品質を確認し、その後に、必要な先読み量、チャンク境界の重複処理、初回音声までの遅延を検証する順番がよいです。**PocketTTS の codec 置換では、ここが実用上の大きな変更点です。**

**5. 女性音声の改善という目的には、特に検討する理由がある**

[実験計画・結果の記録](https://github.com/takuma104/pocket-tts-ja/blob/main/docs/plans/ja_finetune_plan.md) も確認しました。元の Mimi の再構成で女性話者の劣化が大きかったことに加え、decoder FT では、4 kHz 以上の帯域を持ち上げても倍音構造が十分戻らず、雑音的な成分が目立ったという切り分けが記録されています。

これは、**FlowLM によりよい教師データを与えるだけでは取り除けない問題が codec 側にあった**ことを支持します。ただし、その記録から「Mimi latent に必要な情報が存在しない」「decoder 構造上、改善不能」とまで断定はできません。

今回の方式なら、問題があった Mimi の encode/decode 経路自体を外せます。その意味で、latent への直接蒸留と codec の置換は、今回の目的によく合っています。

逆に、**Irodori latent → 学習した変換器 → Mimi latent → Mimi decoder** という構成は、既存システムとの互換性には利点がありますが、問題のある復号経路を残します。女性音声の品質改善を主目的にするなら、優先順位は低いです。

**6. 私なら、次の順番で検証します**

1. **問題が目立つ女性音声で codec を比較する**  
   実録音と Irodori 合成音声について、現在の最良 Mimi と DACVAE の再構成を比較します。教師の直接出力も含め、音量を合わせて試聴し、高域の倍音構造を確認します。ここで DACVAE の利点が明確なら、置換の根拠ができます。

2. **32次元・25 Hz、DACVAE 凍結で小規模 PoC**  
   教師の最終 latent を保存し、既存 trainer の flow/LSD＋EOS 学習を利用します。少数話者で学習経路を確認した後、男女・F0 帯域を分けた未学習話者で評価します。

3. **teacher-forced 条件と自由生成を分けて評価する**  
   正しい過去 latent を与えた条件ではよいのに、自由生成で荒れるなら、主な課題は codec ではなく AR の誤差蓄積です。学習 loss だけで判断せず、早い段階から自由生成を聴きます。

4. **品質を確認してから軽量化する**  
   2フレーム連結、層数削減、少ステップ生成、チャンク復号を順に比較します。初めから全部変更すると、品質低下の原因が分からなくなります。

私の推奨は、**「Irodori の codec を固定したまま、PocketTTS 型の小型 AR 生成器を学習する」方向です。** 提案された teacher forcing はその中心として使えますし、波形経由の二重 codec をなくす点にも明確な意味があります。成功の鍵は「latent 同士を近づける損失」よりも、**共有する latent 空間を選び、その分布を生徒の flow head に正しく学習させること**だと思います。

---

**Sources:**

- [Irodori-TTS/docs/parameters.md at main · Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)