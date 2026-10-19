# Mimi decoder FT: CC BY のデータだけで作る decoder 実験計画 v1

目的: 学習に使う音声を CC BY のもの（自然音声は YODAS 系だけ）に限っても、`mimi_ft_hf_plan.md` の decoder（H2ghSG4p100E、soup12hp。JVS 再構成 1,000 文の UTMOSv2 hp40 は keep4k +0.396 / +0.397、録音 +0.400）と同程度まで持ってこられるかを確かめる。届けば、main の CC 版（B-CC の生徒 D1-CC / D0-CC、`pocket-tts-ja-cc` 案）と一緒に公開する decoder の候補にする（公開するかどうかはユーザーが決める）。

ユーザーの依頼（2026-10-20）: 「自然音声を CC ライセンスのもの（Yodas 系）のみに絞った場合、同程度までもってこれるか」「Yodas 音声はクリップ毎にクオリティにかなりばらつきがあると思われますので、データ選別が重要になってくるかもしれません」。

## 1. ライセンスの前提

| 使える | 条件 |
|---|---|
| 元の Mimi（kyutai/pocket-tts） | CC BY 4.0。出発点 |
| YODAS3 ja（espnet/yodas3） | CC BY 3.0（引用の指定あり）。YouTube の opus、全帯域の自然音声 |
| Emilia-YODAS | CC BY 4.0。32 / 64kbps で帯域が狭い（高域の学習には使えない） |
| CC の声の Irodori 合成（`data/synth/s3_cc`） | Irodori-TTS v4.1 は MIT、参照の声は Emilia-YODAS（main の監査 `scripts/release/audit_cc_mix.py` で確認済み）。110,885 文 |

使えないもの: Hi-Fi-CAPTAIN（CC BY-NC-SA 4.0）、Emilia の高品質分（`data/m2/emilia_hq`、NC）、`data/synth/s3_full`（NC の声を含む）、JVNV（CC BY-SA 4.0。継承の条件が混ざる）、あみたろ（独自の規約）。

**今の decoder はどれも出発点にできない**。keep4k（`M2_gs2gan_keep4k`）は `M2_gan_spec` → `M2_gan_spec2` → keep4k の続きで、どれも s3_full と Emilia HQ を半々で学習している（`scripts/m2/run_gan_keep.sh`、main の計画 10/17 の記録）。HF の段（`run_hf.sh`）は Hi-Fi-CAPTAIN・JVNV・Emilia HQ で学習し、識別器も keep4k のものから始めている。CC 版は元の Mimi からやり直し、識別器も CC のデータだけで学習する。JVS はこれまでどおり評価だけに使う（学習には使わない）。

## 2. データ: YODAS3 の選別

### 2.1 一段目（メタデータ、済み）

`scripts/m2/extract_yodas3.py --select`。main の TTS の学習に使った YODAS3 の採用分（`data/yodas3_kept/train.jsonl`、238,233 発話 / 600.7h。字幕と Whisper が一致する自然音声で、DNSMOS OVRL 2.9 以上、量産の TTS の声に近いものは除外済み）から、

- 動画の音声が 15kHz 以上まで入っている（メタデータの `max_freq_hz`。opus の low-pass で決まる。採用分の時間で 10% 点 13.5kHz、中央値 20kHz）
- DNSMOS P.835 が OVRL 3.2、SIG 3.5、BAK 3.9 以上

で **90,241 クリップ / 234.4h、話者 7,470 人、動画 6,219 本**（107 シャード）。比較: Hi-Fi-CAPTAIN は女性・男性それぞれ約 1.9 万クリップ（各約 20h）。クリップは動画から切り出し直す（`process_shard.py` が見つけた時刻。48kHz で decode、チャンネル平均、24kHz に soxr で変換）。出力は `data/m2/yodas3/audio`（git 管理外）。

### 2.2 二段目（音声から測る）

切り出した音声で、クリップごとに次を測る（`scripts/m2/score_yodas3.py`、1 クリップ約 0.3 秒、16 プロセスで 30 分程度）。

| 指標 | 中身 | 狙い |
|---|---|---|
| 帯域 | 有声フレームの 8–12kHz と 11–12kHz のレベル（1–4kHz 比） | 再エンコードやノイズ除去で高域が欠けたクリップを除く |
| 高域の倍音 | 有声フレームの 4–6 / 6–8kHz の comb（`hf_structure.py` と同じ測り方、自分の F0 で） | decoder に学ばせたいもの（倍音のある高域）を持つクリップを選ぶ |
| 性別 | F0 の中央値 | Hi-Fi-CAPTAIN と同じく女性を多めに混ぜる |
| クリッピング | 振幅が上限近くのサンプルの割合 | 歪んだクリップを除く |
| UTMOSv2（hp40） | クリップそのものの予測 MOS | 主な指標と同じ物差しで選ぶ。時間がかかる（1 秒に数クリップ）ので、上の指標で絞った 2〜3 万クリップだけ |

残響と BGM は DNSMOS の BAK と UTMOSv2 に任せる。選んだあと 50 クリップ程度を試聴ページにして、外れ（BGM、強い残響、ノイズ除去の跡、合成音声）を目で見て確かめる。

### 2.3 HF の段に使う組み合わせ

Hi-Fi-CAPTAIN の比率（女性 0.45、男性 0.35）に合わせ、女性と男性を分けて `--data` に渡す。候補:

- **Y-all**: 二段目の帯域とクリッピングだけで除いた全体（選別の効果を見る対照）
- **Y-hf**: 高域の倍音と帯域の上位
- **Y-mos**: UTMOSv2 の上位

量は性別ごとに 20〜40h を目安にする（Hi-Fi-CAPTAIN と同程度以上。話者は数千人になる）。

## 3. 実験

### C1: HF の段のデータだけを替える（出発点 keep4k、診断用）

H2ghSG4p100E のレシピ（`run_hf.sh` に `--w-high 15 --w-adv 4 --w-fm 40 --w-spec 100 --ema 0.9997` など、10k step、1 本約 1.3 時間）で、Hi-Fi-CAPTAIN・JVNV・Emilia HQ を YODAS3 の組み合わせに替える。出発点と識別器は keep4k のままなので公開はできないが、データの選び方の比較が速く回せる。p100E（+0.396）との同じ文の差（`mos_pair.py`）で見る。選び方をここで決める。

### C2: CC の土台（元の Mimi から）

keep4k までの道のりを CC のデータでなぞる。

1. 元の Mimi から 60k step（`M2_gan_spec` の設定: mel + MPD / MS-STFT 識別器 + 共有スペクトル loss。帯域は最初から 300Hz〜12kHz、つまり `M2_gan_spec2` の修正込み）。データは s3_cc 0.5 + YODAS3（二段目の帯域で除いた全体）0.5。元の連鎖の s3_full 0.5 + Emilia HQ 0.5 に対応する。
2. 続けて 10k step、4kHz 以上を元の Mimi の出力に保つ（`run_gan_keep.sh` の設定）。keep4k に対応する。

keep4k と比べる: 再構成の Kana-CER（keep4k 4.70%）、録音との類似度（0.779）、UTMOSv2 / UTMOS22。元の連鎖の各段の数値（gan60k 4.43% / 0.827 など）とも並べる。学習は 1 本で 4〜5 時間。

### C3: CC の土台 + HF の段

C2 の decoder と識別器から、C1 で決めたデータで HF の段を回す（p100E のレシピ）。soup12hp と同じように、条件違いの 4 本（h7、p100、p50、h7p100）の 5k / 7.5k / 10k を平均した soup も作る。

比べる相手: p100E、soup12hp。指標: JVS 再構成の UTMOSv2 / UTMOS22（hp40）、Kana-CER、類似度、ViSQOL の 8–12kHz、TTS の出力（公開中の生徒 D1b 120k に加えて、CC 版の生徒 D1-CC があればそれでも）。試聴ページ（録音 / keep4k / soup12hp / CC 版）。

## 4. 判断の目安

- C1: YODAS3 の組み合わせで p100E との差が −0.02 以内なら、データの選別で足りている。
- C3: soup12hp との差が UTMOSv2 で −0.02 以内、UTMOS22 が keep4k −0.02 以上、Kana-CER 4.9% 以下、類似度 0.76 以上。最後はユーザーの試聴で決める。
- C1 で大きく足りない場合: 選別を強める（量を減らして質を上げる）、Emilia-YODAS や合成音声を 4kHz 以下だけに使う、などを検討する。

## 5. 日程の目安

| 段階 | 内容 | 時間 |
|---|---|---|
| データ | 切り出し（実行中）、二段目の指標、UTMOSv2、試聴での確認 | 半日 |
| C1 | 組み合わせ 2〜3 本（並列で 2 本ずつ） | 半日 |
| C2 | 元の Mimi から 70k step と評価 | 半日 |
| C3 | HF の段 4 本 + soup、TTS、試聴ページ | 半日〜1 日 |

## 6. この計画で追加したファイル

| ファイル | 内容 |
|---|---|
| `scripts/m2/extract_yodas3.py` | YODAS3 の選別（`--select`）と 24kHz の wav の切り出し |
| `scripts/m2/score_yodas3.py` | 切り出したクリップごとの帯域、高域の倍音、F0、クリッピング（§2.2） |
