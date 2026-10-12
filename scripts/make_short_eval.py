"""Build data/eval/short_ja.jsonl: very short texts (1-8 tokens), for the EOS behaviour.

The stock inference packs sentences into chunks of at most 50 tokens and generates every chunk
on its own, so the last sentence of a paragraph, or the last clause of a sentence longer than 50
tokens, is often generated alone ("認知されている."). Such texts are far shorter than anything in
the training data (0.04% of the rows have 5 tokens or fewer), and a chunk that never reaches
EOS runs on to the length budget.

Two kinds of rows (key, text, kana, kind):
  tail    the last clause of a JSUT eval sentence (after its last "、"), up to --per-length per
          token count
  phrase  everyday short utterances and single words (the list below)
Readings come from pyopenjtalk (good enough for a Kana-CER on texts this short).

    uv run python scripts/make_short_eval.py
"""

import argparse
import json
import random
from pathlib import Path

import pyopenjtalk
import sentencepiece as spm

from pocket_tts_ja.text import normalize

PHRASES = """
はい。 いいえ。 うん。 そう。 ええ。 どうぞ。 どうも。 なるほど。 もちろん。 たぶん。
本当？ 大丈夫？ なぜ？ いつ？ どこ？ 誰？ 何？ そうですか？ いいですか？ 分かりましたか？
ありがとう。 ありがとうございます。 すみません。 ごめんなさい。 お願いします。 こんにちは。
こんばんは。 おはようございます。 おやすみなさい。 さようなら。 お疲れ様でした。 いただきます。
ごちそうさまでした。 了解です。 分かりました。 そうですね。 違います。 知りません。 大丈夫です。
少々お待ちください。 よろしくお願いします。 お待たせしました。 失礼します。 いらっしゃいませ。
東京。 大阪。 北海道。 日本。 富士山。 りんご。 自転車。 図書館。 天気予報。 新幹線。
三時です。 明日です。 月曜日です。 千円です。 二階です。 雨です。 晴れました。 終わりました。
始めます。 行きましょう。 食べました。 見てください。 やめて！ すごい！ 危ない！ 助けて！
""".split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data/eval/jsut_eval_1k.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/ja_4k.model"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--per-length", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("data/eval/short_ja.jsonl"))
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    def tokens(text: str) -> int:
        return len(sp.encode(normalize(text)))

    by_length: dict[int, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    for line in open(args.source):
        row = json.loads(line)
        if "、" not in row["text"]:
            continue
        tail = row["text"].rsplit("、", 1)[1].strip()
        n = tokens(tail)
        if tail in seen or not 1 <= n <= args.max_tokens:
            continue
        seen.add(tail)
        by_length.setdefault(n, []).append((row["key"], tail))
    rng = random.Random(0)
    rows = []
    for n in sorted(by_length):
        for key, tail in sorted(rng.sample(by_length[n], min(args.per_length, len(by_length[n])))):
            rows.append({"key": f"tail_{key}", "text": tail, "kind": "tail"})
    for i, phrase in enumerate(PHRASES):
        assert tokens(phrase) <= args.max_tokens + 2, (phrase, tokens(phrase))
        rows.append({"key": f"phrase_{i:03d}", "text": phrase, "kind": "phrase"})
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            row["kana"] = pyopenjtalk.g2p(row["text"], kana=True)
            row["tokens"] = tokens(row["text"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {n: sum(r["tokens"] == n for r in rows) for n in sorted({r["tokens"] for r in rows})}
    print(f"{len(rows)} rows -> {args.out}; by token count: {counts}")


if __name__ == "__main__":
    main()
