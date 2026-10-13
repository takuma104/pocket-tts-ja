"""Do YODAS3 and Emilia-YODAS contain the same videos?

Both come from YouTube, but YODAS3 names a video by a hash and cuts the utterances differently,
so neither ids nor whole transcripts can be compared. This looks for 10-character runs that the
transcripts of a YODAS3 video and of an Emilia-YODAS video share (runs found in more than
three YODAS3 videos are ignored), then for the Emilia-YODAS utterances whose text is mostly
found in the matched video.

    uv run python scripts/yodas3/overlap_emilia_yodas.py

Result (10/13, kept sets): 541 of 11,581 Emilia-YODAS videos match a YODAS3 video, but only
1,105 utterances / 2.6 h of the 304.3 h are found in both: the filters keep different parts.
"""
import json
import re
from collections import Counter, defaultdict

N = 10
strip = re.compile(r"[\s,.?!]")


def grams(text: str) -> set[int]:
    t = strip.sub("", text)
    return {hash(t[i : i + N]) for i in range(len(t) - N + 1)}


yod: dict[str, set[int]] = defaultdict(set)
hours_y: Counter[str] = Counter()
for line in open("data/yodas3_kept/train_punct.jsonl", encoding="utf-8"):
    r = json.loads(line)
    yod[r["group"]] |= grams(r.get("transcript_raw", r["transcript"]))
    hours_y[r["group"]] += r["duration"] / 3600
index: dict[int, list[str]] = defaultdict(list)
for g, s in yod.items():
    for h in s:
        index[h].append(g)
index = {h: v for h, v in index.items() if len(v) <= 3}  # phrases many videos share say nothing
print(len(yod), "YODAS3 videos,", len(index), "rare 10-grams")

emi: dict[str, set[int]] = defaultdict(set)
hours_e: Counter[str] = Counter()
utt_grams: dict[str, tuple[str, set[int], float]] = {}
for line in open("data/emilia_ja/train_rechecked_yodas.jsonl", encoding="utf-8"):
    r = json.loads(line)
    v = re.match(r"JA_(.+)_W\d+$", r["id"]).group(1)
    g = grams(r["transcript"])
    emi[v] |= g
    hours_e[v] += r["duration"] / 3600
    utt_grams[r["id"]] = (v, g, r["duration"])
pairs: dict[str, tuple[str, int, int]] = {}
for v, s in emi.items():
    c: Counter[str] = Counter()
    for h in s:
        for g in index.get(h, ()):
            c[g] += 1
    if c:
        g, n = c.most_common(1)[0]
        pairs[v] = (g, n, len(s))
hist = Counter(min(n, 50) // 5 * 5 for _, n, _ in pairs.values())
print("Emilia-YODAS videos by shared rare 10-grams with their best YODAS3 video:", sorted(hist.items()))
match = {v: g for v, (g, n, m) in pairs.items() if n >= 10}
print(len(match), "of", len(emi), "Emilia-YODAS videos match a YODAS3 video (>= 10 shared);",
      f"{sum(hours_e[v] for v in match):.1f} h of {sum(hours_e.values()):.1f} h on the Emilia side,",
      f"{sum(hours_y[g] for g in set(match.values())):.1f} h of {sum(hours_y.values()):.1f} h on the YODAS3 side")
# utterance level: Emilia utterances whose text is mostly found in the matched YODAS3 video
dup_h = 0.0
dup_n = 0
for uid, (v, g, dur) in utt_grams.items():
    if v in match and g and len(g & yod[match[v]]) / len(g) >= 0.5:
        dup_h += dur / 3600
        dup_n += 1
print(f"Emilia-YODAS utterances with at least half of their 10-grams in the matched YODAS3 video: {dup_n}, {dup_h:.1f} h")
