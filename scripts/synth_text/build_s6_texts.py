#!/usr/bin/env python3
"""S6 text (C2 hard mining, stage 2): Wikipedia sentences that lift every kanji reading to a
floor of exposures in the training text.

C0 (B1 step 300k) found the accuracy on a JKYB target to follow how often its reading is shown
in training: 75% when the target word occurs fewer than 10 times, 95% from 100 on, and the same
at the level of the (kanji, reading) pair. S1 chose rare *words* (up to 5 sentences each) and
its QC then dropped the sentences the teacher misread, so rare readings are still thin.

Exposure is counted per unit:
  - a kanji reading of the Irodori production lexicon (KANJIDIC2 based, `joyo_key` such as
    汚_けが or 凸_トツ; sound-changed forms count for their key), and
  - a jukujikun word (F:土産_ミヤゲ).
A unit is shown by a sentence when one of its words is a lexicon (JMdict) entry containing it,
read as in that entry. For the real transcripts the reading is Sudachi's; for the accepted S3
sentences the tagged targets count with their ruby/dictionary reading and the rest with
Sudachi's.

`exposure` counts units in a training mix and in the analyzed Wikipedia pool of S1 (cache).
`select` then picks unused pool sentences round by round (first every unit to 5 exposures,
then 10, 20, ...), capped per word so that a reading is shown in different words, with the S1
quality and leakage rules (JKYB, Aozora and JSUT evaluation rows, JVS). No JKYB key is used:
the units come from the lexicon. Every needy word of a chosen sentence is tagged (JKYB schema,
for the QC).

`teacher_text_hira` / `teacher_text_kata` is what the teacher reads: S5 showed that a kanji
input loses exactly the hard readings to the QC, so a tagged word is written in kana when the
lexicon knows a single reading for its written form. A homograph (碑 = ひ / いしぶみ, 貼付,
女形) stays in kanji: its reading here is only the analyzers' default, and the teacher reading
it the same way is then the third opinion the QC asks for. A single kanji next to another
kanji is not tagged at all (mostly pieces of names the analyzers cut up).

    uv run python scripts/synth_text/build_s6_texts.py exposure --mix data/mix/B1_train_v4.jsonl
    uv run python scripts/synth_text/build_s6_texts.py select --floor 100 --budget 150000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_texts import IRODORI, JKYB_FIELDS, REPO, jkyb_error, load_eval_sets  # noqa: E402
from synth_filters import KANJI, leak_key, read_jsonl, to_katakana, write_jsonl  # noqa: E402
from synth_leakage import LeakageIndex  # noqa: E402
from synth_wiki import CONTENT_POS, EXCLUDED_POS1, prefilter, split_stem, stem_reading  # noqa: E402

SYNTH = REPO / "data/synth"
WORK = SYNTH / "work"
ROUNDS = (5, 10, 20, 40, 70, 100, 150, 200)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


# --------------------------------------------------------------------------- lexicon


@dataclass(frozen=True)
class Entry:
    word: str
    reading: str  # katakana
    category: str  # JKYB reading category
    units: tuple[str, ...]
    homograph: bool  # the lexicon has another reading for this written form


def load_lexicon(path: Path) -> dict[str, list[Entry]]:
    lexicon: dict[str, list[Entry]] = {}
    for row in read_jsonl(path):
        kinds = {kr["kind"] for kr in row["kanji_readings"]}
        if "jukujikun" in kinds:
            category = "joyo_appendix_reading"
        else:
            category = "kun_yomi" if "kun" in kinds else "on_yomi"
        units = []
        for kr in row["kanji_readings"]:
            if kr["joyo_key"]:
                units.append(kr["joyo_key"])
            elif kr["kind"] == "jukujikun":
                units.append(f"F:{kr['kanji']}_{kr['reading']}")
        lexicon.setdefault(row["word"], []).append(
            Entry(row["word"], to_katakana(row["reading"]), category, tuple(dict.fromkeys(units)),
                  bool(row["homograph_readings"]))
        )
    return lexicon


@dataclass(frozen=True)
class Match:
    entry: Entry
    stem: str
    core: str  # katakana reading of the stem
    lead: int  # offset of the stem in the morpheme (surface and reading alike: leading kana)


class Matcher:
    """Lexicon entry of a morpheme (surface, reading, dictionary form), memoized."""

    def __init__(self, lexicon: dict[str, list[Entry]]) -> None:
        self.lexicon = lexicon
        self.memo: dict[tuple[str, str, str], Match | None] = {}

    def __call__(self, surface: str, reading: str, lemma: str) -> Match | None:
        key = (surface, reading, lemma)
        if key not in self.memo:
            self.memo[key] = self._match(surface, reading, lemma)
        return self.memo[key]

    def _match(self, surface: str, reading: str, lemma: str) -> Match | None:
        parsed = stem_reading(surface, reading)
        if parsed is None:
            return None
        stem, core, lead = parsed
        if len(stem) > 10 or not KANJI.search(stem):
            return None
        for word in dict.fromkeys((lemma, surface)):
            if split_stem(word)[1] != stem:
                continue
            for entry in self.lexicon.get(word, ()):
                other = stem_reading(word, entry.reading)
                if other is not None and other[1] == core and entry.units:
                    return Match(entry, stem, core, lead)
        return None


# --------------------------------------------------------------------------- exposure

_SUDACHI: Any = None


def _tokens(text: str) -> list[tuple[int, int, str, str, str]]:
    """(begin, end, surface, reading, dictionary form) of the kanji content words (Sudachi C)."""
    global _SUDACHI
    if _SUDACHI is None:
        from sudachipy import Dictionary, SplitMode

        _SUDACHI = (Dictionary().tokenizer(), SplitMode.C)
    tokenizer, mode = _SUDACHI
    out = []
    for m in tokenizer.tokenize(text, mode):
        pos = m.part_of_speech()
        surface = m.surface()
        if pos[0] in CONTENT_POS and pos[1] not in EXCLUDED_POS1 and KANJI.search(surface):
            out.append((m.begin(), m.end(), surface, m.reading_form(), m.dictionary_form()))
    return out


def _count_chunk(chunk: list[tuple[str, list[tuple[int, int]]]]) -> Counter[tuple[str, str, str]]:
    """Morpheme counts of (text, character ranges to skip) pairs."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for text, skip in chunk:
        for begin, end, surface, reading, lemma in _tokens(text):
            if not any(begin < b and a < end for a, b in skip):
                counts[(surface, reading, lemma)] += 1
    return counts


def count_morphemes(items: list[tuple[str, list[tuple[int, int]]]], workers: int) -> Counter[tuple[str, str, str]]:
    chunks = [items[i : i + 4000] for i in range(0, len(items), 4000)]
    counts: Counter[tuple[str, str, str]] = Counter()
    with mp.Pool(workers) as pool:
        for part in pool.imap_unordered(_count_chunk, chunks):
            counts.update(part)
    return counts


def target_span(tagged_text: str) -> tuple[int, int]:
    start = tagged_text.index("<")
    return start, tagged_text.index(">") - 1


def target_entry(target: dict[str, Any], lexicon: dict[str, list[Entry]]) -> Entry | None:
    """Lexicon entry of a tagged S1 target: (word, reading) of the Irodori row key, or the
    entry of the word whose stem reads like the tagged span."""
    parts = (target.get("orig_key") or "").split("_")
    if len(parts) >= 4:
        word, reading = parts[1], to_katakana(parts[2])
        for entry in lexicon.get(word, ()):
            if entry.reading == reading:
                return entry
    start, end = target_span(target["tagged_text"])
    stem = target["tagged_text"].replace("<", "").replace(">", "")[start:end]
    core = target["tagged_yomi"][target["tagged_yomi"].index("<") + 1 : target["tagged_yomi"].index(">")]
    for entry in lexicon.get(target.get("word") or "", ()):
        other = stem_reading(entry.word, entry.reading)
        if other is not None and other[0] == stem and other[1] == core:
            return entry
    return None


def cmd_exposure(args: argparse.Namespace) -> None:
    t0 = time.time()
    lexicon = load_lexicon(args.irodori / "data/yomi_prod/lexicon.jsonl")
    matcher = Matcher(lexicon)
    s1 = {r["id"]: r for r in read_jsonl(SYNTH / "texts.jsonl")}

    # training text: every distinct row of the mix once; S3 rows through their S1 entry so that
    # the tagged targets can be skipped here and counted with their own reading below
    items: list[tuple[str, list[tuple[int, int]]]] = []
    targets: Counter[str] = Counter()
    seen: set[str] = set()
    rows = Counter()
    for line in args.mix.open(encoding="utf-8"):
        r = json.loads(line)
        if r["source_type"] == "emilia_gain" or r["id"] in seen:
            continue
        seen.add(r["id"])
        t = s1.get(r["id"])
        if t is None or not t.get("tagged_text"):
            items.append((r["transcript"] if t is None else t["text"], []))
            rows["plain"] += 1
            continue
        skip = []
        for x in [t] + list(t.get("extra_targets") or []):
            skip.append(target_span(x["tagged_text"]))
            entry = target_entry(x, lexicon)
            rows["targets"] += 1
            rows["targets_in_lexicon"] += entry is not None
            for unit in entry.units if entry else ():
                targets[unit] += 1
        items.append((t["text"], skip))
        rows["tagged"] += 1
    print(f"mix: {dict(rows)} ({time.time() - t0:.0f}s)", flush=True)
    train: Counter[str] = Counter()
    for (surface, reading, lemma), n in count_morphemes(items, args.workers).items():
        m = matcher(surface, reading, lemma)
        for unit in m.entry.units if m else ():
            train[unit] += n
    print(f"training morphemes counted ({time.time() - t0:.0f}s)", flush=True)

    pool: Counter[str] = Counter()
    pool_words: dict[str, Counter[str]] = {}
    n_pool = 0
    for record in read_jsonl(WORK / "wiki_analyzed.jsonl"):
        n_pool += 1
        units = set()
        for w in record["w"]:
            m = matcher(w[2], w[3], w[4])
            if m is not None:
                for unit in m.entry.units:
                    units.add(unit)
                    pool_words.setdefault(unit, Counter())[m.entry.word] += 1
        pool.update(units)
    units = sorted({u for e in lexicon.values() for x in e for u in x.units})
    table = {u: {"train": train[u], "target": targets[u], "pool": pool[u], "pool_words": len(pool_words.get(u, ()))}
             for u in units}
    out = WORK / "s6_exposure.json"
    out.write_text(json.dumps({"mix": str(args.mix), "units": table}, ensure_ascii=False))
    print(f"{len(units):,} units, {n_pool:,} pool sentences -> {out} ({time.time() - t0:.0f}s)")
    report_exposure(table)


def report_exposure(table: dict[str, dict[str, int]]) -> None:
    bins = [(0, 0), (1, 9), (10, 29), (30, 99), (100, 299), (300, 10**9)]
    for kind, pick in (("kanji readings", lambda u: not u.startswith("F:")), ("jukujikun words", lambda u: u.startswith("F:"))):
        print(f"{kind}: exposures in the training text (Sudachi + tagged targets) -> units, of which the pool has sentences")
        for lo, hi in bins:
            us = [u for u, v in table.items() if pick(u) and lo <= v["train"] + v["target"] <= hi]
            with_pool = sum(table[u]["pool"] > 0 for u in us)
            supply = sum(min(table[u]["pool"], 100) for u in us)
            print(f"  {lo:>4}-{hi if hi < 10**8 else '':<5} {len(us):6,} units, {with_pool:6,} in the pool (pool sentences, capped at 100 per unit: {supply:,})")


# --------------------------------------------------------------------------- selection


@dataclass
class Target:
    unit: str
    word: str  # lexicon word
    stem: str
    stem_start: int
    reading: str
    yomi_start: int
    category: str
    kana: bool  # written in kana for the teacher


@dataclass
class Candidate:
    text: str
    yomi: str
    targets: list[Target]


def cmd_select(args: argparse.Namespace) -> None:
    t0 = time.time()
    rng = random.Random(args.seed)
    lexicon = load_lexicon(args.irodori / "data/yomi_prod/lexicon.jsonl")
    matcher = Matcher(lexicon)
    table = json.loads((WORK / "s6_exposure.json").read_text())["units"]
    have = {u: v["train"] + v["target"] for u, v in table.items()}
    needy = {u for u, n in have.items() if n < args.floor}
    print(f"{len(needy):,} of {len(have):,} units below the floor of {args.floor}", flush=True)

    used = {leak_key(r["text"]) for r in read_jsonl(SYNTH / "texts.jsonl")}
    used |= {leak_key(r["text"]) for r in read_jsonl(SYNTH / "short_texts.jsonl")}
    candidates: list[Candidate] = []
    stats: Counter[str] = Counter()
    for record in read_jsonl(WORK / "wiki_analyzed.jsonl"):
        text = record["t"]
        targets = []
        for w in record["w"]:
            m = matcher(w[2], w[3], w[4])
            if m is None:
                continue
            a = w[0] + m.lead
            b = a + len(m.stem)
            if len(m.stem) == 1 and (KANJI.match(text[a - 1 : a]) or KANJI.match(text[b : b + 1])):
                stats["targets_single_kanji_in_run"] += 1
                continue
            for unit in m.entry.units:
                if unit in needy:
                    targets.append(Target(unit, m.entry.word, m.stem, a, m.core, w[7] + m.lead, m.entry.category,
                                          not m.entry.homograph))
        if not targets:
            continue
        if len(text) > args.max_chars or prefilter(text) is not None:
            stats["quality"] += 1
            continue
        if leak_key(text) in used:
            stats["already_used"] += 1
            continue
        candidates.append(Candidate(text, record["yomi"], targets))
    print(f"{len(candidates):,} candidate sentences; skipped {dict(stats)} ({time.time() - t0:.0f}s)", flush=True)

    leakage = LeakageIndex()
    leakage.build(load_eval_sets(args.irodori, REPO / "data/jvs_ver1"))
    rng.shuffle(candidates)
    added: Counter[str] = Counter()
    per_word: Counter[tuple[str, str]] = Counter()
    chosen: list[tuple[Candidate, list[Target], int]] = []
    done = [False] * len(candidates)
    for cap in [c for c in ROUNDS if c < args.floor] + [args.floor]:
        for i, cand in enumerate(candidates):
            if done[i]:
                continue
            if len(chosen) >= args.budget:
                break
            want = [t for t in cand.targets
                    if have[t.unit] + added[t.unit] < cap and per_word[(t.unit, t.word)] < args.max_per_word]
            if not want:
                continue
            done[i] = True
            leak = leakage.check(cand.text)
            if leak is not None:
                stats[f"leak_{leak[0]}:{leak[1]}"] += 1
                continue
            key = leak_key(cand.text)
            if key in used:
                stats["duplicate"] += 1
                continue
            used.add(key)
            # tag every word of the sentence that shows a unit still below the floor
            tagged = [t for t in cand.targets
                      if have[t.unit] + added[t.unit] < args.floor and per_word[(t.unit, t.word)] < args.max_per_word]
            spans: dict[tuple[int, int], Target] = {}
            for t in sorted(tagged, key=lambda t: have[t.unit] + added[t.unit]):
                spans.setdefault((t.stem_start, t.stem_start + len(t.stem)), t)
            for unit, word in {(t.unit, t.word) for t in tagged}:
                added[unit] += 1
                per_word[(unit, word)] += 1
            chosen.append((cand, list(spans.values()), cap))
        print(f"round to {cap}: {len(chosen):,} sentences ({time.time() - t0:.0f}s)", flush=True)
        if len(chosen) >= args.budget:
            break

    rows, jkyb_rows, extra_rows = [], [], []
    for n, (cand, targets, cap) in enumerate(chosen):
        key = f"syn_s6_{n}"
        hira = kata = cand.text
        for t in sorted(targets, key=lambda t: -t.stem_start):  # right to left keeps offsets valid
            if t.kana:
                a, b = t.stem_start, t.stem_start + len(t.stem)
                hira = hira[:a] + to_hiragana(t.reading) + hira[b:]
                kata = kata[:a] + t.reading + kata[b:]
        tagged = []
        for k, t in enumerate(targets):
            a, b = t.stem_start, t.stem_start + len(t.stem)
            ya, yb = t.yomi_start, t.yomi_start + len(t.reading)
            assert cand.text[a:b] == t.stem and cand.yomi[ya:yb] == t.reading, (cand.text, t)
            tagged.append({
                "key": key if k == 0 else f"{key}__t{k}",
                "tagged_text": f"{cand.text[:a]}<{t.stem}>{cand.text[b:]}",
                "tagged_yomi": f"{cand.yomi[:ya]}<{t.reading}>{cand.yomi[yb:]}",
                "readings": {"natural": [t.reading], "marginal": []},
                "reading_category": t.category, "word": t.word, "unit": t.unit, "kana": t.kana,
                "source": "wiki_joyo",
            })
        row = {"id": key, "text": cand.text, "yomi": cand.yomi, "group": "iv", "analyzers_agree": True,
               "round": cap, "teacher_text_hira": hira, "teacher_text_kata": kata,
               **tagged[0], "extra_targets": tagged[1:]}
        rows.append(row)
        jkyb_rows.append({name: row[name] for name in JKYB_FIELDS})
        for extra in tagged[1:]:
            extra_rows.append({name: {**extra, "text": cand.text, "yomi": cand.yomi}[name] for name in JKYB_FIELDS})
    bad = [(e, r["key"]) for r in jkyb_rows + extra_rows if (e := jkyb_error(r)) is not None]
    assert not bad, bad[:10]
    write_jsonl(SYNTH / "s6_texts.jsonl", rows)
    write_jsonl(SYNTH / "s6_tagged_jkyb.jsonl", jkyb_rows)
    write_jsonl(SYNTH / "s6_tagged_jkyb_extra.jsonl", extra_rows)

    reached = Counter()
    for u in needy:
        total = have[u] + added[u]
        reached["floor" if total >= args.floor else "half" if total >= args.floor / 2 else "ten" if total >= 10 else "under 10"] += 1
    chars = sum(len(r["text"]) for r in rows)
    lines = [
        f"S6 text: {len(rows):,} sentences ({chars / max(len(rows), 1):.1f} chars on average), "
        f"{len(jkyb_rows) + len(extra_rows):,} tagged words, floor {args.floor}, at most {args.max_per_word} sentences per word and unit",
        f"units below the floor before: {len(needy):,}; after (if every sentence passes the QC): at the floor {reached['floor']:,}, "
        f"at half of it {reached['half']:,}, at 10 or more {reached['ten']:,}, under 10 {reached['under 10']:,}",
        f"categories of the tagged words: {dict(Counter(r['reading_category'] for r in jkyb_rows + extra_rows))}",
        f"sentences by round: {dict(sorted(Counter(r['round'] for r in rows).items()))}",
        f"tagged words written in kana for the teacher: {sum(x['kana'] for r in rows for x in [r, *r['extra_targets']]):,}; "
        f"sentences with no kana word: {sum(r['teacher_text_hira'] == r['text'] for r in rows):,}",
        f"rejected while selecting: {dict(stats)}",
    ]
    (SYNTH / "s6_texts_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--irodori", type=Path, default=IRODORI)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exposure")
    e.add_argument("--mix", type=Path, default=REPO / "data/mix/B1_train_v3.jsonl")
    e.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("select")
    s.add_argument("--floor", type=int, default=100)
    s.add_argument("--budget", type=int, default=150_000)
    s.add_argument("--max-per-word", type=int, default=12)
    s.add_argument("--max-chars", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"exposure": cmd_exposure, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
