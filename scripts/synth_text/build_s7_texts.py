#!/usr/bin/env python3
"""S7 text (C2 hard mining, stage 3): LLM-written sentences for the kanji readings that the
Wikipedia pool cannot lift.

After S6 about 700 readings with common dictionary words still have fewer than 40 exposures:
kun readings of numerals (七日, 九つ), jukujikun (一昨年, 五月雨), rare kun readings of common
kanji (主 = あるじ, 京 = みやこ). JKYB-mini accuracy on a reading with fewer than 30 exposures
is 32-54%, against 83% from 30 on (C0), so each of them gets example sentences written by an
LLM for a given (word, reading), as the Irodori experiments did (generate_yomi_sentences.py).
The units and words come from the production lexicon (KANJIDIC2 / JMdict); no JKYB key is used.

  tasks    units below --floor after S6 -> up to 3 lexicon words each (frequency tier <= 2),
           split into task files of (word, reading, gloss, n) for the writers
  verify   task files asking which reading each sentence of a homograph word uses
  collect  the writers' sentences -> s7_texts.jsonl. A sentence is kept when the word's stem
           occurs exactly once as its own kanji run, it passes the S1 quality and leakage
           rules, and (homographs) the verifier picked the intended reading. The yomi is the
           Sudachi reading with the word's reading applied; UniDic must agree outside the word.
           The teacher reads the word in kana.

    uv run python scripts/synth_text/build_s7_texts.py tasks
    uv run python scripts/synth_text/build_s7_texts.py verify
    uv run --with fugashi python scripts/synth_text/build_s7_texts.py collect
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_texts import IRODORI, JKYB_FIELDS, REPO, jkyb_error, load_eval_sets  # noqa: E402
from synth_filters import HIRAGANA, KANJI, leak_key, quality_reason, read_jsonl, to_katakana, write_jsonl, yomi_ok  # noqa: E402
from synth_leakage import LeakageIndex  # noqa: E402
from synth_wiki import convention_reading, split_stem, stem_reading  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_jkyb_errors import strip_sound_change  # noqa: E402

SYNTH = REPO / "data/synth"
S7 = SYNTH / "s7"
QC_YIELD = 0.7  # expected share of written sentences that survive collection and the audio QC
ASCII_OR_DIGIT = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
KATAKANA = re.compile(r"[ァ-ヺ]")


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def unit_of(kr: dict[str, Any]) -> str | None:
    if kr["joyo_key"]:
        return kr["joyo_key"]
    return f"F:{kr['kanji']}_{kr['reading']}" if kr["kind"] == "jukujikun" else None


def category_of(row: dict[str, Any]) -> str:
    kinds = {kr["kind"] for kr in row["kanji_readings"]}
    if "jukujikun" in kinds:
        return "joyo_appendix_reading"
    return "kun_yomi" if "kun" in kinds else "on_yomi"


# --------------------------------------------------------------------------- tasks


def cmd_tasks(args: argparse.Namespace) -> None:
    table = json.loads((SYNTH / "work/s6_exposure.json").read_text())["units"]
    added: Counter[str] = Counter()
    for r in read_jsonl(SYNTH / "s6_texts.jsonl"):
        added.update({r["unit"], *[x["unit"] for x in r["extra_targets"]]})
    have = {u: v["train"] + v["target"] + 0.75 * added[u] for u, v in table.items()}

    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    readings_of: dict[str, set[str]] = defaultdict(set)
    entry_seen: set[tuple[str, str]] = set()
    skipped: Counter[str] = Counter()
    for row in read_jsonl(args.irodori / "data/yomi_prod/lexicon.jsonl"):
        readings_of[row["word"]].add(row["reading"])
        # JMdict lists the readings of one entry by commonness: a later one is a variant of the
        # same word (寂しい: さびしい / さみしい) that no context can tell apart, so it is not taught
        variant = (row["word"], row["seq"]) in entry_seen
        entry_seen.add((row["word"], row["seq"]))
        if row["tier"] > args.max_tier or stem_reading(row["word"], to_katakana(row["reading"])) is None:
            continue
        if variant or KATAKANA.search(row["reading"]):
            skipped["variant reading of the same entry" if variant else "katakana reading"] += 1
            continue
        for unit in {u for kr in row["kanji_readings"] if (u := unit_of(kr))}:
            by_unit[unit].append(row)

    words: dict[tuple[str, str], dict[str, Any]] = {}
    units = sorted(u for u, n in have.items() if n < args.floor and by_unit[u])
    for unit in units:
        need = math.ceil((args.floor + 5 - have[unit]) / QC_YIELD)
        picked = sorted(by_unit[unit], key=lambda r: (r["tier"], len(r["word"]), r["word"]))[: args.words_per_unit]
        per_word = min(args.max_per_word, max(args.min_per_word, math.ceil(need / len(picked))))
        for row in picked:
            key = (row["word"], row["reading"])
            item = words.setdefault(key, {
                "word": row["word"], "reading": row["reading"], "gloss": "; ".join((row["gloss"] or [])[:3]),
                "n": 0, "units": [], "category": category_of(row),
                "other_readings": sorted(readings_of[row["word"]] - {row["reading"]}),
            })
            item["n"] = max(item["n"], per_word)
            item["units"].append(unit)
    items = sorted(words.values(), key=lambda w: (w["word"], w["reading"]))
    for i, item in enumerate(items):
        item["id"] = f"w{i:04d}"
    (S7 / "tasks").mkdir(parents=True, exist_ok=True)
    write_jsonl(S7 / "words.jsonl", items)
    for old in (S7 / "tasks").glob("task_*.jsonl"):
        old.unlink()
    chunk: list[dict[str, Any]] = []
    n_tasks = 0
    for item in items + [None]:  # type: ignore[list-item]
        if item is None or (chunk and sum(w["n"] for w in chunk) + item["n"] > args.sentences_per_task):
            write_jsonl(S7 / "tasks" / f"task_{n_tasks:03d}.jsonl",
                        [{k: w[k] for k in ("id", "word", "reading", "gloss", "n")} for w in chunk])
            n_tasks += 1
            chunk = []
        if item is not None:
            chunk.append(item)
    total = sum(w["n"] for w in items)
    print(f"{len(units):,} units below {args.floor} -> {len(items):,} (word, reading) pairs, {total:,} sentences "
          f"in {n_tasks} task files under {S7 / 'tasks'}")
    print(f"homograph words (verified by a second reader): {sum(bool(w['other_readings']) for w in items):,}; "
          f"lexicon rows left out: {dict(skipped)}")


# --------------------------------------------------------------------------- collect


def rivals(item: dict[str, Any]) -> list[str]:
    """Other readings of the written form that are a different word, not a spelling of the same
    sounds (おばあさん / おばーさん) or a sound-changed variant (みっつ / みつ)."""
    own = strip_sound_change(to_katakana(item["reading"]))
    return [r for r in item["other_readings"] if strip_sound_change(to_katakana(r)) != own]


def load_written(words: dict[str, dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """(word item, sentence) pairs from the writers' files; junk lines are skipped."""
    out = []
    for path in sorted((S7 / "out").glob("task_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                item, text = words[row["id"]], row["text"].strip()
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            out.append((item, text))
    return out


def stem_span(item: dict[str, Any], text: str) -> tuple[int, int] | None:
    """The single occurrence of the word's stem as its own kanji run, or None."""
    stem = split_stem(item["word"])[1]
    if text.count(stem) != 1:
        return None
    a = text.index(stem)
    b = a + len(stem)
    lead, _, trail = split_stem(item["word"])
    if lead and text[a - len(lead) : a] != lead:
        return None
    if KANJI.match(text[a - 1 : a]) or KANJI.match(text[b : b + 1]):
        return None
    if trail and not HIRAGANA.match(text[b : b + 1]):  # a conjugating word keeps some okurigana
        return None
    return a, b


def cmd_verify(args: argparse.Namespace) -> None:
    words = {w["id"]: w for w in read_jsonl(S7 / "words.jsonl")}
    rows = []
    seen: set[str] = set()
    for item, text in load_written(words):
        if not rivals(item) or text in seen or stem_span(item, text) is None:
            continue
        seen.add(text)
        choices = sorted([item["reading"], *rivals(item)])
        rows.append({"vid": f"v{len(rows):05d}", "text": text, "word": item["word"], "choices": choices,
                     "intended": item["reading"]})
    (S7 / "verify").mkdir(parents=True, exist_ok=True)
    write_jsonl(S7 / "verify_all.jsonl", rows)
    for old in (S7 / "verify").glob("task_*.jsonl"):
        old.unlink()
    for i in range(0, len(rows), args.rows_per_task):
        write_jsonl(S7 / "verify" / f"task_{i // args.rows_per_task:03d}.jsonl",
                    [{k: r[k] for k in ("vid", "text", "word", "choices")} for r in rows[i : i + args.rows_per_task]])
    print(f"{len(rows):,} sentences of homograph words -> {math.ceil(len(rows) / args.rows_per_task)} verify tasks")


def cmd_collect(args: argparse.Namespace) -> None:
    import synth_wiki

    synth_wiki._init_worker(args.unidic_dir or None)
    words = {w["id"]: w for w in read_jsonl(S7 / "words.jsonl")}
    verdict: dict[str, bool] | None = None
    if (S7 / "verify_all.jsonl").exists():
        intended = {r["vid"]: r for r in read_jsonl(S7 / "verify_all.jsonl")}
        picked: dict[str, str] = {}
        for path in sorted((S7 / "verify_out").glob("task_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    picked[row["vid"]] = str(row["reading"]).strip()
                except (ValueError, KeyError, TypeError):
                    continue
        verdict = {r["text"]: picked.get(vid) == r["intended"] for vid, r in intended.items()}

    used = {leak_key(r["text"]) for name in ("texts.jsonl", "short_texts.jsonl", "s6_texts.jsonl")
            for r in read_jsonl(SYNTH / name)}
    leakage = LeakageIndex()
    leakage.build(load_eval_sets(args.irodori, REPO / "data/jvs_ver1"))
    stats: Counter[str] = Counter()
    per_word: Counter[str] = Counter()
    rows, jkyb_rows = [], []
    for item, text in load_written(words):
        stats["written"] += 1
        span = stem_span(item, text)
        if span is None:
            stats["stem_not_once"] += 1
            continue
        if ASCII_OR_DIGIT.search(text) or quality_reason(text, min_len=8, max_len=60) is not None:
            stats["quality"] += 1
            continue
        if rivals(item):
            if verdict is None or text not in verdict:
                stats["homograph_unverified"] += 1
                continue
            if not verdict[text]:
                stats["homograph_other_reading"] += 1
                continue
        key = leak_key(text)
        if key in used:
            stats["duplicate"] += 1
            continue
        leak = leakage.check(text)
        if leak is not None:
            stats[f"leak_{leak[0]}:{leak[1]}"] += 1
            continue
        a, b = span
        core = stem_reading(item["word"], to_katakana(item["reading"]))[1]  # type: ignore[index]
        parts: list[str] = []
        region: tuple[int, int] | None = None
        ya = 0
        ok = True
        for m in synth_wiki._SUDACHI.tokenize(text, synth_wiki._SPLIT_C):
            s, e = m.begin(), m.end()
            if e <= a or s >= b:
                if m.is_oov():
                    ok = False
                parts.append(convention_reading(m.surface(), m.part_of_speech()[0], m.reading_form()))
                continue
            # the morpheme(s) holding the stem: only kana may surround it inside them
            before, after = text[s:max(s, min(a, e))], text[min(max(b, s), e):e]
            if KANJI.search(before) or KANJI.search(after):
                ok = False
            if region is None:
                region = (s, e)
                parts.append(to_katakana(before))
                ya = sum(len(p) for p in parts)
                parts.append(core)
            else:
                region = (region[0], e)
            parts.append(to_katakana(after))
        yomi = "".join(parts)
        if not ok or region is None or not yomi_ok(yomi):
            stats["yomi_failed"] += 1
            continue
        if not synth_wiki.agree_outside(text, [region]):
            stats["analyzers_disagree"] += 1
            continue
        used.add(key)
        per_word[item["id"]] += 1
        row = {
            "id": f"syn_s7_{len(rows)}", "key": f"syn_s7_{len(rows)}", "text": text, "yomi": yomi,
            "tagged_text": f"{text[:a]}<{text[a:b]}>{text[b:]}",
            "tagged_yomi": f"{yomi[:ya]}<{core}>{yomi[ya + len(core):]}",
            "readings": {"natural": [core], "marginal": []}, "reading_category": item["category"],
            "source": "llm_joyo", "group": "v", "word": item["word"], "units": item["units"], "analyzers_agree": True,
            "teacher_text_hira": text[:a] + to_hiragana(core) + text[b:],
            "teacher_text_kata": text[:a] + core + text[b:],
        }
        rows.append(row)
        jkyb_rows.append({name: row[name] for name in JKYB_FIELDS})
        stats["kept"] += 1
    bad = [(e, r["key"], r["tagged_text"]) for r in jkyb_rows if (e := jkyb_error(r)) is not None]
    assert not bad, bad[:10]
    write_jsonl(SYNTH / "s7_texts.jsonl", rows)
    write_jsonl(SYNTH / "s7_tagged_jkyb.jsonl", jkyb_rows)
    short = sorted((per_word[w["id"]], w["word"], w["reading"], w["n"]) for w in words.values()
                   if per_word[w["id"]] < 0.5 * w["n"])
    print(f"{dict(stats)}")
    print(f"{len(rows):,} sentences for {len(per_word):,} of {len(words):,} words -> {SYNTH / 's7_texts.jsonl'}")
    print(f"words with fewer than half of the sentences asked for: {len(short):,}; e.g. {short[:12]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--irodori", type=Path, default=IRODORI)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--floor", type=int, default=40)
    t.add_argument("--max-tier", type=int, default=2)
    t.add_argument("--words-per-unit", type=int, default=3)
    t.add_argument("--min-per-word", type=int, default=8)
    t.add_argument("--max-per-word", type=int, default=40)
    t.add_argument("--sentences-per-task", type=int, default=600)
    v = sub.add_parser("verify")
    v.add_argument("--rows-per-task", type=int, default=600)
    c = sub.add_parser("collect")
    c.add_argument("--unidic-dir", default=str(IRODORI / ".venv/lib/python3.13/site-packages/unidic/dicdir"))
    args = parser.parse_args()
    {"tasks": cmd_tasks, "verify": cmd_verify, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
