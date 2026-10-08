#!/usr/bin/env python3
"""Build the text corpus for Irodori-TTS synthesis (plan §4.4, step S1).

    uv run --with fugashi --with pyahocorasick python scripts/synth_text/build_texts.py

(``fugashi`` and ``pyahocorasick`` are pulled in with ``--with`` so that the project's
pyproject/uv.lock stay untouched; MeCab uses the full UniDic 3.1.0 dictionary that
Irodori-TTS-Experiments already has, see ``--unidic-dir``.)

Writes to ``data/synth/``:

- ``texts.jsonl``: about 300k sentences. Each line has ``id`` (= ``key``,
  ``syn_<source>_<n>``), ``text`` (raw, as it should be read), ``yomi`` (katakana
  sentence reading, JKYB convention: は/へ particles read ワ/エ), ``tagged_text`` /
  ``tagged_yomi`` / ``readings`` / ``reading_category`` (JKYB fields, or null),
  ``source``, ``group`` (i/ii/iii), ``word`` (target word or null) and, for Irodori
  rows, ``orig_key``.
- ``texts_tagged_jkyb.jsonl``: the tagged rows only, in the exact JKYB-Parakeet schema
  (``key, text, tagged_text, yomi, tagged_yomi, reading_category, readings, source``),
  usable as ``jkyb-eval tts --dataset`` for the QC of the synthesized audio.
- ``texts_stats.md`` and ``texts_removal.json``: counts, removal statistics.
- ``work/``: the analysis cache of the Wikipedia pool and the jkyb-eval self-check.

Sources:

(i) Irodori-TTS-Experiments production rows (KANJIDIC2-based lexicon, no JKYB keys):
    ``rows_prod`` (Wikipedia), ``rows_aozora`` / ``rows_aozora2`` (Aozora ruby),
    ``rows_var`` (LLM sentences for misread corpus words), ``rows_contrast``
    (Wikipedia contrast compounds). The JKYB-key-derived early stages (pilot, m3, kun)
    are not used.
(ii) Wikipedia sentences (Irodori's local shards: ``general_text.jsonl`` = shard 0,
    ``wiki_shard1-5.jsonl``) containing a JMdict word whose stem occurs fewer than
    ``--rare-max-count`` times in the Emilia JA training transcripts; the rare word is
    tagged. See ``synth_wiki.py`` for the analysis (Sudachi yomi, UniDic agreement).
(iii) Plain or conversational-looking Wikipedia sentences without the rarity rule.

Leakage control: exact / 5-gram Jaccard >= 0.6 / shared 15-char substring against
JKYB-Parakeet, the Irodori Aozora and JSUT evaluation rows and the JVS transcripts
(``synth_leakage.py``); Aozora rows from evaluation works are dropped; texts are
deduplicated across sources (priority i > ii > iii).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pocket_tts_ja.text import normalize  # noqa: E402
from synth_filters import (  # noqa: E402
    KANJI,
    kanji_ratio,
    leak_key,
    quality_reason,
    read_jsonl,
    strip_lone_quote,
    write_jsonl,
    yomi_ok,
)
from synth_leakage import LeakageIndex  # noqa: E402
from synth_wiki import (  # noqa: E402
    TargetWord,
    analyze_pool,
    count_in_corpus,
    load_lexicon,
    prefilter,
    target_word,
)

REPO = Path(__file__).resolve().parents[2]
IRODORI = Path("/home/takuma/co/Irodori-TTS-Experiments")
JKYB_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/datasets--Parakeet-Inc--joyo-kanji-yomi-benchmark-parakeet"
    / "snapshots/*/data/common_kanji_source.jsonl"
)
PRODUCTION_FILES = {  # source name -> Irodori rows file (production stage only)
    "prod": "rows_prod.jsonl",
    "aozora": "rows_aozora.jsonl",
    "aozora2": "rows_aozora2.jsonl",
    "var": "rows_var.jsonl",
    "contrast": "rows_contrast.jsonl",
}
WIKI_FILES = {  # pool source -> file in Irodori data/yomi
    "w0": "general_text.jsonl",
    "w1": "wiki_shard1.jsonl",
    "w2": "wiki_shard2.jsonl",
    "w3": "wiki_shard3.jsonl",
    "w4": "wiki_shard4.jsonl",
    "w5": "wiki_shard5.jsonl",
}
JKYB_FIELDS = (
    "key",
    "text",
    "tagged_text",
    "yomi",
    "tagged_yomi",
    "reading_category",
    "readings",
    "source",
)
READING_CATEGORIES = ("on_yomi", "kun_yomi", "joyo_appendix_reading")


# --------------------------------------------------------------------------- helpers


def _strip_tag(value: str) -> str | None:
    if value.count("<") != 1 or value.count(">") != 1 or value.index("<") > value.index(">"):
        return None
    return value.replace("<", "").replace(">", "")


def jkyb_error(row: dict[str, Any]) -> str | None:
    """Why ``row`` is not a valid JKYB-Parakeet row (mirrors jkyb_eval.models), or None.

    Also requires the tagged reading to be one of the listed readings, so that the row's
    own yomi scores as correct.
    """
    for name in ("key", "text", "tagged_text", "yomi", "tagged_yomi"):
        if not isinstance(row.get(name), str) or not row[name]:
            return f"{name}_empty"
    readings = row.get("readings")
    if not isinstance(readings, dict):
        return "readings_missing"
    natural, marginal = readings.get("natural"), readings.get("marginal")
    for items, required in ((natural, True), (marginal, False)):
        if not isinstance(items, list) or not all(isinstance(x, str) and x for x in items):
            return "readings_type"
        if required and not items:
            return "natural_empty"
        if len(items) != len(set(items)):
            return "readings_duplicate"
    if set(natural) & set(marginal):
        return "readings_overlap"
    if _strip_tag(row["tagged_text"]) != row["text"]:
        return "tagged_text_mismatch"
    if _strip_tag(row["tagged_yomi"]) != row["yomi"]:
        return "tagged_yomi_mismatch"
    if row.get("reading_category") not in READING_CATEGORIES:
        return "reading_category"
    tagged = row["tagged_yomi"]
    if tagged[tagged.index("<") + 1 : tagged.index(">")] not in set(natural) | set(marginal):
        return "tag_not_in_readings"
    if not isinstance(row.get("source", ""), str):
        return "source_type"
    return None


def repair_lone_quote(row: dict[str, Any]) -> dict[str, Any] | None:
    """Drop a lone leading 「/『 or trailing 」/』 from text, tagged_text and both yomi."""
    fixed = strip_lone_quote(row["text"])
    if fixed is None:
        return None
    leading = fixed == row["text"][1:]
    ch = row["text"][0] if leading else row["text"][-1]
    out = dict(row)
    for name in ("text", "tagged_text", "yomi", "tagged_yomi"):
        value = row.get(name)
        if value is None:
            continue
        if leading and value.startswith(ch):
            out[name] = value[1:]
        elif not leading and value.endswith(ch):
            out[name] = value[:-1]
        else:
            return None
    return out


@dataclass
class Removal:
    """Removal counts per (stage, source, reason)."""

    counts: dict[str, dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )

    def add(self, stage: str, source: str, reason: str, n: int = 1) -> None:
        self.counts[stage][source][reason] += n

    def to_json(self) -> dict[str, Any]:
        return {
            stage: {src: dict(c.most_common()) for src, c in by_src.items()}
            for stage, by_src in self.counts.items()
        }


def output_row(
    *,
    source: str,
    group: str,
    text: str,
    yomi: str,
    word: str | None,
    tagged_text: str | None = None,
    tagged_yomi: str | None = None,
    readings: dict[str, list[str]] | None = None,
    reading_category: str | None = None,
    orig_key: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "",
        "key": "",
        "text": text,
        "yomi": yomi,
        "tagged_text": tagged_text,
        "tagged_yomi": tagged_yomi,
        "readings": readings,
        "reading_category": reading_category,
        "source": source,
        "group": group,
        "word": word,
    }
    if orig_key is not None:
        row["orig_key"] = orig_key
    return row


# --------------------------------------------------------------------------- inputs


def load_eval_sets(irodori: Path, jvs_dir: Path) -> dict[str, list[str]]:
    paths = sorted(glob.glob(JKYB_GLOB))
    if not paths:
        raise FileNotFoundError(f"JKYB dataset not found: {JKYB_GLOB}")
    sets = {
        "jkyb": [r["text"] for r in read_jsonl(Path(paths[-1]))],
        "aozora_eval": [r["text"] for r in read_jsonl(irodori / "data/yomi/aozora_eval_rows.jsonl")],
        "jsut_eval": [r["text"] for r in read_jsonl(irodori / "data/yomi/jsut_eval_rows.jsonl")],
    }
    jvs: set[str] = set()
    for path in jvs_dir.glob("jvs*/*/transcripts_utf8.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                jvs.add(line.split(":", 1)[1].strip())
    sets["jvs"] = sorted(jvs)
    return sets


def aozora_work_ids(irodori: Path) -> dict[tuple[str, str, str], set[str]]:
    """(word, reading, text) -> Aozora work ids, from Irodori's sentence files."""
    works: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for name in ("sentences_aozora.jsonl", "sentences_aozora2.jsonl"):
        for row in read_jsonl(irodori / "data/yomi" / name):
            source = row.get("source", "")
            if source.startswith("aozora:"):
                works[(row["word"], row["reading"], row["text"])].add(source.split(":", 1)[1])
    return works


def emilia_texts(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["text", "is_first_copy"])
    table = table.filter(pc.equal(table["is_first_copy"], True))
    return [t for t in table["text"].to_pylist() if t]


# --------------------------------------------------------------------------- (i)


def build_production(
    args: argparse.Namespace,
    leakage: LeakageIndex,
    seen: set[str],
    removal: Removal,
) -> list[dict[str, Any]]:
    eval_works = set(
        json.loads((args.irodori / "data/yomi/aozora_eval_rows.works.json").read_text())["works"]
    )
    works = aozora_work_ids(args.irodori)
    rows: list[dict[str, Any]] = []
    for source, filename in PRODUCTION_FILES.items():
        n_in = 0
        for raw in read_jsonl(args.irodori / "data/yomi" / filename):
            n_in += 1
            if args.drop_irodori_dev and raw.get("split") == "dev":
                removal.add("i", source, "irodori_dev_split")
                continue
            reason = quality_reason(raw["text"])
            if reason == "unbalanced_brackets" and args.repair_lone_quote:
                fixed = repair_lone_quote(raw)
                if fixed is not None and quality_reason(fixed["text"]) is None:
                    removal.add("i_repaired", source, "lone_quote_stripped")
                    raw, reason = fixed, None
            if reason is None and not yomi_ok(raw["yomi"]):
                reason = "bad_yomi"
            if reason is None and not normalize(raw["text"]):
                reason = "normalize_empty"
            if reason is None and source.startswith("aozora"):
                ids = works.get((raw["word"], raw["reading"], raw["text"]))
                if not ids:
                    reason = "aozora_work_unknown"
                elif ids & eval_works:
                    reason = "aozora_eval_work"
            if reason is None:
                leak = leakage.check(raw["text"])
                if leak is not None:
                    reason = f"leak_{leak[0]}:{leak[1]}"
            if reason is None:
                key = leak_key(raw["text"])
                if key in seen:
                    reason = "duplicate_text"
                else:
                    seen.add(key)
            if reason is not None:
                removal.add("i", source, reason)
                continue
            row = output_row(
                source=source,
                group="i",
                text=raw["text"],
                yomi=raw["yomi"],
                word=raw.get("word"),
                tagged_text=raw.get("tagged_text"),
                tagged_yomi=raw.get("tagged_yomi"),
                readings=raw.get("readings"),
                reading_category=raw.get("reading_category"),
                orig_key=raw["key"],
            )
            error = jkyb_error({**row, "key": "x"})
            if error is not None:
                removal.add("i_tags_dropped", source, error)
                for name in ("tagged_text", "tagged_yomi", "readings", "reading_category"):
                    row[name] = None
            rows.append(row)
        removal.add("i_input", source, "rows", n_in)
        print(f"(i) {source}: {n_in:,} rows read, total kept so far {len(rows):,}", flush=True)
    return rows


# --------------------------------------------------------------------------- pool


def build_pool_cache(args: argparse.Namespace, removal: Removal) -> Path:
    cache = args.out_dir / "work" / "wiki_analyzed.jsonl"
    stats_path = cache.with_suffix(".stats.json")
    if cache.exists() and stats_path.exists() and not args.rebuild_cache:
        for stage, by_reason in json.loads(stats_path.read_text()).items():
            for reason, n in by_reason.items():
                removal.add(stage, "wiki_pool", reason, n)
        print(f"pool: reusing {cache}", flush=True)
        return cache
    seen_text: set[str] = set()
    items: list[tuple[str, str]] = []
    pre: Counter[str] = Counter()
    for source, filename in WIKI_FILES.items():
        for row in read_jsonl(args.irodori / "data/yomi" / filename):
            text = row["text"]
            pre["input"] += 1
            if text in seen_text:
                pre["duplicate_in_pool"] += 1
                continue
            seen_text.add(text)
            reason = prefilter(text)
            if reason is not None:
                pre[reason] += 1
                continue
            items.append((source, text))
    del seen_text
    pre["passed"] = len(items)
    print(f"pool prefilter: {dict(pre)}", flush=True)
    t0 = time.time()
    analysis = analyze_pool(
        items, cache, unidic_dir=args.unidic_dir, workers=args.workers
    )
    print(f"pool analysis: {dict(analysis)} ({time.time() - t0:.0f}s)", flush=True)
    stats = {"pool_prefilter": dict(pre), "pool_analysis": dict(analysis)}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    for stage, by_reason in stats.items():
        for reason, n in by_reason.items():
            removal.add(stage, "wiki_pool", reason, n)
    return cache


@dataclass
class RareCandidate:
    text: str
    yomi: str
    targets: list[TargetWord]


def conversational(text: str) -> bool:
    return bool(
        re.search(r"(です|ます|ました|でした|ません|でしょう|ください)。$", text)
        or re.search(r"(よ|ね|わ|ぞ|ぜ|な|か|の|さ|です|ます|だ|ろう|たい)[！？!?]?」", text)
    )


def plain(text: str) -> bool:
    return len(text) <= 45 and kanji_ratio(text) <= 0.35


def scan_pool(
    args: argparse.Namespace,
    cache: Path,
    seen: set[str],
    removal: Removal,
) -> tuple[list[RareCandidate], list[tuple[str, str]], list[tuple[str, str]], dict[str, int]]:
    """Rare-word candidates and the conversational / plain general candidates."""
    lexicon = load_lexicon(args.irodori / "data/yomi_prod/lexicon.jsonl")
    print(f"lexicon: {len(lexicon):,} words", flush=True)
    memo: dict[tuple[str, ...], TargetWord | None] = {}

    def targets_of(record: dict[str, Any]) -> list[TargetWord]:
        out: list[TargetWord] = []
        for w in record["w"]:
            key = (w[2], w[3], w[4], w[5], w[6])
            if key not in memo:
                memo[key] = target_word([0, 0, *w[2:7], 0], lexicon)
            tw = memo[key]
            if tw is not None:
                lead = tw.stem_start  # offset of the stem inside the morpheme
                out.append(
                    TargetWord(
                        stem=tw.stem,
                        stem_start=w[0] + lead,
                        reading=tw.reading,
                        yomi_start=w[7] + lead,
                        lemma=tw.lemma,
                        category=tw.category,
                    )
                )
        return out

    stems: set[str] = set()
    for record in read_jsonl(cache):
        stems.update(t.stem for t in targets_of(record))
    print(f"pool: {len(stems):,} distinct JMdict-verified word stems", flush=True)
    t0 = time.time()
    counts = count_in_corpus(stems, emilia_texts(args.emilia_meta))
    print(f"emilia counts done ({time.time() - t0:.0f}s)", flush=True)
    rare_stems = {s for s in stems if counts[s] < args.rare_max_count}
    print(f"rare stems (< {args.rare_max_count} in Emilia): {len(rare_stems):,}", flush=True)

    rare: list[RareCandidate] = []
    conv: list[tuple[str, str]] = []
    plain_list: list[tuple[str, str]] = []
    pool_counts: Counter[str] = Counter()
    for record in read_jsonl(cache):
        text = record["t"]
        pool_counts["analyzed_ok"] += 1
        if leak_key(text) in seen:
            pool_counts["duplicate_of_i"] += 1
            continue
        targets = [t for t in targets_of(record) if t.stem in rare_stems]
        if targets:
            rare.append(RareCandidate(text, record["yomi"], targets))
        if conversational(text):
            conv.append((text, record["yomi"]))
        elif plain(text):
            plain_list.append((text, record["yomi"]))
    pool_counts["rare_candidates"] = len(rare)
    pool_counts["conversational_candidates"] = len(conv)
    pool_counts["plain_candidates"] = len(plain_list)
    for reason, n in pool_counts.items():
        removal.add("pool_scan", "wiki_pool", reason, n)
    print(f"pool scan: {dict(pool_counts)}", flush=True)
    emilia_count = {s: counts[s] for s in rare_stems}
    return rare, conv, plain_list, emilia_count


def select_rare(
    candidates: list[RareCandidate],
    quota: int,
    max_per_word: int,
    emilia_count: dict[str, int],
    leakage: LeakageIndex,
    seen: set[str],
    removal: Removal,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Greedy: maximize distinct rare words, at most ``max_per_word`` sentences each."""
    rng.shuffle(candidates)
    coverage: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    rejected: set[int] = set()
    for cap in range(1, max_per_word + 1):
        for idx, cand in enumerate(candidates):
            if len(chosen) >= quota:
                break
            if idx in rejected:
                continue
            best = min(
                cand.targets,
                key=lambda t: (coverage[t.stem], emilia_count[t.stem], -len(t.stem)),
            )
            if coverage[best.stem] >= cap:
                continue
            leak = leakage.check(cand.text)
            if leak is not None:
                removal.add("ii", "wiki_rare", f"leak_{leak[0]}:{leak[1]}")
                rejected.add(idx)
                continue
            key = leak_key(cand.text)
            if key in seen:
                removal.add("ii", "wiki_rare", "duplicate_text")
                rejected.add(idx)
                continue
            seen.add(key)
            rejected.add(idx)  # selected: never revisit
            for t in {t.stem for t in cand.targets}:
                coverage[t] += 1
            a, b = best.stem_start, best.stem_start + len(best.stem)
            ya, yb = best.yomi_start, best.yomi_start + len(best.reading)
            assert cand.text[a:b] == best.stem and cand.yomi[ya:yb] == best.reading
            chosen.append(
                output_row(
                    source="wiki_rare",
                    group="ii",
                    text=cand.text,
                    yomi=cand.yomi,
                    word=best.lemma,
                    tagged_text=f"{cand.text[:a]}<{best.stem}>{cand.text[b:]}",
                    tagged_yomi=f"{cand.yomi[:ya]}<{best.reading}>{cand.yomi[yb:]}",
                    readings={"natural": [best.reading], "marginal": []},
                    reading_category=best.category,
                )
            )
        print(f"(ii) after cap {cap}: {len(chosen):,} sentences", flush=True)
        if len(chosen) >= quota:
            break
    return chosen


def select_general(
    conv: list[tuple[str, str]],
    plain_list: list[tuple[str, str]],
    quota: int,
    conv_share: float,
    leakage: LeakageIndex,
    seen: set[str],
    removal: Removal,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rng.shuffle(conv)
    rng.shuffle(plain_list)
    chosen: list[dict[str, Any]] = []

    def take(items: list[tuple[str, str]], limit: int, style: str) -> None:
        for text, yomi in items:
            if len(chosen) >= limit:
                return
            key = leak_key(text)
            if key in seen:
                continue  # selected earlier (ii) or already taken
            leak = leakage.check(text)
            if leak is not None:
                removal.add("iii", "wiki_general", f"leak_{leak[0]}:{leak[1]}")
                continue
            seen.add(key)
            row = output_row(source="wiki_general", group="iii", text=text, yomi=yomi, word=None)
            row["style"] = style
            chosen.append(row)

    take(conv, int(quota * conv_share), "conversational")
    take(plain_list, quota, "plain")
    return chosen


# --------------------------------------------------------------------------- outputs


def joyo_kanji() -> set[str]:
    """Joyo kanji inventory from JKYB on/kun targets (scripts/tokenizer_study.py::joyo_kanji)."""
    chars: set[str] = set()
    for row in read_jsonl(Path(sorted(glob.glob(JKYB_GLOB))[-1])):
        if row["reading_category"] != "joyo_appendix_reading":
            m = re.search(r"<(.+?)>", row["tagged_text"])
            if m and len(m.group(1)) == 1:
                chars.add(m.group(1))
    return chars


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def write_stats(path: Path, rows: list[dict[str, Any]], removal: Removal, args: argparse.Namespace) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    joyo = joyo_kanji()
    lines = [
        "# Synthetic-text corpus (S1) statistics",
        "",
        f"- Built by `scripts/synth_text/build_texts.py` on {time.strftime('%Y-%m-%d')}.",
        f"- Total sentences: **{len(rows):,}**",
        "",
        "## Per source",
        "",
        "| group | source | sentences | tagged | tagged % | distinct target words | mean chars | total chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, items in by_source.items():
        tagged = [r for r in items if r["tagged_text"]]
        words = {r["word"] for r in tagged if r["word"]}
        chars = [len(r["text"]) for r in items]
        lines.append(
            f"| {items[0]['group']} | {source} | {len(items):,} | {len(tagged):,} | "
            f"{len(tagged) / len(items):.1%} | {len(words):,} | {statistics.mean(chars):.1f} | "
            f"{sum(chars):,} |"
        )
    for group in ("i", "ii", "iii"):
        items = [r for r in rows if r["group"] == group]
        tagged = [r for r in items if r["tagged_text"]]
        lines.append(
            f"| **{group}** | all | **{len(items):,}** | {len(tagged):,} | "
            f"{len(tagged) / max(len(items), 1):.1%} | {len({r['word'] for r in tagged}):,} | "
            f"{statistics.mean(len(r['text']) for r in items):.1f} | "
            f"{sum(len(r['text']) for r in items):,} |"
        )
    tagged_all = [r for r in rows if r["tagged_text"]]
    target_pairs = {
        (r["tagged_text"][r["tagged_text"].index("<") + 1 : r["tagged_text"].index(">")],
         r["readings"]["natural"][0])
        for r in tagged_all
    }
    lines += [
        "",
        f"- Tagged rows: {len(tagged_all):,} ({len(tagged_all) / len(rows):.1%}); "
        f"distinct target words (`word`): {len({r['word'] for r in tagged_all}):,}; "
        f"distinct (tagged span, reading) pairs: {len(target_pairs):,}",
        f"- Reading categories of tagged rows: "
        + ", ".join(f"{k} {v:,}" for k, v in Counter(r["reading_category"] for r in tagged_all).most_common()),
    ]
    styles = Counter(r.get("style") for r in rows if r["group"] == "iii")
    lines.append(
        "- (iii) styles: " + ", ".join(f"{k} {v:,}" for k, v in styles.most_common())
    )

    lengths = [len(r["text"]) for r in rows]
    bins = [(8, 15), (16, 25), (26, 35), (36, 45), (46, 60), (61, 80)]
    lines += [
        "",
        "## Length distribution (characters of `text`)",
        "",
        f"- min {min(lengths)}, p10 {percentile(lengths, 0.1)}, median {percentile(lengths, 0.5)}, "
        f"p90 {percentile(lengths, 0.9)}, max {max(lengths)}, mean {statistics.mean(lengths):.1f}",
        "",
        "| chars | " + " | ".join(f"{a}-{b}" for a, b in bins) + " |",
        "|---|" + "---:|" * len(bins),
    ]
    for group in ("i", "ii", "iii"):
        gl = [len(r["text"]) for r in rows if r["group"] == group]
        lines.append(
            f"| {group} | "
            + " | ".join(f"{sum(a <= x <= b for x in gl):,}" for a, b in bins)
            + " |"
        )
    lines.append(
        "| all | " + " | ".join(f"{sum(a <= x <= b for x in lengths):,}" for a, b in bins) + " |"
    )
    kana = sum(len(r["yomi"]) for r in rows)
    lines += [
        "",
        f"- Total yomi length {kana:,} kana; at ~7.5 kana/s that is roughly "
        f"{kana / 7.5 / 3600:,.0f} h of speech (rough estimate).",
    ]

    in_text: set[str] = set()
    in_tags: set[str] = set()
    for r in rows:
        in_text.update(ch for ch in r["text"] if ch in joyo)
        if r["tagged_text"]:
            span = r["tagged_text"][r["tagged_text"].index("<") + 1 : r["tagged_text"].index(">")]
            in_tags.update(ch for ch in span if ch in joyo)
    missing = sorted(joyo - in_text)
    all_kanji = {ch for r in rows for ch in r["text"] if KANJI.match(ch)}
    lines += [
        "",
        "## Joyo kanji coverage",
        "",
        f"- Joyo inventory (single-kanji JKYB on/kun targets, as in `scripts/tokenizer_study.py`): "
        f"{len(joyo):,} kanji",
        f"- Joyo kanji appearing in `text`: {len(in_text):,} ({len(in_text) / len(joyo):.1%}); "
        f"inside a tagged target span: {len(in_tags):,} ({len(in_tags) / len(joyo):.1%})",
        f"- Missing from all texts ({len(missing)}): {''.join(missing) or '-'}",
        f"- Distinct kanji in all texts (Joyo or not): {len(all_kanji):,}",
    ]

    lines += ["", "## Removal statistics", ""]
    lines += [
        "Rules are applied in order and each removed sentence is counted once, under the first",
        "rule that fired: quality (`length`, `rare_chars`, `latin_heavy`, `url`,",
        "`unbalanced_brackets`, `multi_digit_number`, `bad_yomi`, `normalize_empty`), then",
        "`aozora_eval_work`, then leakage (`leak_exact`, `leak_jaccard` = 5-gram Jaccard >= 0.6,",
        "`leak_substring` = shared >= 15 chars; the label after `:` is the evaluation set), then",
        "`duplicate_text` (normalized text already taken by an earlier source).",
        "",
    ]
    data = removal.to_json()
    for stage, by_src in data.items():
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| source | reason | count |")
        lines.append("|---|---|---:|")
        for src, reasons in by_src.items():
            for reason, n in reasons.items():
                lines.append(f"| {src} | {reason} | {n:,} |")
        lines.append("")
    leak_totals: Counter[str] = Counter()
    for stage in ("i", "ii", "iii"):
        for reasons in data.get(stage, {}).values():
            for reason, n in reasons.items():
                if reason.startswith("leak_"):
                    leak_totals[reason] += n
    lines += ["### Leakage removals, all stages", "", "| rule:set | count |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(leak_totals.items())]
    lines += [
        "",
        "## Settings",
        "",
        f"- total {args.total:,}, general {args.general:,}, rare minimum {args.rare_min:,}, "
        f"rare = Emilia count < {args.rare_max_count}, at most {args.max_per_word} sentences per rare word, "
        f"conversational share of (iii) {args.conv_share}, seed {args.seed}, "
        f"repair lone quotes {args.repair_lone_quote}, drop Irodori dev split {args.drop_irodori_dev}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_jkyb_check(args: argparse.Namespace, tagged_path: Path, rows: list[dict[str, Any]]) -> str:
    """Score a sample of the tagged rows' own yomi with ``jkyb-eval g2p`` (expects ~100%)."""
    work = args.out_dir / "work" / "jkyb_g2p_check"
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.jkyb_check_rows, len(rows)))
    write_jsonl(work / "dataset_sample.jsonl", sample)
    write_jsonl(work / "predictions.jsonl", [{"key": r["key"], "yomi": r["yomi"]} for r in sample])
    project = REPO / "third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition"
    outputs = []
    for name, dataset, extra in (
        ("sample", work / "dataset_sample.jsonl", []),
        ("full_dataset_allow_missing", tagged_path, ["--allow-missing"]),
    ):
        cmd = [
            "uv", "run", "--project", str(project), "jkyb-eval", "g2p",
            str(work / "predictions.jsonl"), "--dataset", str(dataset),
            "--output-dir", str(work / name), *extra,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        outputs.append(f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr[-2000:]}")
    report = "\n".join(outputs)
    (work / "report.txt").write_text(report, encoding="utf-8")
    return report


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--irodori", type=Path, default=IRODORI)
    parser.add_argument(
        "--unidic-dir",
        default=str(IRODORI / ".venv/lib/python3.13/site-packages/unidic/dicdir"),
        help="MeCab UniDic directory (empty string: fugashi's default dictionary).",
    )
    parser.add_argument("--emilia-meta", type=Path, default=REPO / "exp/emilia_survey/emilia_ja_meta.parquet")
    parser.add_argument("--jvs-dir", type=Path, default=REPO / "data/jvs_ver1")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/synth")
    parser.add_argument("--total", type=int, default=300_000)
    parser.add_argument("--general", type=int, default=30_000)
    parser.add_argument("--rare-min", type=int, default=50_000)
    parser.add_argument("--rare-max-count", type=int, default=3, help="rare: Emilia count < this")
    parser.add_argument("--max-per-word", type=int, default=3)
    parser.add_argument("--conv-share", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--repair-lone-quote",
        action="store_true",
        help="Keep (i) rows whose only bracket problem is a lone leading 「 / trailing 」, "
        "with that character removed (default: drop them).",
    )
    parser.add_argument("--drop-irodori-dev", action="store_true", help="Skip Irodori dev-split rows.")
    parser.add_argument("--jkyb-check-rows", type=int, default=3000)
    parser.add_argument("--skip-jkyb-check", action="store_true")
    args = parser.parse_args()
    args.unidic_dir = args.unidic_dir or None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    removal = Removal()
    t0 = time.time()

    eval_sets = load_eval_sets(args.irodori, args.jvs_dir)
    leakage = LeakageIndex()
    leakage.build(eval_sets)
    print("eval sets: " + ", ".join(f"{k} {len(v):,}" for k, v in eval_sets.items()), flush=True)

    seen: set[str] = set()
    production = build_production(args, leakage, seen, removal)
    print(f"(i) kept {len(production):,} ({time.time() - t0:.0f}s)", flush=True)

    cache = build_pool_cache(args, removal)
    rare_cands, conv, plain_list, emilia_count = scan_pool(args, cache, seen, removal)
    rare_quota = max(args.rare_min, args.total - len(production) - args.general)
    rare_rows = select_rare(
        rare_cands, rare_quota, args.max_per_word, emilia_count, leakage, seen, removal, rng
    )
    del rare_cands
    general_rows = select_general(
        conv, plain_list, args.general, args.conv_share, leakage, seen, removal, rng
    )
    print(f"(ii) {len(rare_rows):,}  (iii) {len(general_rows):,}", flush=True)

    rows = production + rare_rows + general_rows
    per_source: Counter[str] = Counter()
    for row in rows:
        row["id"] = row["key"] = f"syn_{row['source']}_{per_source[row['source']]}"
        per_source[row["source"]] += 1
        assert normalize(row["text"]), row
        assert yomi_ok(row["yomi"]), row
    tagged = [r for r in rows if r["tagged_text"] is not None]
    errors = Counter(e for r in tagged if (e := jkyb_error(r)) is not None)
    assert not errors, errors
    jkyb_rows = [{name: r[name] for name in JKYB_FIELDS} for r in tagged]

    write_jsonl(args.out_dir / "texts.jsonl", rows)
    tagged_path = args.out_dir / "texts_tagged_jkyb.jsonl"
    write_jsonl(tagged_path, jkyb_rows)
    (args.out_dir / "texts_removal.json").write_text(
        json.dumps(removal.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_stats(args.out_dir / "texts_stats.md", rows, removal, args)
    print(
        f"wrote {len(rows):,} rows ({len(jkyb_rows):,} tagged) to {args.out_dir} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    if not args.skip_jkyb_check:
        print(run_jkyb_check(args, tagged_path, jkyb_rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
