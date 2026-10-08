"""Wikipedia pool for the synthetic-text corpus: analysis, rarity and selection.

Steps (driven by ``build_texts.py``):

1. ``prefilter``: cheap rules on the raw sentence (15-60 chars, ends with 。, mostly
   kanji/kana, <= 3 digits and no multi-digit number, few brackets, no URL).
2. ``analyze_pool``: SudachiPy (mode C) and MeCab + UniDic on every prefiltered
   sentence, in worker processes. A sentence is kept when Sudachi has no OOV morpheme,
   its sentence reading is pure katakana (+ punctuation), and UniDic and Sudachi agree
   on the reading of every kanji span. Spans are cut where both segmentations have a
   boundary, as in Irodori-TTS-Experiments ``scripts/analyze_reading_agreement.py``.
   The reading is the one of Irodori's ``prepare_yomi_rows.py``: Sudachi mode C
   ``reading_form`` with the particles は/へ read as ワ/エ. Results are cached as JSONL.
3. ``count_in_corpus``: occurrences of candidate word stems in the training transcripts
   (Emilia JA), with Aho-Corasick.
4. Rare-word sentences (``RareCandidate``) and general sentences (``GeneralCandidate``)
   are then chosen by ``build_texts.py``.

A candidate word is a Sudachi mode-C morpheme with kanji whose POS is a content word
(not a proper noun or numeral). Its *stem* is the surface without leading/trailing
hiragana (嗜んだ -> 嗜ん -> 嗜, お茶 -> 茶) and is the tagged target span. The word must
be a JMdict entry (via the Irodori production lexicon, KANJIDIC2 based) whose reading
of the same stem equals the analyzers' reading.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_filters import (
    BRACKET_CHARS,
    DIGIT,
    HIRAGANA,
    KANJI,
    japanese_ratio,
    quality_reason,
    read_jsonl,
    to_katakana,
    yomi_ok,
)

CONTENT_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞"}
EXCLUDED_POS1 = {"固有名詞", "数詞"}

# Word record in the analysis cache:
# [begin, end, surface, reading, dictionary_form, pos0, pos1, yomi_offset]
WordRecord = list[Any]


def prefilter(text: str) -> str | None:
    """First failed pool rule for a Wikipedia sentence, or None."""
    if not 15 <= len(text) <= 60:
        return "wiki_length"
    if not text.endswith("。"):
        return "wiki_not_sentence"
    reason = quality_reason(text)
    if reason is not None:
        return reason
    if not KANJI.search(text):
        return "wiki_no_kanji"
    if len(DIGIT.findall(text)) > 3:
        return "wiki_digits"
    if japanese_ratio(text) + len(DIGIT.findall(text)) / len(text) < 0.95:
        return "wiki_not_mostly_japanese"
    if sum(1 for ch in text if ch in BRACKET_CHARS) > 2:
        return "wiki_brackets_heavy"
    if text[0] in "・-*＊→":
        return "wiki_list_item"
    return None


# --------------------------------------------------------------------------- analysis

_SUDACHI: Any = None
_TAGGER: Any = None
_SPLIT_C: Any = None


def _init_worker(unidic_dir: str | None) -> None:
    global _SUDACHI, _TAGGER, _SPLIT_C
    import fugashi
    from sudachipy import Dictionary, SplitMode

    _SUDACHI = Dictionary().tokenizer()
    _SPLIT_C = SplitMode.C
    _TAGGER = fugashi.Tagger(f"-d {unidic_dir} -r {unidic_dir}/mecabrc" if unidic_dir else "")


def _unidic_tokens(text: str) -> list[tuple[int, int, str]] | None:
    tokens: list[tuple[int, int, str]] = []
    position = 0
    for word in _TAGGER(text):
        start = text.find(word.surface, position)
        if start < 0:
            return None
        end = start + len(word.surface)
        kana = getattr(word.feature, "kana", None)
        tokens.append((start, end, kana if kana and kana != "*" else to_katakana(word.surface)))
        position = end
    return tokens


def _kanji_spans_agree(
    text: str, uni: list[tuple[int, int, str]], sud: list[tuple[int, int, str]]
) -> bool:
    common = sorted({e for _, e, _ in uni} & {e for _, e, _ in sud})
    start = 0
    for end in common:
        if KANJI.search(text[start:end]):
            u = "".join(r for s, e, r in uni if start <= s and e <= end)
            d = "".join(r for s, e, r in sud if start <= s and e <= end)
            if u != d:
                return False
        start = end
    return True


def analyze(text: str) -> tuple[str, dict[str, Any] | None]:
    """(status, record). The record holds the sentence reading and its kanji words."""
    sud: list[tuple[int, int, str]] = []
    parts: list[str] = []
    words: list[WordRecord] = []
    offset = 0
    for m in _SUDACHI.tokenize(text, _SPLIT_C):
        if m.is_oov():
            return "oov", None
        surface = m.surface()
        reading = m.reading_form()
        pos = m.part_of_speech()
        sud.append((m.begin(), m.end(), reading))
        if pos[0] == "助詞" and surface in ("は", "へ"):
            reading = {"は": "ワ", "へ": "エ"}[surface]
        if KANJI.search(surface) and pos[0] in CONTENT_POS and pos[1] not in EXCLUDED_POS1:
            words.append(
                [m.begin(), m.end(), surface, reading, m.dictionary_form(), pos[0], pos[1], offset]
            )
        parts.append(reading)
        offset += len(reading)
    yomi = "".join(parts)
    if not yomi_ok(yomi):
        return "bad_yomi", None
    uni = _unidic_tokens(text)
    if uni is None:
        return "unidic_align", None
    if not _kanji_spans_agree(text, uni, sud):
        return "reading_disagree", None
    return "ok", {"yomi": yomi, "w": words}


def _analyze_chunk(chunk: list[tuple[str, str]]) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for source, text in chunk:
        status, record = analyze(text)
        counts[status] += 1
        if record is not None:
            out.append({"t": text, "s": source, **record})
    return counts, out


def _chunks(items: list[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def analyze_pool(
    items: list[tuple[str, str]],
    cache: Path,
    *,
    unidic_dir: str | None,
    workers: int,
    chunk_size: int = 2000,
) -> Counter[str]:
    """Analyze (source, text) pairs into the JSONL ``cache``; returns status counts."""
    counts: Counter[str] = Counter()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    done = 0
    with tmp.open("w", encoding="utf-8") as handle, mp.Pool(
        workers, initializer=_init_worker, initargs=(unidic_dir,)
    ) as pool:
        for chunk_counts, records in pool.imap_unordered(
            _analyze_chunk, _chunks(items, chunk_size)
        ):
            counts.update(chunk_counts)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done += sum(chunk_counts.values())
            if done % 200_000 < chunk_size:
                print(f"  analyzed {done:,}/{len(items):,} {dict(counts)}", flush=True)
    tmp.replace(cache)
    return counts


# --------------------------------------------------------------------------- lexicon


@dataclass(frozen=True)
class LexEntry:
    reading: str  # katakana
    category: str  # on_yomi / kun_yomi / joyo_appendix_reading


def load_lexicon(path: Path) -> dict[str, list[LexEntry]]:
    """JMdict word -> readings, with the category rule of make_aozora_ruby_eval.category."""
    lexicon: dict[str, list[LexEntry]] = {}
    for row in read_jsonl(path):
        kinds = {kr["kind"] for kr in row["kanji_readings"]}
        if kinds & {"jukujikun", "appendix"}:
            category = "joyo_appendix_reading"
        else:
            category = "kun_yomi" if "kun" in kinds else "on_yomi"
        lexicon.setdefault(row["word"], []).append(
            LexEntry(to_katakana(row["reading"]), category)
        )
    return lexicon


def split_stem(surface: str) -> tuple[str, str, str]:
    """(leading hiragana, stem, trailing hiragana) of a word surface."""
    i, j = 0, len(surface)
    while i < j and HIRAGANA.match(surface[i]):
        i += 1
    while j > i and HIRAGANA.match(surface[j - 1]):
        j -= 1
    return surface[:i], surface[i:j], surface[j:]


def stem_reading(surface: str, reading: str) -> tuple[str, str, int] | None:
    """(stem, reading of the stem, offset of that reading inside ``reading``)."""
    lead, stem, trail = split_stem(surface)
    lead_k, trail_k = to_katakana(lead), to_katakana(trail)
    if not reading.startswith(lead_k) or not reading.endswith(trail_k):
        return None
    core = reading[len(lead_k) : len(reading) - len(trail_k)]
    if not stem or not core:
        return None
    return stem, core, len(lead_k)


@dataclass(frozen=True)
class TargetWord:
    stem: str
    stem_start: int  # character offset in the sentence
    reading: str  # katakana reading of the stem
    yomi_start: int  # offset of that reading in the sentence yomi
    lemma: str
    category: str


def target_word(record: WordRecord, lexicon: dict[str, list[LexEntry]]) -> TargetWord | None:
    """The tag candidate for a cached word, if it is a JMdict word read as in JMdict."""
    begin, _end, surface, reading, lemma, pos0, pos1, yomi_offset = record
    if pos0 not in CONTENT_POS or pos1 in EXCLUDED_POS1:
        return None
    parsed = stem_reading(surface, reading)
    if parsed is None:
        return None
    stem, core, lead_len = parsed
    if len(stem) > 10 or not KANJI.search(stem):
        return None
    for word in dict.fromkeys((lemma, surface)):
        if split_stem(word)[1] != stem:
            continue
        for entry in lexicon.get(word, ()):
            parsed_entry = stem_reading(word, entry.reading)
            if parsed_entry is not None and parsed_entry[1] == core:
                return TargetWord(
                    stem=stem,
                    stem_start=begin + lead_len,
                    reading=core,
                    yomi_start=yomi_offset + lead_len,
                    lemma=lemma,
                    category=entry.category,
                )
    return None


# --------------------------------------------------------------------------- rarity


def count_in_corpus(stems: Iterable[str], texts: Iterable[str]) -> Counter[str]:
    """Substring occurrences of each stem in ``texts`` (single characters counted directly)."""
    import ahocorasick

    stems = set(stems)
    singles = {s for s in stems if len(s) == 1}
    automaton = ahocorasick.Automaton()
    for stem in stems - singles:
        automaton.add_word(stem, stem)
    automaton.make_automaton()
    counts: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    for text in texts:
        chars.update(text)
        for _, stem in automaton.iter(text):
            counts[stem] += 1
    for stem in singles:
        counts[stem] = chars[stem]
    return counts
