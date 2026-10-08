"""Leakage index: exact and near-duplicate matches against evaluation sentences.

A candidate sentence leaks when, after ``leak_key`` (NFKC, casefold, no punctuation /
spaces / symbols), it

1. ``exact``: equals an evaluation sentence;
2. ``jaccard5``: has a character 5-gram Jaccard similarity >= 0.6 with one;
3. ``jaccard3``: has a character 3-gram Jaccard similarity >= 0.6 with one. The 5-gram
   rule alone misses short near-copies: changing one character in the middle of a
   20-character JKYB sentence removes 5 of its 16 5-grams (J = 0.52), but only 3 of its
   18 3-grams (J = 0.71);
4. ``substring``: shares a substring of >= 15 characters with one.

The rules are checked in this order and a sentence is attributed to the first rule that
fires. The Jaccard rules use prefix filtering (Chaudhuri et al. 2006; PPJoin): grams
are ordered globally by their document frequency in the evaluation set (rarest first),
and two sets with Jaccard >= t must share a gram among their first
``|x| - ceil(t*|x|) + 1`` grams, so only the short posting lists of prefix grams are
probed. Every candidate found this way is verified with the exact Jaccard. Rule 4 is an
exact lookup of all 15-character windows.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from synth_filters import leak_key


def char_ngrams(key: str, n: int) -> frozenset[str]:
    if len(key) < n:
        return frozenset({key}) if key else frozenset()
    return frozenset(key[i : i + n] for i in range(len(key) - n + 1))


@dataclass
class JaccardIndex:
    n: int
    threshold: float
    gram_sets: list[frozenset[str]] = field(default_factory=list)
    df: Counter[str] = field(default_factory=Counter)
    postings: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def prefix(self, grams: frozenset[str]) -> list[str]:
        size = len(grams) - math.ceil(self.threshold * len(grams)) + 1
        return sorted(grams, key=lambda g: (self.df.get(g, 0), g))[:size]

    def build(self, keys: list[str]) -> None:
        self.gram_sets = [char_ngrams(key, self.n) for key in keys]
        for grams in self.gram_sets:
            self.df.update(grams)
        for idx, grams in enumerate(self.gram_sets):
            for gram in self.prefix(grams):
                self.postings[gram].append(idx)

    def match(self, key: str) -> int | None:
        """Index of an indexed set with Jaccard >= threshold, or None."""
        grams = char_ngrams(key, self.n)
        if not grams:
            return None
        candidates: set[int] = set()
        for gram in self.prefix(grams):
            candidates.update(self.postings.get(gram, ()))
        for idx in sorted(candidates):
            other = self.gram_sets[idx]
            inter = len(grams & other)
            if inter / (len(grams) + len(other) - inter) >= self.threshold:
                return idx
        return None


@dataclass
class LeakageIndex:
    jaccard: tuple[tuple[int, float], ...] = ((5, 0.6), (3, 0.6))
    substring_len: int = 15
    exact: dict[str, str] = field(default_factory=dict)
    windows: dict[str, str] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    indexes: list[JaccardIndex] = field(default_factory=list)

    def build(self, sets: dict[str, list[str]]) -> None:
        """``sets``: evaluation-set label -> its sentences."""
        keys = self.keys
        for label, texts in sets.items():
            for text in texts:
                key = leak_key(text)
                if not key:
                    continue
                self.exact.setdefault(key, label)
                for i in range(len(key) - self.substring_len + 1):
                    self.windows.setdefault(key[i : i + self.substring_len], label)
                keys.append(key)
                self.labels.append(label)
        for n, threshold in self.jaccard:
            index = JaccardIndex(n, threshold)
            index.build(keys)
            self.indexes.append(index)

    def check(self, text: str) -> tuple[str, str] | None:
        """(rule, evaluation-set label) of the first matching rule, or None."""
        key = leak_key(text)
        if not key:
            return None
        if key in self.exact:
            return "exact", self.exact[key]
        for index in self.indexes:
            idx = index.match(key)
            if idx is not None:
                return f"jaccard{index.n}", self.labels[idx]
        for i in range(len(key) - self.substring_len + 1):
            label = self.windows.get(key[i : i + self.substring_len])
            if label is not None:
                return "substring", label
        return None

    def explain(self, text: str) -> str | None:
        """The normalized evaluation sentence that ``text`` matched by a Jaccard rule."""
        key = leak_key(text)
        for index in self.indexes:
            idx = index.match(key)
            if idx is not None:
                return self.keys[idx]
        return None
