"""Synthetic workload and the variants under comparison.

The decision being benchmarked: a batch of exact integer membership queries
arrives together with a fresh list of integers. Is it worth building a set
for that batch, or is scanning the list good enough?

The benchmark is a warm, in-memory proxy for that decision: every timed call
reuses the same already-generated case, and the candidate rebuilds its set on
each call. Generating or reading new input and cold caches are not timed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

Variant = Callable[["Case"], list[bool]]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    n_items: int
    n_queries: int
    hit_ratio: float
    loops: int  # batches per timed sample, so tiny cases rise above timer noise


@dataclass(frozen=True)
class Case:
    spec: CaseSpec
    items: list[int]
    queries: list[int]
    prebuilt: frozenset[int]  # only used by the labelled query-only view


CASE_SPECS = (
    CaseSpec("small-items/few-queries", 64, 4, 0.5, 2000),
    CaseSpec("small-items/many-queries", 64, 200, 0.5, 200),
    CaseSpec("large-items/few-queries", 20_000, 4, 0.5, 20),
    CaseSpec("large-items/many-queries", 20_000, 200, 0.5, 1),
)


def make_case(spec: CaseSpec, seed: int) -> Case:
    """Build deterministic items and queries with an exact number of hits.

    Items are drawn with replacement, so duplicates are expected. Misses are
    negative numbers, which can never appear among the non-negative items.
    """
    if not 0.0 <= spec.hit_ratio <= 1.0:
        raise ValueError("hit_ratio must be between 0 and 1")
    rng = random.Random(f"{seed}:{spec.name}")
    items = [rng.randrange(0, 4 * spec.n_items + 1) for _ in range(spec.n_items)]
    n_hits = round(spec.n_queries * spec.hit_ratio) if items else 0
    hits = [rng.choice(items) for _ in range(n_hits)]
    misses = [-1 - rng.randrange(0, 4 * spec.n_items + 1)
              for _ in range(spec.n_queries - n_hits)]
    queries = hits + misses
    rng.shuffle(queries)
    return Case(spec, items, queries, frozenset(items))


def list_scan(case: Case) -> list[bool]:
    """Baseline: ``in`` on the list, a linear scan implemented in C."""
    items = case.items
    return [q in items for q in case.queries]


def set_build_and_probe(case: Case) -> list[bool]:
    """Candidate, end-to-end: building the set is part of the timed work."""
    lookup = set(case.items)
    return [q in lookup for q in case.queries]


def set_probe_only(case: Case) -> list[bool]:
    """Amortized view: the set was built before the timer started."""
    lookup = case.prebuilt
    return [q in lookup for q in case.queries]


BASELINE = "list_scan"
CANDIDATE = "set_build_and_probe"
QUERY_ONLY = "set_probe_only"

VARIANTS: dict[str, Variant] = {
    BASELINE: list_scan,
    CANDIDATE: set_build_and_probe,
    QUERY_ONLY: set_probe_only,
}
