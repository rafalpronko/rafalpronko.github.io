"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

score(d) = sum over input rankings r of 1 / (c + rank_r(d)), rank one-based.
A document missing from a (truncated) ranking contributes nothing for it.
"""

from collections import defaultdict

from retrieval_lab.ranking import top


def rrf(rankings: list[list[str]], c: int = 60, depth: int = 5) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (c + rank)
    return top(dict(scores), depth)
