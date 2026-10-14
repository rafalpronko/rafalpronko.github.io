"""Retrieval-only evaluation: recall@k, MRR@k and unanswerable queries.

This measures whether the right chunks are retrieved. It says nothing about
evidence sufficiency or answer quality, which need their own evaluations.

Each case is one search call, so the corpus must not change during a run;
nothing here pins one snapshot across calls.
"""

from .search import search


def evaluate(conn, cases: list[dict], k: int = 3) -> dict:
    """Score cases {"query": str, "relevant": [[doc_id, chunk_no], ...]}.

    recall_at_k: macro recall, i.e. per answerable query the share of its
    unique relevant chunks found in the top k, averaged over those queries.
    mrr: MRR@k, mean of 1/rank of the first relevant hit (0 if none in k).
    misses: answerable queries with no relevant chunk in the top k.
    """
    answerable = [case for case in cases if case["relevant"]]
    unanswerable = [case for case in cases if not case["relevant"]]
    recalls, reciprocal_ranks, misses = [], [], []
    for case in answerable:
        recall, rank = _score_case(conn, case, k)
        recalls.append(recall)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is None:
            misses.append({"query": case["query"],
                           "expected": case["relevant"]})
    noisy = [case["query"] for case in unanswerable
             if search(conn, case["query"], k)["hits"]]
    return {
        "k": k,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall_at_k": _ratio(sum(recalls), len(recalls)),
        "mrr": _ratio(sum(reciprocal_ranks), len(reciprocal_ranks)),
        "unanswerable_with_hits": len(noisy),
        "misses": misses,
    }


def _score_case(conn, case: dict, k: int) -> tuple[float, int | None]:
    """Return (recall, rank of the first relevant hit or None) for a case."""
    relevant = {tuple(pair) for pair in case["relevant"]}
    hits = search(conn, case["query"], k)["hits"]
    retrieved = [(hit["doc_id"], hit["chunk_no"]) for hit in hits]
    found = relevant.intersection(retrieved)
    rank = next((position for position, key in enumerate(retrieved, start=1)
                 if key in relevant), None)
    return len(found) / len(relevant), rank


def _ratio(numerator, denominator) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
