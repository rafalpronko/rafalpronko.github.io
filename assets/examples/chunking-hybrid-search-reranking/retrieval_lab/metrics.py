"""Evidence-level metrics. Labels enter here and nowhere in retrieval code.

Evidence is a set of (doc_id, sentence_id) pairs, so the same label works for any
chunking. Sentences are de-duplicated: two overlapping chunks carrying the same
evidence sentence count once.
"""

from retrieval_lab.chunking import Chunk

Evidence = frozenset[tuple[str, int]]


def sentences(chunk: Chunk) -> set[tuple[str, int]]:
    return {(chunk.doc_id, s) for s in chunk.sentence_ids}


def query_metrics(ranked: list[Chunk], evidence: Evidence, k: int) -> dict:
    seen: set[tuple[str, int]] = set()
    reciprocal_rank, redundant = 0.0, 0
    for rank, chunk in enumerate(ranked[:k], start=1):
        carried = sentences(chunk) & evidence
        if carried and carried <= seen:
            redundant += 1
        seen |= sentences(chunk)
        if not reciprocal_rank and evidence <= seen:
            reciprocal_rank = 1 / rank
    gold_docs = {doc for doc, _ in evidence}
    return {
        "evidence_recall": len(evidence & seen) / len(evidence),
        "full_support": evidence <= seen,
        "reciprocal_rank": reciprocal_rank,
        "self_contained": any(evidence <= sentences(c) for c in ranked[:k]),
        "parent_hit": any(c.doc_id in gold_docs for c in ranked[:k]),
        "context_sentences": len(seen),
        "redundant_evidence_chunks": redundant,
    }
