from retrieval_lab.chunking import Chunk
from retrieval_lab.metrics import query_metrics


def chunk(doc, sids):
    return Chunk(f"{doc}#{sids}", doc, tuple(sids), "")


EVIDENCE = frozenset({("d", 0), ("d", 1)})


def test_complete_support_needs_all_evidence_sentences():
    m = query_metrics([chunk("d", [1]), chunk("x", [0]), chunk("d", [0])], EVIDENCE, k=3)
    assert m["evidence_recall"] == 1.0 and m["full_support"] is True
    assert m["reciprocal_rank"] == 1 / 3
    assert m["self_contained"] is False and m["parent_hit"] is True


def test_parent_hit_without_evidence_is_not_support():
    m = query_metrics([chunk("d", [3]), chunk("x", [0])], EVIDENCE, k=3)
    assert m["parent_hit"] is True and m["evidence_recall"] == 0.0
    assert m["reciprocal_rank"] == 0.0


def test_duplicated_evidence_in_overlapping_chunks_counts_once():
    m = query_metrics([chunk("d", [0, 1, 2]), chunk("d", [1, 2, 3])], EVIDENCE, k=3)
    assert m["evidence_recall"] == 1.0 and m["self_contained"] is True
    assert m["redundant_evidence_chunks"] == 1
    assert m["context_sentences"] == 4


def test_only_top_k_is_scored():
    m = query_metrics([chunk("x", [0]), chunk("d", [0, 1])], EVIDENCE, k=1)
    assert m["full_support"] is False and m["context_sentences"] == 1
