from docsys.evaluate import evaluate
from docsys.search import search
from docsys.store import delete_document, upsert_document


def load(conn, corpus):
    for doc in corpus:
        upsert_document(conn, doc["doc_id"], doc["version"], doc["title"], doc["body"])


def test_search_returns_ranked_chunks_with_corpus_version(conn, corpus):
    load(conn, corpus)
    result = search(conn, "restore snapshot", limit=3)
    assert result["corpus_version"] == 4
    top = result["hits"][0]
    assert (top["doc_id"], top["chunk_no"], top["version"]) == ("runbook-backup", 1, 1)
    scores = [hit["score"] for hit in result["hits"]]
    assert scores == sorted(scores)


def test_ties_are_broken_deterministically(conn):
    upsert_document(conn, "b", 1, "T", "same words here")
    upsert_document(conn, "a", 1, "T", "same words here")
    hits = search(conn, "same words", limit=5)["hits"]
    assert [hit["doc_id"] for hit in hits] == ["a", "b"]


def test_deleted_document_is_not_returned(conn, corpus):
    load(conn, corpus)
    delete_document(conn, "runbook-backup", 2)
    hits = search(conn, "snapshot", limit=5)["hits"]
    assert all(hit["doc_id"] != "runbook-backup" for hit in hits)


def test_unanswerable_query_returns_no_hits(conn, corpus):
    load(conn, corpus)
    assert search(conn, "kubernetes autoscaling", limit=5)["hits"] == []


def test_operator_words_are_searched_as_plain_terms(conn):
    upsert_document(conn, "d", 1, "T", "this or that")
    hits = search(conn, "OR", limit=5)["hits"]
    assert [hit["doc_id"] for hit in hits] == ["d"]


def test_evaluation_reports_recall_and_unanswerable_handling(conn, corpus, queries):
    load(conn, corpus)
    report = evaluate(conn, queries, k=3)
    assert report["answerable"] == 5
    assert report["unanswerable"] == 1
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["unanswerable_with_hits"] == 0
    assert [row["query"] for row in report["misses"]] == []


def test_recall_is_macro_averaged_over_unique_relevant_chunks(conn):
    upsert_document(conn, "d", 1, "T", "alpha one\n\nalpha two\n\nbeta")
    cases = [
        # two relevant chunks, k=1 retrieves one of them: recall 0.5
        {"query": "alpha", "relevant": [["d", 0], ["d", 1], ["d", 0]]},
        # one relevant chunk, retrieved at rank 1: recall 1.0
        {"query": "beta", "relevant": [["d", 2]]},
    ]
    single = evaluate(conn, cases[:1], k=1)
    assert single["recall_at_k"] == 0.5
    assert single["mrr"] == 1.0
    assert single["misses"] == []
    assert evaluate(conn, cases, k=1)["recall_at_k"] == 0.75
    assert evaluate(conn, cases, k=3)["recall_at_k"] == 1.0
