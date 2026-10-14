"""The labelled evidence must answer the question as asked, not a neighbouring one."""

from retrieval_lab.corpus import DOCUMENTS, QUERIES
from retrieval_lab.labels import EVIDENCE
from retrieval_lab.text import tokenize


def evidence_text(query_id: str) -> str:
    return " ".join(DOCUMENTS[doc][i] for doc, i in sorted(EVIDENCE[query_id]))


def test_q02_asks_for_the_input_limit_that_its_evidence_states():
    assert QUERIES["q02"] == "How many rows can the search endpoint accept per call?"
    text = evidence_text("q02")
    assert "The search endpoint" in text and "It accepts at most 50 rows per call." in text


def test_q04_asks_for_the_timing_that_its_evidence_states():
    assert QUERIES["q04"] == "How soon can I expect a reimbursement for a duplicate charge?"
    assert "issued within five business days" in evidence_text("q04")


def test_q04_shares_no_token_with_the_corpus_so_only_synonyms_can_find_it():
    corpus_tokens = {t for sentences in DOCUMENTS.values() for s in sentences for t in tokenize(s)}
    assert set(tokenize(QUERIES["q04"])) & corpus_tokens == set()
