from retrieval_lab.chunking import sentence_chunks
from retrieval_lab.concepts import ConceptIndex, concepts_of

MAP = {"refund": "REFUND", "reimbursement": "REFUND", "error": "ERROR"}


def test_synonyms_share_a_concept():
    assert concepts_of(["reimbursement"], MAP) == concepts_of(["refund"], MAP)


def test_finds_passage_without_lexical_overlap():
    idx = ConceptIndex(sentence_chunks({"a": ["Refunds take days"], "b": ["Invoices"]}), MAP)
    assert [cid for cid, _ in idx.search("reimbursement", depth=5)] == ["a#s0"]


def test_identifiers_are_invisible_so_ties_fall_back_to_chunk_id():
    docs = {"e": ["Error ERR-1 means expired", "Error ERR-2 means scope"]}
    idx = ConceptIndex(sentence_chunks(docs), MAP)
    ranked = idx.search("error ERR-2", depth=5)
    assert [cid for cid, _ in ranked] == ["e#s0", "e#s1"]
    assert ranked[0][1] == ranked[1][1]


def test_unknown_words_give_no_results():
    idx = ConceptIndex(sentence_chunks({"a": ["Refunds"]}), MAP)
    assert idx.search("dormant purge", depth=5) == []
