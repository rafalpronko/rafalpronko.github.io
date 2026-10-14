from retrieval_lab.chunking import Chunk
from retrieval_lab.rerank import rerank

MAP = {"refund": "REFUND", "reimbursement": "REFUND"}


def chunk(cid, text):
    return Chunk(cid, cid.split("#")[0], (0,), text)


def test_exact_identifier_beats_neighbour_identifier():
    cands = [chunk("a#s0", "Error ERR-4012 means expired"), chunk("b#s0", "Error ERR-4013 means scope")]
    assert [cid for cid, _ in rerank("error ERR-4013", cands, MAP)] == ["b#s0", "a#s0"]


def test_coverage_counts_concept_matches():
    cands = [chunk("a#s0", "Invoices monthly"), chunk("b#s0", "Refund in five days")]
    assert rerank("reimbursement days", cands, MAP)[0][0] == "b#s0"


def test_only_reorders_given_candidates_and_is_stable_on_ties():
    cands = [chunk("z#s0", "same words"), chunk("a#s0", "same words")]
    assert [cid for cid, _ in rerank("same words", cands, MAP)] == ["z#s0", "a#s0"]
