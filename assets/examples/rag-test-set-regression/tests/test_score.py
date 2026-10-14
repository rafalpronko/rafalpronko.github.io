from rageval.score import score_case

from conftest import DATASET, GOOD_RESPONSES

CASES = {c["case_id"]: c for c in DATASET["cases"]}
RESPONSES = {r["case_id"]: r for r in GOOD_RESPONSES}


def test_complete_multi_evidence_answer_passes():
    result = score_case(CASES["q-sso"], RESPONSES["q-sso"])
    assert result.passed
    assert result.fact_coverage == 1.0
    assert result.evidence_recall == 1.0


def test_partial_evidence_set_does_not_count_as_support():
    response = dict(RESPONSES["q-sso"], cited_passage_ids=["P-b"])
    result = score_case(CASES["q-sso"], response)
    assert not result.passed
    assert "citations cover no reference evidence set" in result.reasons


def test_missing_fact_is_reported_by_name():
    response = dict(RESPONSES["q-sso"], answer="You need the Enterprise plan.")
    result = score_case(CASES["q-sso"], response)
    assert not result.passed
    assert result.fact_coverage == 0.5
    assert "missing facts: owner" in result.reasons


def test_any_alternative_evidence_set_is_enough():
    case = dict(CASES["q-logs"], evidence_sets=[["P-b", "P-c"], ["P-a"]])
    result = score_case(case, RESPONSES["q-logs"])
    assert result.passed


def test_abstention_on_unanswerable_passes():
    assert score_case(CASES["q-price"], RESPONSES["q-price"]).passed


def test_answer_on_unanswerable_fails():
    response = dict(RESPONSES["q-price"], action="answer", answer="$10")
    result = score_case(CASES["q-price"], response)
    assert not result.passed
    assert result.reasons == ("expected abstain, got answer",)


def test_abstention_on_answerable_fails():
    response = dict(RESPONSES["q-logs"], action="abstain", answer="")
    result = score_case(CASES["q-logs"], response)
    assert not result.passed
    assert result.reasons == ("expected answer, got abstain",)
