import copy
import json
from pathlib import Path

import pytest

from rag_eval import isolate, load, parse, score

HERE = Path(__file__).resolve().parent.parent
LABELS = HERE / "labels.json"


@pytest.fixture(scope="module")
def results():
    data = load(LABELS)
    return {case.id: score(case, data.k) for case in data.cases}


def raw_labels():
    return json.loads(LABELS.read_text(encoding="utf-8"))


def test_happy_path_scores_every_layer_at_full(results):
    r = results["ok-refund"]
    assert r.recall == 1.0
    assert r.precision == pytest.approx(1 / 3)
    assert (r.sufficient_topk, r.sufficient_context) == (True, True)
    assert (r.correctness, r.completeness, r.relevance) == (1.0, 1.0, 1.0)
    assert (r.faithfulness, r.citation_coverage, r.citation_entailment) == (1.0, 1.0, 1.0)
    assert r.tags == ("ok",)


def test_correct_answer_can_be_unsupported(results):
    r = results["correct-unsupported"]
    assert r.correctness == 1.0 and r.completeness == 1.0
    assert r.faithfulness == 0.0
    assert r.citation_coverage == 0.0
    assert r.citation_entailment is None  # no citations: N/A, not 0 or 1
    assert r.tags == ("retrieval_miss", "unsupported_claim", "uncited_claim")


def test_truncation_is_not_a_retrieval_miss(results):
    r = results["context-dropped"]
    assert r.recall == 1.0
    assert (r.sufficient_topk, r.sufficient_context) == (True, False)
    assert r.completeness == 0.5
    assert r.faithfulness == 1.0
    assert r.tags == ("context_dropped", "incomplete")


def test_gold_context_isolates_the_context_pipeline(results):
    gold = results["context-dropped@gold"]
    assert gold.recall is None and gold.precision is None  # retrieval bypassed
    assert gold.tags == ("ok",)
    verdict = isolate(results["context-dropped"], gold)
    assert verdict.startswith("gold context fixes it")
    assert "suspect context assembly" in verdict
    assert "not proof" in verdict


def test_stale_source_gives_supported_but_wrong_answer(results):
    r = results["stale-but-supported"]
    assert r.faithfulness == 1.0 and r.citation_entailment == 1.0
    assert r.correctness == 0.0 and r.completeness == 0.0
    assert r.tags == ("retrieval_miss", "supported_but_wrong", "incomplete")


def test_precision_style_metrics_reward_a_short_answer(results):
    r = results["short-but-precise"]
    assert (r.correctness, r.relevance, r.faithfulness) == (1.0, 1.0, 1.0)
    assert r.completeness == 0.5
    assert r.tags == ("incomplete",)


def test_any_sufficient_evidence_set_counts(results):
    r = results["two-evidence-paths"]
    assert r.recall == pytest.approx(2 / 3)  # labelled recall is partial...
    assert r.sufficient_topk is True  # ...but the evidence is sufficient
    assert r.tags == ("ok",)


def test_citation_entailment_differs_from_coverage(results):
    r = results["bad-citations"]
    assert r.citation_coverage == 1.0
    assert r.citation_entailment == 0.5  # 2 of 4 (claim, citation) pairs
    assert r.invalid_citations == 1  # doc-999 was never in the context
    assert r.relevance == pytest.approx(2 / 3)
    assert r.faithfulness == 1.0
    assert r.tags == ("bad_citation", "off_topic_claim")


def test_unanswerable_with_abstention_is_na_not_perfect(results):
    r = results["unanswerable-abstains"]
    assert (r.recall, r.precision, r.sufficient_topk) == (None, None, None)
    assert (r.correctness, r.completeness, r.faithfulness) == (None, None, None)
    assert r.abstention == "correct"
    assert r.tags == ("ok",)


def test_unanswerable_with_a_guess_is_flagged(results):
    r = results["unanswerable-guesses"]
    assert r.abstention == "should_abstain"
    assert r.faithfulness == 0.0 and r.citation_entailment == 0.0
    assert r.tags == ("should_abstain", "unsupported_claim", "bad_citation")


def test_refusal_does_not_become_perfect_faithfulness(results):
    r = results["answerable-refuses"]
    assert r.faithfulness is None
    assert r.completeness == 0.0
    assert r.abstention == "missed_answer"
    assert r.tags == ("missed_answer",)


def test_answerable_refusal_without_evidence_is_no_evidence():
    def no_refund_page(r):
        chunks = ["refund-exceptions", "shipping-eu", "store-hours"]
        case(r, "answerable-refuses").update(retrieved=chunks, provided=chunks)
    data = parse(mutate(no_refund_page))
    r = score(next(c for c in data.cases if c.id == "answerable-refuses"), data.k)
    assert r.sufficient_context is False
    assert r.abstention == "no_evidence"
    assert r.tags == ("no_evidence", "retrieval_miss")


def test_stale_chunk_text_carries_no_marker():
    # The article says nothing in the context reveals the page is outdated.
    assert "superseded" not in raw_labels()["corpus"]["refund-2023"]


def mutate(fn):
    raw = raw_labels()
    fn(raw)
    return raw


def case(raw, case_id):
    return next(c for c in raw["cases"] if c["id"] == case_id)


def add_gold(raw, base_id, provided, claims):
    raw["cases"].append({"id": f"{base_id}@gold", "gold_of": base_id,
                         "provided": provided, "abstained": False,
                         "claims": claims})


def claim(claim_id, chunk):
    return {"id": claim_id, "text": claim_id, "correct": True,
            "relevant": True, "supported_by": [chunk], "cited": [chunk]}


def isolate_gold(raw, base_id):
    data = parse(raw)
    results = {c.id: score(c, data.k) for c in data.cases}
    return isolate(results[base_id], results[f"{base_id}@gold"])


def assert_inconclusive(verdict):
    assert verdict.startswith("inconclusive")
    for cause in ("composition", "order", "generation", "variability"):
        assert cause in verdict
    assert "suspect" not in verdict
    assert "not to retrieval" not in verdict


def test_gold_pass_with_identical_sufficient_context_is_inconclusive():
    # Same context as the actual run, a complete answer this time. The
    # harness does not compare the two contexts, so the reading stays as
    # conservative as for a different context.
    raw = raw_labels()
    add_gold(raw, "short-but-precise",
             ["warranty-page", "shipping-fee", "store-hours"],
             [claim("warranty-2y", "warranty-page"),
              claim("claim-via-form", "warranty-page")])
    assert_inconclusive(isolate_gold(raw, "short-but-precise"))


def test_gold_pass_with_different_sufficient_context_keeps_composition():
    # Both contexts hold a full evidence set, but the gold one drops the
    # distractors: composition and order may explain the pass.
    raw = raw_labels()
    add_gold(raw, "short-but-precise", ["warranty-page"],
             [claim("warranty-2y", "warranty-page"),
              claim("claim-via-form", "warranty-page")])
    assert_inconclusive(isolate_gold(raw, "short-but-precise"))


def test_gold_pass_via_another_evidence_path_keeps_composition():
    # Actual run used the combined page; the gold run used the two FAQ
    # entries, another sufficient evidence set in a different order.
    raw = raw_labels()
    add_gold(raw, "short-but-precise", ["warranty-faq-b", "warranty-faq-a"],
             [claim("warranty-2y", "warranty-faq-a"),
              claim("claim-via-form", "warranty-faq-b")])
    assert_inconclusive(isolate_gold(raw, "short-but-precise"))


def test_gold_pass_after_retrieval_miss_suggests_retrieval():
    raw = raw_labels()
    add_gold(raw, "correct-unsupported", ["shipping-eu"],
             [claim("ship-3-5d", "shipping-eu")])
    verdict = isolate_gold(raw, "correct-unsupported")
    assert "suspect retrieval" in verdict
    assert "not proof" in verdict


def test_gold_failure_does_not_rule_out_upstream_failures():
    raw = raw_labels()
    add_gold(raw, "short-but-precise", ["warranty-page"],
             [claim("warranty-2y", "warranty-page")])
    verdict = isolate_gold(raw, "short-but-precise")
    assert verdict.startswith("gold context does not fix it")
    assert "upstream" in verdict


def test_retrieved_list_may_be_longer_than_k():
    raw = mutate(lambda r: case(r, "ok-refund")["retrieved"].append("careers"))
    data = parse(raw)
    r = score(next(c for c in data.cases if c.id == "ok-refund"), data.k)
    assert r.recall == 1.0 and r.tags == ("ok",)


@pytest.mark.parametrize("case_id", ["ok-refund", "unanswerable-abstains"])
def test_ordinary_case_without_retrieved_is_rejected(case_id):
    # Only a gold case may bypass retrieval; a missing log is a label error.
    with pytest.raises(ValueError, match="needs a 'retrieved' list"):
        parse(mutate(lambda r: case(r, case_id).pop("retrieved")))


@pytest.mark.parametrize("value", [None, "refund-2024", {"refund-2024": 1}, 3])
@pytest.mark.parametrize("case_id", ["ok-refund", "unanswerable-abstains"])
def test_ordinary_case_retrieved_must_be_a_json_list(case_id, value):
    with pytest.raises(ValueError, match="'retrieved' must be a JSON list"):
        parse(mutate(lambda r: case(r, case_id).update(retrieved=value)))


def test_empty_retrieved_list_means_retrieval_returned_nothing():
    def nothing_found(r):
        case(r, "answerable-refuses").update(retrieved=[], provided=[])
    data = parse(mutate(nothing_found))
    r = score(next(c for c in data.cases if c.id == "answerable-refuses"), data.k)
    assert (r.recall, r.precision) == (0.0, 0.0)  # measured, not n/a
    assert (r.sufficient_topk, r.sufficient_context) == (False, False)
    assert r.abstention == "no_evidence"
    assert r.tags == ("no_evidence", "retrieval_miss")


def test_empty_retrieved_list_on_unanswerable_case_is_valid():
    def nothing_found(r):
        case(r, "unanswerable-abstains").update(retrieved=[], provided=[])
    data = parse(mutate(nothing_found))
    r = score(next(c for c in data.cases if c.id == "unanswerable-abstains"),
              data.k)
    assert r.abstention == "correct" and r.tags == ("ok",)


def test_gold_case_still_bypasses_retrieval():
    data = parse(raw_labels())
    gold = next(c for c in data.cases if c.id == "context-dropped@gold")
    assert gold.retrieved is None


def set_field(case_id, field, value, claim_index=None):
    def change(r):
        target = case(r, case_id)
        if claim_index is not None:
            target = target["claims"][claim_index]
        target[field] = value
    return change


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
@pytest.mark.parametrize(
    "case_id, field, claim_index",
    [("ok-refund", "correct", 0), ("ok-refund", "relevant", 0),
     ("ok-refund", "abstained", None), ("ok-refund", "answerable", None)],
)
def test_non_boolean_flags_are_rejected(case_id, field, claim_index, value):
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        parse(mutate(set_field(case_id, field, value, claim_index)))


@pytest.mark.parametrize(
    "field, value",
    [("question", "Something else?"), ("relevant", ["refund-2024"]),
     ("evidence_sets", [["refund-exceptions"]]),
     ("required_claims", ["no-gift-card-returns"]), ("answerable", True)],
)
def test_gold_case_cannot_set_reference_fields(field, value):
    with pytest.raises(ValueError, match="must not set"):
        parse(mutate(set_field("context-dropped@gold", field, value)))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: case(r, "ok-refund")["retrieved"].append("refund-2024"),
         "duplicate retrieved ID"),
        (lambda r: r["cases"].append(copy.deepcopy(case(r, "ok-refund"))),
         "duplicate case ID"),
        (lambda r: case(r, "bad-citations")["claims"].append(
            copy.deepcopy(case(r, "bad-citations")["claims"][0])),
         "duplicate claim ID"),
        (lambda r: case(r, "ok-refund")["claims"][0]["cited"].append("refund-2024"),
         "duplicate cited ID"),
        (lambda r: case(r, "ok-refund")["relevant"].append("ghost"),
         "unknown chunk ID"),
        (lambda r: case(r, "ok-refund")["claims"][0]["supported_by"].append("careers"),
         "not in provided context"),
        (lambda r: case(r, "unanswerable-abstains")["relevant"].append("careers"),
         "unanswerable"),
        (lambda r: case(r, "answerable-refuses")["claims"].append(
            copy.deepcopy(case(r, "ok-refund")["claims"][0])),
         "abstained"),
        (lambda r: case(r, "stale-but-supported")["claims"][0].update(id="return-14d"),
         "required claim"),
        (lambda r: r.update(k=0), "k must be"),
        (lambda r: case(r, "context-dropped@gold").update(gold_of="missing"),
         "gold_of"),
        (lambda r: case(r, "context-dropped@gold").update(
            gold_of="unanswerable-abstains"),
         "answerable, non-gold"),
        (lambda r: case(r, "context-dropped@gold").update(
            retrieved=["refund-exceptions", "refund-2024"]),
         "retrieved list"),
        (lambda r: case(r, "context-dropped@gold").update(
            provided=["refund-exceptions"],
            claims=case(r, "context-dropped@gold")["claims"][:1]),
         "full evidence set"),
        (lambda r: case(r, "context-dropped@gold").update(provided=[], claims=[]),
         "full evidence set"),
        (lambda r: case(r, "ok-refund").update(answerable=False),
         "answerable flag"),
        (lambda r: case(r, "unanswerable-abstains").update(answerable=True),
         "answerable flag"),
        (lambda r: (case(r, "ok-refund")["retrieved"].append("careers"),
                    case(r, "ok-refund")["provided"].append("careers")),
         "top k"),
    ],
)
def test_invalid_labels_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        parse(mutate(change))


def test_hallucinated_citation_id_is_a_metric_not_a_label_error():
    raw = mutate(lambda r: case(r, "ok-refund")["claims"][0]["cited"].append("doc-404"))
    data = parse(raw)
    r = score(next(c for c in data.cases if c.id == "ok-refund"), data.k)
    assert r.invalid_citations == 1
    assert r.citation_entailment == 0.5
