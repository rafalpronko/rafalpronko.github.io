from dataclasses import dataclass


@dataclass(frozen=True)
class CaseScore:
    """Deterministic checks for one stored response. These are proxies:
    a fact string in the answer and a cited evidence ID do not prove the
    answer is faithful to that evidence."""

    case_id: str
    passed: bool
    reasons: tuple[str, ...]
    fact_coverage: float | None = None
    evidence_recall: float | None = None


def _normalize(text):
    return " ".join(text.lower().split())


def _evidence_recall(evidence_sets, retrieved):
    return max(len(set(es) & retrieved) / len(es) for es in evidence_sets)


def _score_answer(case, response):
    answer = _normalize(response["answer"])
    missing = [f for f in case["required_facts"] if _normalize(f) not in answer]
    cited = set(response["cited_passage_ids"])
    reasons = []
    if missing:
        reasons.append("missing facts: " + ", ".join(missing))
    if not any(set(es) <= cited for es in case["evidence_sets"]):
        reasons.append("citations cover no reference evidence set")
    coverage = 1 - len(missing) / len(case["required_facts"])
    recall = _evidence_recall(case["evidence_sets"], set(response["retrieved_passage_ids"]))
    return CaseScore(case["case_id"], not reasons, tuple(reasons), coverage, recall)


def score_case(case, response):
    expected, actual = case["expected_behavior"], response["action"]
    if expected != actual:
        return CaseScore(case["case_id"], False, (f"expected {expected}, got {actual}",))
    if expected == "answer":
        return _score_answer(case, response)
    return CaseScore(case["case_id"], True, ())
