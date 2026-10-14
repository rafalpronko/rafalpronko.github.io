import pytest

from rageval.compare import compare
from rageval.gate import evaluate_gate, validate_policy
from rageval.report import render
from rageval.score import score_case
from rageval.validate import ValidationError

from conftest import DATASET, GOOD_RESPONSES

CASES = DATASET["cases"]
POLICY = {
    "max_critical_failures": 0,
    "max_critical_regressions": 0,
    "max_regressions": 2,
    "max_slice_drop": 0.5,
    "min_slice_size": 2,
}


def scores(responses):
    by_id = {r["case_id"]: r for r in responses}
    return {c["case_id"]: score_case(c, by_id[c["case_id"]]) for c in CASES}


def swapped_responses():
    """Baseline fails q-logs; candidate fixes it but breaks critical q-sso."""
    base = [dict(r) for r in GOOD_RESPONSES]
    base[0]["answer"] = "Plan A keeps logs for a while."
    cand = [dict(r) for r in GOOD_RESPONSES]
    cand[1]["answer"] = "You need the Enterprise plan."
    return base, cand


def test_cancellation_is_visible_per_case():
    base, cand = swapped_responses()
    result = compare(CASES, scores(base), scores(cand))
    overall = result.slices[("all", "all")]
    assert (overall.baseline_pass, overall.candidate_pass) == (2, 2)
    assert [p.case_id for p in result.regressions] == ["q-sso"]
    assert [p.case_id for p in result.fixes] == ["q-logs"]


def test_slices_count_all_cases_in_denominator():
    base, cand = swapped_responses()
    result = compare(CASES, scores(base), scores(cand))
    critical = result.slices[("criticality", "critical")]
    assert (critical.n, critical.baseline_pass, critical.candidate_pass) == (2, 2, 1)


def test_gate_fails_on_critical_regression_despite_unchanged_total():
    base, cand = swapped_responses()
    gate = evaluate_gate(compare(CASES, scores(base), scores(cand)), POLICY)
    assert not gate.passed
    assert gate.failures == [
        "critical failures in candidate: 1 > allowed 0",
        "critical regressions: 1 > allowed 0",
    ]


def test_gate_fails_on_critical_failure_shared_by_both_runs():
    _, cand = swapped_responses()
    same = scores(cand)
    gate = evaluate_gate(compare(CASES, same, same), POLICY)
    assert gate.failures == ["critical failures in candidate: 1 > allowed 0"]


def test_gate_fails_on_slice_drop():
    cand = [dict(r) for r in GOOD_RESPONSES]
    cand[1]["answer"] = "You need the Enterprise plan."
    cand[2].update(action="answer", answer="$10")
    policy = dict(POLICY, max_critical_regressions=5, max_critical_failures=5)
    gate = evaluate_gate(compare(CASES, scores(GOOD_RESPONSES), scores(cand)), policy)
    assert "slice criticality=critical drop 1.00 > allowed 0.50" in gate.failures


def test_gate_fails_on_total_regressions():
    cand = [dict(r) for r in GOOD_RESPONSES]
    cand[0]["answer"] = "Plan A keeps logs for a while."
    cand[1]["answer"] = "You need the Enterprise plan."
    policy = dict(POLICY, max_critical_failures=5, max_critical_regressions=5,
                  max_regressions=1, max_slice_drop=1.0)
    gate = evaluate_gate(compare(CASES, scores(GOOD_RESPONSES), scores(cand)), policy)
    assert gate.failures == ["regressions: 2 > allowed 1"]


def one_noncritical_regression():
    cand = [dict(r) for r in GOOD_RESPONSES]
    cand[0]["answer"] = "Plan A keeps logs for a while."
    return compare(CASES, scores(GOOD_RESPONSES), scores(cand))


def test_slices_smaller_than_min_slice_size_are_not_gated():
    policy = dict(POLICY, max_slice_drop=0.0, min_slice_size=4)
    assert evaluate_gate(one_noncritical_regression(), policy).failures == []


def test_slices_at_min_slice_size_are_gated():
    policy = dict(POLICY, max_slice_drop=0.0, min_slice_size=3)
    assert evaluate_gate(one_noncritical_regression(), policy).failures == [
        "slice overall drop 0.33 > allowed 0.00",
    ]


def test_slice_drop_is_measured_in_pass_rate_points():
    # overall 2/3 -> 1/3: an absolute drop of 0.33, although the rate halves (relative 0.5)
    base, _ = swapped_responses()
    cand = [dict(r) for r in base]
    cand[1]["answer"] = "You need the Enterprise plan."
    policy = dict(POLICY, max_critical_failures=5, max_critical_regressions=5,
                  max_slice_drop=0.4, min_slice_size=3)
    assert evaluate_gate(compare(CASES, scores(base), scores(cand)), policy).passed


def test_gate_passes_without_regressions():
    same = scores(GOOD_RESPONSES)
    assert evaluate_gate(compare(CASES, same, same), POLICY).passed


def test_gate_rejects_empty_evaluation():
    with pytest.raises(ValidationError, match="no cases to evaluate"):
        evaluate_gate(compare([], {}, {}), POLICY)


def test_policy_rejects_unknown_keys():
    with pytest.raises(ValidationError, match="unknown policy keys: max_drop"):
        validate_policy(dict(POLICY, max_drop=0.1))


def test_policy_must_be_an_object():
    with pytest.raises(ValidationError, match="policy: expected an object"):
        validate_policy([POLICY])


COUNT_KEYS = ["max_critical_failures", "max_critical_regressions", "max_regressions"]


@pytest.mark.parametrize("key", COUNT_KEYS)
@pytest.mark.parametrize("value", [True, False, "0", None, -1, 0.5, 2.0, float("nan")])
def test_policy_count_limits_must_be_non_negative_integers(key, value):
    with pytest.raises(ValidationError, match=f"{key} must be an integer >= 0"):
        validate_policy(dict(POLICY, **{key: value}))


@pytest.mark.parametrize("value", [True, "0.5", None, -0.1, 1.5,
                                   float("nan"), float("inf"), float("-inf"),
                                   pytest.param(10**309, id="10**309"),
                                   pytest.param(-(10**309), id="-10**309")])
def test_policy_slice_drop_must_be_finite_between_0_and_1(value):
    with pytest.raises(ValidationError, match=r"max_slice_drop must be a finite number in \[0, 1\]"):
        validate_policy(dict(POLICY, max_slice_drop=value))


@pytest.mark.parametrize("value", [True, 0, -2, 1.5, "3", None])
def test_policy_min_slice_size_must_be_positive_integer(value):
    with pytest.raises(ValidationError, match="min_slice_size must be an integer >= 1"):
        validate_policy(dict(POLICY, min_slice_size=value))


@pytest.mark.parametrize("drop", [0, 1, 0.0, 1.0])
def test_policy_accepts_slice_drop_bounds(drop):
    assert validate_policy(dict(POLICY, max_slice_drop=drop, min_slice_size=1))


def test_report_lists_critical_regression_before_totals_hide_it():
    base, cand = swapped_responses()
    result = compare(CASES, scores(base), scores(cand))
    text = render(result, evaluate_gate(result, POLICY))
    assert "[CRITICAL] q-sso" in text
    assert "GATE: FAIL" in text


def test_report_names_failures_shared_by_both_runs():
    _, cand = swapped_responses()
    same = scores(cand)
    result = compare(CASES, same, same)
    text = render(result, evaluate_gate(result, POLICY))
    assert ("Unchanged failures (both runs fail):\n"
            "  [CRITICAL] q-sso (multi_hop, holdout) candidate: missing facts: owner") in text
