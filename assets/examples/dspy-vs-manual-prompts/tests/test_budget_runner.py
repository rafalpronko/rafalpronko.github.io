from __future__ import annotations

import pytest
from conftest import LABELS, make_candidate

from fair_compare.budget import BudgetExceeded, Ledger
from fair_compare.candidates import FATAL, TRANSIENT
from fair_compare.cases import Case
from fair_compare.runner import CacheKey, Outcome, run_case

CASE = Case("k1", "g1", "charged twice", "billing")


def test_ledger_refuses_the_attempt_beyond_the_cap() -> None:
    ledger = Ledger("arm/opt", 2)
    ledger.charge("teacher")
    ledger.charge("student")
    with pytest.raises(BudgetExceeded):
        ledger.charge("retry")
    assert ledger.billable == 2


def test_arms_have_independent_ledgers() -> None:
    manual, dspy = Ledger("manual/opt", 1), Ledger("dspy/opt", 1)
    manual.charge("student")
    with pytest.raises(BudgetExceeded):
        manual.charge("student")
    dspy.charge("student")
    assert (manual.billable, dspy.billable) == (1, 1)


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        Ledger("x", 5).charge("free-lunch")


def test_transient_failure_is_retried_and_counted() -> None:
    ledger = Ledger("arm", 10)
    cand = make_candidate(overrides={"k1": [TRANSIENT, "billing"]})
    out = run_case(cand, CASE, LABELS, ledger, max_attempts=2)
    assert (out.status, out.correct, out.attempts) == ("ok", True, 2)
    assert ledger.by_role["student"] == 1 and ledger.by_role["retry"] == 1


def test_exhausted_retries_become_an_error_not_a_skip() -> None:
    ledger = Ledger("arm", 10)
    cand = make_candidate(overrides={"k1": [TRANSIENT, TRANSIENT]})
    out = run_case(cand, CASE, LABELS, ledger, max_attempts=2)
    assert (out.status, out.correct, out.attempts) == ("error", False, 2)
    assert ledger.billable == 2


def test_fatal_failure_is_not_retried() -> None:
    ledger = Ledger("arm", 10)
    cand = make_candidate(overrides={"k1": [FATAL, "billing"]})
    out = run_case(cand, CASE, LABELS, ledger, max_attempts=2)
    assert (out.status, out.attempts, ledger.billable) == ("error", 1, 1)


def test_invalid_output_is_wrong_and_not_retried() -> None:
    ledger = Ledger("arm", 10)
    cand = make_candidate(overrides={"k1": ["Billing."]})
    out = run_case(cand, CASE, LABELS, ledger, max_attempts=2)
    assert (out.status, out.prediction, out.correct) == ("invalid", None,
                                                         False)
    assert ledger.billable == 1


def test_cache_hit_is_recorded_but_not_billed() -> None:
    ledger = Ledger("arm", 10)
    cache: dict[CacheKey, Outcome] = {}
    cand = make_candidate()
    run_case(cand, CASE, LABELS, ledger, 2, cache)
    hit = run_case(cand, CASE, LABELS, ledger, 2, cache)
    assert hit.cache_hit and hit.attempts == 0
    assert (ledger.billable, ledger.cache_hits) == (1, 1)


def test_errors_are_never_cached() -> None:
    ledger = Ledger("arm", 10)
    cache: dict[CacheKey, Outcome] = {}
    cand = make_candidate(overrides={"k1": [FATAL]})
    run_case(cand, CASE, LABELS, ledger, 2, cache)
    run_case(cand, CASE, LABELS, ledger, 2, cache)
    assert (ledger.billable, ledger.cache_hits, cache) == (2, 0, {})


def test_budget_stops_a_case_mid_retry() -> None:
    ledger = Ledger("arm", 1)
    cand = make_candidate(overrides={"k1": [TRANSIENT, "billing"]})
    with pytest.raises(BudgetExceeded):
        run_case(cand, CASE, LABELS, ledger, max_attempts=2)
