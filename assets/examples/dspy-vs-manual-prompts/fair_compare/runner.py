"""Run one candidate on one case under the shared retry and output policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fair_compare.budget import Ledger
from fair_compare.candidates import FATAL, TRANSIENT, Candidate
from fair_compare.cases import Case, parse_label

Status = Literal["ok", "invalid", "error"]
CacheKey = tuple[str, str]


@dataclass(frozen=True)
class Outcome:
    case_id: str
    group: str
    status: Status
    prediction: str | None
    correct: bool
    attempts: int
    cache_hit: bool = False


def run_case(
    candidate: Candidate,
    case: Case,
    labels: tuple[str, ...],
    ledger: Ledger,
    max_attempts: int,
    cache: dict[CacheKey, Outcome] | None = None,
) -> Outcome:
    """Charge every attempt; retry only transient failures.

    Invalid outputs are not retried: they are scored as wrong and reported
    separately. Only completed responses are cached, never errors.
    """
    key = (candidate.artifact_sha256, case.case_id)
    if cache is not None and key in cache:
        ledger.record_cache_hit()
        hit = cache[key]
        return Outcome(
            hit.case_id, hit.group, hit.status, hit.prediction,
            hit.correct, attempts=0, cache_hit=True,
        )
    for attempt in range(max_attempts):
        ledger.charge("student" if attempt == 0 else "retry")
        raw = candidate.output(case, attempt)
        if raw == TRANSIENT:
            continue
        if raw == FATAL:
            return Outcome(case.case_id, case.group, "error", None, False,
                           attempt + 1)
        label = parse_label(raw, labels)
        status: Status = "ok" if label is not None else "invalid"
        outcome = Outcome(case.case_id, case.group, status, label,
                          label == case.label, attempt + 1)
        if cache is not None:
            cache[key] = outcome
        return outcome
    return Outcome(case.case_id, case.group, "error", None, False,
                   max_attempts)
