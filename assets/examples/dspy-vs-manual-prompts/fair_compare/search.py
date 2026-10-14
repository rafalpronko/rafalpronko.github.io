"""Search one arm on dev under its own budget and feedback cap."""

from __future__ import annotations

from dataclasses import dataclass, field

from fair_compare.budget import BudgetExceeded, Ledger
from fair_compare.candidates import Candidate
from fair_compare.runner import CacheKey, Outcome, run_case
from fair_compare.splits import Split


class LeakageError(ValueError):
    """A candidate's few-shot demos come from outside the train split."""


def check_demos_from_train(candidate: Candidate, train: Split) -> None:
    train_ids = {case.case_id for case in train.cases}
    outside = sorted(set(candidate.demo_ids) - train_ids)
    if outside:
        raise LeakageError(
            f"{candidate.candidate_id} uses non-train demos: {outside}"
        )


@dataclass(frozen=True)
class DevResult:
    candidate: Candidate
    outcomes: tuple[Outcome, ...]

    @property
    def accuracy(self) -> float:
        return sum(o.correct for o in self.outcomes) / len(self.outcomes)

    def count(self, status: str) -> int:
        return sum(o.status == status for o in self.outcomes)


@dataclass
class ArmLog:
    arm: str
    ledger: Ledger
    results: list[DevResult] = field(default_factory=list)
    stop_reason: str = "all candidates evaluated"


def search_arm(
    arm: str,
    candidates: tuple[Candidate, ...],
    train: Split,
    dev: Split,
    labels: tuple[str, ...],
    ledger: Ledger,
    feedback_cap: int,
    max_attempts: int,
) -> ArmLog:
    """Compile (charged to the ledger), then score each candidate on dev.

    Demos must come from train; dev is used only for scoring and choosing.
    A candidate whose compile or dev run hits the cap is discarded: a partial
    dev score is not comparable and cannot be selected.
    """
    if (train.name, dev.name) != ("train", "dev"):
        raise ValueError(
            f"search sees train+dev only, got {train.name!r}/{dev.name!r}"
        )
    log = ArmLog(arm=arm, ledger=ledger)
    cache: dict[CacheKey, Outcome] = {}
    for candidate in candidates:
        if len(log.results) >= feedback_cap:
            log.stop_reason = (
                f"feedback cap of {feedback_cap} dev evaluations reached; "
                f"{candidate.candidate_id} not evaluated"
            )
            break
        check_demos_from_train(candidate, train)
        try:
            for role, count in candidate.compile_attempts.items():
                for _ in range(count):
                    ledger.charge(role)
            outcomes = tuple(
                run_case(candidate, case, labels, ledger, max_attempts, cache)
                for case in dev.cases
            )
        except BudgetExceeded as exc:
            log.stop_reason = f"{exc}; {candidate.candidate_id} discarded"
            break
        log.results.append(DevResult(candidate, outcomes))
    return log


def select_on_dev(log: ArmLog) -> DevResult:
    """Highest dev accuracy; ties go to the earliest (simplest) candidate."""
    if not log.results:
        raise ValueError(f"{log.arm}: no complete dev evaluation to select")
    best = log.results[0]
    for result in log.results[1:]:
        if result.accuracy > best.accuracy:
            best = result
    return best
