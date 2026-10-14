"""Per-arm call ledger: every billable attempt is charged before it runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

ROLES = ("teacher", "student", "retry", "judge")


class BudgetExceeded(RuntimeError):
    """Raised before an attempt that would exceed the arm's cap."""


@dataclass
class Ledger:
    """Counts billable attempts by role, plus cache hits (not billable).

    One ledger per arm and per phase. Arms never share a ledger, so one arm
    running out cannot borrow from, or starve, the other.
    """

    name: str
    max_attempts: int
    by_role: Counter[str] = field(default_factory=Counter)
    cache_hits: int = 0

    @property
    def billable(self) -> int:
        return sum(self.by_role.values())

    @property
    def remaining(self) -> int:
        return self.max_attempts - self.billable

    def charge(self, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if self.billable >= self.max_attempts:
            raise BudgetExceeded(
                f"{self.name}: cap of {self.max_attempts} attempts reached"
            )
        self.by_role[role] += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def summary(self) -> dict[str, object]:
        return {
            "ledger": self.name,
            "cap": self.max_attempts,
            "billable_attempts": self.billable,
            "by_role": {role: self.by_role[role] for role in ROLES},
            "cache_hits": self.cache_hits,
        }
