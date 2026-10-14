"""Strictly accepted local repair: re-pack a small window of bins into fewer.

Neighbourhood: one target bin plus 1..window-1 partner bins. Their items
are pooled and an exact depth-first search tries to fit the pool into as
many bins as there are partners. Success removes (at least) one bin.

Order: targets from least to most loaded (ties: lower index first);
partner sets in itertools.combinations order, smaller sets first.
Acceptance: only a move that removes a bin, so the count never grows.
Budget: one unit of work = one window examined (even if its total size
rules it out at once) or one tentative placement in the search. The
budget is shared by the whole call, so the run is finite and the same
input always spends the same work.
Termination: the lower bound is reached, a full scan finds no improving
move (a local optimum for THIS neighbourhood, not a proof of optimality)
or the budget runs out.
Rollback: searches work on copies; a failed or interrupted search
leaves the last accepted packing untouched.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from packing.model import (
    Bins, Item, PackingError, check_packing, load, lower_bound,
    positive_int, validate_instance,
)


@dataclass(frozen=True)
class RepairResult:
    bins: Bins
    removed_bins: int
    windows: int
    placements: int
    stop_reason: str  # "lower_bound" | "no_improving_move" | "budget"

    @property
    def work(self) -> int:
        return self.windows + self.placements


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, max_work):
        self.max_work = max_work
        self.windows = 0
        self.placements = 0

    def spend(self, kind):
        if self.windows + self.placements >= self.max_work:
            raise BudgetExhausted
        setattr(self, kind, getattr(self, kind) + 1)


def repair(capacity: int, bins: Iterable[Iterable[Item]], *,
           max_work: int, window: int = 3) -> RepairResult:
    """Repair a valid packing; `bins` may be any iterable, read once."""
    positive_int(max_work, "max_work")
    if positive_int(window, "window") < 2:
        raise ValueError("window must be at least 2 bins")
    current = tuple(tuple(bin_) for bin_ in bins)
    original_count = len(current)
    items = validate_instance(capacity, [it for b in current for it in b])
    check_packing(capacity, items, current)
    target = lower_bound(capacity, items)
    budget = Budget(max_work)
    while True:
        if len(current) <= target:
            reason = "lower_bound"
            break
        try:
            improved = improve_once(capacity, current, window, budget)
        except BudgetExhausted:
            reason = "budget"
            break
        if improved is None:
            reason = "no_improving_move"
            break
        check_packing(capacity, items, improved)
        if len(improved) >= len(current):
            raise PackingError("accepted move did not remove a bin")
        current = improved
    return RepairResult(current, original_count - len(current),
                        budget.windows, budget.placements, reason)


def improve_once(capacity, bins, window, budget):
    """First improving move in the documented order, or None."""
    loads = [load(b) for b in bins]
    targets = sorted(range(len(bins)), key=lambda i: (loads[i], i))
    for t in targets:
        others = [i for i in range(len(bins)) if i != t]
        for k in range(1, window):
            for partners in combinations(others, k):
                budget.spend("windows")
                if loads[t] + sum(loads[p] for p in partners) > k * capacity:
                    continue
                pool = bins[t] + tuple(it for p in partners for it in bins[p])
                packed = pack_exact(capacity, pool, k, budget)
                if packed is not None:
                    touched = {t, *partners}
                    kept = tuple(b for i, b in enumerate(bins)
                                 if i not in touched)
                    return kept + packed
    return None


def pack_exact(capacity, pool, k, budget):
    """Exhaustive search: put the pool into k bins, or return None."""
    order = sorted(pool, key=lambda it: -it.size)
    rooms = [capacity] * k
    groups = [[] for _ in range(k)]

    def place(i):
        if i == len(order):
            return True
        item, tried = order[i], set()
        for b in range(k):
            room = rooms[b]
            if room < item.size or room in tried:
                continue
            tried.add(room)  # bins with equal room are interchangeable
            budget.spend("placements")
            rooms[b] -= item.size
            groups[b].append(item)
            if place(i + 1):
                return True
            rooms[b] += item.size
            groups[b].pop()
        return False

    if not place(0):
        return None
    return tuple(tuple(g) for g in groups if g)
