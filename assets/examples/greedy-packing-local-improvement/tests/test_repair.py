import pytest

from packing.ffd import first_fit_decreasing
from packing.model import check_packing, lower_bound
from packing.repair import repair
from tests.helpers import items_of, sizes_of

BUDGET = 10_000


def ffd_then_repair(capacity, items, **kwargs):
    bins = first_fit_decreasing(capacity, items)
    return bins, repair(capacity, bins, **kwargs)


def test_empty_packing_stops_immediately():
    result = repair(10, (), max_work=BUDGET)
    assert result.bins == ()
    assert result.work == 0
    assert result.stop_reason == "lower_bound"


def test_saturated_packing_is_already_at_lower_bound():
    items = items_of(10, 10)
    bins, result = ffd_then_repair(10, items, max_work=BUDGET)
    assert result.bins == bins
    assert result.work == 0
    assert result.stop_reason == "lower_bound"


def test_repacks_three_bins_into_two_on_instructive_fixture():
    # Manufactured: FFD gives [5,4] [4,3,2] [2]; {5,3,2} {4,4,2} is optimal.
    items = items_of(5, 4, 4, 3, 2, 2)
    bins, result = ffd_then_repair(10, items, max_work=BUDGET)
    assert len(bins) == 3
    assert len(result.bins) == 2 == lower_bound(10, items)
    assert result.removed_bins == 1
    assert result.stop_reason == "lower_bound"
    check_packing(10, items, result.bins)


def test_failed_search_rolls_back_to_the_input_packing():
    # Lower bound says 2, but no two 6s share a bin: optimum is 3.
    items = items_of(6, 6, 6)
    bins, result = ffd_then_repair(10, items, max_work=BUDGET)
    assert result.bins == bins
    assert result.removed_bins == 0
    # 3 targets x (2 pairs + 1 triple) windows, 2 placements per triple.
    assert (result.windows, result.placements) == (9, 6)
    assert result.work == 15
    assert result.stop_reason == "no_improving_move"


def test_budget_bounds_window_scanning_even_without_placements():
    items = items_of(6, 6, 6)
    bins, result = ffd_then_repair(10, items, max_work=2)
    assert result.bins == bins
    assert (result.windows, result.placements) == (2, 0)
    assert result.stop_reason == "budget"


def test_budget_exhaustion_returns_last_accepted_packing():
    items = items_of(5, 4, 4, 3, 2, 2)
    bins, result = ffd_then_repair(10, items, max_work=1)
    assert result.bins == bins
    assert result.work == 1
    assert result.stop_reason == "budget"


def test_budget_exhausted_mid_search_rolls_back_partial_placements():
    # 3 windows (two ruled out by load), then 2 placements into [2,5,4,4,3,2].
    items = items_of(5, 4, 4, 3, 2, 2)
    bins = first_fit_decreasing(10, items)
    snapshot = bins
    result = repair(10, bins, max_work=5)
    assert bins == snapshot
    assert result.bins == bins
    assert result.removed_bins == 0
    assert (result.windows, result.placements) == (3, 2)
    assert result.work == 5
    assert result.stop_reason == "budget"


def test_budget_exhaustion_keeps_an_accepted_improvement():
    # First window {2} + {6} fits one bin: 1 window + 2 placements = 3.
    items = items_of(6, 6, 6, 2, 2)
    bins = tuple((item,) for item in items)
    snapshot = bins
    result = repair(10, bins, max_work=3)
    assert bins == snapshot
    assert len(result.bins) == 4
    assert result.removed_bins == 1
    assert (result.windows, result.placements) == (1, 2)
    assert result.work == 3
    assert result.stop_reason == "budget"
    check_packing(10, items, result.bins)


def test_plateau_local_optimum_is_not_global_optimum():
    # Selected by find_plateau.py (seed 1, trial 22725); illustrative only.
    items = items_of(13, 12, 9, 9, 5, 4, 3, 3, 1)
    bins, result = ffd_then_repair(20, items, max_work=BUDGET)
    assert sizes_of(bins) == [[13, 5, 1], [12, 4, 3], [9, 9], [3]]
    assert result.bins == bins
    assert result.stop_reason == "no_improving_move"
    by_size = {}
    for item in items:
        by_size.setdefault(item.size, []).append(item)
    pick = lambda s: by_size[s].pop()
    certificate = ((pick(13), pick(4), pick(3)),
                   (pick(12), pick(5), pick(3)),
                   (pick(9), pick(9), pick(1)))
    check_packing(20, items, certificate)
    assert len(certificate) == len(result.bins) - 1 == lower_bound(20, items)


def test_is_deterministic():
    items = items_of(7, 3, 6, 4, 5, 5, 2, 8, 1, 9, 2, 3)
    first = ffd_then_repair(10, items, max_work=BUDGET)[1]
    second = ffd_then_repair(10, items, max_work=BUDGET)[1]
    assert first == second


def test_never_worse_and_input_bins_unchanged():
    items = items_of(7, 3, 6, 4, 5, 5, 2, 8, 1, 9, 2, 3)
    bins = first_fit_decreasing(10, items)
    snapshot = bins
    result = repair(10, bins, max_work=BUDGET)
    assert bins == snapshot
    assert len(result.bins) <= len(bins)
    check_packing(10, items, result.bins)


def test_one_shot_iterable_counts_removed_bins_from_snapshot():
    items = items_of(5, 4, 4, 3, 2, 2)
    bins = first_fit_decreasing(10, items)
    result = repair(10, (b for b in bins), max_work=BUDGET)
    assert len(result.bins) == len(bins) - 1
    assert result.removed_bins == 1


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_rejects_invalid_budget(budget):
    with pytest.raises((TypeError, ValueError)):
        repair(10, (), max_work=budget)


def test_rejects_invalid_packing():
    a, b = items_of(6, 5)
    with pytest.raises(ValueError):
        repair(10, ((a, b),), max_work=BUDGET)
