import pytest

from packing.ffd import first_fit_decreasing
from packing.model import check_packing
from tests.helpers import ids_of, items_of, sizes_of


def test_empty_input_gives_no_bins():
    assert first_fit_decreasing(10, []) == ()


def test_places_largest_first_into_first_bin_that_fits():
    items = items_of(2, 5, 4, 3, 4, 2)
    bins = first_fit_decreasing(10, items)
    assert sizes_of(bins) == [[5, 4], [4, 3, 2], [2]]
    check_packing(10, items, bins)


def test_duplicate_sizes_keep_distinct_identities_in_input_order():
    items = items_of(4, 4, 4)
    bins = first_fit_decreasing(8, items)
    assert [[i.id for i in b] for b in bins] == [["x0", "x1"], ["x2"]]
    assert ids_of(bins) == ["x0", "x1", "x2"]


def test_saturated_items_each_get_their_own_bin():
    items = items_of(10, 10, 10)
    assert sizes_of(first_fit_decreasing(10, items)) == [[10], [10], [10]]


def test_does_not_mutate_input_list():
    items = items_of(1, 9, 5)
    snapshot = list(items)
    first_fit_decreasing(10, items)
    assert items == snapshot


def test_validates_input():
    with pytest.raises(ValueError):
        first_fit_decreasing(10, items_of(11))
