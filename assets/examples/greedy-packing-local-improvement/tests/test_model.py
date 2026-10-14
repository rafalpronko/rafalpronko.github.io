import pytest

from packing.model import (
    Item, PackingError, check_packing, lower_bound, validate_instance,
)
from tests.helpers import items_of


def test_empty_instance_is_valid_and_needs_zero_bins():
    assert validate_instance(10, []) == ()
    assert lower_bound(10, []) == 0


@pytest.mark.parametrize("capacity", [0, -5, True, 1.5, "10", None])
def test_rejects_capacity_that_is_not_a_positive_int(capacity):
    with pytest.raises((TypeError, ValueError)):
        validate_instance(capacity, items_of(1))


@pytest.mark.parametrize("size", [0, -1, True, False, 2.0, "3", 11])
def test_rejects_item_size_that_is_not_a_positive_int_within_capacity(size):
    with pytest.raises((TypeError, ValueError)):
        validate_instance(10, [Item("a", size)])


def test_rejects_duplicate_ids_and_non_items():
    with pytest.raises(ValueError):
        validate_instance(10, [Item("a", 1), Item("a", 2)])
    with pytest.raises(TypeError):
        validate_instance(10, [("a", 1)])


def test_validation_returns_a_tuple_and_does_not_mutate_input():
    items = items_of(3, 3, 1)
    snapshot = list(items)
    assert validate_instance(10, items) == tuple(snapshot)
    assert items == snapshot


def test_lower_bound_is_ceiling_of_total_over_capacity():
    assert lower_bound(10, items_of(5, 5)) == 1
    assert lower_bound(10, items_of(5, 5, 1)) == 2
    assert lower_bound(10, items_of(6, 6, 6)) == 2  # optimum is 3


def test_check_packing_accepts_valid_packing():
    a, b, c = items_of(6, 4, 4)
    check_packing(10, [a, b, c], ((a, b), (c,)))


def test_check_packing_rejects_overfull_bin():
    a, b = items_of(6, 5)
    with pytest.raises(PackingError, match="capacity"):
        check_packing(10, [a, b], ((a, b),))


def test_check_packing_rejects_lost_or_duplicated_item():
    a, b = items_of(4, 4)
    with pytest.raises(PackingError):
        check_packing(10, [a, b], ((a,),))
    with pytest.raises(PackingError):
        check_packing(10, [a, b], ((a, b), (b,)))


def test_check_packing_tracks_identity_not_just_size():
    a, b = items_of(4, 4)
    impostor = Item("other", 4)
    with pytest.raises(PackingError):
        check_packing(10, [a, b], ((a, impostor),))


def test_check_packing_rejects_empty_bin():
    (a,) = items_of(4)
    with pytest.raises(PackingError, match="empty"):
        check_packing(10, [a], ((a,), ()))
