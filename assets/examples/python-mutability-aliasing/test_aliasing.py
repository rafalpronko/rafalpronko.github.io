"""Regression tests for the aliasing example.

Every behaviour that the article describes is checked here.
Run: pytest -v
"""
import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aliasing import Order, load_batch, tag_orders, tag_orders_shallow


# --- the bug: same annotated types, corrupted input ------------------

def test_shallow_version_corrupts_input_without_changing_types() -> None:
    batch = load_batch()
    result = tag_orders_shallow(batch, "eu")

    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)
    assert batch != load_batch()
    # The caller's lists were sorted and extended in place.
    assert batch[0]["line_cents"] == [300, 1200]
    assert batch[0]["tags"] == ["raw", "eu"]


def test_shallow_version_makes_sibling_results_share_nested_lists() -> None:
    batch = load_batch()
    eu = tag_orders_shallow(batch, "eu")
    vip = tag_orders_shallow(batch, "vip")

    assert eu[0] is not vip[0]  # distinct outer dicts...
    assert eu[0]["tags"] is vip[0]["tags"]  # ...one shared list
    # The "eu" result carries the "vip" tag.
    assert eu[0]["tags"] == ["raw", "eu", "vip"]


def test_shallow_copy_is_equal_but_not_identical_and_shares_children() -> None:
    order = load_batch()[0]
    row = order.copy()

    assert row == order and row is not order
    assert row["tags"] is order["tags"]


# --- the fix: owned reconstruction -----------------------------------

def test_fixed_version_leaves_input_unchanged() -> None:
    batch = load_batch()
    tag_orders(batch, "eu")

    assert batch == load_batch()


def test_fixed_version_returns_expected_values() -> None:
    assert tag_orders(load_batch(), "eu") == [
        {"order_id": "A-1", "line_cents": [300, 1200],
         "tags": ["raw", "eu"]},
        {"order_id": "A-2", "line_cents": [500], "tags": ["raw", "eu"]},
    ]


def test_fixed_version_keeps_sibling_results_independent() -> None:
    batch = load_batch()
    eu = tag_orders(batch, "eu")
    vip = tag_orders(batch, "vip")

    eu[0]["tags"].append("manual-review")
    eu[0]["line_cents"].append(1)

    assert vip[0] == {
        "order_id": "A-1",
        "line_cents": [300, 1200],
        "tags": ["raw", "vip"],
    }
    assert batch == load_batch()


def test_fixed_version_owns_every_mutable_component() -> None:
    batch = load_batch()
    result = tag_orders(batch, "eu")

    for before, after in zip(batch, result, strict=True):
        assert after is not before
        assert after["tags"] is not before["tags"]
        assert after["line_cents"] is not before["line_cents"]


def test_fixed_version_separates_aliases_present_in_the_input() -> None:
    shared = ["raw"]
    batch = [
        Order(order_id="A-1", line_cents=[1], tags=shared),
        Order(order_id="A-2", line_cents=[2], tags=shared),
    ]
    result = tag_orders(batch, "eu")

    assert result[0]["tags"] is not result[1]["tags"]
    result[0]["tags"].append("x")
    assert result[1]["tags"] == ["raw", "eu"]
    assert shared == ["raw"]


# --- mutation versus rebinding ---------------------------------------

def test_plus_equals_mutates_shared_list_but_rebinding_does_not() -> None:
    order = load_batch()[0]

    rebound = order.copy()
    rebound["tags"] = rebound["tags"] + ["eu"]  # new list, new binding
    assert order["tags"] == ["raw"]

    augmented = order.copy()
    augmented["tags"] += ["eu"]  # list.__iadd__ extends the shared list
    assert order["tags"] == ["raw", "eu"]


# --- deepcopy and its limits -----------------------------------------

def test_deepcopy_memo_preserves_internal_aliases() -> None:
    shared = ["raw"]
    batch = [
        Order(order_id="A-1", line_cents=[1], tags=shared),
        Order(order_id="A-2", line_cents=[2], tags=shared),
    ]
    cloned = copy.deepcopy(batch)

    assert cloned[0]["tags"] is not shared  # detached from the original
    assert cloned[0]["tags"] is cloned[1]["tags"]  # alias survives
    cloned[0]["tags"].append("eu")
    assert cloned[1]["tags"] == ["raw", "eu"]


@dataclass
class Step:
    name: str
    log: object


def test_deepcopy_cannot_copy_an_open_text_file(tmp_path: Path) -> None:
    with open(tmp_path / "export.log", "w", encoding="utf-8") as handle:
        step = Step(name="export", log=handle)
        with pytest.raises(
            TypeError, match=r"^cannot pickle 'TextIOWrapper' instances$"
        ):
            copy.deepcopy(step)


# --- frozen dataclasses and tuples are shallowly immutable -----------

@dataclass(frozen=True)
class FrozenBatch:
    source: str
    orders: tuple[Order, ...]


def test_frozen_dataclass_blocks_rebinding_but_not_nested_mutation() -> None:
    frozen = FrozenBatch(source="eu", orders=tuple(load_batch()))

    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.orders = ()  # type: ignore[misc]
    frozen.orders[0]["tags"].append("eu")
    assert frozen.orders[0]["tags"] == ["raw", "eu"]


def test_tuple_value_changes_when_a_contained_list_changes() -> None:
    tags = ["raw"]
    pair = ("A-1", tags)
    before = copy.deepcopy(pair)

    tags.append("eu")
    assert pair != before


def test_frozen_batch_hash_fails_on_the_dicts_in_orders() -> None:
    frozen = FrozenBatch(source="eu", orders=tuple(load_batch()))

    with pytest.raises(TypeError, match=r"^unhashable type: 'dict'$"):
        hash(frozen)


@dataclass(frozen=True)
class Cents:
    """Immutable: one int field, frozen, but deliberately unhashable."""

    value: int
    __hash__ = None  # type: ignore[assignment]


class Counter:
    """Mutable, yet hashable through the default identity hash."""

    def __init__(self) -> None:
        self.count = 0


def test_hashability_is_not_an_immutability_test() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Cents(1).value = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match=r"unhashable type"):
        hash(Cents(1))

    counter = Counter()
    before = hash(counter)
    counter.count += 1
    assert hash(counter) == before


# --- default_factory versus caller-supplied objects ------------------

@dataclass
class RunReport:
    warnings: list[str] = field(default_factory=list)


@dataclass
class OwningRunReport:
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.warnings = list(self.warnings)


def test_default_factory_gives_each_instance_a_new_list() -> None:
    assert RunReport().warnings is not RunReport().warnings


def test_default_factory_does_not_copy_a_caller_supplied_list() -> None:
    shared: list[str] = []
    a, b = RunReport(shared), RunReport(shared)

    a.warnings.append("late file")
    assert b.warnings == ["late file"]
    assert shared == ["late file"]


def test_post_init_copy_takes_ownership_of_caller_supplied_list() -> None:
    shared: list[str] = []
    a, b = OwningRunReport(shared), OwningRunReport(shared)

    a.warnings.append("late file")
    assert b.warnings == [] and shared == []
