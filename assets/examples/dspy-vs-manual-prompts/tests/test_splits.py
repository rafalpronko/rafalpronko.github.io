from __future__ import annotations

import pytest
from conftest import DATA, LABELS

from fair_compare.cases import Case, load_dataset, parse_label
from fair_compare.splits import (
    Split,
    check_disjoint,
    check_label_coverage,
    split_cases,
)


def test_split_is_deterministic_and_group_disjoint() -> None:
    cases = load_dataset(DATA / "cases.json").cases
    first = split_cases(cases, "article-36-split-12", 50, 25)
    second = split_cases(cases, "article-36-split-12", 50, 25)
    assert first == second
    groups = [s.groups for s in first.values()]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(len(s.cases) for s in first.values()) == len(cases)
    check_label_coverage(first, LABELS)


def test_shared_group_across_splits_is_rejected() -> None:
    a = Split("train", (Case("x1", "g1", "one", "billing"),))
    b = Split("dev", (Case("x2", "g1", "two", "billing"),))
    with pytest.raises(ValueError, match="groups=\\['g1'\\]"):
        check_disjoint({"train": a, "dev": b})


def test_duplicate_text_across_splits_is_rejected() -> None:
    a = Split("train", (Case("x1", "g1", "same text", "billing"),))
    b = Split("test", (Case("x2", "g2", "same text", "billing"),))
    with pytest.raises(ValueError, match="texts=1"):
        check_disjoint({"train": a, "test": b})


def test_missing_label_in_a_split_is_rejected() -> None:
    only_billing = Split("test", (Case("x1", "g1", "t", "billing"),))
    with pytest.raises(ValueError, match="no cases for"):
        check_label_coverage({"test": only_billing}, LABELS)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("billing", "billing"), ("  Billing\n", "billing"),
     ("Billing.", None), ("shipping/delivery", None), ("", None)],
)
def test_output_contract(raw: str, expected: str | None) -> None:
    assert parse_label(raw, LABELS) == expected
