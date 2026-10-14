"""Deterministic group-level train/dev/test split with disjointness checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from fair_compare.cases import Case

SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class Split:
    name: str
    cases: tuple[Case, ...]

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(case.group for case in self.cases)


def bucket(group: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(group: str, salt: str, train_pct: int, dev_pct: int) -> str:
    value = bucket(group, salt)
    if value < train_pct:
        return "train"
    if value < train_pct + dev_pct:
        return "dev"
    return "test"


def split_cases(
    cases: tuple[Case, ...], salt: str, train_pct: int, dev_pct: int
) -> dict[str, Split]:
    """Split by group, so related tickets never straddle two splits."""
    by_name: dict[str, list[Case]] = {name: [] for name in SPLIT_NAMES}
    for case in cases:
        name = assign_split(case.group, salt, train_pct, dev_pct)
        by_name[name].append(case)
    splits = {n: Split(n, tuple(items)) for n, items in by_name.items()}
    check_disjoint(splits)
    return splits


def check_label_coverage(
    splits: dict[str, Split], labels: tuple[str, ...]
) -> None:
    """Every split must contain every label (checked on labels only)."""
    for name, split in splits.items():
        missing = set(labels) - {case.label for case in split.cases}
        if missing:
            raise ValueError(
                f"{name} split has no cases for {sorted(missing)}"
            )


def check_disjoint(splits: dict[str, Split]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = splits[left], splits[right]
            shared_ids = {c.case_id for c in a.cases} & {
                c.case_id for c in b.cases
            }
            shared_text = {c.text for c in a.cases} & {c.text for c in b.cases}
            shared_groups = a.groups & b.groups
            if shared_ids or shared_groups or shared_text:
                raise ValueError(
                    f"{left}/{right} overlap: ids={sorted(shared_ids)} "
                    f"groups={sorted(shared_groups)} texts={len(shared_text)}"
                )
