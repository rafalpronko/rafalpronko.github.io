"""Labelled routing cases and the output contract shared by both arms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    text: str
    label: str


@dataclass(frozen=True)
class Dataset:
    labels: tuple[str, ...]
    cases: tuple[Case, ...]


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = tuple(raw["labels"])
    cases = tuple(Case(**item) for item in raw["cases"])
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique")
    unknown = {case.label for case in cases} - set(labels)
    if unknown:
        raise ValueError(f"cases use labels outside the contract: {unknown}")
    return Dataset(labels=labels, cases=cases)


def parse_label(raw_output: str, labels: tuple[str, ...]) -> str | None:
    """Apply the output contract: exactly one known label, nothing else.

    Returns None for an invalid output. Both arms use this same parser.
    """
    candidate = raw_output.strip().lower()
    return candidate if candidate in labels else None
