"""Predeclared candidate fixtures. They are lookup tables, not models.

A candidate answers every case with its true label unless an override says
otherwise. Overrides are written by hand in data/candidates.json. Two special
outputs simulate failures: TRANSIENT (retryable, e.g. a rate limit) and FATAL
(not retryable, e.g. a rejected request).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from fair_compare.cases import Case

TRANSIENT = "!transient"
FATAL = "!fatal"


@dataclass(frozen=True)
class Candidate:
    arm: str
    candidate_id: str
    artifact: str
    compile_attempts: Mapping[str, int]
    human_minutes: int | None
    overrides: Mapping[str, tuple[str, ...]]
    demo_ids: tuple[str, ...] = ()

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.artifact.encode()).hexdigest()

    def output(self, case: Case, attempt: int) -> str:
        scripted = self.overrides.get(case.case_id)
        if scripted is None:
            return case.label
        return scripted[min(attempt, len(scripted) - 1)]


def _as_tuple(value: str | list[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def load_candidates(path: Path) -> dict[str, tuple[Candidate, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arms: dict[str, tuple[Candidate, ...]] = {}
    for arm, items in raw["arms"].items():
        arms[arm] = tuple(
            Candidate(
                arm=arm,
                candidate_id=item["candidate_id"],
                artifact=item["artifact"],
                compile_attempts=MappingProxyType(
                    dict(item.get("compile_attempts", {}))
                ),
                human_minutes=item.get("human_minutes"),
                overrides=MappingProxyType(
                    {k: _as_tuple(v) for k, v in item["overrides"].items()}
                ),
                demo_ids=tuple(item.get("demo_ids", ())),
            )
            for item in items
        )
    _check_same_artifact_same_behaviour(arms)
    return arms


def _check_same_artifact_same_behaviour(
    arms: dict[str, tuple[Candidate, ...]],
) -> None:
    """A cache keyed by artifact hash is only valid if equal artifacts
    behave identically (deterministic decoding is part of the contract)."""
    seen: dict[tuple[str, str], Candidate] = {}
    for arm, candidates in arms.items():
        for candidate in candidates:
            key = (arm, candidate.artifact_sha256)
            first = seen.setdefault(key, candidate)
            if dict(first.overrides) != dict(candidate.overrides):
                raise ValueError(
                    f"{candidate.candidate_id} reuses the artifact of "
                    f"{first.candidate_id} but scripts different outputs"
                )
