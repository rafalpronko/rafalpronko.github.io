from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from fair_compare.candidates import Candidate
from fair_compare.cases import Case
from fair_compare.splits import Split

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = ("billing", "technical", "account", "shipping")


def make_candidate(
    cid: str = "c",
    overrides: dict[str, list[str]] | None = None,
    compile_attempts: dict[str, int] | None = None,
    demo_ids: tuple[str, ...] = (),
    artifact: str | None = None,
) -> Candidate:
    return Candidate(
        arm="test-arm",
        candidate_id=cid,
        artifact=artifact or f"artifact for {cid}",
        compile_attempts=MappingProxyType(compile_attempts or {}),
        human_minutes=None,
        overrides=MappingProxyType(
            {k: tuple(v) for k, v in (overrides or {}).items()}
        ),
        demo_ids=demo_ids,
    )


def make_split(name: str, n: int, prefix: str) -> Split:
    cases = tuple(
        Case(f"{prefix}{i}", f"{prefix}g{i // 2}", f"{prefix} text {i}",
             LABELS[i % len(LABELS)])
        for i in range(n)
    )
    return Split(name, cases)


@pytest.fixture
def data_dir() -> Path:
    return DATA
