"""Freeze both selections, then allow exactly one holdout evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from fair_compare.candidates import Candidate
from fair_compare.splits import Split


class HoldoutError(RuntimeError):
    """The holdout was opened twice or with a manifest that does not match."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ids_digest(split: Split) -> str:
    return sha256_text("\n".join(sorted(c.case_id for c in split.cases)))


@dataclass(frozen=True)
class Manifest:
    """Immutable record written before any test case is scored."""

    selected: Mapping[str, str]
    artifact_sha256: Mapping[str, str]
    test_ids_sha256: str
    fixtures_sha256: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "selected": dict(self.selected),
                "artifact_sha256": dict(self.artifact_sha256),
                "test_ids_sha256": self.test_ids_sha256,
                "fixtures_sha256": self.fixtures_sha256,
            },
            indent=2,
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return sha256_text(self.to_json())


def freeze(
    selected: Mapping[str, Candidate],
    test_ids_sha256: str,
    fixtures_sha256: str,
) -> Manifest:
    return Manifest(
        selected=MappingProxyType(
            {arm: c.candidate_id for arm, c in selected.items()}
        ),
        artifact_sha256=MappingProxyType(
            {arm: c.artifact_sha256 for arm, c in selected.items()}
        ),
        test_ids_sha256=test_ids_sha256,
        fixtures_sha256=fixtures_sha256,
    )


class HoldoutGuard:
    """Holds the test split; hands it out once, to a matching manifest.

    The receipt file makes the one-shot rule survive a process restart.
    This is a cooperative sequential workflow, not a tamper-proof seal or
    a concurrent security boundary. Preserve receipts and failed run dirs.
    """

    def __init__(self, test: Split, receipt: Path) -> None:
        if test.name != "test":
            raise ValueError("HoldoutGuard only wraps the test split")
        self._test = test
        self._receipt = receipt

    @property
    def test_ids_sha256(self) -> str:
        """A digest of the test ids; reveals no text and no labels."""
        return ids_digest(self._test)

    def open(
        self, manifest: Manifest, candidates: Mapping[str, Candidate]
    ) -> Split:
        if self._receipt.exists():
            raise HoldoutError(f"holdout already used: {self._receipt}")
        if manifest.test_ids_sha256 != self.test_ids_sha256:
            raise HoldoutError("test split changed after freeze")
        for arm, candidate in candidates.items():
            if manifest.selected.get(arm) != candidate.candidate_id:
                raise HoldoutError(f"{arm}: candidate differs from manifest")
            if manifest.artifact_sha256.get(arm) != candidate.artifact_sha256:
                raise HoldoutError(f"{arm}: artifact changed after freeze")
        self._receipt.write_text(manifest.sha256 + "\n", encoding="utf-8")
        return self._test
