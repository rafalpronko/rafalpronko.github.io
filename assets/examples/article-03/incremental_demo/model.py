"""Row and batch types shared by the example modules.

All times are integer Unix seconds in UTC. Integers compare unambiguously,
which ISO-8601 strings with different offsets do not.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    """One version of a source row.

    ``visible_at`` exists only to simulate the source: it is the moment a
    reader can first see this version. A real consumer never receives it.
    ``currency`` belongs to contract version 2; ``None`` means "unknown".
    """

    order_id: str
    updated_at: int
    visible_at: int
    status: str
    amount_minor: int
    currency: str | None = None
    deleted: bool = False


@dataclass(frozen=True)
class Batch:
    """Rows extracted under one declared source contract version."""

    contract_version: int
    rows: tuple[Change, ...]


class VersionConflict(Exception):
    """Same key and same updated_at, but contradictory content."""


class ContractError(ValueError):
    """A row or batch violates the declared source contract."""


class MigrationRequired(RuntimeError):
    """The batch needs a target schema that has not been migrated yet."""


class CheckpointMoved(RuntimeError):
    """The stored watermark is not the one this run started from."""
