"""Publish validated invoice lines into SQLite: whole batch or nothing.

Python 3.11.x only, standard library only. The transaction checks describe
the Python 3.11 sqlite3 model, so connect() and publish() refuse any other
Python version. No STRICT tables and no UPSERT, so the schema does not raise
the minimum SQLite version needed by sqlite3.
"""

from __future__ import annotations

import sqlite3
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from invoice_contract import (AMOUNT_FIELDS, AMOUNT_MAX, CONTRACT_VERSION,
                              ID_FIELDS, ContractViolation, validate_batch)

SUPPORTED_PYTHON = (3, 11)
ADVISORY_FACTOR = 10  # tenfold band; an illustration, not a validated threshold

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoice_lines (
    line_id TEXT NOT NULL PRIMARY KEY
        CHECK (line_id <> '' AND line_id = trim(line_id)),
    invoice_id TEXT NOT NULL
        CHECK (invoice_id <> '' AND invoice_id = trim(invoice_id)),
    net_minor INTEGER NOT NULL
        CHECK (typeof(net_minor) = 'integer' AND net_minor >= 0),
    tax_bp INTEGER NOT NULL
        CHECK (typeof(tax_bp) = 'integer' AND tax_bp >= 0),
    tax_minor INTEGER NOT NULL
        CHECK (typeof(tax_minor) = 'integer' AND tax_minor >= 0),
    gross_minor INTEGER NOT NULL
        CHECK (typeof(gross_minor) = 'integer'),
    contract_version TEXT NOT NULL,
    CHECK (gross_minor = net_minor + tax_minor)
)
"""
COLUMNS = (*ID_FIELDS, *AMOUNT_FIELDS, "contract_version")
INSERT_SQL = (f"INSERT INTO invoice_lines ({', '.join(COLUMNS)}) "
              f"VALUES ({', '.join('?' for _ in COLUMNS)})")


@dataclass(frozen=True)
class PublishResult:
    rows_written: int
    advisories: list[str]  # soft signals for a human; never a gate


def connect(path: str = ":memory:") -> sqlite3.Connection:
    """Open a Python 3.11 connection with the default isolation_level."""
    require_supported_runtime()
    con = sqlite3.connect(path)
    try:
        con.execute(SCHEMA)
    except BaseException:
        con.close()
        raise
    return con


def count_rows(con: sqlite3.Connection) -> int:
    return int(con.execute("SELECT count(*) FROM invoice_lines").fetchone()[0])


def publish(con: sqlite3.Connection, envelope: object,
            baseline_median_net: int) -> PublishResult:
    """Validate the whole batch in memory, then insert it in one transaction.

    Before touching the database: RuntimeError or ValueError for an
    unsupported runtime, connection or baseline, ContractViolation for the
    batch. After rolling back: sqlite3.IntegrityError (e.g. a stored key).
    Advisories are computed after the commit and never block it.
    """
    require_supported_runtime()
    require_transaction_control(con)
    require_baseline(baseline_median_net)
    errors = validate_batch(envelope)
    if errors:
        raise ContractViolation(errors)
    rows = cast(list[dict[str, Any]], cast(dict[str, Any], envelope)["rows"])
    params = [(*(row[c] for c in COLUMNS[:-1]), CONTRACT_VERSION)
              for row in rows]
    with con:  # commit on success, rollback on any exception
        con.executemany(INSERT_SQL, params)  # INSERT opens the transaction
    return PublishResult(len(params),
                         distribution_advisories(rows, baseline_median_net))


def require_supported_runtime() -> None:
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        found = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(f"Python 3.11.x required, found {found}: the "
                           "transaction checks cover only the 3.11 sqlite3 "
                           "transaction model")


def require_baseline(baseline_median_net: object) -> None:
    """Same exact-int range as net_minor, so the advisory cannot overflow."""
    limit = AMOUNT_MAX["net_minor"]
    if (type(baseline_median_net) is not int
            or not 1 <= baseline_median_net <= limit):
        raise ValueError(f"baseline_median_net must be an int in [1, {limit}]")


def require_transaction_control(con: sqlite3.Connection) -> None:
    if con.isolation_level is None:
        raise ValueError("isolation_level is None (autocommit): each INSERT "
                         "would commit on its own and a failure could leave "
                         "a partial batch")
    if con.in_transaction:
        raise RuntimeError("caller has an open transaction; publish() would "
                           "commit or roll back the caller's pending work")


def distribution_advisories(rows: Sequence[Mapping[str, Any]],
                            baseline_median_net: int) -> list[str]:
    """Flag a batch median far from a reference value. Advisory only.

    Expects rows that passed the contract (0 <= net_minor <= AMOUNT_MAX)
    and enforces the baseline range itself; within these bounds the float
    arithmetic below cannot overflow.
    """
    require_baseline(baseline_median_net)
    if not rows:
        return []
    median = statistics.median([row["net_minor"] for row in rows])
    low = baseline_median_net / ADVISORY_FACTOR
    high = baseline_median_net * ADVISORY_FACTOR
    if low <= median <= high:
        return []
    return [f"median net_minor {median} outside advisory band "
            f"[{low:.12g}, {high}] around baseline {baseline_median_net}"]
