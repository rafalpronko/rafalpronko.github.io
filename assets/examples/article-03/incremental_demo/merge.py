"""Row validation and the conditional UPSERT with its conflict policy.

Version order is ``updated_at``. For an existing key:

* higher ``updated_at``  -> the incoming version replaces the stored one;
* lower ``updated_at``   -> no-op, an old extract must not regress state;
* equal ``updated_at``   -> the core payload must be identical, otherwise
  VersionConflict. The one permitted difference is enrichment: a stored
  NULL (unknown) currency may be filled by a known one. A known currency is
  never replaced at the same version, and an incoming NULL never erases it.

Enrichment trusts the source: it assumes the v2 extract reports the currency
the source holds for that same business version. The merge cannot tell a
recorded value from one the producer guessed or derived.
"""
import re
import sqlite3

from incremental_demo.model import Change, ContractError, VersionConflict

SUPPORTED_CONTRACTS = (1, 2)
CURRENCY_CODE = re.compile(r"[A-Z]{3}")

UPSERT_V1 = """
INSERT INTO orders (order_id, updated_at, status, amount_minor, deleted)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (order_id) DO UPDATE SET
    updated_at = excluded.updated_at,
    status = excluded.status,
    amount_minor = excluded.amount_minor,
    deleted = excluded.deleted
WHERE excluded.updated_at > orders.updated_at
"""

UPSERT_V2 = """
INSERT INTO orders (order_id, updated_at, status, amount_minor, deleted,
                    currency)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (order_id) DO UPDATE SET
    updated_at = excluded.updated_at,
    status = excluded.status,
    amount_minor = excluded.amount_minor,
    deleted = excluded.deleted,
    currency = excluded.currency
WHERE excluded.updated_at > orders.updated_at
   OR (excluded.updated_at = orders.updated_at
       AND orders.currency IS NULL
       AND excluded.currency IS NOT NULL)
"""


def check_contract_version(contract_version: object) -> None:
    """A batch-level check; ``True == 1`` in Python, so bool is refused."""
    if (type(contract_version) is not int
            or contract_version not in SUPPORTED_CONTRACTS):
        raise ContractError(f"unsupported contract {contract_version!r}")


def validate(row: Change, contract_version: int) -> None:
    """Reject rows that break the declared contract before they touch SQL."""
    check_contract_version(contract_version)
    if type(row.order_id) is not str or not row.order_id:
        raise ContractError(f"order_id {row.order_id!r}: non-empty str required")
    for name in ("updated_at", "amount_minor"):
        if type(getattr(row, name)) is not int:
            raise ContractError(f"{row.order_id}.{name}: int required")
    if type(row.status) is not str or type(row.deleted) is not bool:
        raise ContractError(f"{row.order_id}: bad status/deleted type")
    if row.amount_minor < 0:
        raise ContractError(f"{row.order_id}.amount_minor: negative")
    if contract_version == 1 and row.currency is not None:
        raise ContractError(f"{row.order_id}: v1 rows carry no currency")
    if contract_version == 2 and not _is_code(row.currency):
        raise ContractError(f"{row.order_id}.currency: 3-letter code required")


def _is_code(value: object) -> bool:
    return isinstance(value, str) and CURRENCY_CODE.fullmatch(value) is not None


def check_same_version(con: sqlite3.Connection, row: Change,
                       has_currency: bool) -> None:
    """Raise VersionConflict when an equal version carries other content."""
    extra = ", currency" if has_currency else ""
    stored = con.execute(
        f"SELECT updated_at, status, amount_minor, deleted{extra} "
        "FROM orders WHERE order_id = ?", (row.order_id,)).fetchone()
    if stored is None or stored[0] != row.updated_at:
        return
    if stored[1:4] != (row.status, row.amount_minor, int(row.deleted)):
        raise VersionConflict(f"{row.order_id} @ {row.updated_at}: payload")
    known = stored[4] if has_currency else None
    if known is not None and row.currency is not None and known != row.currency:
        raise VersionConflict(f"{row.order_id} @ {row.updated_at}: currency")


def merge_row(con: sqlite3.Connection, row: Change, has_currency: bool) -> None:
    check_same_version(con, row, has_currency)
    params = (row.order_id, row.updated_at, row.status, row.amount_minor,
              int(row.deleted))
    if has_currency:
        con.execute(UPSERT_V2, params + (row.currency,))
    else:
        con.execute(UPSERT_V1, params)
