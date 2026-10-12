"""Target schema, the currency migration, transactions and the checkpoint.

The connection uses ``isolation_level=None`` so that this module issues
BEGIN/COMMIT/ROLLBACK itself and the SELECT that reads the stored version
belongs to the same transaction as the write.
"""
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from incremental_demo.model import CheckpointMoved

PIPELINE = "orders"
CURRENCY_MIGRATION = "0002_add_nullable_currency"

BASE_SCHEMA = (
    """CREATE TABLE orders (
        order_id TEXT NOT NULL PRIMARY KEY,
        updated_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        amount_minor INTEGER NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE checkpoint (
        pipeline TEXT NOT NULL PRIMARY KEY,
        watermark INTEGER NOT NULL)""",
    """CREATE TABLE schema_migrations (
        name TEXT NOT NULL PRIMARY KEY)""",
)


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[None]:
    """One explicit write transaction: commit on success, roll back on error.

    COMMIT is inside the ``try``: SQLite can refuse it (SQLITE_BUSY, a
    deferred foreign key) and leave the transaction open. SQLite may also
    have rolled back on its own already, hence the ``in_transaction`` check.
    A failure whose outcome is unknown, such as a crash during COMMIT, is
    not handled here; the next run has to converge by re-reading.
    """
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise


def connect(start_watermark: int) -> sqlite3.Connection:
    """In-memory target with the contract-v1 schema and a first checkpoint.

    The target starts empty, so the first run's lower bound
    ``start_watermark - overlap`` must lie at or below the oldest
    ``updated_at`` the target has to hold. Otherwise older rows are never
    in any window; load them first with ``run_backfill`` below the watermark.
    """
    con = sqlite3.connect(":memory:", isolation_level=None)
    with transaction(con):
        for ddl in BASE_SCHEMA:
            con.execute(ddl)
        con.execute("INSERT INTO checkpoint VALUES (?, ?)",
                    (PIPELINE, start_watermark))
    return con


def columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(orders)")}


def add_currency_column(con: sqlite3.Connection) -> bool:
    """Contract-v2 migration. Returns False when it was already applied.

    The column is nullable and has no default: rows loaded before the
    migration keep NULL, which this pipeline reads as "currency unknown".
    """
    with transaction(con):
        done = con.execute("SELECT 1 FROM schema_migrations WHERE name = ?",
                           (CURRENCY_MIGRATION,)).fetchone()
        if done:
            return False
        con.execute("ALTER TABLE orders ADD COLUMN currency TEXT")
        con.execute("INSERT INTO schema_migrations VALUES (?)",
                    (CURRENCY_MIGRATION,))
    return True


def watermark(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT watermark FROM checkpoint WHERE pipeline = ?",
                      (PIPELINE,)).fetchone()
    return int(row[0])


def advance_checkpoint(con: sqlite3.Connection, old: int, new: int) -> None:
    """Compare-and-set; must run inside the transaction that wrote the rows."""
    cur = con.execute("UPDATE checkpoint SET watermark = ? "
                      "WHERE pipeline = ? AND watermark = ?",
                      (new, PIPELINE, old))
    if cur.rowcount != 1:
        raise CheckpointMoved(f"expected watermark {old}")
