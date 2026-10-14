"""Synthetic stand-in for an external system that honours idempotency keys.

It opens an independent connection to its own SQLite file, so the worker's
checkpoint transaction does not cover the receiver's commit. The separate
file alone would not do that: in rollback-journal mode one connection can
ATTACH several files and commit them atomically. This independent connection
models the gap to an external API, which no local transaction can include.
"""
import hashlib
import sqlite3

SCHEMA = """CREATE TABLE IF NOT EXISTS delivery (
    operation_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    receipt TEXT NOT NULL)"""


class PayloadConflict(Exception):
    """Same operation id arrived with a different payload."""


def deliver(path: str, operation_id: str, payload: str) -> str:
    """Apply the effect at most once per operation id; return a receipt.

    Callers must use a unique id for each logical operation in this receiver's
    namespace, and reuse that same id only for retries of the same operation.
    """
    con = sqlite3.connect(path, isolation_level=None)
    try:
        con.execute(SCHEMA)
        con.execute("BEGIN IMMEDIATE")
        try:
            receipt = _apply(con, operation_id, payload)
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")
        return receipt
    finally:
        con.close()


def _apply(con: sqlite3.Connection, operation_id: str, payload: str) -> str:
    row = con.execute(
        "SELECT payload, receipt FROM delivery WHERE operation_id = ?",
        (operation_id,)).fetchone()
    if row is None:
        digest = hashlib.sha256(operation_id.encode()).hexdigest()
        receipt = f"rcpt-{digest[:12]}"
        con.execute("INSERT INTO delivery VALUES (?, ?, 1, ?)",
                    (operation_id, payload, receipt))
        return receipt
    if row[0] != payload:
        raise PayloadConflict(operation_id)
    con.execute("UPDATE delivery SET attempts = attempts + 1 "
                "WHERE operation_id = ?", (operation_id,))
    return row[1]
