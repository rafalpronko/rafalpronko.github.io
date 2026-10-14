"""Database connection and schema.

Needs SQLite 3.24.0 or newer (2018-06-04, plain SQL UPSERT in store.py) built
with FTS5 enabled. The FTS5 features used were already in its first release,
SQLite 3.9.0 (2015): an ordinary (full content) FTS5 table, UNINDEXED columns
and the bm25() function.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    version      INTEGER NOT NULL,
    content_hash TEXT,
    title        TEXT,
    deleted      INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    text, doc_id UNINDEXED, version UNINDEXED, chunk_no UNINDEXED
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('corpus_version', 0);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection in autocommit mode; transactions are explicit.

    If a PRAGMA or the schema fails, the connection is closed before the
    original error propagates.
    """
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(SCHEMA)
    except BaseException:
        close_quietly(conn)
        raise
    return conn


def is_usable(conn: Connection) -> bool:
    """False if conn is closed or still inside a transaction."""
    try:
        return not conn.in_transaction
    except sqlite3.Error:  # ProgrammingError on a closed connection
        return False


def close_quietly(conn: Connection) -> None:
    """Close conn while another exception propagates; never replace it."""
    try:
        conn.close()
    except Exception:
        pass


@contextmanager
def write_transaction(conn: Connection) -> Iterator[Connection]:
    """BEGIN IMMEDIATE ... COMMIT; roll back if the body or COMMIT fails."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        _rollback_keeping_error(conn)
        raise


@contextmanager
def read_snapshot(conn: Connection) -> Iterator[Connection]:
    """A read transaction so that several SELECTs see one corpus version."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        _rollback_keeping_error(conn)
        raise


def _rollback_keeping_error(conn: Connection) -> None:
    """Roll back if a transaction is still open.

    Called while an exception is propagating: a failing ROLLBACK must not
    replace that original exception. If the transaction is still open after
    ROLLBACK, the connection is closed, because every later BEGIN on it would
    fail. Callers detect that with is_usable().
    """
    try:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
    except Exception:
        pass
    if not is_usable(conn):
        close_quietly(conn)
