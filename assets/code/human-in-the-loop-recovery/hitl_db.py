"""Explicit SQLite transaction control shared by the device and the coordinator."""
import sqlite3
from contextlib import contextmanager


def connect(path) -> sqlite3.Connection:
    # isolation_level=None: the sqlite3 module opens no implicit transactions;
    # every BEGIN / COMMIT / ROLLBACK below is ours.
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("PRAGMA synchronous = FULL")
    return con


@contextmanager
def transaction(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
