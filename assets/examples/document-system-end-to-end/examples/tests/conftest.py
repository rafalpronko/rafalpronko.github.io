import json
import sqlite3
from pathlib import Path

import pytest

from docsys.schema import connect

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "docs.db")
    yield connection
    connection.close()


@pytest.fixture
def corpus():
    return json.loads((FIXTURES / "corpus.json").read_text())


@pytest.fixture
def queries():
    return json.loads((FIXTURES / "queries.json").read_text())


class FailingCommit:
    """Connection proxy whose next COMMIT raises once, before reaching SQLite.

    With fail_rollback=True the following ROLLBACK raises too, so the real
    connection stays inside the transaction. It checks the error paths of the
    transaction helpers and the API; it does not simulate I/O errors, a full
    disk or a crash inside SQLite.
    """

    def __init__(self, conn, fail_rollback=False):
        self._conn = conn
        self.armed = True
        self.fail_rollback = fail_rollback

    def execute(self, sql, *args):
        if self.armed and sql == "COMMIT":
            self.armed = False
            raise sqlite3.OperationalError("simulated COMMIT failure")
        if self.fail_rollback and sql == "ROLLBACK":
            raise sqlite3.OperationalError("simulated ROLLBACK failure")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def failing_commit():
    return FailingCommit
