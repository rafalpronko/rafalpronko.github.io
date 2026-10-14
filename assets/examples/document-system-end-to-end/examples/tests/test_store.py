import sqlite3

import pytest

from docsys import schema, store
from docsys.search import search
from docsys.store import VersionConflict, delete_document, upsert_document


def chunk_rows(conn, doc_id):
    sql = "SELECT version, chunk_no, text FROM chunks WHERE doc_id = ? ORDER BY chunk_no"
    return [tuple(row) for row in conn.execute(sql, (doc_id,))]


def corpus_version(conn):
    return conn.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()[0]


def test_insert_writes_one_chunk_per_paragraph(conn):
    assert upsert_document(conn, "d1", 1, "T", "alpha one\n\nbeta two") == "inserted"
    assert chunk_rows(conn, "d1") == [(1, 0, "alpha one"), (1, 1, "beta two")]
    assert corpus_version(conn) == 1


def test_replaying_same_version_is_idempotent(conn):
    upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta")
    assert upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta") == "unchanged"
    assert len(chunk_rows(conn, "d1")) == 2
    assert corpus_version(conn) == 1


def test_same_version_with_different_content_conflicts(conn):
    upsert_document(conn, "d1", 1, "T", "alpha")
    with pytest.raises(VersionConflict):
        upsert_document(conn, "d1", 1, "T", "different")
    assert chunk_rows(conn, "d1") == [(1, 0, "alpha")]


def test_newer_version_replaces_all_chunks(conn):
    upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta\n\ngamma")
    assert upsert_document(conn, "d1", 2, "T", "delta") == "updated"
    assert chunk_rows(conn, "d1") == [(2, 0, "delta")]


def test_older_version_is_ignored(conn):
    upsert_document(conn, "d1", 2, "T", "new text")
    assert upsert_document(conn, "d1", 1, "T", "old text") == "stale"
    assert chunk_rows(conn, "d1") == [(2, 0, "new text")]


def test_delete_leaves_tombstone_and_removes_chunks(conn):
    upsert_document(conn, "d1", 1, "T", "alpha")
    assert delete_document(conn, "d1", 2) == "deleted"
    assert chunk_rows(conn, "d1") == []
    row = conn.execute("SELECT version, deleted FROM documents WHERE doc_id = 'd1'").fetchone()
    assert tuple(row) == (2, 1)


def test_stale_replay_after_delete_cannot_resurrect(conn):
    upsert_document(conn, "d1", 1, "T", "alpha")
    delete_document(conn, "d1", 2)
    assert upsert_document(conn, "d1", 1, "T", "alpha") == "stale"
    assert delete_document(conn, "d1", 2) == "unchanged"
    with pytest.raises(VersionConflict):
        upsert_document(conn, "d1", 2, "T", "alpha")
    assert chunk_rows(conn, "d1") == []


def test_delete_before_first_write_blocks_late_older_insert(conn):
    assert delete_document(conn, "d9", 3) == "deleted"
    assert upsert_document(conn, "d9", 2, "T", "late arrival") == "stale"
    assert upsert_document(conn, "d9", 4, "T", "recreated") == "updated"
    assert chunk_rows(conn, "d9") == [(4, 0, "recreated")]


def test_failure_mid_write_rolls_back_chunks_and_metadata(conn, monkeypatch):
    upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta")
    original = store._insert_chunks

    def insert_then_fail(connection, doc_id, version, paragraphs):
        original(connection, doc_id, version, paragraphs[:1])
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(store, "_insert_chunks", insert_then_fail)
    with pytest.raises(RuntimeError):
        upsert_document(conn, "d1", 2, "T", "gamma\n\ndelta")
    assert chunk_rows(conn, "d1") == [(1, 0, "alpha"), (1, 1, "beta")]
    assert conn.execute("SELECT version FROM documents WHERE doc_id='d1'").fetchone()[0] == 1
    assert corpus_version(conn) == 1


@pytest.mark.parametrize(
    "doc_id, version, title, body",
    [("", 1, "T", "x"), ("d", 0, "T", "x"), ("d", True, "T", "x"), ("d", 1, "T", "  \n\n ")],
)
def test_invalid_documents_are_rejected(conn, doc_id, version, title, body):
    with pytest.raises(ValueError):
        upsert_document(conn, doc_id, version, title, body)


def test_failed_commit_rolls_back_and_connection_stays_usable(conn, failing_commit):
    upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta")
    proxy = failing_commit(conn)
    with pytest.raises(sqlite3.OperationalError, match="simulated COMMIT"):
        upsert_document(proxy, "d1", 2, "T", "gamma\n\ndelta")
    assert not conn.in_transaction
    assert chunk_rows(conn, "d1") == [(1, 0, "alpha"), (1, 1, "beta")]
    sql = "SELECT version, content_hash FROM documents"
    row = conn.execute(sql).fetchone()
    assert row["version"] == 1 and row["content_hash"] is not None
    assert corpus_version(conn) == 1
    assert upsert_document(conn, "d1", 2, "T", "gamma") == "updated"
    assert search(conn, "gamma", limit=5)["corpus_version"] == 2


def test_failure_before_commit_leaves_connection_usable(conn, monkeypatch):
    upsert_document(conn, "d1", 1, "T", "alpha")

    def fail(*args):
        raise RuntimeError("simulated failure before COMMIT")

    monkeypatch.setattr(store, "_bump_corpus_version", fail)
    with pytest.raises(RuntimeError, match="before COMMIT"):
        upsert_document(conn, "d1", 2, "T", "beta")
    monkeypatch.undo()
    assert not conn.in_transaction
    assert chunk_rows(conn, "d1") == [(1, 0, "alpha")]
    assert upsert_document(conn, "d1", 2, "T", "beta") == "updated"
    assert search(conn, "beta", limit=5)["corpus_version"] == 2


def test_failed_snapshot_commit_does_not_leave_transaction_open(conn, failing_commit):
    upsert_document(conn, "d1", 1, "T", "alpha")
    with pytest.raises(sqlite3.OperationalError, match="simulated COMMIT"):
        search(failing_commit(conn), "alpha", limit=5)
    assert not conn.in_transaction
    assert search(conn, "alpha", limit=5)["hits"][0]["doc_id"] == "d1"


def test_tombstone_hash_is_sql_null(conn):
    delete_document(conn, "d1", 1)
    row = conn.execute("SELECT content_hash FROM documents").fetchone()
    assert row["content_hash"] is None


def test_version_range_is_sqlite_signed_64_bit(conn):
    assert upsert_document(conn, "d1", 2**63 - 1, "T", "alpha") == "inserted"
    for version in (2**63, 10**20):
        with pytest.raises(ValueError, match="version"):
            upsert_document(conn, "d2", version, "T", "beta")
        with pytest.raises(ValueError, match="version"):
            delete_document(conn, "d1", version)
    assert chunk_rows(conn, "d1") == [(2**63 - 1, 0, "alpha")]
    assert chunk_rows(conn, "d2") == []
    assert corpus_version(conn) == 1


def test_failed_rollback_closes_connection_and_keeps_commit_error(tmp_path, failing_commit):
    path = tmp_path / "docs.db"
    conn = schema.connect(path)
    upsert_document(conn, "d1", 1, "T", "alpha\n\nbeta")
    proxy = failing_commit(conn, fail_rollback=True)
    with pytest.raises(sqlite3.OperationalError, match="simulated COMMIT"):
        upsert_document(proxy, "d1", 2, "T", "gamma")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    assert not schema.is_usable(conn)
    fresh = schema.connect(path)
    try:
        assert chunk_rows(fresh, "d1") == [(1, 0, "alpha"), (1, 1, "beta")]
        assert corpus_version(fresh) == 1
    finally:
        fresh.close()


class FailingSetup:
    """Wraps the connection that connect() opens: one setup step raises."""

    def __init__(self, conn, step, close_fails):
        self.real, self.step, self.close_fails = conn, step, close_fails

    def execute(self, sql, *args):
        if sql == self.step:
            raise sqlite3.OperationalError("simulated setup failure")
        return self.real.execute(sql, *args)

    def executescript(self, script):
        if self.step == "schema":
            raise sqlite3.OperationalError("simulated setup failure")
        return self.real.executescript(script)

    def close(self):
        self.real.close()
        if self.close_fails:
            raise sqlite3.OperationalError("simulated close failure")


@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize(
    "step", ["PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=5000", "schema"]
)
def test_connect_closes_connection_when_setup_fails(tmp_path, monkeypatch, step, close_fails):
    opened = []
    real_connect = sqlite3.connect

    def connect_with_failing_setup(*args, **kwargs):
        opened.append(FailingSetup(real_connect(*args, **kwargs), step, close_fails))
        return opened[-1]

    monkeypatch.setattr(schema.sqlite3, "connect", connect_with_failing_setup)
    with pytest.raises(sqlite3.OperationalError, match="simulated setup failure"):
        schema.connect(tmp_path / "docs.db")
    monkeypatch.undo()
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].real.execute("SELECT 1")
