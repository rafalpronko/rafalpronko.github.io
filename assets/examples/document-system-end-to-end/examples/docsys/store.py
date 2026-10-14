"""Versioned, idempotent document writes.

Rules (per doc_id, versions are integers 1..2**63-1 supplied by the source):
  * version lower than the stored one          -> "stale", ignored
  * same version, same content (or tombstone)  -> "unchanged"
  * same version, different content            -> VersionConflict
  * higher version                             -> replace chunks / tombstone
A delete stores a versioned tombstone, so replaying an older upsert after the
delete cannot resurrect the document. Every write is one transaction covering
document metadata, all chunks and the corpus version counter.
"""

import sqlite3

from .chunking import content_hash, split_paragraphs
from .schema import write_transaction

TOMBSTONE = None  # content_hash stored for deleted documents (SQL NULL)
MAX_VERSION = 2**63 - 1  # largest value SQLite can store in an INTEGER


class VersionConflict(Exception):
    """Same doc_id and version arrived with different content."""


def upsert_document(conn, doc_id, version, title, body) -> str:
    _validate(doc_id, version)
    paragraphs = split_paragraphs(body) if isinstance(body, str) else []
    if not isinstance(title, str) or not paragraphs:
        raise ValueError("title must be a string and body must contain text")
    new_hash = content_hash(title, body)
    with write_transaction(conn):
        current = _current(conn, doc_id)
        outcome = _decide(current, version, new_hash)
        if outcome in ("inserted", "updated"):
            _delete_chunks(conn, doc_id)
            _insert_chunks(conn, doc_id, version, paragraphs)
            _save_document(conn, doc_id, version, new_hash, title, deleted=0)
            _bump_corpus_version(conn)
    return outcome


def delete_document(conn, doc_id, version) -> str:
    _validate(doc_id, version)
    with write_transaction(conn):
        current = _current(conn, doc_id)
        outcome = _decide(current, version, TOMBSTONE)
        if outcome in ("inserted", "updated"):
            _delete_chunks(conn, doc_id)
            _save_document(conn, doc_id, version, TOMBSTONE, None, deleted=1)
            _bump_corpus_version(conn)
            outcome = "deleted"
    return outcome


def _validate(doc_id: object, version: object) -> None:
    if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > 200:
        raise ValueError("doc_id must be a non-empty string up to 200 chars")
    if type(version) is not int or not 1 <= version <= MAX_VERSION:
        raise ValueError(f"version must be an integer from 1 to {MAX_VERSION}")


def _current(conn, doc_id) -> sqlite3.Row | None:
    sql = "SELECT version, content_hash FROM documents WHERE doc_id = ?"
    return conn.execute(sql, (doc_id,)).fetchone()


def _decide(current, version, new_hash) -> str:
    if current is None:
        return "inserted"
    if version < current["version"]:
        return "stale"
    if version == current["version"]:
        if current["content_hash"] == new_hash:
            return "unchanged"
        raise VersionConflict(f"version {version} already stored with other content")
    return "updated"


def _delete_chunks(conn, doc_id) -> None:
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def _insert_chunks(conn, doc_id, version, paragraphs) -> None:
    rows = [(text, doc_id, version, n) for n, text in enumerate(paragraphs)]
    conn.executemany(
        "INSERT INTO chunks (text, doc_id, version, chunk_no) VALUES (?, ?, ?, ?)", rows
    )


def _save_document(conn, doc_id, version, digest, title, deleted) -> None:
    conn.execute(
        "INSERT INTO documents (doc_id, version, content_hash, title, deleted)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(doc_id) DO UPDATE SET version = excluded.version,"
        " content_hash = excluded.content_hash, title = excluded.title,"
        " deleted = excluded.deleted",
        (doc_id, version, digest, title, deleted),
    )


def _bump_corpus_version(conn) -> None:
    conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'corpus_version'")
