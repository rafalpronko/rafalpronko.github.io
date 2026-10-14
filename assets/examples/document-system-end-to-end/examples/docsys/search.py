"""Lexical retrieval over paragraph chunks with bm25()."""

from .query import to_match_expression
from .schema import read_snapshot

MAX_LIMIT = 50

# bm25() returns lower (more negative) values for better matches, so ascending
# order puts the best chunk first; doc_id and chunk_no make ties deterministic.
SEARCH_SQL = """
SELECT doc_id, version, chunk_no, text, bm25(chunks) AS score
FROM chunks
WHERE chunks MATCH ?
ORDER BY score ASC, doc_id ASC, chunk_no ASC
LIMIT ?
"""


def search(conn, raw_query, limit=5) -> dict:
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    expression = to_match_expression(raw_query)
    with read_snapshot(conn):
        corpus_version = _corpus_version(conn)
        rows = conn.execute(SEARCH_SQL, (expression, limit)).fetchall()
    return {
        "query": expression,
        "corpus_version": corpus_version,
        "hits": [_hit(row) for row in rows],
    }


def _corpus_version(conn) -> int:
    sql = "SELECT value FROM meta WHERE key = 'corpus_version'"
    return conn.execute(sql).fetchone()[0]


def _hit(row) -> dict:
    return {
        "doc_id": row["doc_id"],
        "version": row["version"],
        "chunk_no": row["chunk_no"],
        "score": round(row["score"], 6),
        "text": row["text"],
    }
