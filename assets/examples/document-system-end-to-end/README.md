# Document system end to end: runnable example

Companion code for the article "A document system end to end: ingestion,
search, API and evaluation". A small lexical document retrieval system in
standard library Python with SQLite FTS5. Synthetic data only. It is not a
RAG pipeline (nothing is generated) and not a production server.

## Compatibility

- Python 3.12 (`requires-python = ">=3.12,<3.13"`). No runtime dependencies.
- SQLite **3.24.0 or newer** (plain SQL `INSERT ... ON CONFLICT DO UPDATE`,
  released 2018-06-04) as linked into the Python that uv selects, **with FTS5
  enabled**. FTS5 is a compile-time option, so check it rather than trusting
  the version number (command below).
- pytest 8.3.3 (dev dependency, pinned in `uv.lock`). It is declared in a
  `[dependency-groups]` table (PEP 735), which uv reads and older tools
  ignore; without uv, install pytest 8.3.3 yourself and run
  `python -m pytest` from this directory.
- [uv](https://docs.astral.sh/uv/) to create the environment.
- Verified on 13 October 2026 on macOS: tests, demo and the commands below
  with CPython 3.12.11 / SQLite 3.47.1 (uv 0.7.8); the test suite only with
  CPython 3.12.8 / SQLite 3.45.3. Other platforms were not tested.
- `uv.lock` pins packages resolved in 2026 (for example packaging 26.3). It
  is a record of the 2026 run, not a promise of a 2024 environment.

## Layout

| Path | Responsibility |
| --- | --- |
| `examples/docsys/schema.py` | connection, schema (ordinary FTS5 table), explicit transactions |
| `examples/docsys/chunking.py` | paragraph chunks, length-prefixed SHA-256 content hash |
| `examples/docsys/store.py` | versioned idempotent upsert, versioned tombstone delete, one transaction per write |
| `examples/docsys/query.py` | restricted query: `[a-z0-9]` terms, each emitted as a quoted FTS5 string |
| `examples/docsys/search.py` | `bm25()` ranking (lower is better), deterministic ties, corpus version in a read snapshot |
| `examples/docsys/evaluate.py` | retrieval-only macro recall@k, MRR@k, unanswerable queries (run on an unchanging corpus) |
| `examples/docsys/api.py` | `http.server` JSON API, loopback only |
| `examples/demo.py` | ingest, replay, conflict, delete, stale replay, search, evaluate, one HTTP request |
| `examples/tests/` | 61 pytest tests |
| `examples/fixtures/` | synthetic corpus (4 documents) and 6 evaluation queries |

## Run

From the directory that contains `pyproject.toml`:

```bash
# 1. Creates the environment (Python 3.12, pytest) and checks SQLite in the
#    interpreter uv selected: fails if older than 3.24.0 or FTS5 is missing
uv run --python 3.12 python -c "import sqlite3 as s; assert s.sqlite_version_info >= (3, 24, 0), s.sqlite_version; s.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('SQLite', s.sqlite_version, 'with FTS5')"

# 2. Tests
uv run pytest -v          # expected: 61 passed

# 3. Demo (uses a temporary database that is removed at the end)
uv run python examples/demo.py
```

Without the project environment, against a specific interpreter:

```bash
uv run --isolated --no-project --python /path/to/python3.12 --with pytest==8.3.3 pytest -q
```

## Manual server

Start your own server on an unused port. `port=0` lets the OS pick a free one;
the command prints that port and the server's process id, binds to
`127.0.0.1` only and creates `docs.db` in the current directory.

```bash
uv run python -c "import os, sys; sys.path.insert(0, 'examples'); from docsys.api import make_server; s = make_server('docs.db', port=0); print('pid', os.getpid(), 'port', s.server_address[1], flush=True); s.serve_forever()"
```

In a second terminal, set `PORT` to the printed port and check that the
listener is exactly that process before sending anything:

```bash
PORT=<printed port>
lsof -nP -iTCP:$PORT -sTCP:LISTEN     # PID must match the printed pid
curl -s 127.0.0.1:$PORT/health        # must print {"status": "ok"}
```

Only if both checks pass:

```bash
curl -s -X PUT 127.0.0.1:$PORT/documents/d1 -d '{"version":1,"title":"T","body":"alpha\n\nbeta"}'
curl -s "127.0.0.1:$PORT/search?q=alpha&limit=5"
curl -s -X DELETE "127.0.0.1:$PORT/documents/d1?version=2"
curl -s "127.0.0.1:$PORT/search?q=alpha"
```

If the server failed to start, or another PID owns the port, do not send the
requests. Stop the server with Ctrl+C. Without `port=...` the default is
8780; use it only after the same two checks.

API summary: `GET /health`, `GET /search?q=&limit=` (limit 1 to 50, default 5),
`PUT /documents/<doc_id>` with `{"version", "title", "body"}`,
`DELETE /documents/<doc_id>?version=<n>`. Versions are integers from 1 to
2^63-1 (SQLite INTEGER range). Errors are `{"error": "..."}` with 400
(invalid input, including a version outside that range), 404, 409 (same
version, different content), 413 (`Content-Length` header over 1,000,000
bytes), 500 (generic message for unexpected errors) or 503 (`database
unavailable`, see below).

If a failed request leaves the SQLite connection closed or inside a
transaction (for example `COMMIT` and then `ROLLBACK` both fail), that request
gets 500, the server closes the connection, and `/health` and every database
request answer 503 until you restart it. There is no automatic reconnect.

## Cleanup

```bash
rm -f docs.db docs.db-wal docs.db-shm   # only if you ran the manual server
rm -rf .venv .pytest_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

## Known limits

ASCII-only query terms (`on-call` is rejected; `oncall` does not match the
tokenised `on call`), no stemming, implicit AND only, titles are not indexed,
no authentication, TLS or ACL, no request timeouts, rate limits or concurrency
controls (a client that sends a slow or short body blocks the single-threaded
server), chunk deletion scans the FTS5 table by an UNINDEXED column, deletes
are logical (not forensic erasure), tombstones are kept forever, no rebuild or
backfill tooling, no reconnect after the server has discarded a broken
database connection.
