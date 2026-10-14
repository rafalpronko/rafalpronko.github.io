---
layout: post
lang: en
title: "A document system end to end: ingestion, search, API and evaluation"
date: 2024-10-10 09:00:00 +0200
permalink: /blog/document-system-end-to-end/
author: Rafał Prońko
categories: [search, data-engineering]
description: "A small runnable document retrieval system in Python 3.12 and SQLite FTS5: versioned ingestion, atomic index writes, tombstones, a restricted query grammar, a local JSON API and a retrieval evaluation with an unanswerable question."
image: /assets/images/document-system-end-to-end.svg
---

# A document system end to end: ingestion, search, API and evaluation

![Diagram of the example system: a source document with id and version is split into paragraph chunks and written in one SQLite transaction to a documents table, an FTS5 chunks table and a corpus version counter. Search ranks chunks with bm25 inside one read snapshot. A loopback-only JSON API writes through the same transaction and reads through search, and a retrieval evaluation calls search with an unanswerable question among its cases.]({{ '/assets/images/document-system-end-to-end.svg' | relative_url }})

<time datetime="2024-10-10">10 October 2024</time>

> **About the dates.** 10 October 2024 is the editorial date of this article in the series. The code was written and run on 13 October 2026. Every command shown below ran on CPython 3.12.11 with SQLite 3.47.1; the test suite alone was also run on CPython 3.12.8 with SQLite 3.45.3. The outputs are from those runs, not from 2024. The example uses only APIs that existed by the editorial date: Python 3.12.0, the FTS5 features of its first release (SQLite 3.9.0, 2015) and plain SQL UPSERT (SQLite 3.24.0, 2018). The compatibility section lists what was checked.

"Search over our documents" sounds like one feature. In practice it is a chain of contracts: what counts as a document and its version, what happens when the same update arrives twice, what a reader may see while an update is half written, which query syntax users are allowed to type, and how you know that a change to chunking did not make results worse.

In this article I build that chain in a small, complete form. There is no language model and no generated answer. It is a lexical retrieval baseline: documents are split into paragraphs, indexed with SQLite FTS5 and ranked with `bm25()`. A baseline like this is worth having before anything more complex, because it gives you a working ingestion path, an API contract and an evaluation set that later variants have to beat.

The whole project is standard library Python plus pytest for the tests. You can download it as an archive: [document-system-end-to-end.zip]({{ '/assets/examples/document-system-end-to-end.zip' | relative_url }}) ([README]({{ '/assets/examples/document-system-end-to-end/README.md' | relative_url }})). Below I show only the fragments that carry a decision.

## The pieces and their contracts

The project has seven small modules, none longer than 180 lines:

| Module | Responsibility |
| --- | --- |
| `chunking.py` | split a body into paragraph chunks, compute a content hash |
| `schema.py` | connection, schema, explicit write and read transactions |
| `store.py` | versioned upsert and versioned delete, one transaction per write |
| `query.py` | restricted query grammar translated into an FTS5 expression |
| `search.py` | `bm25()` ranking with deterministic ties and the corpus version |
| `evaluate.py` | retrieval-only macro recall@k, MRR@k and unanswerable questions |
| `api.py` | JSON API on `http.server`, loopback only |

The input contract is deliberately narrow. A document is a `doc_id` (a non-empty string up to 200 characters), a `version` (an integer from 1 to 2^63−1, the positive half of SQLite's signed 64-bit `INTEGER` range), a `title` (a string) and a `body` (text with at least one non-blank paragraph). The version is **supplied by the source system**. The index does not invent versions and does not use arrival time to decide which write is newer, because arrival order is exactly what retries and backfills break.

Extraction is outside this contract. PDF parsing, OCR or HTML cleanup would be a separate adapter that produces `(doc_id, version, title, body)`. I have not implemented such an adapter here, and nothing in the evaluation below says anything about extraction quality.

## Ingestion: paragraphs and a content fingerprint

Chunking is the simplest thing that keeps a citation meaningful: split on blank lines, collapse internal whitespace, drop empty blocks. Each paragraph becomes one row in the index with `chunk_no` starting at 0, so a hit points to a specific place in a specific version of a document.

```python
def split_paragraphs(body: str) -> list[str]:
    paragraphs = (" ".join(block.split()) for block in _BLANK_LINE.split(body))
    return [paragraph for paragraph in paragraphs if paragraph]
```

Next to the chunks, every document gets a SHA-256 hash of its title and raw body. Each part is prefixed with its length, so `("a", "bc")` and `("ab", "c")` produce different hashes. The hash is computed over the raw body, not over the normalised paragraphs. That makes the contract strict: if the source sends version 1 again with only different whitespace, it is treated as different content. I prefer a loud conflict to silently deciding that two payloads are "basically the same".

The title is stored in the `documents` table but is not indexed. Searching titles would be a reasonable extension; it is not in this example.

## One transaction per document write

The schema has three tables:

```sql
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
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
```

`chunks` is an ordinary FTS5 table that stores its own content. FTS5 also supports external-content tables, where the index points at rows of another table and you keep the two in sync yourself. That saves space, but a missed or misordered sync leaves an index that disagrees with its content. For a baseline I prefer to pay with duplicated text and get one place where chunk text and its index entries cannot drift apart. `UNINDEXED` columns are stored and returned but are not searchable, which is exactly what metadata like `doc_id` should be.

The important property is that a document write is atomic **including the decision**. The connection is opened with `isolation_level=None`, so Python does not open transactions implicitly, and the code issues them itself:

```python
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
```

`COMMIT` sits inside the `try`. SQLite's documentation warns that a failed `COMMIT` can leave the transaction open, and on the server's single shared connection that would make every later `BEGIN` fail with "cannot start a transaction within a transaction". So a failure in the body *or* in `COMMIT` rolls back, but only if a transaction is still open, and a failing `ROLLBACK` does not replace the original exception. If `ROLLBACK` fails and the transaction is still open, the helper closes the connection instead of handing a broken one back: `is_usable()` returns `False` for a connection that is closed or still inside a transaction, `close_quietly()` swallows any error from `close()` so the original exception still wins, and the API uses the same check to stop serving (see below). `read_snapshot`, the read-side helper, has the same shape. `connect()` follows the same rule for setup: if a `PRAGMA` or the schema script fails after the connection was opened, it closes the connection before re-raising the original error.

```python
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
```

`BEGIN IMMEDIATE` takes SQLite's write lock at the start, before the current version is read. Reading the version outside the transaction and writing inside it would leave a window where two writers both see version 1 and both decide they are newer. Inside one transaction the code reads the stored version, deletes all old chunks, inserts the new ones, updates the metadata row and increments the corpus version counter. Either all of that is committed or none of it is.

Three tests check the error paths. One replaces `_insert_chunks` with a function that inserts the first new paragraph and then raises. After the exception the document still has its two old version-1 chunks, the metadata row still says version 1 and the corpus version has not moved. A second raises just before `COMMIT`. A third wraps the connection in a proxy that makes the next `COMMIT` raise once; afterwards the old chunks, metadata and corpus version are intact, the connection is not left in a transaction, and the next write and search succeed. A fourth makes the following `ROLLBACK` raise as well: the caller still sees the `COMMIT` error, the connection is closed, and a fresh connection to the same file finds version 1 and the old corpus version. Six more make each `connect()` setup step fail, with and without a failing `close()`, and check that the original error propagates and the connection is closed. These are exceptions raised in Python. They do not simulate I/O errors, a full disk or a crash inside SQLite; for those the example relies on SQLite's atomic commit under the durability settings the connection runs with (the code does not change `synchronous`), which no test here exercises.

## Versions, conflicts and tombstones

The decision for each write is a small function:

| Incoming write compared to stored state | Result |
| --- | --- |
| no stored row | `inserted` (or `deleted` for a delete) |
| lower version | `stale`, nothing changes |
| same version, same hash | `unchanged`, nothing changes |
| same version, different hash | `VersionConflict`, transaction rolled back |
| higher version | `updated` (or `deleted`), all chunks replaced |

A delete is not a `DELETE FROM documents`. It removes the chunks but **keeps the metadata row** with the delete's version, `deleted = 1` and a SQL `NULL` hash. That row is a tombstone. Without it, this sequence would resurrect a document: v1 is written, v2 deletes it, then a retry of v1 arrives late. With the tombstone, the late v1 is simply `stale`.

The tests cover the less obvious cases as well:

- replaying the same delete returns `unchanged`;
- an upsert with the same version as the tombstone raises a conflict, because "version 2 is a delete" and "version 2 has this content" cannot both be true;
- a delete can arrive *before* the first write. Deleting `d9` at version 3 creates a tombstone; a late upsert at version 2 is `stale`; an upsert at version 4 recreates the document.

Tombstones never expire in this example. A real system needs a retention rule for them, and that rule has to be longer than the longest delay with which an old message can still be replayed. Remove a tombstone too early and you are back to resurrection.

## Query grammar: SQL parameters are not enough

The search query is passed to SQLite as a bound parameter, so it cannot inject SQL. That is a separate question from what FTS5 does with the string. The bound value of `MATCH ?` is parsed by the FTS5 query grammar, which has `OR`, `NOT`, `NEAR(...)`, column filters such as `title:`, prefix queries with `*`, `+` and quoted strings. A user who types `back*` or an unbalanced quote either changes the meaning of the query or gets an FTS5 syntax error, which an API would otherwise surface as a 500.

So the API accepts its own, much smaller grammar and translates it:

```python
_TERM = re.compile(r"[a-z0-9]+")

def to_match_expression(raw: object) -> str:
    return " ".join(f'"{term}"' for term in parse_terms(raw))
```

`parse_terms` lowercases the input, splits it on whitespace and rejects the query if it is empty, has more than 8 terms, or any term is longer than 40 characters or contains anything other than ASCII letters and digits. Each accepted term is wrapped in double quotes, which makes it an FTS5 string rather than an operator. Adjacent strings are combined by FTS5's implicit AND. `NOT Backup` becomes `"not" "backup"`: a search for two words, not a negation.

The cost is visible in the demo. `on-call` is rejected with `unsupported term: 'on-call'`. `oncall` is accepted but finds nothing, because the default tokenizer indexed the text "on-call" as two tokens, `on` and `call`. A user has to type `on call`. That is a real usability gap, and I would rather have it explicit in a baseline than hide it behind a query parser that half-supports FTS5 syntax.

## Ranking: `bm25()`, ties and what statistics mean here

```sql
SELECT doc_id, version, chunk_no, text, bm25(chunks) AS score
FROM chunks
WHERE chunks MATCH ?
ORDER BY score ASC, doc_id ASC, chunk_no ASC
LIMIT ?
```

FTS5's `bm25()` returns the BM25 score multiplied by -1, so better matches have numerically lower values and `ORDER BY score ASC` puts the best chunk first. Equal scores are ordered by `doc_id` and `chunk_no`. Without that, two identical paragraphs could swap places between runs, and a ranking test or an evaluation report would flicker for no reason. One test inserts two documents with identical text in the order `b`, `a` and asserts that `a` comes first.

Two properties of this ranking are worth knowing before you read any number:

- **Statistics are per chunk, not per document.** Document frequency and average length are computed over FTS5 rows, which are paragraphs. A long document contributes many rows, and `LIMIT` applies to chunks, so one document can fill all result slots. There is no per-document grouping.
- **Tiny corpora produce odd scores.** With two indexed paragraphs, `alpha` and `beta`, of which only one contains the term, `alpha` scored `-1e-06`. In the SQLite source the IDF of a term that appears in at least half of the rows is replaced by `1e-6` instead of going negative. Scores are useful for ordering within one query, not as a confidence value or a threshold.

## Snapshot consistency and the corpus version

Every committed write that changes the index increments `meta.corpus_version`. Search reads that counter and runs the `MATCH` query inside one read transaction:

```python
with read_snapshot(conn):
    corpus_version = _corpus_version(conn)
    rows = conn.execute(SEARCH_SQL, (expression, limit)).fetchall()
```

The database runs in WAL mode, so a read transaction sees one consistent state of the database while writers continue. The returned `corpus_version` therefore describes exactly the state the hits came from. Because chunks, metadata and the counter change in one write transaction, a reader can never see new chunks with an old version number, or half of a document's paragraphs.

The guarantee covers one request. Two requests can see different corpus versions, and the API has no pagination or snapshot token. The version in the response is what lets a client notice that the corpus changed between two calls. It is a counter in one database file, not a globally unique snapshot id: a rebuild into a new file starts again from 0, so a production system would pair it with a database generation id.

## The API

The API is built on `http.server` from the standard library. The Python documentation states plainly that `http.server` is not recommended for production and only implements basic security checks. This server is a local harness for the article, not a service design. `make_server` raises `ValueError` for any host other than `127.0.0.1` or `localhost`. The default port is 8780; `port=0` lets the OS pick a free one, which is what the demo and the tests use. `server_close()` closes both the socket and the SQLite connection and is safe to call twice. `DocServer` sets its connection attribute to `None` before calling the `HTTPServer` constructor, because that constructor calls `server_close()` itself when `bind` or `listen` fails; otherwise cleanup would raise `AttributeError` and hide the real error, such as a port already in use. If the database cannot be opened, the already bound socket is closed before the original error propagates. Loopback binding limits who can connect; it is not authentication.

The exact endpoints:

| Request | Success response |
| --- | --- |
| `GET /health` | `200 {"status": "ok"}`, or `503` once the database connection has been discarded |
| `GET /search?q=<terms>&limit=<1..50>` (limit defaults to 5) | `200 {"query", "corpus_version", "hits": [{"doc_id", "version", "chunk_no", "score", "text"}]}` |
| `PUT /documents/<doc_id>` with JSON `{"version", "title", "body"}` | `200 {"status": ...}` with `inserted`, `updated`, `unchanged` or `stale` |
| `DELETE /documents/<doc_id>?version=<n>` | `200 {"status": ...}` with `deleted`, `unchanged` or `stale` |

Errors are JSON objects `{"error": "..."}`: 400 for an invalid query, limit, document or JSON body (including a `version` outside 1..2^63−1), 404 for unknown paths, 409 for a version conflict, 413 when the `Content-Length` header announces more than 1,000,000 bytes (the check uses the header, the server then reads at most that many bytes), 500 with a generic message for anything unexpected and 503 `{"error": "database unavailable"}` after the server has discarded its connection. A stale write is a 200 with `"stale"`, not an error: the source did nothing wrong by delivering an old message, and the caller should not retry it.

The server fails closed when its one connection breaks. After every request it checks the connection; if it is closed or still inside a transaction, which is what a failed `COMMIT` followed by a failed `ROLLBACK` leaves behind, the server closes it and drops it. That request has already returned 500. From then on `/health`, search, `PUT` and `DELETE` answer 503 with `{"error": "database unavailable"}` and touch nothing, until the process is restarted; there is no automatic reconnect. A failure where `ROLLBACK` succeeds keeps the connection, and later requests work normally. Both paths have API tests that inject the failures with the proxy described above. The guarantee is narrow: the server will not run requests on a connection it can see is broken, and `/health` will not report a discarded database as healthy. It is not a database health probe, and it says nothing about I/O errors or crashes, which the tests do not simulate.

This is a manual session recorded on 13 October 2026 against a server started from an earlier version of the README, which used the default port 8780. Before it, the recording checked that port 8780 had no listener and that `/health` answered from this server:

```text
$ curl -X PUT localhost:8780/documents/d1 -d '{"version":1,"title":"T","body":"alpha\n\nbeta"}'
{"status": "inserted"}
$ curl 'localhost:8780/search?q=alpha'
{"query": "\"alpha\"", "corpus_version": 1, "hits": [{"doc_id": "d1", "version": 1, "chunk_no": 0, "score": -1e-06, "text": "alpha"}]}
$ curl -X DELETE 'localhost:8780/documents/d1?version=2'
{"status": "deleted"}
$ curl 'localhost:8780/search?q=alpha'
{"query": "\"alpha\"", "corpus_version": 2, "hits": []}
```

If you repeat it, do not send requests to a fixed port blindly. The current README starts the server with `port=0`, prints its process id and port, and asks you to confirm with `lsof` that this process owns the listener and that `/health` answers before any `PUT`. I followed that procedure on the extracted archive with an OS-chosen port; the same four requests gave the same responses.

The server uses `HTTPServer`, which handles one request at a time over one SQLite connection. That keeps the example free of connection-per-thread questions. It also means one slow request blocks everyone else: a client that sends a `Content-Length` larger than its body, or sends the body slowly, holds the server until it gives up, because there are no request timeouts.

## Evaluation: retrieval, with a question that has no answer

The fixture corpus is four synthetic documents: a backup runbook, a retention policy, a deployment guide and an on-call FAQ. The query set has six entries. Five name the expected `(doc_id, chunk_no)`; one, `kubernetes autoscaling`, has an empty list because nothing in the corpus answers it.

`evaluate()` reports three numbers. `recall_at_k` is macro recall: for each answerable query, the share of its unique relevant `(doc_id, chunk_no)` pairs found in the top k, averaged over those queries. If a query has two relevant chunks and only one comes back, it contributes 0.5, not 1.0; a regression test checks exactly that case, a mixed average and duplicate entries in the expected list. `mrr` is MRR@k: the mean of 1/rank of the first relevant chunk, counted as 0 when none is in the top k. The third is the number of unanswerable queries that still returned hits. Each query is a separate search call, and nothing pins one snapshot across the run, so evaluate an unchanging corpus. For a lexical engine with implicit AND, "returned hits" is a meaningful signal: if all terms appear somewhere, it returns something, relevant or not.

The test on the full corpus asserts 5 answerable, 1 unanswerable, recall@3 = 1.0, MRR@3 = 1.0 and 0 unanswerable queries with hits. The demo deletes `faq-oncall` first and removes its query, so it reports four answerable queries:

```text
evaluation (faq-oncall deleted, its case removed):
{
  "k": 3,
  "answerable": 4,
  "unanswerable": 1,
  "recall_at_k": 1.0,
  "mrr": 1.0,
  "unanswerable_with_hits": 0,
  "misses": []
}
```

Read these numbers for what they are. I wrote the queries myself, using words that appear in the target paragraphs, against a corpus of eleven paragraphs. A perfect score here is not evidence that BM25 over paragraphs is good for any real collection. It is a **regression gate**: if a change to chunking, tokenisation or query parsing drops one of these cases, the test fails and the `misses` list says which.

It is also only retrieval evaluation. It checks whether the right paragraph comes back. It does not check whether that paragraph is sufficient evidence for an answer, and there is no answer to evaluate. If a generator is added later, those are two further evaluations with their own data: evidence support and answer quality. A good retrieval score does not imply either.

## Running it

The project expects Python 3.12, SQLite 3.24.0 or newer with FTS5, and [uv](https://docs.astral.sh/uv/) for the environment. pytest 8.3.3 is the only dependency, in the dev group. That group is declared in a `[dependency-groups]` table (PEP 735), a packaging standard newer than the APIs this article relies on and ignored by older tools. It is part of how I ran the project in 2026, not of the 2024 claim. Without uv, install pytest 8.3.3 yourself and run `python -m pytest` from the project root.

```bash
unzip document-system-end-to-end.zip
cd document-system-end-to-end
uv run --python 3.12 python -c "import sqlite3 as s; assert s.sqlite_version_info >= (3, 24, 0), s.sqlite_version; s.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('SQLite', s.sqlite_version, 'with FTS5')"
uv run pytest -v
uv run python examples/demo.py
```

The command after `cd` creates the environment and checks SQLite in the same interpreter that uv selected for the project: it fails if SQLite is older than 3.24.0 or FTS5 is not enabled. A version number alone does not tell you about FTS5, because it is a compile-time option. The demo uses a temporary database and removes it; the README lists cleanup for the manual server.

Recorded results from 13 October 2026:

- `uv sync --python 3.12` with uv 0.7.8 selected CPython 3.12.11 (SQLite 3.47.1) and installed pytest 8.3.3, pluggy 1.6.0, iniconfig 2.3.1 and packaging 26.3.
- The tests were written before the implementation. The first run failed at collection with `ModuleNotFoundError: No module named 'docsys.schema'`.
- The first run after implementation: 34 passed, 6 failed. All six failures came from a test helper that compared `sqlite3.Row` objects with tuples; the helper was fixed, not the system code.
- Then `40 passed in 1.66s`. A second interpreter, CPython 3.12.8 with SQLite 3.45.3, run with `uv run --isolated --no-project --python /usr/local/bin/python3.12 --with pytest==8.3.3 pytest -q`, also gave `40 passed`.
- Ten later regression tests cover a failing `COMMIT`, macro recall with several relevant chunks, the `version` range, server cleanup and an encoded `/` in a `doc_id`. Against the first implementation eight of them failed (`8 failed, 42 passed`): a failed `COMMIT` left the transaction open, `recall_at_k` counted any hit as full recall, a `version` above 2^63−1 gave a 500 and `server_close()` left SQLite open. After the fixes, `50 passed` on both interpreters, and the demo output did not change apart from the random port.
- Eleven further tests cover a `ROLLBACK` that fails after a failed `COMMIT` (in the store and through the API), the API path where `ROLLBACK` succeeds, a `bind` or `listen` failure in the server constructor, and a failing `connect()` setup step. Against the previous code ten of them failed (`10 failed, 51 passed`): the connection stayed open, the bind error was replaced by `AttributeError`, and a failed setup step left the connection open. The one that already passed checks that an ordinary failure does not switch the server off. After the fixes, `61 passed` on both interpreters, and the demo output again did not change apart from the random port.

Part of the demo output:

```text
replay same version: unchanged
conflicting v1: version 1 already stored with other content
delete faq-oncall v2: deleted
stale replay faq-oncall v1: stale

search 'restore snapshot' -> match "restore" "snapshot" (corpus_version=5)
  -2.211288  runbook-backup#1 v1  Restore drills happen every quarter. The on-call engineer re

search 'on-call' rejected: unsupported term: 'on-call'

search 'oncall' -> match "oncall" (corpus_version=5)
  (no hits)
```

## What this example does not do

Everything above holds for a single SQLite file on one machine. These are the gaps I would have to close before calling it a service, roughly in order of how badly they would hurt.

**Access control before retrieval.** There are no users and no permissions. In a real document system, permission filtering has to happen *inside* the retrieval query, before ranking and before `LIMIT`. Filtering after retrieval leaks information through counts and scores and returns fewer than `limit` results. If the retrieved text is later passed to a language model, an unauthorised paragraph that reached the prompt has already leaked.

**Retrieved text is untrusted input.** The documents come from a source system, and the API returns their text verbatim. If a later stage puts that text into a prompt, a paragraph can contain instructions. The restricted query grammar protects FTS5 from the user; nothing here protects a downstream model from the documents.

**Authentication, TLS, timeouts and concurrency.** There is no authentication or TLS. The only limits are the ones in the code: request bodies up to 1,000,000 bytes, at most 8 query terms and at most 50 hits. There are no production-grade request timeouts, rate limits or concurrency controls. Writes are serialised by the SQLite write lock with a 5-second busy timeout, and the HTTP server is single-threaded, so one slow client blocks it.

**Recovery, rebuild and backfill.** Atomicity across a crash during a write rests on SQLite's transaction and WAL under the connection's durability settings; the tests only raise exceptions, they do not crash the process. A lost or corrupted database file is not covered by anything in this example. A server whose connection broke stays at 503 until someone restarts it; there is no reconnect or alerting. Because the index is derived data, a rebuild means creating a new database file, replaying a full export of the source with its versions, comparing each `doc_id` and version against the source, and then pointing the server at the new file. The new file's `corpus_version` starts again from 0. That export must include deletes. If you rebuild only from current documents, the tombstones are gone, and a late replay of an old version can resurrect a deleted document. A backfill of history is safe to replay because older versions come back as `stale`, but it replays one document at a time; there is no bulk loader, no progress record and no rebuild command in the project.

**Index maintenance and deletion.** `DELETE FROM chunks WHERE doc_id = ?` filters on an `UNINDEXED` column, which means a scan of the FTS5 table on every update. That is fine for this fixture and would be the first performance problem on a large corpus. Tombstones accumulate without a retention rule. A delete is logical, not forensic erasure: removed text can remain in free database pages or the WAL file until SQLite overwrites them.

**Language.** The query grammar accepts only ASCII letters and digits, so `zażółć` is rejected, and there is no stemming.

## Compatibility and dates

The example is meant to be valid for its editorial date, so I checked when the primitives it relies on became available rather than relying on current documentation alone:

- FTS5 first shipped in SQLite 3.9.0, released on 14 October 2015, as an option enabled with `SQLITE_ENABLE_FTS5`. In the 3.9.0 source, `UNINDEXED` is already parsed, `bm25` is registered as an auxiliary function and multiplies its score by -1, and adjacent phrases are joined by implicit AND. The current FTS5 page has gained features since then; none of the newer ones are used.
- Python 3.12.0 was released on 2 October 2023. `isolation_level=None` with explicit `BEGIN`/`COMMIT`, and the `http.server` classes used here, are in its documentation.
- pytest 8.3.3 was uploaded to PyPI on 10 September 2024.
- The code also uses `ON CONFLICT ... DO UPDATE` in plain SQL. That UPSERT syntax arrived in SQLite 3.24.0, released on 4 June 2018, so the real requirement is SQLite 3.24.0 or newer **and** FTS5 enabled, not just 3.9.0.

uv is the tool I used to run the project in 2026. It is not part of the 2024 claim, and the project has no runtime dependencies besides the standard library. The dates above cover the APIs only. `uv.lock` pins the versions resolved in 2026, including transitive pytest dependencies such as packaging 26.3; it records the environment that ran, not a promise of a working 2024 environment.

## Where to go from here

The useful part of a baseline is not its score. It is the set of contracts that every later variant has to keep: versions supplied by the source, one atomic write per document, tombstones for deletes, a query grammar that does not expose the engine's grammar, ranking with deterministic ties, a corpus version on every response, and an evaluation set with at least one question that should return nothing.

If I replace BM25 with embeddings or add an answer generator, those contracts stay. The evaluation grows into three: retrieval, evidence and answer.

### Sources

- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html): query syntax, strings and implicit AND, `UNINDEXED`, external-content tables, `bm25()` and its sign. Read in 2026; the page is current, not a 2024 snapshot.
- [SQLite 3.9.0 release notes](https://www.sqlite.org/releaselog/3_9_0.html): FTS5 introduced on 14 October 2015.
- [SQLite 3.24.0 release notes](https://www.sqlite.org/releaselog/3_24_0.html): PostgreSQL-style UPSERT added on 4 June 2018.
- [SQLite transactions](https://www.sqlite.org/lang_transaction.html): `BEGIN IMMEDIATE`, read transactions and the warning that a failed `COMMIT` may leave the transaction active.
- SQLite 3.9.0 source: [fts5parse.y](https://raw.githubusercontent.com/sqlite/sqlite/version-3.9.0/ext/fts5/fts5parse.y), [fts5_aux.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.9.0/ext/fts5/fts5_aux.c), [fts5_config.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.9.0/ext/fts5/fts5_config.c): implicit AND, the `bm25` implementation and the `UNINDEXED` option as they existed in 2015.
- [Python 3.12: sqlite3](https://docs.python.org/3.12/library/sqlite3.html): placeholders and transaction control.
- [Python 3.12: http.server](https://docs.python.org/3.12/library/http.server.html): server classes and the warning that the module is not recommended for production.
- [Python 3.12.0 release](https://www.python.org/downloads/release/python-3120/): released on 2 October 2023.
- [pytest 8.3.3 on PyPI](https://pypi.org/project/pytest/8.3.3/): uploaded on 10 September 2024.
