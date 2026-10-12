# Incremental loading, merge and schema evolution — example code

Synthetic, in-memory example for the article
"Incremental loading, merge and schema evolution: where correctness breaks".
It is a teaching example, not a production library.

- Python 3.11+ standard library only (`sqlite3`, `dataclasses`, `re`).
- SQLite 3.24.0+ for the basic UPSERT clause. Check the library your
  interpreter was built with: `python -c "import sqlite3; print(sqlite3.sqlite_version)"`.
- `pytest` is needed only for the tests.
- No SQL `MERGE` statement, no `STRICT` tables, no `RETURNING`, no
  `ALTER COLUMN`; the only schema change is `ALTER TABLE ... ADD COLUMN`.

## Layout

| File | Role |
|---|---|
| `incremental_demo/model.py` | `Change`, `Batch` and the exception types |
| `incremental_demo/source.py` | simulated source: snapshot, window, keyset pagination |
| `incremental_demo/store.py` | schema, nullable `currency` migration, transactions, checkpoint |
| `incremental_demo/merge.py` | contract validation, conditional UPSERT, equal-version policy |
| `incremental_demo/pipeline.py` | incremental run, explicit backfill, one-transaction apply |
| `incremental_demo/demo.py` | printed walkthrough of one timeline |
| `tests/` | pytest scenarios for windows, merge, failures and schema evolution |

## Run

The article links `article-03.zip`, which unpacks to this directory with
every file, including `incremental_demo/__init__.py`. From this directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "pytest==7.4.2"
.venv/bin/python -m pytest -q
.venv/bin/python -m incremental_demo.demo
```

`uv` is a contemporary tool used to reproduce the example in an isolated
environment. It is not claimed to have existed when the article is dated,
and the code does not depend on it: any Python 3.11+ interpreter with
`pytest` installed runs the same commands.

## Assumptions the code relies on

- Per key, `updated_at` strictly increases with every modification, and each
  extract reads one consistent snapshot. The pipeline then converges to the
  source's *current* state; versions replaced before an extract are not kept.
- Visibility delay is bounded by the `overlap` passed to a run.
- The target starts empty, so the first run's lower bound
  (`start_watermark - overlap`) must lie at or below the oldest `updated_at`
  the target has to hold, or older rows must be loaded first with an explicit
  backfill below the watermark. Rows older than that bound are otherwise
  never read.
- A same-version currency enrichment is trusted as the source's value for that
  version; the code cannot tell it from a guess.

## What is not covered

Concurrent runs, process crashes, a COMMIT whose outcome is unknown, disk
failures, a real network source, clock skew between source and pipeline, and
SQL engines other than SQLite.
