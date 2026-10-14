# Checkpoint and recovery demo (synthetic, single worker)

Companion code for the article "State, checkpoints and recovery: what must
survive a restart". Python 3.11 standard library only (`sqlite3`, `json`,
`hashlib`, `argparse`). pytest is the only test dependency. The SQLite
library linked into Python must be 3.24.0 or newer (for `ON CONFLICT ... DO
NOTHING`); check with `python3.11 -c 'import sqlite3; print(sqlite3.sqlite_version)'`.

This is a teaching example, not a production library. It does not provide
exactly-once external effects, it has no lease or fencing for multiple
workers, and its tests kill a *process*; they do not simulate power loss or
a lying disk.

## Files

| File | Purpose |
|---|---|
| `worker.py` | Four-step workflow (`normalize`, `classify`, `summarize`, `notify`). Step output, operation acknowledgement and cursor are committed in one SQLite transaction. |
| `receiver.py` | Synthetic "external system" with an independent SQLite connection. Deduplicates by operation id and rejects a reused id with a different payload. |
| `tests/test_worker.py` | Runs the worker as a child process, kills it with `os._exit(77)` at chosen points, restarts it and checks the stored state. |
| `demo.sh` | The same scenarios as a readable transcript. |

The receiver commits through its own connection, outside the worker's
checkpoint transaction. The separate file itself is not the barrier: in
rollback-journal mode (the default) SQLite can commit atomically across files
attached to one connection with `ATTACH`. The independent connection stands in
for an external API that no local transaction can include.

An `operation_id` must identify one logical operation uniquely within the
receiver's namespace. The worker builds it as `run_id:step:name`, so choose a
`run_id` unique across all state files that send to that receiver. A new
logical run needs a new id; a restart or retry keeps the original id. The
receiver rejects a reused id with a different payload, but the same payload
would be treated as a retry even if the caller meant a new operation.

## Run

The commands use [uv](https://docs.astral.sh/uv/) as present-day
reproduction tooling.

```bash
uv run --python 3.11 --extra test pytest -v   # 21 tests
bash demo.sh
```

Without uv, any Python 3.11 works for the worker itself. From the examples
directory, create the same `tickets.json` input used by `demo.sh`, then run:

```bash
cat > tickets.json <<'JSON'
[{"id": "T-3", "text": "  Refund for DOUBLE charge  "},
 {"id": "T-1", "text": "Cannot log in after password reset"},
 {"id": "T-2", "text": "Invoice shows the wrong VAT number"}]
JSON
python3.11 worker.py --db state.db --receiver receiver.db \
    --input tickets.json --run-id run-1
```

## Crash points

Set `CRASH_AT=<point>:<step>` to make the worker call `os._exit(77)`:

| Point | Where |
|---|---|
| `before_commit:<step>` | Inside the transaction, after the writes, before `COMMIT` |
| `after_commit:<step>` | Right after `COMMIT` returns |
| `after_intent:notify` | Operation recorded as `pending`, nothing sent yet |

`before_commit:notify` is the "lost acknowledgement" case: the receiver has
applied the effect, the worker has not recorded it.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Run finished (or was already finished) |
| 3 | Checkpoint is incompatible: different input digest, workflow version or schema version |
| 4 | State file unreadable as SQLite; the worker refuses to start over |
| 77 | Injected crash |
