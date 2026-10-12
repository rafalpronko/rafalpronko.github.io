# Data contracts and validation: synthetic invoice-lines example

Companion code for the article "Why a model cannot fix bad data: contracts
and validation". All data is synthetic. The tax rates and the half-up rounding
rule belong to this example contract; they are not tax advice for any
jurisdiction.

## Files

- `invoice_contract.py` - the hard contract: envelope, exact row fields,
  exact `int` types, bounds, identifiers, tax/gross relations, duplicate keys.
  `validate_batch()` returns a list of errors and never raises on bad input.
  It reports every independently checkable error: envelope errors stop before
  the rows are read, and malformed values skip the relation checks that
  depend on them.
- `invoice_store.py` - SQLite schema with `CHECK` constraints, `publish()`
  (validate everything first, then insert the whole batch in one transaction),
  the runtime, connection and baseline guards, and the advisory distribution
  monitor.
- `fixtures.py` - synthetic batches, including the "consistent unit lie" that
  the contract cannot detect, and an invented baseline median (not a benchmark).
- `demo.py` - runs eight scenarios and prints each outcome; exit code 0 means
  every outcome matched expectations.
- `tests/` - pytest suite (86 test cases).

## Requirements

Python 3.11.x only (`requires-python = ">=3.11,<3.12"`). The transaction
checks in `invoice_store.py` describe the Python 3.11 `sqlite3` transaction
model; later Python versions add transaction controls those checks do not
cover. `connect()` and `publish()` therefore raise `RuntimeError` on any other
Python version before anything is written. The code uses only the standard
library. The tests need pytest; the lock file pins pytest 7.4.2 (released
2023-09-07) and mypy 1.5.1.

## Running

With uv (a modern reproduction tool; it was not part of the 2023 setup):

```
uv run --python 3.11 --locked pytest -q
uv run --python 3.11 --locked python demo.py
uv run --python 3.11 --locked mypy --strict invoice_contract.py invoice_store.py fixtures.py demo.py tests
```

Without uv (equivalent commands; the recorded verification used uv only):

```
python3.11 -m venv .venv
.venv/bin/pip install pytest==7.4.2
.venv/bin/python -m pytest -q
.venv/bin/python demo.py
```

`pyproject.toml` sets `exclude-newer = "2023-10-19T00:00:00Z"`, so uv only
resolves distributions uploaded before that date.

## Scope

Not implemented here: replay acknowledgement (an exact replay leaves the table
unchanged, but the caller receives `IntegrityError`, not "already applied"),
quarantine or retention of rejected batches, freshness and missing-batch
detection, concurrency and crash testing.

`publish()` accepts `baseline_median_net` only as an exact `int` in
`[1, AMOUNT_MAX["net_minor"]]` and checks it before writing, so a call that
passes every check cannot fail in the advisory step after the commit.
