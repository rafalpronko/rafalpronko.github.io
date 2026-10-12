---
layout: post
lang: en
title: "A small API-to-database pipeline with dlt — and a rerun test"
date: 2023-10-26 09:00:00 +0200
permalink: /blog/dlt-api-duckdb-rerun/
author: Rafał Prońko
categories: [data-engineering]
tags: [dlt, duckdb, python, idempotency, testing, pipelines]
description: "A paginated HTTP API loaded into DuckDB with dlt 0.3.22: validation, timeouts, merge on a primary key, and tests that run the pipeline twice."
image: /assets/images/dlt-api-duckdb-rerun.svg
---

# A small API-to-database pipeline with dlt — and a rerun test

![Diagram: a local paginated orders API is read by a client with a timeout and validation, dlt extracts, normalizes and merges the rows into DuckDB by order_id; a second run with the same data leaves the business columns unchanged, while an older version loaded later overwrites a newer one unless a version guard drops it.]({{ '/assets/images/dlt-api-duckdb-rerun.svg' | relative_url }})

<time datetime="2023-10-26">26 October 2023</time>

In the [previous post]({{ '/blog/pipeline-idempotencja/' | relative_url }}) I wrote a conditional UPSERT by hand in SQLite. This time I want to see what changes when a library does the loading for me.

The setup is deliberately small: an HTTP API that returns orders page by page, [dlt](https://dlthub.com/) as the loader and DuckDB as the database. One question drives the whole example: **what is in the table after I run the pipeline a second time?**

I answer it with tests, not with a diagram. The tests also show where the answer stops being comfortable.

## What I am building

- A **fake orders API** on `127.0.0.1`. It is synthetic, uses only the standard library and can be switched into failure modes: HTTP 500, a slow answer, HTML instead of JSON, a missing `data` key, a missing `next_page` key, a wrong `page` number, a skipped page.
- A **client** that reads all pages with a bounded timeout and checks every order before it reaches dlt.
- A **dlt pipeline** that merges orders into DuckDB by `order_id`.
- **pytest tests** that run the pipeline twice against a temporary DuckDB file and compare the table.

Versions: Python 3.11, `dlt[duckdb]==0.3.22` (on PyPI since 24 October 2023), `duckdb==0.9.1`, `requests` 2.31, `pytest` 7.4.3. All dependencies are pinned to packages published up to 25 October 2023. Things I say about dlt below refer to this version; newer releases added other merge strategies, so check the documentation of whatever you run.

The complete project — with the fake API, tests and `uv.lock` — is available as a [zip archive]({{ '/assets/examples/dlt-orders-pipeline.zip' | relative_url }}). The core code is also in this post.

## The API contract

Before writing a line of loading code I write down what I expect from the source. A request `GET /orders?page=N` returns:

```json
{"data": [{"order_id": 1, "status": "paid", "amount_cents": 1999,
           "version": 1, "updated_at": "2023-10-20T10:00:00Z"}],
 "page": 1, "next_page": 2}
```

The rules:

- Pages start at 1. Every page body carries `data` (a list), `page` (the number I asked for) and `next_page`. `next_page` is either the next number or `null` on the last page. Anything else — a missing key, a wrong `page`, a jump, a repeat — is a broken contract, not something to guess around.
- `order_id` is a positive integer and identifies the order.
- `status` is one of `pending`, `paid`, `refunded`, `cancelled`.
- `amount_cents` is a non-negative integer. Money in cents, no floats.
- `version` is a positive integer that grows with every change of the order.
- `updated_at` is an ISO 8601 timestamp with a date **and** a time, and with a timezone: `Z` or a numeric offset such as `+02:00`. A date alone, a number or a time without a timezone is rejected.

`version` is an assumption about the source, the same one I made in the SQLite post. dlt does not know about it. You'll see why that matters.

## The client: timeout, pagination, validation

dlt can consume any Python iterable. That makes it tempting to put `requests.get(...).json()["data"]` straight into a resource. I prefer a separate function that I can test without dlt and that fails loudly.

```python
"""HTTP client for the orders API: pagination contract, bounded timeout, business validation."""
from datetime import datetime

import requests

STATUSES = {"pending", "paid", "refunded", "cancelled"}
REQUIRED = ("order_id", "status", "amount_cents", "version", "updated_at")


class ContractError(ValueError):
    """The API answered, but not in the shape (or with the values) we agreed on."""


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def parse_updated_at(value):
    """ISO 8601 with date AND time and an explicit timezone (`Z` or a numeric offset)."""
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("expected an ISO 8601 string with date and time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone required (Z or +HH:MM)")
    return parsed


def validate_order(order):
    missing = [f for f in REQUIRED if f not in order]
    if missing:
        raise ContractError(f"order missing fields: {missing}")
    if not _positive_int(order["order_id"]):
        raise ContractError(f"bad order_id: {order['order_id']!r}")
    if order["status"] not in STATUSES:
        raise ContractError(f"order {order['order_id']}: unknown status {order['status']!r}")
    amount = order["amount_cents"]
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ContractError(f"order {order['order_id']}: bad amount_cents {amount!r}")
    if not _positive_int(order["version"]):
        raise ContractError(f"order {order['order_id']}: bad version {order['version']!r}")
    try:
        parse_updated_at(order["updated_at"])
    except ValueError as exc:
        raise ContractError(f"order {order['order_id']}: bad updated_at {order['updated_at']!r} ({exc})") from None
    return {f: order[f] for f in REQUIRED}  # drop undeclared fields


def fetch_orders(base_url, timeout=5.0, max_pages=100, session=None):
    """Yield validated orders page by page. Any HTTP/contract problem raises; nothing is skipped."""
    http = session or requests.Session()
    seen = set()
    page = 1
    for _ in range(max_pages):
        response = http.get(f"{base_url}/orders", params={"page": page}, timeout=timeout)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            raise ContractError(f"page {page}: body is not JSON") from None
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ContractError(f"page {page}: expected an object with a 'data' list")
        missing = [k for k in ("page", "next_page") if k not in body]
        if missing:
            raise ContractError(f"page {page}: missing pagination fields {missing}")
        if not isinstance(body["page"], int) or isinstance(body["page"], bool) or body["page"] != page:
            raise ContractError(f"page {page}: page echo is {body['page']!r}, expected {page}")
        next_page = body["next_page"]
        if next_page is not None and (not isinstance(next_page, int) or isinstance(next_page, bool)):
            raise ContractError(f"page {page}: next_page={next_page!r} is not an integer or null")
        for raw in body["data"]:
            order = validate_order(raw)
            if order["order_id"] in seen:
                raise ContractError(f"duplicate order_id {order['order_id']} within one run")
            seen.add(order["order_id"])
            yield order
        if next_page is None:
            return
        if next_page != page + 1:
            raise ContractError(f"page {page}: next_page={next_page!r}, expected {page + 1} or null")
        page = next_page
    raise ContractError(f"more than max_pages={max_pages} pages; refusing to continue")
```

A few decisions worth spelling out:

**`timeout=` on every request.** `requests` has no default timeout, so a server that accepts the connection and never answers can hang the job indefinitely. The value here applies to connecting and to each wait for data from the socket, not to the whole download. It bounds a stalled request; it is not a deadline for the run.

**`raise_for_status()`.** Without it, a 500 response with a JSON error body would only fail later, on the shape check (no `data` list), and the error would talk about the contract instead of the server. With it, the run stops on the HTTP status itself. Either way the run fails loudly: a body without a `data` list is a `ContractError` in this client, never an empty page.

**The page loop checks the contract.** Each page body must carry the keys `data`, `page` and `next_page`. A missing `next_page` key is not the same as `next_page: null`: a truncated response would otherwise look like the last page and end the run early without an error. If the `page` echo is not the page I asked for, or `next_page` is not an integer or `null`, I stop. If page 1 says `next_page: 3`, I stop too. Silently following the pointer would lose page 2. `max_pages` is a guard against an API that keeps returning a next page forever.

**Duplicates in one run are an error.** If the same `order_id` arrives twice during one pass, something in the pagination is wrong — for example, the underlying data shifted while I was reading. I want to know about it. This also matters for dlt, as you'll see in the merge section.

**The whole run fails on one bad record.** I don't skip invalid orders and load the rest. Skipping is a valid policy, but then I need a place to put rejected records and someone who looks at them. In a small example the honest choice is to stop.

**`updated_at` needs a date, a time and a timezone.** `datetime.fromisoformat` alone accepts `"2023-10-20"`, and converting with `str()` first would let a number like `20231020` through. The client requires a string with a `T` between date and time and rejects a parsed value without `tzinfo`. `+02:00` and `Z` are both fine; the string is kept as sent.

**Undeclared fields are dropped.** dlt infers the schema from the data. If the API starts sending a new field, I'd rather add it on purpose than have a new column appear on its own.

## The pipeline: one resource, merge on `order_id`

```python
"""Load orders from the HTTP API into DuckDB with dlt (merge on order_id)."""
import os

os.environ.setdefault("RUNTIME__DLTHUB_TELEMETRY", "false")  # no anonymous telemetry from the demo

import dlt  # noqa: E402
import duckdb  # noqa: E402

from orders_pipeline.client import fetch_orders  # noqa: E402

DATASET = "shop"
BUSINESS_COLUMNS = ["order_id", "status", "amount_cents", "version", "updated_at"]


def current_versions(db_path):
    """order_id -> version already in the destination (empty before the first load)."""
    if not os.path.exists(db_path):
        return {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return dict(con.execute(f"SELECT order_id, version FROM {DATASET}.orders").fetchall())
    except duckdb.CatalogException:
        return {}
    finally:
        con.close()


def drop_stale(orders, known):
    """Version guard: skip records whose version is not newer than what is loaded."""
    for order in orders:
        if order["version"] > known.get(order["order_id"], 0):
            yield order


def run_pipeline(base_url, db_path, pipelines_dir, timeout=5.0, version_guard=False):
    orders = fetch_orders(base_url, timeout=timeout)
    if version_guard:
        orders = drop_stale(orders, current_versions(db_path))
    resource = dlt.resource(
        orders,
        name="orders",
        primary_key="order_id",
        write_disposition="merge",
        columns={"updated_at": {"data_type": "text"}},  # keep the API string as-is
    )
    pipeline = dlt.pipeline(
        pipeline_name="orders",
        pipelines_dir=str(pipelines_dir),
        destination="duckdb",
        credentials=str(db_path),
        dataset_name=DATASET,
    )
    return pipeline.run(resource)


def _query(db_path, sql):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def business_rows(db_path, pipelines_dir=None):
    cols = ", ".join(BUSINESS_COLUMNS)
    return _query(db_path, f"SELECT {cols} FROM {DATASET}.orders ORDER BY order_id")


def load_ids(db_path, pipelines_dir=None):
    return sorted(r[0] for r in _query(db_path, f"SELECT DISTINCT _dlt_load_id FROM {DATASET}.orders"))


def completed_loads(db_path):
    """Number of load packages dlt recorded as completed (status 0) in this dataset."""
    return _query(db_path, f"SELECT count(*) FROM {DATASET}._dlt_loads WHERE status = 0")[0][0]
```

(The last three functions — `business_rows`, `load_ids` and `completed_loads` — are small query helpers used by the tests and the demo.)

Ignore `version_guard` for now; it comes back at the end.

The first line disables dlt's anonymous usage telemetry. In 0.3.22 it is on by default (`dlthub_telemetry: bool = True` in `RunConfiguration`), and the environment variable `RUNTIME__DLTHUB_TELEMETRY=false` turns it off. A demo that phones home has no place in a blog post.

`pipelines_dir` keeps dlt's working state — extracted files, schemas, load packages — in a directory I choose. In the tests it's a temporary directory, so the runs do not share state with anything else on the machine.

I also pin `updated_at` as text. Without the hint dlt would try to recognise the string as a timestamp. For this example I only want to show that the business value survives a rerun unchanged, so I keep exactly what the API sent.

### What `pipeline.run` does

`run` hides three steps:

1. **Extract** — iterates over `fetch_orders` and writes the records to files in `pipelines_dir`. This is where the HTTP requests and my validation happen.
2. **Normalize** — infers or updates the schema and adds dlt's own columns. Every row gets `_dlt_load_id` (which load package it came from) and `_dlt_id`. In 0.3.22 `_dlt_id` for a top-level row is random: the source literally says `# row_id is always random, no matter if primary_key is present or not`.
3. **Load** — creates tables if needed and writes the rows to DuckDB. With `write_disposition="merge"` the rows go first to a staging dataset (`shop_staging`) and are then merged into `shop.orders`.

### What “merge” means in this version

“Merge” sounds like `MERGE` or `INSERT … ON CONFLICT`. In dlt 0.3.22 it is neither. In [`sql_jobs.py`](https://github.com/dlt-hub/dlt/blob/0.3.22/dlt/destinations/sql_jobs.py), for a table without child tables, the generated SQL is:

1. `DELETE` from the destination table every row whose primary key exists in the staging table;
2. `INSERT` the staging rows, numbered with `ROW_NUMBER() OVER (PARTITION BY <primary key> ORDER BY (SELECT NULL))` and keeping only row number 1.

So it's **delete-insert by key**. Three consequences:

- A rerun with the same data deletes the three orders and inserts them again. Business columns stay the same, but dlt's columns don't: `_dlt_load_id` and `_dlt_id` are new.
- If one batch contains the same key twice, one row is kept, and `ORDER BY (SELECT NULL)` means there is no rule for which one. That's why my client rejects duplicates inside a run instead of relying on deduplication.
- There is no comparison of anything except the key. The newest row loaded wins — not the row with the highest `version`.

Rows whose keys are *not* in the incoming batch are left untouched: merge deletes only keys present in the staging batch, so the table converges to a union of everything ever loaded. The source cannot remove rows through this pipeline.

The [current dlt documentation on merge loading](https://dlthub.com/docs/general-usage/merge-loading) describes the same default delete-insert behaviour and the same caveat about deduplication within a batch, along with features that 0.3.22 does not have. For this post I rely on the tagged source of 0.3.22.

## The rerun test

The tests start the fake API on a random port and point the pipeline at a DuckDB file in a temporary directory. This is the central one:

```python
def test_identical_replay_preserves_business_columns(api, db):
    run_pipeline(api.url, **db)
    first = business_rows(**db)
    first_load_ids = load_ids(**db)

    run_pipeline(api.url, **db)
    second = business_rows(**db)

    assert len(first) == 3
    assert second == first  # business columns identical, no duplicates
    # dlt metadata is NOT stable: delete-insert rewrote every row with the new load id
    assert load_ids(**db) != first_load_ids
```

`business_rows` selects only the five business columns, ordered by `order_id`. I don't compare `SELECT *`. If I did, the test would fail on `_dlt_load_id` and `_dlt_id`, and I might be tempted to “fix” the test instead of understanding why. The last assertion documents that the rewrite really happens.

The second case is an ordinary incremental change: order 2 becomes `paid` with `version` 2, and order 4 appears.

```python
def test_update_plus_new_row(api, db):
    run_pipeline(api.url, **db)
    api.orders[1].update(status="paid", version=2, updated_at="2023-10-21T09:00:00Z")
    api.orders.append({"order_id": 4, "status": "pending", "amount_cents": 800, "version": 1, "updated_at": "2023-10-21T09:01:00Z"})

    run_pipeline(api.url, **db)
    rows = {r[0]: dict(zip(BUSINESS_COLUMNS, r)) for r in business_rows(**db)}

    assert sorted(rows) == [1, 2, 3, 4]
    assert rows[2]["status"] == "paid" and rows[2]["version"] == 2
    assert rows[4]["amount_cents"] == 800
```

The demo script runs the same three steps and prints the table. Real output, from the reproduction run of the pinned project (the load ids are Unix timestamps of that run):

```text
--- run 1: initial load
    ('order_id', 'status', 'amount_cents', 'version', 'updated_at')
    (1, 'paid', 1999, 1, '2023-10-20T10:00:00Z')
    (2, 'pending', 4500, 1, '2023-10-20T10:05:00Z')
    (3, 'paid', 1200, 1, '2023-10-20T10:10:00Z')
    _dlt_load_id values in orders: ['1791821076.401187']
--- run 2: identical replay
    ('order_id', 'status', 'amount_cents', 'version', 'updated_at')
    (1, 'paid', 1999, 1, '2023-10-20T10:00:00Z')
    (2, 'pending', 4500, 1, '2023-10-20T10:05:00Z')
    (3, 'paid', 1200, 1, '2023-10-20T10:10:00Z')
    _dlt_load_id values in orders: ['1791821076.872548']
--- run 3: update order 2, add order 4
    ('order_id', 'status', 'amount_cents', 'version', 'updated_at')
    (1, 'paid', 1999, 1, '2023-10-20T10:00:00Z')
    (2, 'paid', 4500, 2, '2023-10-21T09:00:00Z')
    (3, 'paid', 1200, 1, '2023-10-20T10:10:00Z')
    (4, 'pending', 800, 1, '2023-10-21T09:01:00Z')
    _dlt_load_id values in orders: ['1791821077.249258']
```

The replay leaves the business rows identical, and every row carries the new load id. In run 3 the unchanged orders 1 and 3 also get the new load id: the whole batch is deleted and inserted, not just the changed row.

## When the source misbehaves

Failure tests use the same pattern: load once, change the source, break it, run again, compare.

```python
@pytest.mark.parametrize("mode", ["not_json", "missing_data", "http_500"])
def test_failed_extract_leaves_destination_unchanged(api, db, mode):
    run_pipeline(api.url, **db)
    before = business_rows(**db)

    api.orders[0]["status"] = "refunded"
    api.mode = mode
    with pytest.raises(Exception):
        run_pipeline(api.url, **db)

    assert business_rows(**db) == before
```

There is also a test in which page 1 is valid and contains a real change, while page 2 has an order with `amount_cents = -5`. The change from page 1 does not reach the table either. One more test covers recovery: after such a failure, the source is fixed and the pipeline runs again. The table is correct, and `_dlt_loads` contains exactly two completed loads — the first one and the recovery. No half-extracted package from the failed attempt was loaded.

Why does this work? All my checks run inside the extract step. When `fetch_orders` raises, dlt never reaches normalize or load, so the destination is not touched. That is a property of *where* I validate, not a general guarantee of dlt. A failure during the load step — a lost database connection in the middle of the merge, for example — is a different case, and these tests do not cover it.

The client tests (no dlt involved) check the rest of the contract: pagination stops on `next_page: null`, a missing `next_page` key raises, a wrong `page` echo raises, a skipped page raises, `max_pages` raises, a duplicate `order_id` raises, HTTP 500 raises `requests.HTTPError`, and a server that sleeps 3 seconds with `timeout=0.5` raises `requests.Timeout` in under 2.5 seconds. Each validation rule has a parametrised case, including a date-only `updated_at`, an integer, a boolean and a time without a timezone; a separate test checks that `+02:00` is accepted and parses to the same instant as the equivalent `Z` time.

## The limitation: `primary_key` is not freshness

Here is the test I find most useful, because it passes by documenting a problem:

```python
def test_stale_version_regresses_state_without_guard(api, db):
    """Documents the limitation: primary_key is identity, not freshness."""
    api.orders[0].update(status="refunded", version=3)
    run_pipeline(api.url, **db)

    api.orders[0].update(status="paid", version=2)  # late, older event
    run_pipeline(api.url, **db)

    row = [r for r in business_rows(**db) if r[0] == 1][0]
    assert row[BUSINESS_COLUMNS.index("version")] == 2  # state went backwards
```

The order was refunded (version 3). Then an older snapshot of it (version 2, `paid`) arrived — a replica lagging behind, a backfill from an old export, a retried request that was slow. Delete-insert by key did exactly what it promises: it replaced the row. The refund disappeared from the table.

`primary_key` tells dlt *which* row to replace. It says nothing about *whether* to replace it. In the SQLite post the condition `WHERE excluded.version > orders.version` lived inside the write. Here nothing equivalent exists, so I have to add it myself.

### A simple version guard, and its limits

The guard in `run_pipeline` reads the current `version` for every `order_id` from DuckDB before the run and drops incoming records whose version is not higher:

```python
def test_version_guard_drops_stale_rows(api, db):
    api.orders[0].update(status="refunded", version=3)
    run_pipeline(api.url, version_guard=True, **db)

    api.orders[0].update(status="paid", version=2)
    api.orders[1].update(status="paid", version=2)  # a genuine newer update still lands
    run_pipeline(api.url, version_guard=True, **db)

    rows = {r[0]: dict(zip(BUSINESS_COLUMNS, r)) for r in business_rows(**db)}
    assert rows[1]["version"] == 3 and rows[1]["status"] == "refunded"
    assert rows[2]["version"] == 2
```

It works in this test, and it has clear limits:

- **It is a read followed by a write, not one atomic operation.** Two runs at the same time can both read version 1 and both write. It is only correct if runs for this table never overlap. The conditional UPSERT from the previous post doesn't have this gap, because the comparison happens in the same statement as the write.
- **It reads the whole key → version map into memory.** Fine for thousands of rows, not for hundreds of millions.
- **A record with the same version and different content is silently dropped.** As before, that is a contradiction in the source, and it deserves an alert, not a skip.
- **It trusts the `version` contract.** If the source doesn't increase `version` reliably, the guard protects nothing.

A rerun of this pipeline is safe for the *business columns* when the source returns the same or newer data. That is all. It isn't exactly-once delivery, and it isn't protection against stale data unless I add the guard and run one load at a time.

## Running it yourself

Download the [zip]({{ '/assets/examples/dlt-orders-pipeline.zip' | relative_url }}) and unpack it. With Python 3.11 and a plain virtual environment:

```bash
cd dlt-orders-pipeline
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
export RUNTIME__DLTHUB_TELEMETRY=false
.venv/bin/python -m orders_pipeline.demo
.venv/bin/python -m pytest -v
```

`requirements.txt` pins every package, including transitive ones, with hashes. There are no credentials and no external services: the API listens on loopback, the database is a local file. The suite has 33 tests. In the re-run described below a full run of all 33 took about 20 seconds; most of the time goes into starting dlt several times per test.

The archive also contains `pyproject.toml` and `uv.lock` for the same versions; see the note below.

## A note on reproduction

The outputs in this post come from a later re-run of the archived project, not from a run in October 2023. That is why the load ids above are 2026 Unix timestamps. The re-run used Python 3.11.12 on macOS (arm64), the exact pinned package versions listed above, and [uv](https://docs.astral.sh/uv/) as an installer on top of the same lock — a tool that did not exist when this post is dated. The `pip` route above was checked in the same re-run. I make no claim about other platforms or other dlt versions.

## What I'd take from this

- Validate in the extract step, and fail the run instead of loading part of it. In this setup a failed extract never touches the destination.
- Compare business columns in rerun tests. dlt's columns change on every load by design.
- `write_disposition="merge"` with `primary_key` in dlt 0.3.22 is delete-insert by key. It removes duplicates across runs. It does not decide which version of a record is newer.
- If the source can send older data after newer data, the rule that rejects it must live somewhere. In this example it's an explicit guard with known limits.

The next post in this series goes deeper into incremental loading: cursors, late data and schema changes, and where they lose correctness.

### Sources

- [dlt 0.3.22 on PyPI](https://pypi.org/project/dlt/0.3.22/) — release uploaded on 24 October 2023.
- [dlt at tag 0.3.22: `dlt/destinations/sql_jobs.py`](https://github.com/dlt-hub/dlt/blob/0.3.22/dlt/destinations/sql_jobs.py) — merge SQL: delete by key, `ROW_NUMBER()` deduplication within the staging data.
- [dlt at tag 0.3.22: `relational.py`](https://github.com/dlt-hub/dlt/blob/0.3.22/dlt/common/normalizers/json/relational.py) — `_dlt_id` and `_dlt_load_id` columns.
- [dlt at tag 0.3.22: `run_configuration.py`](https://github.com/dlt-hub/dlt/blob/0.3.22/dlt/common/configuration/specs/run_configuration.py) — `dlthub_telemetry` setting.
- [dlt documentation: merge loading](https://dlthub.com/docs/general-usage/merge-loading) — current documentation; describes later versions too.
- [DuckDB 0.9.1 on PyPI](https://pypi.org/project/duckdb/0.9.1/).
- [Requests: timeouts](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts) — no default timeout; connect and read timeouts.
