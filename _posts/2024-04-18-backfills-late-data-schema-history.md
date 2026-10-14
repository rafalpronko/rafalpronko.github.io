---
layout: post
lang: en
title: "Backfills, late data and schema changes without silently losing history"
date: 2024-04-18 09:00:00 +0200
permalink: /blog/backfills-late-data-schema-history/
author: Rafał Prońko
categories: [data-engineering]
description: "Keep raw revisions, normalize schema versions without guessing, rebuild the days that late data touches and replay backfills safely — a Python and SQLite example."
image: /assets/images/backfills-late-data-schema-history.svg
---

# Backfills, late data and schema changes without silently losing history

![Diagram: raw revisions in schema v1 and v2 are normalized, rejected records are kept aside, accepted revisions go into an append-only ledger, new revisions mark old and new UTC days as dirty, and a single transaction rebuilds the lookback window, the dirty days or a bounded backfill range into daily partitions.]({{ '/assets/images/backfills-late-data-schema-history.svg' | relative_url }})

<time datetime="2024-04-18">18 April 2024</time>

*This is an editorial date label. The runnable example and recorded test results are a modern reproduction, not evidence of a run on 18 April 2024.*

A daily revenue table can look perfectly healthy while it slowly drifts away from the truth.

The job is green every night. The row counts look plausible. Nobody notices that an order corrected two weeks later still sits on the wrong day, that a record from the old schema was counted as zero because a field was renamed, or that last month's backfill quietly overwrote the version of the data a report was built on.

None of these is a crash. Each is a small decision that drops information without telling anyone.

In the [idempotency article]({{ '/blog/pipeline-idempotencja/' | relative_url }}) I cared about one thing: the current state of an order after a retry. A conditional UPSERT kept the newest version and threw older ones away. That is fine when the current state is all you need. It is not fine when someone asks *how the numbers looked last Tuesday*, or when a late correction has to move revenue from one day to another.

This article goes one step further. I build a small revision ledger in Python and SQLite that:

- keeps every accepted revision exactly as received,
- normalizes two schema versions without inventing values,
- rejects what it does not understand, loudly,
- knows which daily partitions a late record has invalidated,
- rebuilds a bounded backfill range in a single transaction that can be replayed.

Python 3.11, SQLite and the standard library are enough. The code is complete further down; the tests are run with pytest.

## Three questions that sound like one

Before any code, I separate three questions that people often ask as if they were the same.

1. **Current state.** What do I believe about order `A-1` now? The latest accepted version.
2. **Current-best revenue by event day.** How much revenue do I now assign to 31 March? Grouped by the time the business event took place (*event time*), using the latest accepted version of each order.
3. **What was known then.** What would the pipeline have reported on 5 April, before a correction arrived on 12 April? Filtered by the time the pipeline first saw each revision (*first observed time*).

An overwrite-in-place table answers only the first question. A daily aggregate that is rebuilt from current state answers the second. The ledger retains effective-time history through the event time on every accepted revision; the daily tables summarize current best knowledge and do not retain earlier aggregate snapshots. The third needs the history of revisions *and* the moment each one arrived.

So I keep two times on every revision, and I never let one stand in for the other: `event_time` comes from the payload, `first_observed_at` comes from the pipeline clock when the revision was first stored. Both stored timestamps and historical query cutoffs use fixed-width UTC text with six fractional digits, such as `2024-04-12T06:00:00.100000+00:00`. This preserves microsecond precision and makes text comparisons chronological. A revision is eligible at its exact observation time, never at an earlier cutoff within the same second.

I want to be precise about what this is not. A table with versions and an ingestion timestamp is not a complete bitemporal model. It has no validity intervals, it does not record when the *source* changed its mind (only when I noticed), and it cannot tell me anything about the time before the pipeline existed. It is enough to answer "what did this pipeline know at time T", and nothing more.

## Raw revisions first, normalized views second

The ledger has a stable key: `(source, entity_id, version)`.

`source` matters because two systems can both have an order `1001`. `entity_id` identifies the business object. `version` is a number the source increases with every change. As in the earlier article, this is a contract with the source, not something the database can work out for itself.

Every accepted record is stored twice, in two tables with the same key:

- `raw_revisions` holds the payload as received, its schema version, a SHA-256 fingerprint and `first_observed_at`. Rows are only ever inserted.
- `normalized_revisions` holds the canonical columns derived from the raw payload, plus the name of the normalizer that produced them.

Why both? Because normalization rules change. The raw payloads provide the input for a separate re-normalization migration, which this demo does not implement. Such a migration would need to reprocess revisions and invalidate all affected days. Replaying an accepted record leaves its stored normalization unchanged, even after changing `NORMALIZER_VERSION`. Replacing existing normalized rows could also rewrite answers to "what was known then", so preserving historical answers requires retaining the earlier normalization or explicitly versioning those answers. If I had only kept the normalized values, the original would be gone.

## Schema change without guessing

The example has two schema versions of the same order event.

Schema v1:

```json
{"order_id": "A-1", "version": 1, "amount_cents": 1000,
 "event_time": "2024-04-01T10:00:00+02:00"}
```

Schema v2:

```json
{"order_id": "B-1", "version": 1, "amount_minor": 2500, "currency": "EUR",
 "event_time": "2024-04-01T12:00:00+00:00"}
```

v2 renamed the amount field and added a currency. The canonical form has `amount_minor` (an integer in the currency's minor unit) and `currency`.

Two shortcuts are tempting here, and both lose information silently.

The first is to treat a missing amount as zero. A record without an amount is not a free order. It is an order whose amount I don't know. In the canonical table it becomes `NULL`, and the daily aggregate counts it in a separate `missing_amount` column instead of adding 0 to revenue.

The second is to give v1 records a default currency. v1 never said what currency `amount_cents` was in. If I fill in "PLN" because most orders were in złoty, I have invented data and made it look like a fact. The example allows inference only for a source with an explicit contract — a synthetic `legacy-pl` source documented as single-currency. Every other v1 record keeps `currency = NULL`, and it is counted in `missing_currency`, not summed into some currency's revenue.

Everything else that does not fit is rejected:

- a `schema_version` that is not an integer or is unknown (a dictionary lookup would otherwise treat `True` as schema 1 and `2.0` as schema 2),
- a payload that is not a JSON object,
- an amount that is a string or a boolean (in Python, `True` is an `int`, so the check is explicit),
- an amount outside SQLite's signed 64-bit integer range, `-2**63` through `2**63 - 1`, or a version outside `1` through `2**63 - 1`,
- a currency that is not three capital letters,
- an `event_time` without a UTC offset.

A rejection is not a log line nobody reads. `ingest` raises `RejectedRecord` and, before re-raising, writes the original payload and the reason into `rejected_records`. The record is not lost, and it is not quietly half-loaded either. The rejected `schema_version` is stored as JSON text in `schema_version_json`, so even a value that SQLite's `INTEGER` cannot hold, such as `2**63`, survives quarantine exactly as received.

## The same version with different content

A replay of a revision I already have is harmless. A *different* payload under a key I already have is not.

To tell them apart, I fingerprint each record: the schema version plus the payload parsed as JSON and serialized again with sorted keys. Two payloads that differ only in key order or whitespace are the same revision. This is a canonical-fingerprint replay. If the fingerprint matches, `ingest` returns `False` and leaves `first_observed_at` alone. The first observation stays the first observation.

If the fingerprint differs, the source has broken its own version contract. In the idempotency article I mentioned that a plain conditional UPSERT would ignore such a record without any alarm. Here it is an error, and the stored revision stays untouched.

One consequence is worth making explicit: if a producer re-sends the same version in a new schema during a migration, the fingerprints differ and the record is rejected. I would rather resolve that case deliberately than have the pipeline decide which one is right.

## Late data and corrections invalidate days

Most daily pipelines are incremental by event time: every night they rebuild the last *N* days and leave older partitions alone. That cutoff is where late data disappears.

Suppose order `C-1` happened on 25 March but arrives on 12 April. A two-day lookback rebuilds 11 and 12 April. The order is safely stored in the ledger, and it never reaches the 25 March partition.

Corrections are worse. Order `A-1` was first recorded at 10:00 on 1 April at a UTC+02:00 offset. Version 2 corrects the event time to 23:50 UTC on 31 March. Rebuilding only the new day adds the order to 31 March, but 1 April still counts the old version. The same order is now counted twice.

The fix is to track which partitions a revision invalidates at the moment it is stored. When a new revision arrives, `store` marks as dirty:

- the UTC day of the new revision's event time,
- the UTC day of the entity's current latest version, if there is one.

Both marks are written in the same transaction as the revision. The nightly job then rebuilds the union of its normal lookback window and all dirty days, however old they are. Rebuilding a day clears its mark in the same transaction.

The marking is deliberately conservative. A late *lower* version that will not change the current state still marks its days. Rebuilding a day that did not need it costs some work; missing a day that did need it costs correctness.

Partitions are UTC calendar days. `2024-04-01T01:30:00+02:00` is 23:30 UTC on 31 March and goes to the 31 March partition. If the business reports in local days, that is a different partition key, and it has to be chosen explicitly, because daylight-saving changes make local days 23 or 25 hours long.

## A backfill is a bounded, versioned, atomic rebuild

A backfill is the same operation as the nightly rebuild, over a range I choose by hand. Three properties matter.

**The range is explicit and half-open.** `backfill_days(date(2024, 3, 25), date(2024, 4, 13))` means every UTC day from 25 March up to, but not including, 13 April. Half-open ranges compose without overlap: `[a, b)` followed by `[b, c)` covers each day exactly once. The function refuses empty and reversed ranges, ranges longer than 31 days and `datetime` bounds where a calendar date is expected. The 31-day limit is an arbitrary guard for the demo. The point is that a backfill should never be accidentally unbounded.

**The output records which logic produced it.** Every row in `daily_partitions` carries a `transformation_version`. After a backfill with `"daily/2"`, each day in the range says so, including days that are now empty. The label records the aggregation logic the caller says it is running; changing the label alone neither changes that logic nor re-normalizes raw revisions. Rebuilds read the existing normalized rows. An empty day with a partition row means "rebuilt, nothing there". A day without a row means "never built". Those are different statements, and I don't want them to look the same.

**The rebuild is all or nothing and safe to repeat.** `rebuild` opens one `BEGIN IMMEDIATE` transaction, deletes and recomputes each day from the current state of the ledger, and commits at the end. A failure during the rebuild or `COMMIT` triggers rollback if the connection still has an open transaction: no day is half-rebuilt, and no dirty mark is cleared. Running the same backfill twice leaves exactly the same tables, provided nothing it reads has changed. That means the same source snapshot, the same stored normalization, the same normalizer and aggregation logic, and the same transformation version. New revisions or a logic change can change the result.

That last property is idempotency in the sense of the earlier article: repeating the operation with those inputs held constant does not change the final effect. The repeated-backfill test keeps the ledger and logic unchanged between runs.

## The complete example

**[Download the complete example (ZIP)]({{ '/assets/downloads/backfills-late-data-schema-history.zip' | relative_url }})** — `backfill_demo.py`, both test files, `pyproject.toml`, `uv.lock`, `.python-version` and a README with the exact reproduction commands. With [uv](https://docs.astral.sh/uv/) installed, run `uv run --frozen --isolated --python 3.11 python backfill_demo.py` and `uv run --frozen --isolated --python 3.11 python -m pytest` in the unpacked directory. uv and the pinned pytest are modern reproduction tooling; the example itself needs only Python 3.11.

Or save the code below as `backfill_demo.py` and run it with `python3.11 backfill_demo.py` (it was written and tested for Python 3.11). I keep the ledger and demonstration in one complete, copyable file so running the standard-library example requires no multi-file import setup. It needs no third-party packages. The database lives in memory and disappears when the script ends; this is a demonstration of semantics, not durable storage.

```python
"""Backfills, late data and schema changes on a SQLite revision ledger.

Synthetic demo for Python 3.11+ and SQLite 3.24+; standard library only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

NORMALIZER_VERSION = "orders-normalizer/1"
MAX_BACKFILL_DAYS = 31
SQLITE_INT_MIN = -(2**63)
SQLITE_INT_MAX = 2**63 - 1
# Synthetic, explicitly documented contract: only this v1 source is single-currency.
V1_CURRENCY_CONTRACTS = {"legacy-pl": "PLN"}
CURRENCY_CODE = re.compile(r"[A-Z]{3}")

SCHEMA = """
CREATE TABLE raw_revisions (
    source TEXT NOT NULL, entity_id TEXT NOT NULL, version INTEGER NOT NULL,
    schema_version INTEGER NOT NULL, payload TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL, first_observed_at TEXT NOT NULL,
    PRIMARY KEY (source, entity_id, version));
CREATE TABLE normalized_revisions (
    source TEXT NOT NULL, entity_id TEXT NOT NULL, version INTEGER NOT NULL,
    event_time TEXT NOT NULL, event_day TEXT NOT NULL,
    amount_minor INTEGER, currency TEXT, normalizer_version TEXT NOT NULL,
    PRIMARY KEY (source, entity_id, version),
    FOREIGN KEY (source, entity_id, version) REFERENCES raw_revisions);
CREATE TABLE rejected_records (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, schema_version_json TEXT NOT NULL,
    payload TEXT NOT NULL, observed_at TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE dirty_days (day TEXT PRIMARY KEY);
CREATE TABLE daily_revenue (
    day TEXT NOT NULL, currency TEXT NOT NULL,
    amount_minor INTEGER NOT NULL, orders INTEGER NOT NULL,
    PRIMARY KEY (day, currency));
CREATE TABLE daily_partitions (
    day TEXT PRIMARY KEY, orders INTEGER NOT NULL,
    missing_amount INTEGER NOT NULL, missing_currency INTEGER NOT NULL,
    transformation_version TEXT NOT NULL);
"""

# Latest accepted version per (source, entity_id); no window functions needed.
CURRENT = """
SELECT n.* FROM normalized_revisions n
WHERE n.version = (SELECT MAX(m.version) FROM normalized_revisions m
                   WHERE m.source = n.source AND m.entity_id = n.entity_id)
"""


class RejectedRecord(ValueError):
    """A record we refuse to normalize; its payload goes to rejected_records."""


@dataclass(frozen=True)
class RawRecord:
    source: str
    schema_version: int
    payload: str  # kept byte-for-byte as received


@dataclass(frozen=True)
class Revision:
    entity_id: str
    version: int
    event_time: datetime  # aware, UTC
    amount_minor: int | None  # None means unknown, never zero
    currency: str | None  # None means unknown


def connect(path: str = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(path, isolation_level=None)  # we issue BEGIN ourselves
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[None]:
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise


def utc_text(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError(f"naive datetime is ambiguous: {moment!r}")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_event_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise RejectedRecord("event_time must be an ISO 8601 string")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RejectedRecord(f"event_time is not ISO 8601: {value!r}") from exc
    if moment.tzinfo is None:
        raise RejectedRecord(f"event_time has no UTC offset: {value!r}")
    return moment.astimezone(timezone.utc)


def sqlite_int(value: object, field: str) -> int:
    """Reject unsupported integers before binding, so quarantine keeps the payload."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RejectedRecord(f"{field} must be an integer")
    if not SQLITE_INT_MIN <= value <= SQLITE_INT_MAX:
        raise RejectedRecord(f"{field} is outside the signed SQLite 64-bit range")
    return value


def optional_int(body: dict[str, object], field: str) -> int | None:
    value = body.get(field)
    if value is None:
        return None  # missing means unknown, not zero
    return sqlite_int(value, field)


def entity_key(body: dict[str, object]) -> tuple[str, int]:
    entity_id = body.get("order_id")
    if not isinstance(entity_id, str) or not entity_id:
        raise RejectedRecord("order_id must be a non-empty string")
    version = sqlite_int(body.get("version"), "version")
    if version < 1:
        raise RejectedRecord("version must be a positive integer")
    return entity_id, version


def money_v1(source: str, body: dict[str, object]) -> tuple[int | None, str | None]:
    # v1 never carried a currency; only an explicit source contract may supply one.
    return optional_int(body, "amount_cents"), V1_CURRENCY_CONTRACTS.get(source)


def money_v2(source: str, body: dict[str, object]) -> tuple[int | None, str | None]:
    currency = body.get("currency")
    if currency is not None and not (
        isinstance(currency, str) and CURRENCY_CODE.fullmatch(currency)
    ):
        raise RejectedRecord(f"currency must be a 3-letter code: {currency!r}")
    return optional_int(body, "amount_minor"), currency


MONEY_BY_SCHEMA = {1: money_v1, 2: money_v2}


def parse(record: RawRecord) -> Revision:
    if not isinstance(record.schema_version, int) or isinstance(record.schema_version, bool):
        raise RejectedRecord("schema_version must be an integer")  # True == 1, 2.0 == 2
    money = MONEY_BY_SCHEMA.get(record.schema_version)
    if money is None:
        raise RejectedRecord(f"unknown schema_version {record.schema_version!r}")
    try:
        body = json.loads(record.payload)
    except json.JSONDecodeError as exc:
        raise RejectedRecord("payload is not valid JSON") from exc
    if not isinstance(body, dict):
        raise RejectedRecord("payload must be a JSON object")
    entity_id, version = entity_key(body)
    amount, currency = money(record.source, body)
    return Revision(entity_id, version, parse_event_time(body.get("event_time")),
                    amount, currency)


def fingerprint(record: RawRecord) -> str:
    canonical = json.dumps([record.schema_version, json.loads(record.payload)],
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mark_dirty(con: sqlite3.Connection, days: Iterable[str]) -> None:
    con.executemany("INSERT INTO dirty_days (day) VALUES (?) "
                    "ON CONFLICT (day) DO NOTHING", [(day,) for day in days])


def current_day(con: sqlite3.Connection, source: str, entity_id: str) -> list[str]:
    row = con.execute("SELECT event_day FROM normalized_revisions WHERE source = ? "
                      "AND entity_id = ? ORDER BY version DESC LIMIT 1",
                      (source, entity_id)).fetchone()
    return [] if row is None else [row[0]]


def store(con: sqlite3.Connection, record: RawRecord, rev: Revision,
          observed: str) -> bool:
    key = (record.source, rev.entity_id, rev.version)
    digest = fingerprint(record)
    row = con.execute("SELECT payload_sha256 FROM raw_revisions WHERE source = ? "
                      "AND entity_id = ? AND version = ?", key).fetchone()
    if row is not None:
        if row[0] != digest:
            raise RejectedRecord(f"conflicting payload for existing revision {key}")
        return False  # canonical-fingerprint replay: keep the first observation
    new_day = rev.event_time.date().isoformat()
    mark_dirty(con, current_day(con, record.source, rev.entity_id) + [new_day])
    con.execute("INSERT INTO raw_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (*key, record.schema_version, record.payload, digest, observed))
    con.execute("INSERT INTO normalized_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, utc_text(rev.event_time), new_day, rev.amount_minor,
                 rev.currency, NORMALIZER_VERSION))
    return True


def quarantine(con: sqlite3.Connection, record: RawRecord, observed: str,
               reason: str) -> None:
    # JSON text keeps any rejected schema_version, even one SQLite INTEGER cannot hold.
    con.execute("INSERT INTO rejected_records (source, schema_version_json, payload, "
                "observed_at, reason) VALUES (?, ?, ?, ?, ?)",
                (record.source, json.dumps(record.schema_version), record.payload,
                 observed, reason))


def ingest(con: sqlite3.Connection, record: RawRecord, observed_at: datetime) -> bool:
    """Append a revision. True: new revision; False: canonical-fingerprint replay."""
    observed = utc_text(observed_at)
    try:
        revision = parse(record)
        with transaction(con):
            return store(con, record, revision, observed)
    except RejectedRecord as exc:
        quarantine(con, record, observed, str(exc))
        raise


def rebuild_day(con: sqlite3.Connection, day: str, transformation: str) -> None:
    con.execute("DELETE FROM daily_revenue WHERE day = ?", (day,))
    con.execute("DELETE FROM daily_partitions WHERE day = ?", (day,))
    con.execute(f"""INSERT INTO daily_revenue
        SELECT event_day, currency, SUM(amount_minor), COUNT(*) FROM ({CURRENT})
        WHERE event_day = ? AND amount_minor IS NOT NULL AND currency IS NOT NULL
        GROUP BY event_day, currency""", (day,))
    con.execute(f"""INSERT INTO daily_partitions
        SELECT ?, COUNT(*), COALESCE(SUM(amount_minor IS NULL), 0),
               COALESCE(SUM(currency IS NULL), 0), ?
        FROM ({CURRENT}) WHERE event_day = ?""", (day, transformation, day))
    con.execute("DELETE FROM dirty_days WHERE day = ?", (day,))


def rebuild(con: sqlite3.Connection, days: Iterable[str], transformation: str) -> list[str]:
    """Rebuild every given UTC day in one all-or-nothing transaction."""
    ordered = sorted(set(days))
    with transaction(con):
        for day in ordered:
            rebuild_day(con, day, transformation)
    return ordered


def backfill_days(start: date, end: date) -> list[str]:
    """UTC calendar days in the half-open range [start, end)."""
    if isinstance(start, datetime) or isinstance(end, datetime):
        raise TypeError("backfill bounds are calendar dates, not datetimes")
    span = (end - start).days
    if not 0 < span <= MAX_BACKFILL_DAYS:
        raise ValueError(f"backfill must cover 1..{MAX_BACKFILL_DAYS} days, got {span}")
    return [(start + timedelta(days=offset)).isoformat() for offset in range(span)]


def incremental_days(con: sqlite3.Connection, today: date, lookback: int) -> list[str]:
    """Recent event-time window plus every day invalidated by late data."""
    window = {(today - timedelta(days=offset)).isoformat() for offset in range(lookback)}
    dirty = {day for (day,) in con.execute("SELECT day FROM dirty_days")}
    return sorted(window | dirty)


def current_state(con: sqlite3.Connection) -> list[tuple[object, ...]]:
    return con.execute(f"""SELECT source, entity_id, version, event_day, amount_minor,
        currency FROM ({CURRENT}) ORDER BY source, entity_id""").fetchall()


def state_known_at(con: sqlite3.Connection, as_of: datetime) -> list[tuple[object, ...]]:
    """Latest version per entity among revisions this pipeline had observed by as_of."""
    return con.execute("""
        SELECT n.source, n.entity_id, n.version, n.event_day, n.amount_minor, n.currency
        FROM normalized_revisions n JOIN raw_revisions r USING (source, entity_id, version)
        WHERE r.first_observed_at <= :t AND n.version = (
            SELECT MAX(m.version) FROM normalized_revisions m
            JOIN raw_revisions q USING (source, entity_id, version)
            WHERE m.source = n.source AND m.entity_id = n.entity_id
              AND q.first_observed_at <= :t)
        ORDER BY n.source, n.entity_id""", {"t": utc_text(as_of)}).fetchall()


def history(con: sqlite3.Connection, source: str, entity_id: str) -> list[tuple[object, ...]]:
    return con.execute("""SELECT r.version, r.schema_version, n.event_day, n.amount_minor,
        n.currency, r.first_observed_at FROM raw_revisions r
        JOIN normalized_revisions n USING (source, entity_id, version)
        WHERE source = ? AND entity_id = ? ORDER BY r.version""",
                       (source, entity_id)).fetchall()


def snapshot(con: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {table: sorted(con.execute(f"SELECT * FROM {table}"))
            for table in ("daily_revenue", "daily_partitions", "dirty_days")}


def make_record(source: str, schema_version: int, **body: object) -> RawRecord:
    return RawRecord(source, schema_version, json.dumps(body))


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def check(ok: bool, name: str) -> None:
    if not ok:
        raise AssertionError(f"FAIL: {name}")
    print(f"PASS: {name}")


def expect_rejected(con: sqlite3.Connection, raw: RawRecord, observed: datetime) -> str:
    try:
        ingest(con, raw, observed)
    except RejectedRecord as exc:
        return str(exc)
    raise AssertionError(f"FAIL: record was accepted: {raw}")


OBSERVED_FIRST = at("2024-04-02T06:00:00+00:00")
FIRST_LOAD = [
    make_record("shop", 1, order_id="A-1", version=1, amount_cents=1000,
                event_time="2024-04-01T10:00:00+02:00"),
    make_record("legacy-pl", 1, order_id="L-1", version=1, amount_cents=500,
                event_time="2024-04-01T01:30:00+02:00"),
    make_record("shop", 2, order_id="B-1", version=1, amount_minor=2500,
                currency="EUR", event_time="2024-04-01T12:00:00+00:00"),
    make_record("shop", 2, order_id="B-2", version=1, currency="EUR",
                event_time="2024-04-01T13:00:00+00:00"),
]
LATE = make_record("shop", 2, order_id="C-1", version=1, amount_minor=700,
                   currency="EUR", event_time="2024-03-25T09:00:00+00:00")
CORRECTION = make_record("shop", 2, order_id="A-1", version=2, amount_minor=1200,
                         currency="PLN", event_time="2024-03-31T23:50:00+00:00")
CONFLICT = make_record("shop", 2, order_id="B-1", version=1, amount_minor=9999,
                       currency="EUR", event_time="2024-04-01T12:00:00+00:00")
UNKNOWN_SCHEMA = make_record("shop", 3, order_id="D-1", version=1)


def partition(con: sqlite3.Connection, day: str) -> tuple[object, ...] | None:
    return con.execute("SELECT * FROM daily_partitions WHERE day = ?", (day,)).fetchone()


def first_load(con: sqlite3.Connection) -> None:
    for raw in FIRST_LOAD:
        ingest(con, raw, OBSERVED_FIRST)
    rebuild(con, incremental_days(con, date(2024, 4, 2), lookback=2), "daily/1")
    check(partition(con, "2024-03-31") == ("2024-03-31", 1, 0, 0, "daily/1"),
          "01:30+02:00 lands in the 2024-03-31 UTC partition")
    check(partition(con, "2024-04-01") == ("2024-04-01", 3, 1, 1, "daily/1"),
          "missing amount and v1 currency are counted as unknown, not zero")
    replays = [ingest(con, raw, at("2024-04-03T06:00:00+00:00")) for raw in FIRST_LOAD]
    check(replays == [False] * len(FIRST_LOAD), "canonical-fingerprint replay adds no revisions")


def late_data(con: sqlite3.Connection) -> None:
    ingest(con, LATE, at("2024-04-12T08:00:00+00:00"))
    ingest(con, CORRECTION, at("2024-04-12T08:05:00+00:00"))
    days = rebuild(con, incremental_days(con, date(2024, 4, 12), lookback=2), "daily/1")
    check({"2024-03-25", "2024-03-31", "2024-04-01"} <= set(days),
          "late data and the moved correction rebuild old and new days")
    check(partition(con, "2024-04-01") == ("2024-04-01", 2, 1, 0, "daily/1"),
          "the corrected order left its old day")


def bad_input(con: sqlite3.Connection) -> None:
    for raw in (CONFLICT, UNKNOWN_SCHEMA):
        print("rejected:", expect_rejected(con, raw, at("2024-04-12T09:00:00+00:00")))
    check(con.execute("SELECT COUNT(*) FROM rejected_records").fetchone() == (2,),
          "rejected payloads are preserved for inspection")
    check(history(con, "shop", "B-1")[0][3] == 2500, "the accepted revision is untouched")


def backfill(con: sqlite3.Connection) -> None:
    days = backfill_days(date(2024, 3, 25), date(2024, 4, 13))
    rebuild(con, days, "daily/2")
    before = snapshot(con)
    rebuild(con, days, "daily/2")
    check(snapshot(con) == before, "same snapshot and logic produce identical backfill output")
    check(con.execute("SELECT COUNT(*) FROM daily_partitions WHERE "
                      "transformation_version = 'daily/2'").fetchone() == (len(days),),
          "every day in [2024-03-25, 2024-04-13) carries the requested transformation label")


def report(con: sqlite3.Connection) -> None:
    print("A-1 history:", history(con, "shop", "A-1"))
    print("known on 2024-04-05:", state_known_at(con, at("2024-04-05T00:00:00+00:00")))
    print("current state:", current_state(con))
    for row in con.execute("SELECT * FROM daily_revenue ORDER BY day, currency"):
        print("daily_revenue:", row)


def main() -> None:
    con = connect()
    try:
        first_load(con)
        late_data(con)
        bad_input(con)
        backfill(con)
        report(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
```

The script prints:

```text
PASS: 01:30+02:00 lands in the 2024-03-31 UTC partition
PASS: missing amount and v1 currency are counted as unknown, not zero
PASS: canonical-fingerprint replay adds no revisions
PASS: late data and the moved correction rebuild old and new days
PASS: the corrected order left its old day
rejected: conflicting payload for existing revision ('shop', 'B-1', 1)
rejected: unknown schema_version 3
PASS: rejected payloads are preserved for inspection
PASS: the accepted revision is untouched
PASS: same snapshot and logic produce identical backfill output
PASS: every day in [2024-03-25, 2024-04-13) carries the requested transformation label
A-1 history: [(1, 1, '2024-04-01', 1000, None, '2024-04-02T06:00:00.000000+00:00'), (2, 2, '2024-03-31', 1200, 'PLN', '2024-04-12T08:05:00.000000+00:00')]
known on 2024-04-05: [('legacy-pl', 'L-1', 1, '2024-03-31', 500, 'PLN'), ('shop', 'A-1', 1, '2024-04-01', 1000, None), ('shop', 'B-1', 1, '2024-04-01', 2500, 'EUR'), ('shop', 'B-2', 1, '2024-04-01', None, 'EUR')]
current state: [('legacy-pl', 'L-1', 1, '2024-03-31', 500, 'PLN'), ('shop', 'A-1', 2, '2024-03-31', 1200, 'PLN'), ('shop', 'B-1', 1, '2024-04-01', 2500, 'EUR'), ('shop', 'B-2', 1, '2024-04-01', None, 'EUR'), ('shop', 'C-1', 1, '2024-03-25', 700, 'EUR')]
daily_revenue: ('2024-03-25', 'EUR', 700, 1)
daily_revenue: ('2024-03-31', 'PLN', 1700, 2)
daily_revenue: ('2024-04-01', 'EUR', 2500, 1)
```

The `PASS` lines are business assertions on synthetic data, not measurements. They check the behaviour described above: the UTC partition boundary, unknowns that are counted rather than zeroed, canonical-fingerprint replays that add nothing, late data and a moved correction that rebuild both affected days, preserved rejections, and a backfill that can be repeated with unchanged inputs.

## How the pieces fit

A few details are easy to miss when reading the code for the first time.

**Explicit transactions.** I open the connection with `isolation_level=None`. In Python 3.11 that means the `sqlite3` module does not open transactions implicitly, so I can issue `BEGIN IMMEDIATE`, `COMMIT` and `ROLLBACK` myself. This matters in `store`: the `SELECT` that checks for an existing revision and the `INSERT` that follows have to be in the same transaction. With the module's default behaviour, the implicit `BEGIN` is only issued before the first `INSERT`, `UPDATE`, `DELETE` or `REPLACE`, so the check would run outside it. I also don't use `with con:` here. The connection context manager commits or rolls back an open transaction, but it does not open one.

**`BEGIN IMMEDIATE` and one writer.** SQLite allows many readers but only one write transaction at a time. `IMMEDIATE` takes the write lock at the start instead of at the first write, and it can fail with `SQLITE_BUSY` if another connection is already writing. In this single-process demo that cannot happen. With several writers you would need a busy timeout and a retry policy.

**No window functions.** The "latest version per entity" query uses a correlated `MAX(version)` subquery. Window functions would read nicer, but they arrived in SQLite 3.25.0, while UPSERT (`ON CONFLICT ... DO NOTHING` on `dirty_days`) needs 3.24.0. I wanted the example to depend on as few features as possible.

**Foreign keys are opt-in.** `normalized_revisions` references `raw_revisions`, but SQLite only enforces that after `PRAGMA foreign_keys = ON` on each connection.

**Rejections happen outside the failed transaction.** When `store` raises, the transaction is rolled back first. Only then does `ingest` write the payload into `rejected_records`, so the evidence survives even though the revision itself was refused.

## Tests: the failures are the interesting part

The happy path is the least informative test here. The tests that would catch a silent loss of history are the ones about late data, corrections and failure.

In the example directory, save the following as `tests/test_article_excerpt.py`, with `backfill_demo.py` in the parent directory. From that parent directory, run `python3.11 -m pytest tests/test_article_excerpt.py` after installing pytest:

```python
import json
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime

import pytest

import backfill_demo as demo


def eur(entity_id: str, version: int, event_time: str, amount: int) -> demo.RawRecord:
    return demo.make_record("shop", 2, order_id=entity_id, version=version,
                            event_time=event_time, amount_minor=amount, currency="EUR")


def observed(text: str) -> datetime:
    return demo.at(text)


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    con = demo.connect()
    yield con
    con.close()


def test_late_arrival_outside_lookback_is_rebuilt() -> None:
    con = demo.connect()
    demo.ingest(con, eur("A", 1, "2024-03-25T10:00:00+00:00", 100),
                observed("2024-04-12T08:00:00+00:00"))
    days = demo.incremental_days(con, date(2024, 4, 12), lookback=2)
    assert days == ["2024-03-25", "2024-04-11", "2024-04-12"]
    demo.rebuild(con, days, "t/1")
    assert demo.partition(con, "2024-03-25") == ("2024-03-25", 1, 0, 0, "t/1")


def test_correction_invalidates_old_and_new_day() -> None:
    con = demo.connect()
    demo.ingest(con, eur("A", 1, "2024-04-01T10:00:00+00:00", 100),
                observed("2024-04-02T06:00:00+00:00"))
    demo.rebuild(con, ["2024-04-01"], "t/1")
    demo.ingest(con, eur("A", 2, "2024-03-31T23:50:00+00:00", 120),
                observed("2024-04-12T08:00:00+00:00"))
    days = demo.incremental_days(con, date(2024, 4, 12), lookback=1)
    assert days == ["2024-03-31", "2024-04-01", "2024-04-12"]
    demo.rebuild(con, days, "t/1")
    assert demo.partition(con, "2024-04-01") == ("2024-04-01", 0, 0, 0, "t/1")
    assert len(demo.history(con, "shop", "A")) == 2


def test_failed_backfill_leaves_no_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    con = demo.connect()
    demo.ingest(con, eur("A", 1, "2024-04-01T10:00:00+00:00", 100),
                observed("2024-04-02T06:00:00+00:00"))
    demo.rebuild(con, ["2024-04-01"], "t/1")
    before = demo.snapshot(con)
    real_rebuild_day = demo.rebuild_day

    def crash_on_second_day(c, day: str, transformation: str) -> None:
        if day == "2024-04-02":
            raise RuntimeError("simulated crash")
        real_rebuild_day(c, day, transformation)

    monkeypatch.setattr(demo, "rebuild_day", crash_on_second_day)
    with pytest.raises(RuntimeError):
        demo.rebuild(con, demo.backfill_days(date(2024, 4, 1), date(2024, 4, 3)), "t/2")
    assert demo.snapshot(con) == before


@pytest.mark.parametrize(("cutoff", "versions"), [
    ("050000", []),
    ("100000", [1]),
    ("500000", [1]),
    ("899999", [1]),
    ("900000", [2]),
])
def test_known_at_never_includes_a_future_observation(
        connection: sqlite3.Connection, cutoff: str, versions: list[int]) -> None:
    demo.ingest(connection, eur("A", 1, "2024-04-01T10:00:00+00:00", 100),
                observed("2024-04-12T08:00:00.100000+00:00"))
    demo.ingest(connection, eur("A", 2, "2024-04-01T10:00:00+00:00", 120),
                observed("2024-04-12T08:00:00.900000+00:00"))
    rows = demo.state_known_at(
        connection, observed(f"2024-04-12T08:00:00.{cutoff}+00:00"))
    assert [row[2] for row in rows] == versions


def test_commit_failure_rolls_back_partitions_and_dirty_marks(
        connection: sqlite3.Connection) -> None:
    demo.ingest(connection, eur("A", 1, "2024-04-01T10:00:00+00:00", 100),
                observed("2024-04-02T06:00:00+00:00"))
    demo.rebuild(connection, ["2024-04-01"], "t/1")
    demo.ingest(connection, eur("A", 2, "2024-04-02T10:00:00+00:00", 120),
                observed("2024-04-12T08:00:00+00:00"))
    before = demo.snapshot(connection)
    denied_commits = []

    def deny_commit(action: int, arg1: str | None, arg2: str | None,
                    database: str | None, trigger: str | None) -> int:
        if action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT":
            denied_commits.append(arg1)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_commit)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            demo.rebuild(connection, ["2024-04-01", "2024-04-02"], "t/2")
    finally:
        connection.set_authorizer(None)
    assert denied_commits == ["COMMIT"]
    assert not connection.in_transaction
    assert demo.snapshot(connection) == before
    # The same connection must be usable for a subsequent successful rebuild.
    demo.rebuild(connection, ["2024-04-01", "2024-04-02"], "t/2")
    assert demo.partition(connection, "2024-04-01") == ("2024-04-01", 0, 0, 0, "t/2")
    assert demo.partition(connection, "2024-04-02") == ("2024-04-02", 1, 0, 0, "t/2")


@pytest.mark.parametrize(("schema", "field"), [
    (1, "amount_cents"), (2, "amount_minor"), (2, "version"),
])
@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63, True, False])
def test_unsupported_integers_preserve_the_original_payload_in_quarantine(
        connection: sqlite3.Connection, schema: int, field: str,
        value: int | bool) -> None:
    # Deliberate whitespace must survive quarantine as well as the numeric value.
    body = ('{ "order_id": "A", "version": 1, '
            '"event_time": "2024-04-01T10:00:00+00:00", "currency": "EUR" }')
    payload = json.loads(body)
    payload[field] = value
    raw = demo.RawRecord("shop", schema, json.dumps(payload, indent=2) + "\n")
    with pytest.raises(demo.RejectedRecord):
        demo.ingest(connection, raw, observed("2024-04-02T06:00:00+00:00"))
    assert connection.execute("SELECT payload FROM rejected_records").fetchall() == [
        (raw.payload,)]
    assert connection.execute("SELECT COUNT(*) FROM raw_revisions").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM normalized_revisions").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM dirty_days").fetchone() == (0,)
    assert not connection.in_transaction
```

The third test replaces `rebuild_day` with a version that crashes on the second day of a two-day backfill. The first day has already been recomputed inside the transaction. After the exception, the snapshot of `daily_revenue`, `daily_partitions` and `dirty_days` must be exactly as it was before. If `rebuild` committed day by day, the first day would carry the new transformation version, and this test would fail.

The full suite in `examples/tests/test_article_excerpt.py` and `examples/tests/test_backfill_demo.py` has 63 passing test cases, including parameterized cases. Together, they check:

- v1 currency stays unknown unless the source has a contract,
- a missing amount is counted in `missing_amount` and not added as zero,
- unknown schemas and wrongly typed fields are rejected and stored in `rejected_records`; for unknown, out-of-range, boolean or non-integer schema versions the original payload, schema metadata and reason are kept while the ledger and dirty marks stay unchanged,
- naive timestamps are refused, both in the payload and as the observation time,
- a conflicting payload for an existing version is rejected and the original revision is kept,
- a canonical-fingerprint replay with reordered JSON keys is recognized and keeps the first `first_observed_at`,
- a late lower version is kept in history but does not become current,
- "known at" differs from current state, and cutoffs before, between and exactly at subsecond observations exclude future revisions,
- stored event and observation times preserve fixed-width UTC microseconds,
- a denied `COMMIT` restores all partitions and dirty marks and leaves the connection usable,
- an automatic SQLite rollback preserves the original database error,
- signed 64-bit amount bounds and positive version bounds are accepted; out-of-range integers and booleans are rejected with the original payload quarantined,
- backfill ranges are half-open and bounded,
- a repeated backfill with an unchanged ledger and logic produces an identical snapshot.

Observed in the modern reproduction on 12 October 2026, running `uv run --frozen --isolated --python 3.11 python -m pytest` in the unpacked download: CPython 3.11.12, SQLite 3.47.1 and pytest 9.1.1; exit `0`. The final line from the actual full-suite stdout was:

```text
============================== 63 passed in 0.07s ==============================
```

Like the earlier example, these tests say nothing about process crashes, disk failures or concurrent writers. They check semantics on an in-memory database.

## What I would add outside a demo

The demo publishes by committing into the same tables readers query. That works in SQLite because a reader on another connection does not see uncommitted changes: while its read transaction is open, it keeps seeing the snapshot from before the writer's changes. In a warehouse with large partitions, I would split the backfill into three explicit stages. This is a recommendation, not something the example implements:

1. **Stage.** Write the rebuilt partitions into a staging location labelled with the backfill range and transformation version. Nothing that readers use changes yet.
2. **Reconcile.** Compare staging with independent numbers before publishing: order counts per day against the ledger, totals per currency, the number of unknowns, the days that changed and by how much. A large unexplained difference should stop the publish step.
3. **Publish.** Swap the reconciled partitions in as one operation, in whatever form the storage supports atomic replacement, and record which range, transformation version and ledger state were published.

Reconciliation deserves its own article. The short version: row counts alone don't prove much.

## Where the guarantees end

**Deletion and retention.** The ledger only grows. That conflicts with retention rules and with requests to erase personal data. "Never overwrite history" applies to business revisions, not to an obligation to delete. A real design needs a deliberate deletion path: tombstone revisions for business deletes, and physical deletion or crypto-shredding of payloads where the law requires it, followed by a rebuild of the affected days. The demo implements none of that, and it keeps nothing personal in its synthetic payloads.

**Currency.** The demo never converts currencies, and it never sums across them. Revenue is per currency, and records without a known currency are counted separately. Conversion needs exchange rates with their own effective time, and that is a separate history problem.

**The late cutoff still exists somewhere.** Dirty days make the nightly job rebuild old partitions, but someone downstream may have already exported or closed that month. Rebuilding the partition does not reopen a closed financial period. Decide which partitions are frozen, and route corrections for them into an explicit adjustment rather than a silent rewrite.

**Delivery.** The ledger can only reflect revisions that arrived. Idempotent ingestion makes replays safe; it does not make a missing message appear. If a source can lose events, you need reconciliation against that source, not a stronger ledger.

**No exactly-once.** Nothing here gives exactly-once processing across systems. The SQLite transaction is atomic for this one database file. If ingestion is fed from a queue, a crash between the commit and the acknowledgement means the message comes back, and the fingerprint check turns that redelivery into a no-op. That is effectively-once storage of an idempotent operation, not distributed exactly-once.

**Observed time is my clock.** `first_observed_at` says when this pipeline stored the revision. The example preserves microseconds; observations within the same microsecond remain indistinguishable. If I load a historical export today, all of its revisions are "known" today. "What was known then" means what *this pipeline* knew, not what the source knew.

## Where I would start

Not with a backfill script.

First I would decide what identifies a revision and where its version comes from. Then I would write down which fields each schema version has and what a missing field *means*. After that, I would make sure every new revision records which days it invalidates.

Once those three are in place, a backfill is no longer a special operation. It is the nightly rebuild over a range I choose: bounded, labelled with the logic that produced it, all or nothing, and safe to repeat with unchanged inputs and logic.

### Sources

- [Python 3.11 `sqlite3` documentation (tagged 3.11.0 source)](https://github.com/python/cpython/blob/v3.11.0/Doc/library/sqlite3.rst) — `isolation_level=None` and explicit transactions, implicit `BEGIN` only before `INSERT`/`UPDATE`/`DELETE`/`REPLACE`, the connection context manager neither opening a transaction nor closing the connection. Current versioned page: [docs.python.org/3.11/library/sqlite3.html](https://docs.python.org/3.11/library/sqlite3.html).
- [Python 3.11.0 release](https://www.python.org/downloads/release/python-3110/) — released 24 October 2022.
- [SQLite: transactions](https://www.sqlite.org/lang_transaction.html) — explicit `BEGIN`/`COMMIT`/`ROLLBACK`, `BEGIN IMMEDIATE`, one write transaction at a time, readers seeing a snapshot.
- [SQLite: UPSERT](https://www.sqlite.org/lang_upsert.html) and [SQLite 3.24.0 release notes](https://www.sqlite.org/releaselog/3_24_0.html) — `ON CONFLICT ... DO NOTHING`, available since 3.24.0 of 4 June 2018.
- [SQLite 3.25.0 release notes](https://www.sqlite.org/releaselog/3_25_0.html) — window functions, added on 15 September 2018, which is why the example avoids them.
- [SQLite: foreign key support](https://www.sqlite.org/foreignkeys.html) — foreign key enforcement is off by default and enabled per connection with `PRAGMA foreign_keys = ON`.
