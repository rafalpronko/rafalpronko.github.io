---
layout: post
lang: en
title: "A pipeline you can run twice. Idempotency in practice"
date: 2023-10-12 09:00:00 +0200
permalink: /blog/pipeline-idempotencja/
author: Rafał Prońko
categories: [data-engineering]
description: "Idempotent order writes in Python and SQLite: key, version, conditional UPSERT and transactions."
image: /assets/images/pipeline-idempotencja.svg
---

# A pipeline you can run twice. Idempotency in practice

![Diagram of an idempotent order write: a batch with a key and a version passes through a conditional UPSERT, and a retry does not create duplicates.]({{ '/assets/images/pipeline-idempotencja.svg' | relative_url }})

<time datetime="2023-10-12">12 October 2023</time>

Sooner or later a pipeline will be run a second time. The scheduler retries a job after a timeout, someone manually clicks “retry”, a backfill goes over a day that has already been loaded.

So I don't just ask whether the import works. I ask what will be left in the table after such a repeat.

If the first run writes the orders and the second appends copies of them, I have a problem. If replaying an older batch rolls back a newer order amount, I have a less visible but equally serious problem. A green job status then says little about whether the data is correct.

I want to build a write that I can safely repeat. I'll start with a small example in Python and SQLite — no orchestrator, no broker and no extra libraries. First I need the right semantics; the tools come later.

## Retry does not mean idempotency

Retry is a decision: “try again”. Idempotency is a property of an operation: executing it again with the same input does not change its final effect.

This distinction matters especially with timeouts. No response does not mean the previous attempt wrote nothing. It may have committed the transaction and then lost the connection before the client received the acknowledgement.

That's why I don't want correctness to depend on whether I can guess the outcome of the previous attempt. I design the write so that a repeat is safe.

The first step is to establish **what identifies the same business object**.

If the orders table only has an auto-generated identifier, every import can create a new row. The database doesn't know that two records describe the same order. A transaction alone won't fix that.

In the example I choose `order_id` from the source system as the primary key. If I combine several sources with independent numbering, I need a composite key, for example made of the source identifier and the order identifier.

## The same key does not always mean a duplicate

An order may come back with the same content because I re-ran the import. It may also come back with a changed amount because the source system updated it.

Rejecting every existing `order_id` will protect me from duplicates, but it will block updates. Unconditional overwriting will allow data to be updated, but it creates a different problem: an older batch can roll back a newer state.

So I need information about the order of changes.

In this example I assume a simple contract: every order has a numeric `version` that increases with each subsequent change.

- The same identifier and the same version mean a repeat.
- A higher version means an update.
- A lower version means older data that should not replace newer data.

This is an assumption about the source, not something the database will guess for me. If the source doesn't provide a reliable version, I have to design a different conflict-resolution rule.

## A small example: key, version and conditional UPSERT

I use Python 3.11 and SQLite 3.24.0 or later. The standard library is enough.

I store the amount as an integer number of cents. That way I don't bring the problem of representing money with floating-point numbers into the example.

The database lives in memory. I run both loads on the same connection; closing the connection deletes the data. This is a demonstration of write semantics, not durable storage.

The code below can be saved as `pipeline.py` and run with `python3 pipeline.py`.

```python
import sqlite3
Order = tuple[str, int, int]  # (order_id, amount_cents, version)
SCHEMA = """CREATE TABLE orders (
    order_id TEXT PRIMARY KEY NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    version INTEGER NOT NULL)"""
UPSERT = """INSERT INTO orders VALUES (?, ?, ?)
    ON CONFLICT (order_id) DO UPDATE SET amount_cents = excluded.amount_cents,
    version = excluded.version WHERE excluded.version > orders.version"""


def load(con: sqlite3.Connection, batch: list[Order]) -> None:
    with con:  # commit or rollback; does not close the connection
        con.executemany(UPSERT, batch)


def state(con: sqlite3.Connection) -> list[Order]:
    return con.execute("SELECT * FROM orders ORDER BY order_id").fetchall()


def check(ok: bool, name: str) -> None:
    if not ok:
        raise AssertionError(f"FAIL: {name}")
    print(f"PASS: {name}")


con = sqlite3.connect(":memory:")
try:
    con.execute(SCHEMA)
    batch = [("A-1", 1000, 1), ("A-2", 2500, 1)]
    load(con, batch)
    load(con, batch)
    check(state(con) == batch, "reloading the batch yields an identical state")
    updated = [("A-1", 1200, 2), ("A-2", 2500, 1)]
    load(con, [("A-1", 1200, 2)])
    check(state(con) == updated, "higher version updates amount")
    load(con, [("A-1", 900, 1)])
    check(state(con) == updated, "older event does not roll back state")
    try:
        load(con, [("A-3", 500, 1), ("A-4", -1, 1)])
        raise AssertionError("FAIL: negative amount did not raise IntegrityError")
    except sqlite3.IntegrityError:
        check(state(con) == updated, "batch with an error rolled back entirely")
finally:
    con.close()
```

The most important part is the condition:

```sql
WHERE excluded.version > orders.version
```

`excluded` means the record I'm trying to insert. `orders` means the record already stored.

If the order doesn't exist, `INSERT` creates it. If it exists, `ON CONFLICT` switches to an update — but only when the incoming version is higher.

Repeating the same version changes nothing. Older data doesn't change the row either. A newer version updates the amount and the version number.

I don't need a separate “does the record exist?” check in Python. The database enforces the rule inside the write statement.

## A transaction solves a different problem

In the `load` function the whole batch is wrapped in a transaction.

With the default `sqlite3` configuration in Python 3.11, write statements open a transaction. The `with con:` block commits it on successful completion or rolls it back when an exception occurs. It does not close the connection — that's why I do that separately in `finally`.

In the last test the first record of the batch is valid and the second has a negative amount. The `CHECK` constraint raises an error. After the rollback, the first record of that batch should not remain in the table either.

This is a check of **atomicity**, not idempotency.

Atomicity means: the batch is written in full or not at all. Idempotency means: repeating the operation does not change the final effect.

Without atomicity I can leave partially loaded data behind after an error. An idempotent write may allow the import to be completed by a retry, but consumers will see an intermediate state for a while.

Conversely, an atomic transaction won't protect me from duplicates if I commit a batch based on plain row appends twice.

I need both properties, but for different reasons.

## What do the tests actually check?

I don't limit myself to the row count. I compare the entire state of the table.

I check whether:

1. Loading the same batch again leaves identical data.
2. A higher version updates the order amount.
3. An older event does not roll back the current state.
4. An error in the second part of a batch also rolls back the earlier write from that batch.

It's a small set, but it hits the behaviours the example relies on.

It is not a test of resilience to process crashes, concurrent writes or disk problems. I don't derive such guarantees from four checks run against an in-memory database.

## Where does this guarantee end?

The first boundary is the version contract.

If the source sends the same version of an order with a different amount, the example will reject the change without any alarm. Yet this is not an ordinary duplicate — it's a data contradiction.

In a larger solution I would want to detect it: for an equal version, compare the content and report a conflict. A conditional UPSERT is no substitute for quality control of the source.

The second boundary is data delivery.

Idempotency lets me process repeats safely. It doesn't make a missing message reach the system. Nor does it mean I've applied every intermediate version. In this example I care about the current state of the order, not the full history of its changes.

The third boundary is effects outside the database.

If I send an email or initiate a payment during the import, rolling back the transaction won't undo that operation. A retry may execute it again. The primary key of the orders table doesn't protect a mailbox or a bank account.

In such a case I consider the **outbox** pattern: together with the data change, I store in a single transaction the intent to perform the external operation. A separate process reads that intent and carries it out.

But an outbox is not a magical “exactly once”. The process may send the message and then stop before recording the acknowledgement. The next attempt will send it again.

So I also need a stable operation identifier and deduplication on the receiver's side, if the receiver supports it. With payments, I check the conditions under which the idempotency key works in the specific API instead of assuming a universal guarantee.

## Where do I start when designing a pipeline?

Not with setting the number of retries.

First I establish what identifies a record, how I recognise its newer version and what should happen after the same data is processed again.

Then I define the transaction boundary and check partial failure. Only then does retry become a recovery mechanism rather than a way to multiply the problem.

In this example the guarantee covers the state of a single table and relies on a correct version contract. It's a limitation, but a concrete and testable one.

I prefer such a guarantee to a general assurance that the pipeline is “fault-tolerant”.

### Sources

- [SQLite: UPSERT](https://www.sqlite.org/lang_upsert.html) — syntax, `excluded`, conditional update; UPSERT available since SQLite 3.24.0 of 4 June 2018.
- [Python 3.11: sqlite3](https://docs.python.org/3.11/library/sqlite3.html) — transactions and the behaviour of the connection context manager.
