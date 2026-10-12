---
layout: post
lang: en
title: "Incremental loading, merge and schema evolution: where correctness breaks"
date: 2023-11-02 09:00:00 +0100
permalink: /blog/incremental-loading-merge-schema-evolution/
author: Rafał Prońko
categories: [data-engineering]
description: "A synthetic Python and SQLite deep dive: watermark windows with a visibility-delay contract, conditional UPSERT with an equal-version policy, and a nullable schema change that does not invent data."
image: /assets/images/incremental-loading-merge-schema-evolution.svg
---

# Incremental loading, merge and schema evolution: where correctness breaks

![Three panels: the extraction window from the watermark minus the overlap up to a fixed upper bound, the conditional UPSERT decision for newer, older and equal versions, and a nullable currency column whose old rows stay unknown.]({{ '/assets/images/incremental-loading-merge-schema-evolution.svg' | relative_url }})

<time datetime="2023-11-02">2 November 2023</time>

"Load only what changed since last time" sounds like one decision. It is at least three, and each one can be wrong while the job stays green:

1. **Extraction** decides which source rows you read. If it misses a row, nothing downstream will ever see it.
2. **Merge** decides what a row you did read does to the target. If it lets an old version win, the target regresses.
3. **Schema evolution** decides what the target means after the source contract changes. If it fills a new column with a guess, the data looks complete and is wrong.

The [previous article]({{ '/blog/pipeline-idempotencja/' | relative_url }}) covered a conditional UPSERT that makes a batch safe to repeat. This one starts earlier, at the question of which rows end up in the batch, and finishes later, when the table itself changes shape.

Everything below is a small synthetic example in Python 3.11 and SQLite, using only the standard library plus pytest for the tests. It uses SQLite's `INSERT ... ON CONFLICT DO UPDATE`, not a SQL `MERGE` statement. The rules are about the *contract*; the exact behaviour of `MERGE` in a particular warehouse, its concurrency semantics and its handling of duplicates in the source are separate questions that this example does not answer.

The complete, runnable code and tests are linked at the end.

## Part 1: extraction

### Three different times

Pipelines often collapse three moments into one column called `timestamp`:

- **Event time** is when the business event happened, such as the order being placed. It can be arbitrarily old.
- **`updated_at`** is set by the source whenever the row changes. This is the extraction cursor.
- **Visibility time** is when a reader can actually see that change. It can be later than `updated_at`, for example when the timestamp is assigned at the start of a transaction that commits later.

The cursor has to be `updated_at`, not event time. An order placed a month ago and refunded today has an old event time and a fresh `updated_at`; a window over event time would never see the refund.

All times in the example are integer Unix seconds in UTC, because integers compare unambiguously. ISO-8601 strings with different offsets do not: `10:00:00+01:00` sorts after `09:30:00+00:00` as text, although it is the earlier instant.

### The contract with the source

The guarantees below hold only under explicit assumptions about the source:

- per key, `updated_at` **strictly increases** with every modification, so two different versions of one row never share a timestamp and a newer version never carries an older one;
- one read sees a consistent snapshot;
- the delay between `updated_at` and visibility is bounded by a known value **D**;
- runs keep happening, so every window is eventually read;
- the first run covers everything the target has to hold: the initial lower bound (starting watermark minus overlap) lies at or below the oldest `updated_at` you need, or the older rows are already in the target.

Under these assumptions the target converges to the source's **current** state of every row. It does not capture history. A snapshot holds only the latest version, so if a row changes twice between two runs, the intermediate version is never extracted; one test loads a row that went from `paid` to `refunded` before the first run and finds only `refunded`. If you need every version, you need change data capture or an append-only log, not a watermark over a mutable table. In the simulated source, two versions of one row can also become visible in the same second; the snapshot then keeps the one with the higher `updated_at`, and a test feeds the two versions in both orders to check that the result does not depend on input order.

The bootstrap condition is easy to miss because the overlap argument below only looks *forward* from a watermark. The example's target starts empty with a chosen first watermark. A row last modified long before it is never inside any window, however many runs succeed. One test starts with a lower bound below the oldest row and loads everything; another starts above an old row, shows that three successful runs never load it, and then loads it with an explicit backfill below the watermark, which is how you preload history.

The code can detect one violation: two different contents under the same key and `updated_at` fail loudly. It cannot detect a timestamp that goes backwards (that looks exactly like an old replay and is ignored), and it cannot verify the other assumptions in a single run. They are agreements with whoever owns the source, and when they are not written down, the pipeline is relying on luck.

### A fixed upper bound and a half-open window

Let **W** be the last committed watermark. Each run reads the half-open window `[W − overlap, U)` and, on success, sets W to U.

The upper bound U is chosen **once**, at the start of the run, and checked before anything is read:

```python
def check_bounds(current: int, upper: object, read_at: object) -> None:
    """The upper bound is fixed once per run: W < U <= read time."""
    if type(upper) is not int or type(read_at) is not int:
        raise TypeError("upper and read_at must be int seconds")
    if not current < upper <= read_at:
        raise ValueError(f"need watermark {current} < upper {upper} <= read_at")
```

U is not "the largest `updated_at` I happened to read", because that depends on what was visible at that moment. It is not "now" evaluated separately in several places either. `U ≤ read_at` matters for the argument below: the window must not claim a range the snapshot could not yet see. In a real system both values should come from the source's clock; clock skew between source and pipeline moves the bounds directly, and the example does not model it. The explicit `type(...) is not int` check also rejects `True`, which Python would otherwise accept as the integer 1.

The window is half-open on purpose. Rows whose `updated_at` equals U do not belong to this run, so the next run takes all of them together, rather than some now and some later.

The other two run parameters are checked before anything is read as well:

```python
def check_run_params(overlap: object, page_size: object) -> None:
    """A negative overlap misses late rows; page_size 0 reads nothing."""
    if type(overlap) is not int or overlap < 0:
        raise ValueError(f"overlap {overlap!r}: int >= 0 required")
    if type(page_size) is not int or page_size < 1:
        raise ValueError(f"page_size {page_size!r}: int >= 1 required")
```

Both failures are quiet without the check. A negative overlap starts the window *after* the last watermark and skips rows nobody has read. A page size of zero returns no pages, the batch is empty, and the run would still commit W := U, moving the watermark over data it never saw.

### Why the overlap must be at least D

Suppose a row has `updated_at = t` and becomes visible after a delay `d ≤ D`. A run with upper bound U reads at `R ≥ U`. If `t < U` and the row was not yet visible, then `t + d > R ≥ U`, so `t > U − D`. The next run starts at `U − overlap`. If `overlap ≥ D`, the row falls inside the next window. If it is still not visible then, the same argument carries it to the run after. So the row is read by the first run that reads at or after the moment it becomes visible, which is not necessarily the next run: with `D = 300`, a row with `updated_at = 900` that becomes visible at 1,200 is missed by runs reading at 1,000 and at 1,100, and found by the first run that reads at 1,200 or later.

This is an argument under the contract, not a universal proof. It needs each read to be one consistent snapshot, the delay to be really bounded by D, and U and the read time to come from the same clock as `updated_at`. If any of these fails, the window can miss rows without any error, and nothing in a single run will tell you.

So the overlap is not a tuning knob. It is the visibility-delay contract expressed in seconds, and it has hard limits:

- It guarantees nothing for rows delayed by more than D. Those *may* be missed: whether a run happens to read late enough to see them is luck, not a property of the design.
- Every run re-reads the overlap, so the merge must tolerate seeing the same version again.
- A larger overlap costs more reading on every run, so "just make it bigger" has a price and still does not cover an unbounded delay.
- It does not reveal deletes at all, which is covered below.

The tests use `D = 300` seconds. Row B is visible 250 seconds after its `updated_at`: the first run misses it and the second picks it up through the overlap. With `overlap=0` the same row is lost for good. Row E becomes visible 1,100 seconds late, outside the contract, and in this timeline the incremental runs never load it. That test asserts the miss on purpose, because that is how the pattern behaves outside its contract.

### Late beyond the contract: explicit backfill

For data later than D, the answer is not an ever-growing overlap. It is an explicit backfill over a range below the watermark:

```python
def run_backfill(con: sqlite3.Connection, changes: Iterable[Change], *,
                 contract_version: int, lo: int, hi: int,
                 read_at: int) -> list[str]:
    """Explicit re-read of an already committed range. The checkpoint stays."""
    if type(lo) is not int or type(hi) is not int:
        raise TypeError("lo and hi must be int seconds")
    if not lo < hi <= watermark(con):
        raise ValueError("backfill range must lie below the watermark")
    rows = extract(changes, read_at, lo, hi)
    apply_batch(con, Batch(contract_version, tuple(rows)))
    return [row.order_id for row in rows]
```

A backfill reads the source the same way as an incremental run and goes through the same merge, but it never moves the checkpoint. Deciding *when* and *which range* to backfill belongs to an operator or a reconciliation process. The window itself cannot know that it missed something.

### Pagination needs a tie-breaker

Sources rarely return a window in one response. A common cursor is "rows with `updated_at` greater than the last one I saw, limit N". With equal timestamps this skips rows: if a page ends in the middle of a group sharing one `updated_at`, the next page starts strictly after that value and the rest of the group disappears. The test has three rows with the same timestamp and a page size of two; the timestamp-only cursor returns two of them.

The fix is a total order: paginate on `(updated_at, order_id)`.

```python
def pages_keyset(rows: list[Change], page_size: int) -> Iterator[list[Change]]:
    """Keyset pagination on ``(updated_at, order_id)``.

    The composite cursor is a total order, so rows sharing a timestamp
    cannot fall between two pages.
    """
    check_page_input(rows, page_size)
    cursor: Cursor | None = None
    while True:
        page = [r for r in rows if cursor is None or sort_key(r) > cursor]
        page = page[:page_size]
        if not page:
            return
        yield page
        cursor = sort_key(page[-1])
```

A keyset cursor is only correct over input that is already sorted and unique by that key. In the example the extract produces exactly that, and `check_page_input` refuses anything else, together with a page size below one, instead of trusting every caller.

In the example all pages of a run are applied in one transaction and only then does the watermark move. If you checkpoint per page instead, the stored cursor must be the composite `(updated_at, order_id)`, not the timestamp alone, or a restart reintroduces the same gap. Pagination over a live source also needs a stable snapshot across pages; the simulated source has one by construction, a real API may not.

## Part 2: merge

### The conditional UPSERT

Overlap, backfill and retries all mean the same versions arrive more than once, and occasionally an older version arrives after a newer one. The merge uses SQLite's UPSERT with a condition on the version:

```python
UPSERT_V2 = """
INSERT INTO orders (order_id, updated_at, status, amount_minor, deleted,
                    currency)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (order_id) DO UPDATE SET
    updated_at = excluded.updated_at,
    status = excluded.status,
    amount_minor = excluded.amount_minor,
    deleted = excluded.deleted,
    currency = excluded.currency
WHERE excluded.updated_at > orders.updated_at
   OR (excluded.updated_at = orders.updated_at
       AND orders.currency IS NULL
       AND excluded.currency IS NOT NULL)
"""
```

`excluded` is the incoming row; `orders` is the row already stored. The SQLite documentation describes the `WHERE` clause on `DO UPDATE` as the way to turn the update into a no-op depending on the old and new values. It also states that UPSERT only reacts to uniqueness constraints (a `PRIMARY KEY`, `UNIQUE` or unique index), not to failed `NOT NULL` or `CHECK` constraints. Basic UPSERT support arrived in SQLite 3.24.0 (2018-06-04); the library version depends on how your Python was built, so check `sqlite3.sqlite_version`.

The first line of the `WHERE` gives three cases for an existing key: a newer `updated_at` replaces the row, an older one is a no-op, and an equal one is a no-op too. The second branch is the enrichment rule for the schema change in Part 3.

### Equal versions are where the contract leaks

A no-op for an equal version is right when the content is identical, which is exactly what overlap re-reads produce. It is wrong when the same `updated_at` arrives with different content. Then nobody knows which version is true, and silently keeping the first one hides a broken source contract. So before the UPSERT, the code compares the stored row:

```python
def check_same_version(con: sqlite3.Connection, row: Change,
                       has_currency: bool) -> None:
    """Raise VersionConflict when an equal version carries other content."""
    extra = ", currency" if has_currency else ""
    stored = con.execute(
        f"SELECT updated_at, status, amount_minor, deleted{extra} "
        "FROM orders WHERE order_id = ?", (row.order_id,)).fetchone()
    if stored is None or stored[0] != row.updated_at:
        return
    if stored[1:4] != (row.status, row.amount_minor, int(row.deleted)):
        raise VersionConflict(f"{row.order_id} @ {row.updated_at}: payload")
    known = stored[4] if has_currency else None
    if known is not None and row.currency is not None and known != row.currency:
        raise VersionConflict(f"{row.order_id} @ {row.updated_at}: currency")
```

In the test, order A is loaded, and then the same `updated_at` reappears through the overlap with a different amount. The run raises `VersionConflict`, the other row in that batch is not written, and the watermark stays where it was.

This read-then-write is safe here because the read happens inside the same write transaction (`BEGIN IMMEDIATE`, below). Whether the same pattern is safe in another engine depends on that engine's isolation level, which is exactly the kind of guarantee that does not transfer from SQLite automatically.

### Checkpoint atomicity and partial failure

Rows and the new watermark are written in one transaction of the same database:

```python
def apply_batch(con: sqlite3.Connection, batch: Batch,
                checkpoint: tuple[int, int] | None = None) -> None:
    """Validate and merge every row, then optionally move the checkpoint.

    Everything happens in one transaction: an unsupported contract version
    (checked even for an empty batch), one bad row, a version conflict or a
    moved checkpoint rolls back all rows of the batch and the watermark.
    """
    with transaction(con):
        check_contract_version(batch.contract_version)
        has_currency = "currency" in columns(con)
        if batch.contract_version >= 2 and not has_currency:
            raise MigrationRequired("contract v2 needs the currency column")
        for row in batch.rows:
            validate(row, batch.contract_version)
            merge_row(con, row, has_currency)
        if checkpoint is not None:
            advance_checkpoint(con, *checkpoint)
```

If the watermark lived somewhere else, a crash between the two writes could leave it ahead of data that was never committed, and the next run would start after the gap. Moving the checkpoint is a compare-and-set: the `UPDATE` includes the watermark the run started from, and if it affects zero rows, the whole batch rolls back with `CheckpointMoved`. That is a cheap guard, not a concurrency test; concurrent runs are not tested here.

The contract version is checked once per batch, inside the write transaction and before any row is merged, not only inside the row loop. Otherwise a batch with zero rows and an unknown version would pass every check that never ran and still move the watermark. The check also refuses `True`: `True in (1, 2)` is true in Python, because `bool` is a subclass of `int` and `True == 1`. Note the timing: the version check happens after extraction, because `apply_batch` receives the rows that were already read. Run parameters are checked before reading; the contract version is checked before writing.

The connection is opened with `isolation_level=None`, and the `transaction` helper issues `BEGIN IMMEDIATE`, `COMMIT` and `ROLLBACK` itself. In Python 3.11, with the default setting, `sqlite3` opens a transaction implicitly only before `INSERT`, `UPDATE`, `DELETE` or `REPLACE`, not before a `SELECT`. The existing-version check is a `SELECT`, and it has to be inside the same transaction as the write.

```python
@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[None]:
    """One explicit write transaction: commit on success, roll back on error.

    COMMIT is inside the ``try``: SQLite can refuse it (SQLITE_BUSY, a
    deferred foreign key) and leave the transaction open. SQLite may also
    have rolled back on its own already, hence the ``in_transaction`` check.
    A failure whose outcome is unknown, such as a crash during COMMIT, is
    not handled here; the next run has to converge by re-reading.
    """
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
```

`COMMIT` itself can fail. The SQLite documentation says that when `COMMIT` returns `SQLITE_BUSY` the transaction remains active, and that a `COMMIT` fails while a deferred foreign key constraint is violated. A helper that runs `COMMIT` after the `try` block would leave the connection inside an open transaction. A test forces the deferred foreign-key case and checks that the rows and the checkpoint update are rolled back and no transaction is left open. That is the limit of the claim, though: an error that SQLite reports is rolled back; a process that dies during `COMMIT`, or a caller that cannot tell whether it succeeded, is not something this helper can resolve. It is the case below, where the next run has to converge.

Partial-failure tests follow from this. In one, row G is valid and row H carries the amount `"12.50"` as text. G is merged first, H fails validation, and afterwards the table is empty and the watermark unchanged. The validation sits in Python because the SQLite table would not do it: SQLite uses dynamic typing, and in an ordinary (non-`STRICT`) table a column declared `INTEGER` converts text that is a well-formed number and stores other text as text. In a probe on SQLite 3.47.1, `"12.50"` was stored as the real number 12.5 and `"12,50"` as text, both in an `INTEGER NOT NULL` column. SQLite also has no separate Boolean storage class; `deleted` is stored as 0 or 1. In another test, the run commits but the caller never hears about it, as after a timeout. The next run re-reads the overlap, applies the same version as a no-op, and the table is identical. That is convergence through an idempotent merge, not exactly-once delivery.

Atomicity covers the target and the checkpoint only. The read from the source is a different system with its own snapshot and its own guarantees.

### Deletes

A physically deleted row has no `updated_at` to find. A timestamp window cannot see an absence, so in the test a row removed from the source simply stays in the target. The two usual answers are tombstones, where the source marks a row deleted and bumps `updated_at`, or change data capture. With a tombstone the delete is just another version, and the same ordering applies: in the test, a late, older non-deleted version arrives after the tombstone and does not resurrect the row. For sources that offer neither, the remaining tool is periodic reconciliation of the full key set, which is a different job from incremental loading.

## Part 3: schema evolution

### A nullable addition is the easy case — if you leave it nullable

Suppose the source starts sending a currency with each order. This is contract version 2. The target needs a new column, and the tempting migration is:

```sql
ALTER TABLE orders ADD COLUMN currency TEXT NOT NULL DEFAULT 'PLN'
```

It succeeds and every existing row now claims to be in PLN. Nobody knows that; the old rows were loaded under a contract that carried no currency at all. A test in the example runs exactly this statement to show the invented value appearing. The SQLite documentation says that a `NOT NULL` column added this way must have a default other than NULL, and on the populated test table SQLite does refuse the statement without one. So the constraint pushes you towards a default, and the default is where the invention happens.

The migration in the example uses only `ADD COLUMN`, keeps the column nullable, and records itself in the same transaction:

```python
def add_currency_column(con: sqlite3.Connection) -> bool:
    """Contract-v2 migration. Returns False when it was already applied.

    The column is nullable and has no default: rows loaded before the
    migration keep NULL, which this pipeline reads as "currency unknown".
    """
    with transaction(con):
        done = con.execute("SELECT 1 FROM schema_migrations WHERE name = ?",
                           (CURRENCY_MIGRATION,)).fetchone()
        if done:
            return False
        con.execute("ALTER TABLE orders ADD COLUMN currency TEXT")
        con.execute("INSERT INTO schema_migrations VALUES (?)",
                    (CURRENCY_MIGRATION,))
    return True
```

NULL here has one meaning: *unknown*. It does not mean "no currency" and it is not a placeholder for a default. Consumers that need a currency must handle unknown explicitly, which is less convenient and more honest.

If you add a `CHECK` constraint together with a new column, note that SQLite tests the new constraint against existing rows only since 3.37.0 (2021-11-27); older versions accepted the column without checking. The example avoids the question by validating in Python, where the rule is visible next to the contract version.

### Validation is part of the transaction

Contract version 2 requires a three-letter uppercase currency code on every row. A v2 batch that arrives before the migration fails with `MigrationRequired` and writes nothing. In the main schema test, a v2 incremental run brings a valid row C with `"EUR"` and an invalid row D with `"eur"`. Validation of D fails, C is rolled back with it, and the watermark stays at its previous value. The schema change did not weaken the batch's atomicity.

### Same version, new field: the enrichment policy

Old rows now hold NULL. To fill them, you backfill the old range from the source under contract v2. Those rows come back with the **same** `updated_at` as before, because the business data did not change; only the extract now carries one more field. The plain "newer wins" rule would ignore them. Treating every same-version difference as a conflict would reject them.

So the example states a policy for equal versions:

| Stored currency | Incoming currency (same `updated_at`, same core payload) | Result |
|---|---|---|
| NULL (unknown) | known code | **enrichment**: fill it, keep the version |
| known code | the same code | no-op |
| known code | a different code | `VersionConflict`, batch rolled back |
| known code | NULL (old v1 extract replayed) | no-op, the known value stays |

Any difference in the core payload (`status`, `amount_minor`, `deleted`) at an equal version is still a conflict. Enrichment is the single permitted difference, and it only moves from unknown to known.

The policy rests on an assumption the merge cannot check: that the v2 extract reports the currency the source actually recorded for that same business version. If the source never stored a currency for old orders and the producer fills one in from a country code or a default, the backfill writes a guess with the authority of a fact. Enrichment recovers values that exist upstream; it cannot recover values that were never recorded, and in that case NULL should stay. The tests cover each row of the table: the backfill fills A with `"PLN"` without moving the watermark or the version, a later attempt to change A to `"EUR"` at the same version fails, and a replayed v1 extract does not erase `"PLN"`.

One more choice needs saying out loud. A **newer** version from a producer still on contract v1 carries no currency, and the example stores NULL for it. Carrying the previous currency forward would claim that the newer version confirmed a value it never mentioned. Which choice is right depends on the domain; what matters is that it is a decision and a test, not an accident of how the `SET` clause happens to be written.

### Semantic changes do not show up in types

A nullable addition changes the shape of the data. The dangerous changes keep the shape and change the meaning:

- **Units.** If the source switches `amount_minor` from cents to whole euros, `12` is still a valid non-negative integer. The example contains a test that loads exactly that row and asserts that it is accepted. It documents a limit, not a feature: type validation cannot see a unit change.
- **Keys.** If the source starts reusing `order_id` values, or the business key becomes `(source_system, order_id)`, the UPSERT will happily merge two different orders into one row. A conflict target is only as correct as the meaning of the key.
- **Version semantics.** If `updated_at` stops changing on every modification, or a second system starts writing rows with its own clock, the merge rules above quietly stop holding.

None of these can be inferred from the data in one batch. The only lever in the example is the declared contract version: a batch tagged with an unknown version (3, say, "amounts in major units") is rejected before any row is written, and so is an empty batch with that tag, so it cannot move the watermark either. That works only if producers bump the version when meaning changes. A semantic change needs a new contract version, a migration of the stored data or a new column, and a backfill with a plan, not an in-place reinterpretation of existing values.

## What the tests check, and what they do not

The example has 56 pytest test cases (39 test functions, some parametrized) across four files: extraction windows, bootstrap coverage, visibility ties, run parameters and pagination; merge semantics and deletes; partial failure, valid and unsupported empty batches, contract-version checks, a refused `COMMIT` and checkpoint atomicity; and schema evolution. Most compare the whole target state, not row counts.

They do not test concurrent runs, process crashes at arbitrary points, a `COMMIT` whose outcome is unknown, disk failures, a real network source, clock skew, or any SQL engine other than SQLite. The minimum SQLite version for basic UPSERT is 3.24.0 according to the documentation; the tests were not run against that version. The code uses nothing beyond basic UPSERT and `ADD COLUMN` on the SQLite side (no `STRICT` tables, `RETURNING`, `DROP COLUMN` or `ALTER COLUMN`), and on the Python side it controls transactions through `isolation_level=None`, not through the `autocommit` connection attribute, which is not part of Python 3.11's `sqlite3`.

## Code and reproduction

Download [`article-03.zip`]({{ '/assets/examples/article-03.zip' | relative_url }}). It unpacks to one `article-03/` directory with everything needed to run the example:

- `README.md` and `pyproject.toml`;
- the package `incremental_demo/`: `__init__.py`, `model.py`, `source.py`, `store.py`, `merge.py`, `pipeline.py`, `demo.py`;
- the tests `tests/`: `conftest.py`, `test_windows.py`, `test_merge.py`, `test_failures.py`, `test_schema_evolution.py`.

Then, from the unpacked `article-03/` directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "pytest==7.4.2"
.venv/bin/python -m pytest -q
.venv/bin/python -m incremental_demo.demo
```

`uv` is contemporary tooling used for the reproduction, not something that was available when this article is dated, and the code does not depend on it. Any Python 3.11+ interpreter with pytest installed runs the same two `python -m` commands.

*Editorial note.* The date on this article is an editorial label for the series, not a record of when the code was executed. The test run and the demo output in the companion files were produced on 12 October 2026 with CPython 3.11.12, SQLite 3.47.1 and pytest 7.4.2, in an isolated environment created with `uv`. The SQLite and Python documentation pages cited below were read in their live versions on the same date.

## Takeaways

- Extraction completeness is a contract: cursor on a per-key strictly increasing `updated_at`, a first window (or a preload) that covers all existing history, a fixed upper bound per run, a half-open window, and an overlap at least as large as the guaranteed visibility delay. A late row within that delay is read by the first run that reads at or after the moment it becomes visible; beyond that delay it may be missed, so use an explicit backfill. What converges is current state, not history.
- Validate run parameters before anything is read, and the batch's contract version before anything is written. A valid empty batch moves the watermark, because the window was read and no matching rows were visible in that snapshot; an empty batch with an unsupported contract version must not.
- Paginate on `(updated_at, key)`, never on the timestamp alone.
- The conditional UPSERT stops old versions from regressing state. Equal versions with different content are a contract violation and should stop the run.
- Commit rows and the watermark together, and let one bad row roll back both. Put `COMMIT` inside the error handling, and accept that an unknown commit outcome is resolved by an idempotent re-run, not by the helper.
- Deletes need tombstones, CDC or reconciliation; a timestamp window cannot see them.
- Add new columns as nullable and keep "unknown" explicit. Fill them through a same-version enrichment rule, never by a default that invents data.
- Unit, key and version-meaning changes are invisible to type checks. They need a new contract version and a plan, not a quiet reinterpretation.

### Sources

- [SQLite: UPSERT](https://www.sqlite.org/lang_upsert.html) — syntax, `excluded`, the `WHERE` clause on `DO UPDATE`, uniqueness constraints only.
- [SQLite Release 3.24.0 (2018-06-04)](https://www.sqlite.org/releaselog/3_24_0.html) — "Add support for PostgreSQL-style UPSERT".
- [SQLite: ALTER TABLE](https://www.sqlite.org/lang_altertable.html) — `ADD COLUMN` restrictions, including `NOT NULL` requiring a non-NULL default.
- [SQLite Release 3.37.0 (2021-11-27)](https://www.sqlite.org/releaselog/3_37_0.html) — `ADD COLUMN` checks new `CHECK` constraints against existing rows.
- [SQLite: Datatypes In SQLite](https://www.sqlite.org/datatype3.html) — dynamic typing, type affinity of `INTEGER` columns, no separate Boolean storage class.
- [SQLite: Transaction](https://www.sqlite.org/lang_transaction.html) — `BEGIN IMMEDIATE`; a `COMMIT` that fails with `SQLITE_BUSY` leaves the transaction active.
- [SQLite: Foreign Key Support](https://www.sqlite.org/foreignkeys.html) — deferred constraints make `COMMIT` fail while they are violated.
- [Python 3.11: sqlite3](https://docs.python.org/3.11/library/sqlite3.html) — transaction control, `isolation_level`, the connection context manager, `sqlite_version`.
- [Python 3.11.0 release](https://www.python.org/downloads/release/python-3110/) — released 24 October 2022.
