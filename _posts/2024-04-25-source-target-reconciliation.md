---
layout: post
lang: en
title: "Source-to-target reconciliation: row counts are not enough"
date: 2024-04-25 09:00:00 +0200
permalink: /blog/source-target-reconciliation/
author: Rafał Prońko
categories: [data-engineering]
description: "Count and sum checks can pass when the copied rows are wrong. A Python multiset reconciliation with key diagnostics, a fixed event-time window and a fail-closed gate."
image: /assets/images/source-target-reconciliation.svg
---

# Source-to-target reconciliation: row counts are not enough

![Diagram: the reference and the target both have six rows and equal count and sum per segment, but the target is missing t4 and has t2 twice. The row difference and the key checks find both problems and the gate blocks publication.]({{ '/assets/images/source-target-reconciliation.svg' | relative_url }})

<time datetime="2024-04-25">25 April 2024</time>

A load finishes and the check after it compares two numbers: 6 rows in the source window, 6 rows in the target. Often there is a second pair too: the total amount on both sides. Both pairs match, the check is green and the data goes to the dashboard.

That check tells me the target has the right *number* of rows and the right *total*. It does not tell me it has the right *rows*. One lost record and one duplicated record with the same amount leave the count and the sum unchanged. So do two errors in opposite directions within the same segment, and a row that was reloaded under a new key.

Reconciliation is the check that answers the harder question: does the target contain exactly the rows that the reference says it should, for the same population, the same window and the same grain? In this article I build a small version of it in Python 3.11 with the standard library only, and I'm explicit about where its guarantee ends.

## First fix the question, then compare

Most false alarms and most missed errors in reconciliation come from comparing two sides that answer different questions. Before writing any comparison, I pin down five things.

**A fixed snapshot.** The reference must not change while I compare. If I count the source at 10:00 and read it again at 10:05 for the row diff, rows committed in between show up as differences that the load never caused. I extract the reference once, from one consistent read, and compare everything against that extract.

**The window, in event time.** I define the window as a half-open interval `[start, end)` in canonical UTC: both bounds and every timestamp must be exact `datetime` values with `tzinfo is timezone.utc`. A row belongs to it when `start <= event_at < end`. A row stamped exactly at midnight belongs to the next day, not to both days. Canonical UTC avoids local-time gaps and repeats when clocks change.

The window must use the same clock on both sides. If the reference selects by *event time* (when the payment happened) and the target selects by *load time* (when the row arrived in the warehouse), a payment from 23:58 that was loaded at 00:03 belongs to one side's window and not the other's. The difference is real, but it measures the query, not the load.

**The grain.** The comparison below assumes the transformation keeps the grain: one target row per reference row, with a selected set of fields copied without aggregation. If the target is aggregated (daily totals per segment), then I compare at the target's grain: I aggregate the reference independently and reconcile aggregates. Comparing raw source rows with aggregated target rows compares two different things.

**An independent reference.** If "expected" is computed by the same query or the same code path that produced the target, the comparison checks the code against itself and passes. The reference has to come from the source system's data, read through a separate path.

**A policy for nulls, types and currency.** Every compared field needs a decided representation. In the example: amounts are integers in cents, so I never compare floating-point values for equality; `True` is rejected as an amount, even though in Python `bool` is a subclass of `int`; timestamps must be exact `datetime` values carrying the `timezone.utc` singleton; nulls are not allowed in compared fields; and currency is part of the compared row. Amounts in different currencies are never summed or converted inside the check. A conversion is a transformation with its own rate and timestamp, and it deserves its own reconciliation.

## Why the cheap checks pass wrong data

The synthetic reference window has six payments. Three retail payments have the same amount, 4900 cents, which is ordinary for a subscription price:

| key | segment | amount (cents) |
|---|---|---|
| t1 | retail | 4900 |
| t2 | retail | 4900 |
| t3 | retail | 12550 |
| t4 | retail | 4900 |
| t5 | business | 120000 |
| t6 | business | 30000 |

Now four broken targets. Each has six rows, and each has the same count *and* the same sum *per segment* as the reference — a stricter version of the cheap check than a single global count and total:

1. **Loss plus duplication.** `t4` is missing and `t2` was loaded twice. Same count, and since both rows are 4900 cents in retail, the same retail total.
2. **Offsetting errors in one segment.** `t5` lost 1000 cents and `t6` gained 1000 cents. Same keys, same count, same business total.
3. **Key reassignment.** `t3` arrived under the key `t9`, for example after a surrogate key was regenerated. The content matches; the key does not.
4. **Wrong type.** `t6` arrived with `30000.0`, a float. In Python `30000.0 == 30000`, so a total does not notice.

There is also one target that *should* pass: the same six rows in a different order. Row order is not part of the contract, so the comparison must not depend on it.

Comparing sets of keys catches the first and third case, but not the second. It also misses multiplicity: a set does not distinguish one occurrence of `t2` from two. A dictionary keyed by `row_id` is worse, because building it from a list silently overwrites the first `t2` with the second.

What I need is a **multiset**: a collection that remembers how many times each element occurs. In Python that is [`collections.Counter`](https://docs.python.org/3.11/library/collections.html#collections.Counter). Its documentation describes it as a `dict` subclass for counting hashable objects, similar to bags or multisets in other languages. A missing element has a count of zero, and subtraction keeps only positive counts. That makes `Counter(reference) - Counter(target)` exactly the reference rows that have no match in the target, with repeats taken into account, and the reverse subtraction gives target rows that have no match in the reference.

`Counter` uses hashing to find candidates, but a match still requires equality of the whole row. Two different rows that happen to share a hash value are not treated as equal. That matters later, when I compare it with digest-based approaches.

## The example

The whole program is below. It needs Python 3.11 and nothing else. You can [download `reconcile.py`]({{ '/assets/examples/source-target-reconciliation/reconcile.py' | relative_url }}) and the [tests]({{ '/assets/examples/source-target-reconciliation/test_reconcile.py' | relative_url }}), then run:

```sh
python3.11 reconcile.py
python3.11 -m unittest -v
```

```python
"""Source-to-target reconciliation for one closed event-time window.

Rows are compared as multisets with collections.Counter, keys are
checked for uniqueness on both sides and the publish decision fails
closed. Python 3.11, standard library only, amounts in integer cents.
Timestamps and window bounds must use canonical timezone.utc.

Run:   python3.11 reconcile.py
Tests: python3.11 -m unittest -v
"""
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

UTC = timezone.utc


class Row(NamedTuple):
    row_id: str
    event_at: datetime  # event time, not load time; canonical timezone.utc
    segment: str
    currency: str  # compared as-is, never converted
    amount_cents: int


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    duplicate_reference_keys: dict[str, int] = field(default_factory=dict)
    duplicate_target_keys: dict[str, int] = field(default_factory=dict)
    only_reference: list[Row] = field(default_factory=list)
    only_target: list[Row] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    publish: bool
    reasons: tuple[str, ...]
    report: Report


def is_utc(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is timezone.utc


def check_window(start: datetime, end: datetime) -> None:
    if not (is_utc(start) and is_utc(end)):
        raise ValueError("window bounds must be UTC datetimes")
    if not start < end:
        raise ValueError("window must satisfy start < end")


def find_problems(rows: object, side: str, start: datetime, end: datetime,
                  currencies: frozenset[str]) -> list[str]:
    if type(rows) is not list:  # a fixed, materialized snapshot
        return [f"{side}: expected a list of rows, got {type(rows).__name__}"]
    found = []
    for i, row in enumerate(rows):
        where = f"{side}[{i}]"
        if not isinstance(row, tuple) or len(row) != len(Row._fields):
            found.append(f"{where}: expected a {len(Row._fields)}-field row")
            continue
        row_id, event_at, segment, currency, amount = row
        for name, value in (("row_id", row_id), ("segment", segment)):
            if type(value) is not str or not value or value != value.strip():
                found.append(f"{where}: {name} must be a trimmed, non-empty "
                             "str")
        if not is_utc(event_at):
            found.append(f"{where}: event_at must be a UTC datetime")
        elif not start <= event_at < end:
            found.append(f"{where}: event_at outside [start, end)")
        if type(currency) is not str or currency not in currencies:
            found.append(f"{where}: currency {currency!r} not allowed")
        if type(amount) is not int:  # rejects bool, float, Decimal, None
            found.append(f"{where}: amount_cents must be int")
    return found


def duplicates(keys: Counter[str]) -> dict[str, int]:
    return {key: n for key, n in sorted(keys.items()) if n > 1}


def counted(keys: dict[str, int]) -> list[str]:
    return [f"{key} x{n}" for key, n in keys.items()]


def reconcile(reference: list[Row], target: list[Row], start: datetime,
              end: datetime, currencies: frozenset[str]) -> Report:
    check_window(start, end)
    problems = (find_problems(reference, "reference", start, end, currencies)
                + find_problems(target, "target", start, end, currencies))
    if problems:
        return Report(problems=problems)
    ref_rows = Counter(Row._make(row) for row in reference)
    tgt_rows = Counter(Row._make(row) for row in target)
    ref_keys = Counter(row[0] for row in reference)
    tgt_keys = Counter(row[0] for row in target)
    return Report(
        missing_keys=sorted(ref_keys.keys() - tgt_keys.keys()),
        extra_keys=sorted(tgt_keys.keys() - ref_keys.keys()),
        duplicate_reference_keys=duplicates(ref_keys),
        duplicate_target_keys=duplicates(tgt_keys),
        only_reference=sorted((ref_rows - tgt_rows).elements()),
        only_target=sorted((tgt_rows - ref_rows).elements()),
    )


def gate(reference: list[Row], target: list[Row], start: datetime,
         end: datetime, *, currencies: frozenset[str] = frozenset({"EUR"}),
         allow_empty: bool = False) -> Decision:
    report = reconcile(reference, target, start, end, currencies)
    reasons = list(report.problems)
    if not report.problems and not reference and not allow_empty:
        reasons.append("reference window is empty and allow_empty is False")
    key_checks = [
        ("missing from target", report.missing_keys),
        ("unexpected in target", report.extra_keys),
        ("duplicated in reference", counted(report.duplicate_reference_keys)),
        ("duplicated in target", counted(report.duplicate_target_keys)),
    ]
    reasons += [f"keys {text}: {', '.join(keys[:5])}"
                for text, keys in key_checks if keys]
    row_checks = [
        ("reference row(s) without exact match", report.only_reference),
        ("target row(s) without exact match", report.only_target),
    ]
    reasons += [f"{len(rows)} {text}" for text, rows in row_checks if rows]
    return Decision(not reasons, tuple(reasons), report)


def control_totals(rows: list[Row]) -> dict[tuple[str, str], tuple[int, int]]:
    """The weak check: row count and cents per (segment, currency)."""
    totals: dict[tuple[str, str], tuple[int, int]] = {}
    for _, _, segment, currency, cents in rows:
        count, total = totals.get((segment, currency), (0, 0))
        totals[(segment, currency)] = (count + 1, total + cents)
    return totals


START = datetime(2024, 4, 1, tzinfo=UTC)
END = START + timedelta(days=1)


def at(hour: int, minute: int = 0) -> datetime:
    return START + timedelta(hours=hour, minutes=minute)


REFERENCE = [
    Row("t1", at(0, 5), "retail", "EUR", 4900),
    Row("t2", at(3, 10), "retail", "EUR", 4900),
    Row("t3", at(7, 45), "retail", "EUR", 12550),
    Row("t4", at(9, 0), "retail", "EUR", 4900),
    Row("t5", at(12, 30), "business", "EUR", 120000),
    Row("t6", at(18, 15), "business", "EUR", 30000),
]
t1, t2, t3, t4, t5, t6 = REFERENCE

SCENARIOS = [  # (name, reference, target, allow_empty, should_publish)
    ("same rows, reordered", REFERENCE, REFERENCE[::-1], False, True),
    ("t4 lost, t2 loaded twice", REFERENCE,
     [t1, t2, t2, t3, t5, t6], False, False),
    ("offsetting errors in one segment", REFERENCE,
     [t1, t2, t3, t4, t5._replace(amount_cents=119000),
      t6._replace(amount_cents=31000)], False, False),
    ("t3 reloaded under key t9", REFERENCE,
     [t1, t2, t3._replace(row_id="t9"), t4, t5, t6], False, False),
    ("amount arrives as float", REFERENCE,
     [t1, t2, t3, t4, t5, t6._replace(amount_cents=30000.0)], False, False),
    ("empty window, default policy", [], [], False, False),
    ("empty window, allow_empty=True", [], [], True, True),
]


def show(row: Row) -> str:
    return f"{row.row_id} {row.segment} {row.currency} {row.amount_cents}"


def main() -> int:
    print(f"window [{START.isoformat()}, {END.isoformat()}) event time, UTC")
    unexpected = 0
    for name, reference, target, allow_empty, should_publish in SCENARIOS:
        decision = gate(reference, target, START, END, allow_empty=allow_empty)
        weak = control_totals(reference) == control_totals(target)
        verdict = "PUBLISH" if decision.publish else "BLOCK"
        print(f"\n{name}\n  count+sum per segment equal: {weak}"
              f"\n  gate: {verdict}")
        for reason in decision.reasons:
            print(f"  - {reason}")
        report = decision.report
        for row in report.only_reference:
            print(f"    only in reference: {show(row)}")
        for row in report.only_target:
            print(f"    only in target:    {show(row)}")
        unexpected += decision.publish != should_publish
    print(f"\nunexpected decisions: {unexpected}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
```

A few decisions in this code deserve a word.

**Validation collects problems instead of guessing.** `find_problems` walks every row and records everything it doesn't accept: a row with the wrong number of fields, an empty or padded key, a naive or non-UTC timestamp, a timestamp outside `[start, end)`, a currency outside the allowed set, an amount whose type is not exactly `int`. A padded key like `" t1"` is rejected rather than silently stripped. Canonicalization belongs in an explicit, tested step before reconciliation, not hidden inside the check that is supposed to catch differences.

The UTC check enforces canonical UTC: `type(value) is datetime` and `value.tzinfo is timezone.utc`. A zero offset alone is insufficient. During London's repeated autumn hour, `2023-10-29 01:30` with `fold=1` has a zero offset and denotes the same instant as `2023-10-29 01:30 UTC`, yet Python compares those two datetimes as unequal. Accepting both representations would produce a false row difference. The [Python 3.11 datetime documentation](https://docs.python.org/3.11/library/datetime.html#datetime-objects) describes the comparison exception for repeated intervals.

Normalization is an explicit preprocessing step outside reconciliation. It applies to the timestamps on both sides and to `start` and `end`.

First require an exact, aware `datetime`: reject a value if `type(value) is not datetime`, `value.tzinfo is None` or `value.utcoffset() is None`. Only then convert it with `value.astimezone(timezone.utc)`.

Two shortcuts do not count as normalization. Calling `astimezone` on a naive value guesses its zone, because Python assumes the host's local timezone. Attaching UTC with `replace(tzinfo=timezone.utc)` is not a conversion and does not resolve an unknown zone. The source contract must supply the zone and, for an ambiguous local time, the correct fold before this preprocessing can begin.

`is_utc` itself neither converts nor guesses. A row timestamp whose `tzinfo` is anything other than `timezone.utc` is reported as invalid, even for a regional zone such as `Europe/London` at zero offset. A `start` or `end` bound whose `tzinfo` is not `timezone.utc` raises `ValueError`.

The window check inside validation is deliberate. The inputs are already selected for the window, so a row outside it means one side was selected with a different rule, for example an inclusive `end`. That is a defect of the comparison setup, and the gate refuses to approve it.

Validation also requires a `list`. A generator can be consumed once, and a lazy query can be re-executed against a newer state. A materialized list is the snapshot the comparison talks about.

**Two levels of diagnostics.** The row-level difference says *that* the sides disagree. The key-level view says *how*: which keys are missing, which are unexpected, which occur more than once. I check key uniqueness on both sides. A duplicate key in the target is a load error. A duplicate key in the reference means the contract "one row per `row_id` in the window" is broken at the source, or the reference query has the wrong grain. The test `test_duplicate_reference_key_blocks_even_if_sides_agree` covers the second case: both sides contain the same two rows with the same key, the row multisets are equal, and the gate still blocks, because the key contract is violated.

The `reasons` list names at most five keys per check (`keys[:5]`), so an alert stays readable when a load loses thousands of rows. Only the message is shortened: the full lists stay in `decision.report`, and that is what an investigation or an alert attachment should use.

**An empty window needs an explicit decision.** An empty reference may be correct (no payments on a public holiday) or a symptom (an expired credential, a wrong filter, a missed partition). With empty input on both sides every difference is empty too, so a naive gate would approve it. Here an empty reference blocks unless the caller passes `allow_empty=True` for windows where emptiness is expected. That flag does not hide anything else: an empty reference with rows in the target still fails on unexpected keys.

**The weak check stays for contrast.** `control_totals` computes count and cents per `(segment, currency)`. The gate does not need it — equal row multisets imply equal totals — but the demo prints it next to each decision to show what the cheap check would have said.

## What the run shows

A note on dates: the date on this post is an editorial placement, not a historical publication date. The output and test results below come from a run in October 2026 with CPython 3.11.12, a later patch release of Python 3.11; the code uses only the Python 3.11 standard library.

This is the complete output of `reconcile.py` from that run:

```text
window [2024-04-01T00:00:00+00:00, 2024-04-02T00:00:00+00:00) event time, UTC

same rows, reordered
  count+sum per segment equal: True
  gate: PUBLISH

t4 lost, t2 loaded twice
  count+sum per segment equal: True
  gate: BLOCK
  - keys missing from target: t4
  - keys duplicated in target: t2 x2
  - 1 reference row(s) without exact match
  - 1 target row(s) without exact match
    only in reference: t4 retail EUR 4900
    only in target:    t2 retail EUR 4900

offsetting errors in one segment
  count+sum per segment equal: True
  gate: BLOCK
  - 2 reference row(s) without exact match
  - 2 target row(s) without exact match
    only in reference: t5 business EUR 120000
    only in reference: t6 business EUR 30000
    only in target:    t5 business EUR 119000
    only in target:    t6 business EUR 31000

t3 reloaded under key t9
  count+sum per segment equal: True
  gate: BLOCK
  - keys missing from target: t3
  - keys unexpected in target: t9
  - 1 reference row(s) without exact match
  - 1 target row(s) without exact match
    only in reference: t3 retail EUR 12550
    only in target:    t9 retail EUR 12550

amount arrives as float
  count+sum per segment equal: True
  gate: BLOCK
  - target[5]: amount_cents must be int

empty window, default policy
  count+sum per segment equal: True
  gate: BLOCK
  - reference window is empty and allow_empty is False

empty window, allow_empty=True
  count+sum per segment equal: True
  gate: PUBLISH

unexpected decisions: 0
```

For every broken target, `count+sum per segment equal` is `True`, and the gate blocks anyway. The report points at the fault: `t4` missing and `t2` duplicated; two business rows with changed amounts; `t3` gone and `t9` unexpected; a float in the amount column. The reordered target passes.

The program finishes with `unexpected decisions: 0` and exit code 0. If any scenario got a different decision than expected, it would exit with code 1.

## What the tests check

The 24 tests in `test_reconcile.py` use `unittest` from the standard library. They cover:

- the first three broken targets (loss plus duplication, offsetting errors, key reassignment), each with an explicit assertion that row count and per-segment totals are equal *and* the gate blocks, plus assertions on the report: the missing, unexpected and duplicated keys and the differing rows (for the offsetting case, the exact reference rows and the number of target rows);
- the float target separately: the validation tests reject `4900.0` as an amount, and the demo test runs every scenario through `main()` and expects the blocking decision; there is no totals assertion for it in the tests;
- that the reordered target publishes;
- that `Counter` keeps multiplicity, and that a duplicate key in the reference blocks even when both sides agree;
- the empty-window policy, including that `allow_empty` does not hide extra target rows and that an empty target is a loss, not an empty window;
- invalid rows: `True`, `4900.0`, `None` and `"4900"` as amounts; empty, padded and non-string keys; unknown, lowercase and unhashable currencies; naive and `+02:00` timestamps and ISO strings; wrong row shapes; an iterator instead of a list;
- the half-open window: a row at `start` is accepted, a row at `end` or one microsecond before `start` is rejected;
- invalid window bounds (`start >= end`, naive datetimes), which raise `ValueError`;
- the canonical UTC contract: exact `datetime` types, rejection of a different fixed zero-offset zone, rejection of London's autumn `fold=1` timestamp on either side, an exact match after explicit normalization, rejection of a raw regional start or end bound, and acceptance after both bounds are normalized.

The four London tests use the standard library's `zoneinfo` and the host's IANA timezone data. If `Europe/London` data is unavailable, those four tests are explicitly skipped with a reason; no dependency is installed and the DST regression remains unverified on that host. In the October 2026 run, all 24 tests executed with no skips.

The tests use the `assert*` methods of `unittest.TestCase` rather than bare `assert` statements. Python removes bare `assert` statements when it runs with `-O`; the methods keep checking.

## Fail closed, and what it costs

The gate returns a decision; it does not publish. Publishing is a separate step, and it only runs on an explicit `publish=True`:

```python
# Fragment, not part of reconcile.py: how a caller uses the decision.
# reference, target, start, end and window (the caller's name for
# [start, end)) come from the caller; the two functions are hypothetical.
decision = gate(reference, target, start, end)
if decision.publish:
    switch_consumers_to(window)  # hypothetical publish step
else:
    alert(decision.reasons)      # consumers keep the last approved window
```

Fail closed means the absence of a positive result is a negative result. Invalid rows do not raise an exception that some wrapper might catch and log; they produce `publish=False` with the reasons. A misconfigured window does raise `ValueError`, because no comparison was possible at all — and the caller must treat an exception the same way: no decision, no publish. A check that timed out, crashed or never ran is not a pass.

That promise covers the rows, not every argument. The configuration is trusted input: `allow_empty` must be a real `bool` and `currencies` a `frozenset[str]`, as the signature says, and the example does not re-check those types at run time. A wrong type does not fail safely. The string `"False"` is truthy, so `allow_empty="False"` would approve an empty window, and `currencies="EUR"` would accept the currency `"E"` through substring membership. So the caller builds and validates the configuration (from a reviewed file, with explicit types) before it calls `gate`, and a configuration error stops the run the same way an exception does: nothing is published.

To make that safe, the new data has to land somewhere consumers don't read yet: a staging table, a new partition, a new version behind a view. Only an approved window gets switched in.

The cost is freshness. When the gate blocks, consumers keep seeing the last approved window, which may be a day old. I accept that trade, but I want it visible: consumers should be able to see which window the data is approved up to, and the alert should carry the report — missing keys, duplicates, the differing rows — not just a red status. A gate that blocks too often for false reasons gets bypassed, so the setup decisions above (snapshot, event-time window, grain) matter as much as the comparison itself.

Not every difference has to block in every system. A team may decide, for example, that a target lagging behind a still-open window is a warning. That is a policy decision, made per check and written down. In this example every difference blocks, because the window is closed and the transformation is a copy: any difference means the target is not what the reference says.

## Where the exact comparison stops

**One snapshot does not span two databases.** In PostgreSQL, a transaction at the `REPEATABLE READ` isolation level sees a snapshot as of the start of its first non-transaction-control statement, so successive `SELECT`s within it see the same data ([PostgreSQL 16, Transaction Isolation](https://www.postgresql.org/docs/16/transaction-iso.html)). That is a guarantee about one transaction in one database. If I open one such transaction on the source and another on the warehouse, I get two stable views taken at two different moments, and nothing links them. The usual way out is to extract the reference from a single snapshot and treat that extract as the fixed point. It also helps to compare only closed windows, which the source no longer changes, or to record the source's commit position or watermark with the extract.

**Deletes.** If a row is hard-deleted in the source after it was loaded, a reference built from the current source state no longer contains it, and the target shows an unexpected key. That is a correct signal if the target is supposed to mirror the current state. If the target is deliberately append-only history, the reference has to be built from the change history, not from the current table. Soft deletes are simpler: the deletion flag is a compared field like any other, or both sides filter it out with the same rule.

**Late data and backfills.** Event-time windows have a catch: a row for Monday can arrive on Wednesday. Reconciling Monday on Tuesday morning compares an incomplete reference with an incomplete target and may pass; Wednesday's late row then lands without any check. I decide when a window counts as closed (for example, after an allowed lateness), and I re-run reconciliation for every window that a late load or a backfill touched. An approval is valid for a window as of a snapshot, not forever.

**Cost at scale.** The example holds both sides in memory, and each `Counter` stores every distinct row. That works only while both sides of a window fit comfortably in one process, and the limit depends on row width and hardware, not on a magic number. The first step is to partition the comparison so each partition fits and a failure points at a small slice.

Partitioning has a trap: the row difference survives it, but key uniqueness does not by itself. If the same `row_id` appears once in retail and once in business on both sides, the whole-window gate blocks on the duplicate key, while a retail gate and a business gate each see the key once and both pass. Partitioning by day or by segment alone is therefore not enough. Either partition deterministically by key, for example by a stable digest of `row_id` (not Python's salted `hash()`, see below) modulo the number of partitions, so every occurrence of a key lands in the same partition, or keep a separate key-uniqueness check (it needs only the keys, which are narrower than full rows) across the whole original window and block when it fails.

Equal partition-level counts and sums can narrow the search when they *differ*, but as the examples above show, equal totals do not prove a partition is clean. Skipping the exact comparison for partitions with equal totals brings back the weakness this article is about. A random sample of rows catches gross errors quickly, but a single missing or duplicated row easily falls outside the sample.

**The same idea in SQL.** When both sides are in the same database, for example after the reference extract has been loaded into a staging table next to the target, the multiset difference has a direct SQL form. PostgreSQL documents that `EXCEPT` returns the rows of the first query that are not in the result of the second and eliminates duplicates unless `EXCEPT ALL` is used. Both queries must be union compatible: the same number of columns, with compatible types ([PostgreSQL 16, Combining Queries](https://www.postgresql.org/docs/16/queries-union.html)). The sketch below is a concept only; I did not run it for this article:

```sql
-- Concept, not executed here: reference rows without a match in the target.
SELECT row_id, event_at, segment, currency, amount_cents
FROM reference_window
EXCEPT ALL
SELECT row_id, event_at, segment, currency, amount_cents
FROM target_window;
```

`ALL` matters for the same reason `Counter` matters. The [`SELECT` reference](https://www.postgresql.org/docs/16/sql-select.html) spells out the semantics: with `ALL`, a row that occurs *m* times on the left and *n* times on the right appears max(*m* − *n*, 0) times in the result, which is `Counter` subtraction. Without `ALL`, the result has no duplicate rows, so the reverse query (target `EXCEPT` reference) would not report the second `t2` at all: one copy exists in the reference, and that is enough to remove it. The reverse query with `EXCEPT ALL` gives the target rows without a match. The key checks are a separate `GROUP BY row_id HAVING count(*) > 1` per side. In the example, compared fields cannot be null. If they can be in your data, check how your engine treats nulls in set operations and in joins before relying on either.

**Hashes and canonicalization.** To avoid moving full rows between systems, a common approach is to compute a digest of each row, or of each partition, on both sides and compare the digests. This has two limits.

First, both sides must serialize a row in exactly the same way before hashing: the same field order and separator, escaping of the separator inside values, the same representation of null versus an empty string, the same number formatting, timestamp precision and time zone, trimming, character encoding and normalization. Two systems that format `12550` and `125.50`, or `2024-04-01 00:05:00` and `2024-04-01T00:05:00Z`, produce different digests for the same row. Two values joined without escaping can produce the same string for different rows. Python's built-in `hash()` is not an option at all: by default, hash values of `str` and `bytes` are salted with a random value, constant within one process but not predictable between runs ([Python 3.11, Data model, `__hash__`](https://docs.python.org/3.11/reference/datamodel.html#object.__hash__)).

Second, a digest is shorter than the row, so different rows can share one. With a good cryptographic hash an accidental collision is very unlikely, but a collision means a real difference goes unreported. Combining row digests into a partition digest adds a new failure mode: XOR of row digests, for instance, cancels any row that occurs twice, which is exactly the duplicate I wanted to catch. Digest comparison is a reasonable way to find *which* partitions differ. It is weaker evidence that they don't, and the exact comparison remains the reference behavior.

**Agreement is not truth.** Reconciliation shows that the target matches the reference for the compared fields, population and window. If the source recorded a wrong amount, the copy repeats it faithfully and the gate passes. Fields I don't compare can be broken, and rows outside the window or filter never take part. The check is as wide as its definition, and I keep that definition next to the report.

## Summary

Row count and total sum are cheap and worth keeping as a first signal, but they cannot approve a load: loss plus duplication, offsetting errors within a segment and reassigned keys all leave them unchanged. What approves a load is an exact comparison of rows as multisets against an independent reference, extracted once from a fixed snapshot, for the same half-open event-time window in UTC and at the same grain. Around it go key uniqueness checks on both sides, an explicit policy for empty windows, strict types and currency rules, and a gate that publishes only on an explicit pass.

The guarantee has limits I name rather than hide: it doesn't span two databases' snapshots by itself, late data and backfills reopen windows, scale forces partitioning that must keep each key's rows together, and digests trade certainty for cost. Within those limits, the decision rests on the rows themselves, not on two numbers that happen to match.

### Sources

- [Python 3.11: `collections.Counter`](https://docs.python.org/3.11/library/collections.html#collections.Counter) — `dict` subclass for counting hashable objects, zero count for missing elements, multiset subtraction keeping only positive counts, `elements()`.
- [Python 3.11: Data model, `object.__hash__`](https://docs.python.org/3.11/reference/datamodel.html#object.__hash__) — salted hash values of `str` and `bytes`.
- [Python 3.11: `datetime`](https://docs.python.org/3.11/library/datetime.html#datetime-objects) — aware and naive values, repeated-interval comparison rules and explicit `astimezone` conversion.
- [PostgreSQL 16: Transaction Isolation](https://www.postgresql.org/docs/16/transaction-iso.html) — the snapshot seen by a `REPEATABLE READ` transaction.
- [PostgreSQL 16: Combining Queries (`UNION`, `INTERSECT`, `EXCEPT`)](https://www.postgresql.org/docs/16/queries-union.html) — `EXCEPT` versus `EXCEPT ALL` and union compatibility.
- [PostgreSQL 16: `SELECT`](https://www.postgresql.org/docs/16/sql-select.html) — the max(*m* − *n*, 0) multiplicity rule of `EXCEPT ALL`.
