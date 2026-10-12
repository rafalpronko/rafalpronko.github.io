---
layout: post
lang: en
title: "A quality gate before migration: checking the data, not the job status"
date: 2023-11-09 09:00:00 +0100
permalink: /blog/quality-gate-before-migration/
author: Rafał Prońko
categories: [data-engineering]
description: "A pre-cutover reconciliation gate in Python: same scope on both sides, multiset comparison with Counter, fail-closed decisions and keeping the old approved dataset live."
image: /assets/images/quality-gate-before-migration.svg
---

# A quality gate before migration: checking the data, not the job status

![Diagram of a pre-cutover quality gate: the approved reference and the staged candidate are validated against the same scope and compared as multisets of rows. Only a pass may move readers to the checked candidate; a block, an error or an unapproved empty window keeps the old approved dataset live.]({{ '/assets/images/quality-gate-before-migration.svg' | relative_url }})

<time datetime="2023-11-09">9 November 2023</time>

> **Editorial note.** The date above places this post in the series. It isn't a historical publication or execution date: the article was written in 2026, and the example was reproduced on 12 October 2026 with CPython 3.11.12 on macOS (arm64), not with Python 3.11.0 or on a 2023 system. In that run, the module printed the output shown below and all 37 unit tests passed, both normally and under `python -O`. The tests cover only the decision logic of the module. No database, concurrent writer, real cutover or rendering of this site was tested.

A migration usually ends with a moment that looks like a formality. The new pipeline has run in parallel with the old one, the job is green, and someone asks: can we switch the readers over?

A green job tells me that the process finished without raising an error. It doesn't tell me that the new pipeline produced the same data as the old one. A copy can complete successfully while losing the rows of one segment, writing a record twice or assigning an amount to the wrong customer. None of those mistakes has to end in an exception.

So before the cutover I want a gate that looks at the data itself. It compares the dataset readers rely on today with the candidate produced by the new path, for exactly the same question. If they don't match, or if the comparison can't be performed, the answer is "don't switch", and readers keep the old, approved dataset.

In this article I build such a gate in Python 3.11, using only the standard library and synthetic data. It is a demonstration of the decision logic. It doesn't migrate any database, and I'm not describing a particular deployment.

## Where the gate sits

I'm assuming a shadow or staging run before cutover. The old pipeline still feeds the dataset readers use. The new pipeline writes its output to a separate staging location that nobody reads yet.

The gate compares two sides:

- **reference**: the approved dataset produced by the old path,
- **candidate**: the output of the new path, sitting in staging.

There are only two outcomes. Either the candidate passes and may become the dataset readers see, or it is blocked and readers keep the reference. "The check didn't run" or "the check crashed" is not a third outcome. It is a block.

The price is freshness. While the gate blocks, readers stay on the old approved state. I accept that price deliberately, but I want it visible: the team gets the report, not just a red tile.

## Same question on both sides

Before I compare anything, I make sure that both sides answer the same question. A difference between two datasets is only meaningful if they describe the same data:

- the same **population**: the same filters, for example business unit or record type,
- the same **source snapshot**: one consistent cut of the source that both extracts read, so that neither side sees writes the other side missed,
- the same **time window**, in UTC, as a half-open interval `[start, end)`, so that a record at the boundary belongs to exactly one window,
- the same **currency**,
- the same **selected fields**.

If the reference counts transactions by event time and the candidate by load time, the difference isn't a data error. It is the answer to two different questions.

The snapshot needs a closer look. Writing the same watermark, for example "everything up to 23:59:59", on both extracts doesn't by itself make them a consistent snapshot. On a mutable source, rows can be updated or arrive late while an extract is running, and two reads with the same label can still see different data. The consistency has to come from the source, and it has to be **one** cut that both extracts read, not two cuts that happen to carry the same name. How depends on the source:

- an immutable export file that both sides read,
- a versioned table that both sides read at the same fixed version,
- one database snapshot shared by both extracts, for example a snapshot exported by one transaction and imported by the other reader, or reads pinned to the same log position or timestamp. Two separate repeatable-read transactions started at different moments are two different snapshots, even with the same isolation level and the same label,
- a change-data-capture stream, but only if both sides start from the same consistent base snapshot with a clean handoff to the log at the position where that snapshot was taken, and apply every committed transaction completely, in commit order, up to the same end position. When the data comes from several streams, topics or partitions, a matching offset in each of them isn't enough: the end positions have to form one coordinated cut across all of them.

Matching labels, watermarks or terminal offsets don't prove any of this on their own. If the source offers none of these guarantees, the label describes an intention, not a cut.

In the code, that scope is a frozen `Scope` object passed in for each side. The gate requires both to be equal, requires the snapshot identifier and the currency to be non-blank strings, and requires all four window bounds to be `datetime` objects, not a `time` of day or a `date`, that are timezone-aware with a zero UTC offset. The type check comes first: an aware `time` also has a zero UTC offset and can be ordered, but it names no day at all.

One limitation is important here. `Scope` is metadata **asserted by the caller**. The gate checks that the two descriptions agree and are well-formed. It doesn't prove that the rows were actually filtered by that window, read at a consistent snapshot or converted to that currency. That proof belongs to the code that extracts the data, or to checks against the rows themselves. Matching labels on two wrongly filtered datasets will pass.

I also keep amounts as integer numbers of cents. As the Python tutorial puts it, most decimal fractions cannot be represented exactly as binary fractions, so I don't compare floats for equality.

## Why a count and a total are not enough

The most common migration check compares the number of rows and the sum of the amounts. It is cheap and better than nothing, but it lets whole classes of errors through.

If one row is lost and another is written twice, the count still matches. If 10.00 PLN moves from one transaction to another, the global total still matches. Errors compensate each other, and the aggregate shows nothing.

Comparing sets of identifiers is better, but blind to multiplicity. A set cannot tell one occurrence of a key from three. A dictionary keyed by identifier has the same problem in a different form: when I build it from a list, a later occurrence silently overwrites the earlier one, and the duplicate disappears before I can see it.

Even a perfect identifier match says nothing about content. If transaction `t3` is reassigned from segment B2C to B2B, the count, the global total and the set of identifiers are all unchanged. Only the per-segment totals move, and only if I happen to check them.

What I need is a **multiset** of whole rows: a structure that remembers how many times each exact row occurs. In Python that is `collections.Counter`. The documentation describes it as a dict subclass for counting hashable objects, similar to bags or multisets in other languages. A missing element has a count of zero. Subtracting one counter from another keeps only positive counts, and `elements()` repeats each element as many times as its count.

So `Counter(reference) - Counter(candidate)` gives exactly the rows that exist only on the reference side, including repetitions. The opposite subtraction gives the rows that exist only on the candidate side. Order doesn't matter, multiplicity does, and every compared field takes part.

## Validate before you compare

A multiset comparison is only as good as its input. Python won't stop me from putting a `bool` where an amount should be: the documentation describes Booleans as a subtype of integers, so `True` is an instance of `int`, and `True == 1`. A blank string looks like an identifier. A list instead of a tuple isn't hashable and will raise in the middle of the comparison.

Before reconciliation I therefore validate every row on both sides:

- it must be a tuple of exactly three elements,
- `row_id` and `segment` must be non-blank strings,
- `amount_cents` must be exactly `int`. I check `type(amount) is not int`, because `isinstance(True, int)` is true.

A duplicate `row_id` on the reference side gets special treatment. In this contract `row_id` is unique within the window. If the reference itself violates that, I have no trustworthy baseline, and the result is "invalid reference", not a diff. The gate doesn't try to guess which of the two copies is the real one. A duplicate on the candidate side is a regular finding: the new pipeline wrote a row twice.

## Fail closed, and decide about empty windows explicitly

Every check produces a list. An empty list means "nothing found". Any non-empty list blocks.

Fail closed means that the absence of a result is not a positive result. If the comparison raised an exception, the gate returns a blocking decision with the error in the report. Validation errors are not exceptions someone can catch and ignore. They are a decision: "does not pass".

An empty window deserves its own policy. Two empty sides are equal, so a naive comparison passes. But an empty candidate is also what a broken extraction looks like, and when both sides are empty, nothing in the data distinguishes a quiet window from a window that was never extracted. So by default the gate blocks an empty reference. The caller can pass `allow_empty=True` when an empty window is legitimate, for example a day with no transactions in a small segment. That flag allows emptiness. It doesn't hide differences: if the reference is empty and the candidate isn't, the extra rows still block.

## The example

The module below contains the gate, a simulated promotion step and a `main` function that runs a few scenarios on synthetic data. It requires Python 3.11 and only the standard library.

The module, its tests, a short README and a `pyproject.toml` with no dependencies are available as a [ZIP archive]({{ '/assets/examples/quality-gate-before-migration.zip' | relative_url }}). It unpacks into a `quality-gate-before-migration/` directory with the module in `examples/quality_gate.py`. From that directory I run:

```
python3.11 examples/quality_gate.py
```

```python
"""Pre-cutover quality gate: compare a reference snapshot with a candidate.

Synthetic data only. Python 3.11, standard library only.
Run from the directory that contains examples/:
    python3.11 examples/quality_gate.py
"""
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

Row = tuple[str, str, int]  # (row_id, segment, amount_cents)
FIELDS = ("row_id", "segment", "amount_cents")


@dataclass(frozen=True)
class Scope:
    snapshot: str    # id of the consistent source cut both sides were read at
    start: datetime  # UTC, inclusive
    end: datetime    # UTC, exclusive
    currency: str
    fields: tuple[str, ...] = FIELDS


@dataclass(frozen=True)
class Decision:
    passed: bool
    report: dict[str, list] = field(default_factory=dict)  # mutable, advisory
    digest: str | None = None  # SHA-256 of the exact candidate that passed


def scope_errors(reference: Scope, candidate: Scope) -> list[str]:
    errors = []
    if reference != candidate:
        errors.append("scope: reference and candidate describe different data")
    for name in ("snapshot", "currency"):
        value = getattr(reference, name)
        if type(value) is not str or not value.strip():
            errors.append(f"scope: {name} must be a non-blank str")
    bounds = (reference.start, reference.end, candidate.start, candidate.end)
    if not all(isinstance(bound, datetime) for bound in bounds):
        errors.append("scope: window bounds must be datetime")  # not time/date
    elif any(bound.utcoffset() != timedelta(0) for bound in bounds):
        errors.append("scope: window bounds must be timezone-aware UTC")
    elif not reference.start < reference.end:
        errors.append("scope: window is empty or inverted")
    if reference.fields != FIELDS:
        errors.append(f"scope: compared fields must be {FIELDS}")
    return errors


def row_errors(side: str, i: int, row: object) -> list[str]:
    where = f"{side}[{i}]"
    if type(row) is not tuple or len(row) != 3:
        return [f"{where}: expected a 3-tuple, got {row!r}"]
    row_id, segment, amount = row
    errors = []
    for name, value in (("row_id", row_id), ("segment", segment)):
        if type(value) is not str or not value.strip():
            errors.append(f"{where}: {name} must be a non-blank str")
    if type(amount) is not int:  # bool is a subclass of int, so no isinstance
        errors.append(f"{where}: amount_cents must be int, "
                      f"got {type(amount).__name__}")
    return errors


def freeze(side: str, rows: Sequence[object]) -> tuple[tuple, list[str]]:
    frozen = tuple(rows)  # private copy: later edits to the caller's list
    errors = [e for i, row in enumerate(frozen)  # cannot leak into the check
              for e in row_errors(side, i, row)]
    return frozen, errors


def duplicated(rows: tuple[Row, ...]) -> list[str]:
    counts = Counter(row_id for row_id, _, _ in rows)
    return sorted(row_id for row_id, n in counts.items() if n > 1)


def reconcile(reference: tuple[Row, ...],
              candidate: tuple[Row, ...]) -> dict[str, list]:
    ref_ids = {row[0] for row in reference}
    cand_ids = {row[0] for row in candidate}
    ref_rows, cand_rows = Counter(reference), Counter(candidate)
    return {
        "missing": sorted(ref_ids - cand_ids),
        "extra": sorted(cand_ids - ref_ids),
        "duplicate_candidate": duplicated(candidate),
        "only_reference": sorted((ref_rows - cand_rows).elements()),
        "only_candidate": sorted((cand_rows - ref_rows).elements()),
    }


def digest(scope: Scope, rows: Sequence[Row]) -> str:
    payload = [scope.snapshot, scope.start.isoformat(), scope.end.isoformat(),
               scope.currency, list(scope.fields), sorted(rows)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def gate(ref_scope: Scope, reference: Sequence[Row],
         cand_scope: Scope, candidate: Sequence[Row],
         *, allow_empty: bool = False) -> Decision:
    try:
        ref_rows, invalid = freeze("reference", reference)
        cand_rows, cand_invalid = freeze("candidate", candidate)
        invalid += cand_invalid + scope_errors(ref_scope, cand_scope)
        if not invalid:
            invalid = [f"reference: duplicate row_id {d!r}"
                       for d in duplicated(ref_rows)]
        if invalid:
            return Decision(False, {"invalid": invalid})
        report = reconcile(ref_rows, cand_rows)
        empty = not ref_rows and not allow_empty
        report["empty"] = ["empty window without allow_empty"] if empty else []
        if any(report.values()):
            return Decision(False, report)
        return Decision(True, report, digest(cand_scope, cand_rows))
    except Exception as exc:  # fail closed: a check that crashed did not pass
        return Decision(False, {"error": [f"{type(exc).__name__}: {exc}"]})


def promote(decision: Decision, scope: Scope, staged: Sequence[Row],
            switch: Callable[[], None]) -> bool:
    """Simulated cutover hook, not a race-free production gate.

    The digest proves only that `staged` matched when it was hashed; data can
    still change before or during switch(). A real system checks an immutable
    staged version and moves readers to it with a switch that its store makes
    atomic and conditional on the expected current version (compare-and-swap).
    Only `passed` and `digest` are trusted here; `report` is not consulted.
    An exception raised by switch() is not caught: it propagates to the
    caller, who must treat the state of the cutover as unknown.
    """
    if not decision.passed or decision.digest is None:
        return False
    try:
        rows, invalid = freeze("staged", staged)
        unchanged = not invalid and digest(scope, rows) == decision.digest
    except Exception:  # cannot prove it is the same data, so do not switch
        return False
    if unchanged:
        switch()
    return unchanged


def main() -> None:
    utc = timezone.utc
    scope = Scope("legacy-export-0001", datetime(2023, 10, 1, tzinfo=utc),
                  datetime(2023, 11, 1, tzinfo=utc), "PLN")
    reference = [("t1", "B2B", 10000), ("t2", "B2C", 2550), ("t3", "B2C", 700)]
    scenarios = {
        "same rows, different order": reference[::-1],
        "one row lost": reference[:2],
        "one row loaded twice": reference + [reference[0]],
        "10.00 moved between rows": [("t1", "B2B", 9000),
                                     ("t2", "B2C", 3550), ("t3", "B2C", 700)],
        "t3 reassigned to B2B": reference[:2] + [("t3", "B2B", 700)],
        "bool instead of amount": reference[:2] + [("t3", "B2C", True)],
    }
    for name, candidate in scenarios.items():
        decision = gate(scope, reference, scope, candidate)
        found = {k: v for k, v in decision.report.items() if v}
        print("PASS " if decision.passed else "BLOCK", f"{name}: {found}")
    staging = list(reference)
    decision = gate(scope, reference, scope, staging)
    staging.append(("t4", "B2C", 100))  # a late write after the check
    switched = promote(decision, scope, staging, lambda: print("cutover"))
    print(f"promote after a late write: {switched}; old dataset stays live")
    switched = promote(decision, scope, reference, lambda: print("cutover"))
    print(f"promote the copy that was checked: {switched}")


if __name__ == "__main__":
    main()
```

`gate` is the decision. It freezes private copies of both inputs, validates them and the scope, rejects a reference with duplicate identifiers, and only then reconciles. It returns a `Decision` with a report, and, only when the candidate passed, a SHA-256 digest of exactly the scope and rows that passed.

When the inputs are valid, the report has six entries, five comparison lists and one policy entry:

- `missing` and `extra`: identifiers present on only one side,
- `duplicate_candidate`: identifiers the candidate contains more than once,
- `only_reference` and `only_candidate`: whole rows from the multiset differences,
- `empty`: the empty-window policy, non-empty when the reference is empty and `allow_empty` wasn't set.

When validation fails, the report has a single `invalid` entry instead, and when the check itself crashes, a single `error` entry. The demo prints only the non-empty entries.

`Decision` is a frozen dataclass, but its `report` is an ordinary dictionary that a caller can still modify. The report is advisory, written for the people who read it. `promote` doesn't consult it and relies only on `passed` and the digest. And since any code can construct a `Decision`, the digest is a statement about data, not proof that the gate ran. In a real system the decision comes from the gate process itself and refers to an immutable version.

## What the output shows

Running the module prints:

```
PASS  same rows, different order: {}
BLOCK one row lost: {'missing': ['t3'], 'only_reference': [('t3', 'B2C', 700)]}
BLOCK one row loaded twice: {'duplicate_candidate': ['t1'], 'only_candidate': [('t1', 'B2B', 10000)]}
BLOCK 10.00 moved between rows: {'only_reference': [('t1', 'B2B', 10000), ('t2', 'B2C', 2550)], 'only_candidate': [('t1', 'B2B', 9000), ('t2', 'B2C', 3550)]}
BLOCK t3 reassigned to B2B: {'only_reference': [('t3', 'B2C', 700)], 'only_candidate': [('t3', 'B2B', 700)]}
BLOCK bool instead of amount: {'invalid': ['candidate[2]: amount_cents must be int, got bool']}
promote after a late write: False; old dataset stays live
cutover
promote the copy that was checked: True
```

The reversed candidate passes, because a multiset doesn't depend on order.

The two most instructive lines are "10.00 moved between rows" and "t3 reassigned to B2B". In both, the candidate has the same number of rows, the same global total and the same set of identifiers as the reference. A count-and-sum gate would let both through. The multiset comparison blocks them and points at the exact rows that differ.

The `bool` case never reaches the comparison. Without the exact type check, `("t3", "B2C", True)` would be a well-formed row with an amount of one cent.

## Binding the decision to the data that was checked

A gate has a weak spot that is easy to miss: the time between the check and the switch. If the candidate in staging can change after it passed, for example because a late write arrives, the readers end up on data that nobody checked. This is a time-of-check to time-of-use problem.

The example does two things about it.

First, `gate` works on private copies. `freeze` turns each input into a tuple, so later edits to the caller's list can't affect the rows being validated and compared.

Second, a passing decision carries the digest of the scope and rows that passed. The `promote` helper recomputes the digest of the staged rows and calls the `switch` callback only if it matches. In the output, a row appended to staging after the check makes `promote` return `False`. Promoting the copy that was actually checked succeeds.

I want to be precise about what this proves. **`promote` is a simulation, not a race-free production gate.** The recomputed digest only shows that the staged rows matched at the moment they were hashed. Between that moment and the end of `switch()`, a writer can still change the staging data. Rechecking a hash narrows the window. It doesn't close it.

There is one more boundary. `promote` turns errors in its own check into "don't switch", but it doesn't catch an exception raised by `switch()` itself. That exception propagates to the caller. A failed switch may have changed nothing or may have been partly applied, so the caller has to treat the state of the cutover as unknown and verify what readers actually see, rather than assume either outcome.

In a real system I close it with two properties together:

- **the checked version is immutable.** The candidate is written as a new version, for example a table or partition with its own identifier, and nothing writes to it after the gate starts reading it. A late write goes into a new version, which has to pass its own check.
- **the switch is atomic and conditional.** Readers reach data through a pointer, for example a view, an alias or a "current version" row. The cutover changes that pointer to the identifier of the checked version in a single transaction or atomic swap, and nothing else. The change is also conditional on the version the promoter expects to replace, a compare-and-swap: "point readers at v42 only if they still point at v41". If two promoters run at the same time, the second one finds an unexpected current version and fails, instead of silently overwriting the first cutover.

Atomicity here is a property of a specific store, not of views or aliases in general. Some databases let me redefine a view or update a pointer row inside a transaction, some catalogs and search engines document an atomic alias swap, and some systems offer neither, or don't say what a reader in the middle of a query will see. Before I rely on the switch, I check what my store actually guarantees.

With both in place, the decision refers to a version identifier, not to "whatever is in staging right now". The digest is then a useful additional assertion, not the only protection.

## What the tests check

Next to the module I keep a `unittest` suite in `examples/test_quality_gate.py`, which is also in the ZIP archive. From the same directory I run it with:

```
python3.11 -m unittest discover -s examples -v
```

The tests cover:

- **reconciliation**: the same rows in a different order pass; a missing row, an extra row and a duplicated row block; a lost row hidden by a duplicate of another row blocks even though the counts match; duplicates on both sides make the reference invalid; compensating amounts block although the count, total and identifier set are equal; a segment reassignment blocks,
- **validation**: `bool`, string and float amounts; rows of the wrong shape or type; non-string or blank identifiers and segments; an invalid reference row; a scope mismatch; blank or non-string snapshot identifiers and currencies; naive or non-UTC window bounds, including a candidate bound that denotes the same instant with a different offset; an inverted window; a bound that isn't a `datetime`, including an aware time-of-day window used identically on both sides, and a `time`, `date` or string in each of the four bounds,
- **empty policy**: an empty window blocks by default and passes only when allowed; `allow_empty` doesn't hide extra rows,
- **promotion**: a passed and unchanged candidate calls `switch` exactly once; a blocked decision, a candidate changed after the check, malformed staged rows, rows with the wrong shape and a different scope don't call it; a crash inside the check, or input that isn't iterable, produces a blocking decision and no switch; an exception raised by `switch` itself propagates to the caller.

The tests check the decision logic of this module. They don't test concurrency, crashes during a real cutover or the behaviour of any database.

## Where the guarantee ends

The first limit is the most important one. **Source parity is not source truth.** The gate proves that the candidate matches the reference for the compared scope. If the old pipeline has been writing a wrong amount for years, the new one is expected to reproduce it, and the gate will pass. Whether the reference itself is correct is a separate question for the data owner.

The second limit is coverage. I compare the selected fields and the selected population. A field I don't compare can be broken, and a record outside the window or the filters takes no part in the comparison. And, as already said, the scope is asserted by the caller rather than proven from the rows.

The third limit is the kind of transformation. This gate fits a non-aggregating copy of selected fields, where every reference row has exactly one counterpart. If the new pipeline aggregates or derives new fields, the expected result has to be computed by an independent path, and that is a separate design.

The fourth limit is size. The example keeps both sides in memory. For large datasets I would split the comparison into partitions, for example by day and segment, and compute the differences in the database. The query has to keep multiplicity, as the `Counter` subtraction does: either `EXCEPT ALL` in both directions, reference minus candidate and candidate minus reference, where the database supports it, or both sides grouped by all compared fields with a count of occurrences, reporting every group whose counts differ. An ordinary anti-join (`NOT EXISTS` or `LEFT JOIN ... IS NULL`) isn't equivalent. For a reference with one row `r` and a candidate with `r` twice, anti-joins in both directions find nothing, while the multiset difference reports one surplus `r`. I'd use anti-joins only after uniqueness of `row_id` has been established on both sides, and I'd keep the separate duplicate checks in the gate either way. A random sample helps catch gross errors quickly, but it gives no guarantee: a single missing record or a rare duplicate easily falls outside the sample. I'm not quoting any performance numbers here, because I haven't measured any.

## Conclusion

A green job says that a process finished. Whether readers move to the new data is decided, in my setup, by a comparison with the approved reference for the same population, snapshot, UTC window, currency and fields.

I compare multisets of whole rows instead of counts and totals, because those can match despite compensating errors. I validate rows before comparing them, treat a broken reference as invalid rather than as a diff, fail closed on errors and require an explicit decision about empty windows. And I keep the old approved dataset live until a specific, immutable version has passed and an atomic, conditional switch points readers at it.

What the gate confirms is that the copy agrees with the reference. It doesn't confirm that the data is true.

### Sources

- [Python 3.11.0 release page](https://www.python.org/downloads/release/python-3110/) — release date 24 October 2022.
- [Python 3.11.0 documentation source, `collections` module (`Doc/library/collections.rst` at tag `v3.11.0`)](https://raw.githubusercontent.com/python/cpython/v3.11.0/Doc/library/collections.rst) — the `Counter` section: a dict subclass for counting hashable objects, similar to bags or multisets; zero count for missing elements; `elements()`; subtraction keeping only positive counts.
- [Python 3.11.0 documentation source, built-in types (`Doc/library/stdtypes.rst` at tag `v3.11.0`)](https://raw.githubusercontent.com/python/cpython/v3.11.0/Doc/library/stdtypes.rst) — "Booleans are a subtype of integers".
- [Python 3.11.0 tutorial source, floating-point arithmetic (`Doc/tutorial/floatingpoint.rst` at tag `v3.11.0`)](https://raw.githubusercontent.com/python/cpython/v3.11.0/Doc/tutorial/floatingpoint.rst) — "most decimal fractions cannot be represented exactly as binary fractions".

All sources were read again on 12 October 2026. The tagged documentation files show the text as of the 3.11.0 release.
