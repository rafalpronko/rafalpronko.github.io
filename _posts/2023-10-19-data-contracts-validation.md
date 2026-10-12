---
layout: post
lang: en
title: "Why a model cannot fix bad data: contracts and validation"
date: 2023-10-19 09:00:00 +0200
permalink: /blog/data-contracts-validation/
author: Rafał Prońko
categories: [data-engineering, machine-learning]
description: "A hard data contract for invoice lines in Python 3.11 and SQLite: declared units, exact types, bounds, keys, relations and an all-or-nothing write, plus why monitoring is a different tool."
image: /assets/images/data-contracts-validation.svg
---

# Why a model cannot fix bad data: contracts and validation

![Diagram: a batch of invoice lines passes a hard contract before one SQLite transaction; violations write nothing; an advisory monitor runs after the commit; a dashed box marks the semantic blind spot of a consistent unit lie.]({{ '/assets/images/data-contracts-validation.svg' | relative_url }})

<time datetime="2023-10-19">19 October 2023</time>

A column called `net_minor` holds the value `4550`. Is that 45.50 euros in cents, or 4,550 euros? The integer does not say. The column name suggests cents, but a name is a convention, not a guarantee. If the producer quietly switches to whole euros, the type stays `int`, the name stays the same and every downstream job keeps running.

A model trained on this column cannot settle that question either. It sees numbers, not units, and without independent information about what the producer meant it cannot reliably recover the intended unit. A statistical check may flag a shift as an anomaly; it cannot prove which unit was meant. This article is about the checks that have to happen *before* data becomes a feature or a target: what a data contract is, what it can verify, what it cannot, and how it differs from distribution monitoring and from tracking model performance.

The example is a small, synthetic batch of invoice lines validated in Python 3.11 and written to SQLite. The complete code and tests are available for download; the article shows only the central fragments.

## What a model inherits from its input

Every feature and every target carries the unit and meaning of the column it came from. If `gross_minor` is the target of a revenue model and the producer starts sending net amounts in that field, the model is now learning a different quantity under the same name. If the evaluation set comes from the same pipeline, it shifts along with the training data, so the offline metrics can look perfectly normal.

Two common tools can create a false sense of safety here.

**Robust training.** A tree-based model splits on thresholds, so it is not bothered by the absolute scale of a feature *as long as the scale is the same in training and serving*. A unit change that hits only one side breaks that assumption. A model that learned its thresholds on cents and receives euros in production compares values a hundred times smaller against those thresholds, and the rows end up in leaves that were never meant for them. Robustness to noise is not robustness to a change of meaning.

**Imputation.** Imputing a missing value replaces it with a guess based on other rows. It answers "what is a plausible number here?", not "what does this number mean?". If the unit of a whole batch is wrong, nothing is missing. And if the unit is not written down anywhere, statistics can at best suggest that something changed: a hundredfold drop looks suspicious, but so would a real hundredfold drop in prices. The intended unit cannot be determined from the values alone, because that information is not in the data.

So the meaning of a field must be declared somewhere outside the values themselves, and checked before the values are used. That is the job of a data contract.

## Three checks that answer different questions

It helps to separate three kinds of checks that are often lumped together as "data quality":

| Check | Question it answers | When it runs | Typical action |
|---|---|---|---|
| Contract validation | Does this batch satisfy the rules the producer and the consumer agreed on? | Before the write, for every batch | Block the batch |
| Distribution monitoring | Does this batch look like earlier batches? | After the write or on a schedule | Alert a human |
| Model performance | Are the predictions still good? | After true labels arrive, often much later | Investigate, retrain or roll back |

A contract rule is binary and written down in advance. A distribution alert is statistical: a median that moved tenfold may mean a unit change, but it may also mean a sale, a new product line or a holiday. Model performance is the most direct signal, but it often arrives days or weeks after the data, when the bad batch has already been used.

None of these replaces the others. This article builds the first one and adds a deliberately weak version of the second, to show where the line between them runs.

## The example: a batch of invoice lines

Each batch is a JSON-like object with an *envelope* and a list of rows. The envelope declares what the numbers mean:

```json
{
  "contract_version": "invoice_lines.v2",
  "currency": "EUR",
  "unit": "minor",
  "amount_basis": "net_tax_gross",
  "rows": [
    {"line_id": "L-3", "invoice_id": "INV-2", "net_minor": 4550,
     "tax_bp": 500, "tax_minor": 228, "gross_minor": 4778}
  ]
}
```

Amounts are integers in minor units (cents), so there are no floating-point rounding issues. The tax rate is an integer in basis points: `500` means 5%. The allowed rates `{0, 500, 1000, 2000}` and the rounding rule below are choices of this synthetic contract, not a description of any country's tax law.

The contract for version `invoice_lines.v2` says:

- **Envelope.** Exactly these five fields, with the declared values. Any difference rejects the batch. If the producer switches to whole euros and honestly sets `"unit": "major"`, the batch stops at the envelope, before any amount is read. A new contract version means a new, deliberately deployed validator, not a best-effort reading of `v3` with `v2` rules.
- **Grain and key.** One row is one invoice line. `line_id` is unique within the batch and in the target table.
- **Fields.** Every row has exactly the six fields of the contract. No nulls, no extras.
- **Identifiers.** Non-empty strings without leading or trailing whitespace, at most 64 characters.
- **Amounts.** Exact `int` values within declared bounds. All amounts are non-negative in this version.
- **Relations.** `tax_minor` equals `net_minor × tax_bp / 10000` rounded half up, and `gross_minor = net_minor + tax_minor`.
- **Owner.** Someone is responsible for the meaning of the fields and approves new versions. This is not code, but without it a rejected batch has nobody to go to.

## The contract in code

This is the entry point of `invoice_contract.py`. The helpers it calls (`check_envelope`, `identifier_error`, `amount_errors`, `relation_errors` and `duplicate_key_errors`) live in the same file.

```python
def validate_batch(envelope: object) -> list[str]:
    """Return the violations that can be checked independently.

    Envelope errors stop before the rows are read; within a row, malformed
    values skip the relation checks that depend on them. Empty list = valid.
    """
    if not isinstance(envelope, dict):
        return [f"envelope: object required, got {type(envelope).__name__}"]
    errors = check_envelope(envelope)
    if errors:
        return errors
    rows: list[object] = envelope["rows"]
    for i, row in enumerate(rows):
        errors.extend(check_row(i, row))
    errors.extend(duplicate_key_errors(rows))
    return errors
```

Each row then goes through `check_row`, further down in the same file:

```python
def check_row(i: int, row: object) -> list[str]:
    where = f"rows[{i}]"
    if not isinstance(row, dict):
        return [f"{where}: object required, got {type(row).__name__}"]
    if set(row) != ROW_FIELDS:
        return [f"{where}: fields must be exactly {sorted(ROW_FIELDS)}"]
    errors = [error for key in ID_FIELDS
              if (error := identifier_error(f"{where}.{key}", row[key]))]
    errors += amount_errors(where, row)
    if errors:  # relations are meaningless on malformed values
        return errors
    return relation_errors(where, row)
```

A few decisions in this code are worth spelling out.

**Malformed input is a result, not a crash.** The function accepts `object`, not `dict`. If someone passes a list, `None` or a string, the answer is an error message, not an `AttributeError` from deep inside the validator. The producer gets something they can act on.

**Exact fields mean exact.** The envelope check reports missing fields and also *unexpected* ones. A new field such as `"unit_note": "cents, mostly"` is a sign that something changed on the producer side. If the contract promises an exact shape, silently ignoring extras would break that promise.

**Identifiers are compared as stored.** `"L-1"` and `" L-1"` are different strings, so they would be different primary keys. Rejecting surrounding whitespace prevents a duplicate that no key constraint can see.

**`type(x) is int`, not `isinstance`.** In Python, `bool` is a subtype of `int`, so `isinstance(True, int)` is true. With an `isinstance` check, `True` in `tax_bp` would pass the integer type check as one basis point and be stopped only by the allowed-rate rule. `False` would get further: in place of a zero rate, with `tax_minor` 0, it satisfies the allowed-rate rule and both relations, so the whole row would be accepted. The exact type check rejects `bool`, `float` and numeric strings before any of that, and a test covers the `False` case.

**Bounds come before arithmetic.** Python integers have unlimited precision, so `10**30` is a perfectly valid `int`. SQLite stores integers in at most 8 bytes. In the recorded test run, binding `2**63` to a query parameter raised `OverflowError` inside `sqlite3`. The validator checks every amount against an explicit upper bound first, so values far outside the 64-bit range become a contract error instead of a driver exception, and the relations are never computed on absurd inputs.

**Relations are computed in integers.** The rounding rule is written once:

```python
def rounded_tax(net_minor: int, tax_bp: int) -> int:
    """Tax in minor units, rounded half up. Defined for net_minor >= 0 only."""
    return (net_minor * tax_bp + 5_000) // 10_000
```

For 45.50 at 5% the exact tax is 227.5 cents, so the contract requires 228. A producer that truncates sends 227 and the batch is rejected. The rule is only valid for non-negative amounts: Python's `//` rounds towards negative infinity, so credit notes with negative amounts would need their own, explicitly written rule. Version 2 of this contract rejects them instead of computing something different from the accounting system.

The relation `gross = net + tax` catches a typical swap of meaning: the producer starts sending the net amount in `gross_minor`. The type and the name are still correct; the equation is not.

**Duplicates are found even in broken rows.** `duplicate_key_errors` runs over all rows that have a well-formed `line_id`, whether or not the rest of the row is valid. If row 1 repeats `L-1` *and* has a float amount, both problems are reported at once. Otherwise the producer would fix the float, resend, and only then learn about the duplicate.

**Independent errors are collected.** Within a valid envelope, the validator does not stop at the first problem: it reports every error that can be checked independently, in every row. Checks that depend on something already broken are skipped. Envelope errors stop before the rows are read, and a row with a malformed amount gets no relation checks, because `gross = net + tax` means nothing when `net` is a float. One round trip with the producer is cheaper than five.

## Writing: the whole batch or nothing

Validation runs entirely in memory, before the database is touched. Only a fully valid batch reaches the write. This is the core of `publish()` in `invoice_store.py`:

```python
    require_supported_runtime()
    require_transaction_control(con)
    require_baseline(baseline_median_net)
    errors = validate_batch(envelope)
    if errors:
        raise ContractViolation(errors)
    rows = cast(list[dict[str, Any]], cast(dict[str, Any], envelope)["rows"])
    params = [(*(row[c] for c in COLUMNS[:-1]), CONTRACT_VERSION)
              for row in rows]
    with con:  # commit on success, rollback on any exception
        con.executemany(INSERT_SQL, params)  # INSERT opens the transaction
    return PublishResult(len(params),
                         distribution_advisories(rows, baseline_median_net))
```

The `with con:` block deserves precision, because it does less than it seems. According to the Python 3.11 `sqlite3` documentation, when the connection's `isolation_level` is not `None`, the module implicitly opens a transaction before an `INSERT` executed by `execute()` or `executemany()`. The connection used as a context manager commits on success and rolls back on an exception, but it *neither opens a transaction nor closes the connection*. Here it is the `INSERT` that opens the transaction, and the caller closes the connection (the demo and the tests use `contextlib.closing`).

That reliance is fragile, so `publish()` checks its preconditions before anything else:

- **Runtime.** The transaction checks below describe the Python 3.11 `sqlite3` transaction model. Later Python versions add other ways to control transactions, which these checks do not cover. `require_supported_runtime` compares `sys.version_info` with 3.11 and raises `RuntimeError` on any other version, in `publish()` and in `connect()`. Supporting a newer version means extending the checks deliberately, not assuming they still hold.
- If `isolation_level` is `None`, the connection is in autocommit mode and every inserted row would be committed on its own. A failure on row 500 would leave 499 rows behind. `publish()` refuses such a connection with `ValueError`.
- If the caller already has an open transaction, `with con:` would commit or roll back the caller's pending work together with the batch. `publish()` refuses that too, with `RuntimeError`.
- **Baseline.** `require_baseline` accepts only an exact `int` from 1 to the upper bound of `net_minor`. The advisory step runs after the commit and divides in floating point. Without the bound, a huge but perfectly valid Python integer would raise `OverflowError` *after* the batch was committed, and the caller would see an exception for a write that succeeded. Every argument is checked before the write, so nothing after the commit can fail on an accepted input.

One error can still happen inside the transaction: a `line_id` that is valid in the batch but already exists in the table. The in-memory validator cannot see that. The primary key raises `IntegrityError`, the context manager rolls back, and a new row that was inserted just before the conflict disappears with it. A test checks exactly that.

### The table is a second line of defence

A column declared as `INTEGER` in SQLite does not guarantee an integer. SQLite uses dynamic typing: the declared type gives the column an *affinity*, and a text value that is not a well-formed number is simply stored as text. One of the tests inserts `'12.5 EUR'` into a plain `INTEGER` column and reads back `typeof(x) = 'text'`.

The target table therefore repeats the most important rules as `CHECK` constraints: `typeof(net_minor) = 'integer'`, non-negative amounts and `gross_minor = net_minor + tax_minor`. If some future code path skips the validator, the database still refuses text that cannot be converted losslessly to an integer, such as `'12.5'` or `'12.5 EUR'`. It is not a ban on every string: the column affinity converts `'12'` to the integer 12 before the check runs, and a test records that as well. The database also rejects empty identifiers and surrounding ASCII spaces in `line_id` and `invoice_id`; the Python validator enforces the broader whitespace and length rules. I do not use `STRICT` tables, available since SQLite 3.37.0 (2021-11-27), because the Python 3.11 `sqlite3` module only requires SQLite 3.7.15 and the example should not raise that minimum.

### Rejecting the whole batch is a policy

One bad row blocks all the good rows in the same batch. For financial lines, where a partial batch makes totals wrong, I think that is the right default. For other data, a reasonable alternative is to load valid rows and send invalid ones to a quarantine table with the error messages. Either way, I would keep rejected batches for a defined period so that the producer can diagnose them. Neither quarantine nor retention is implemented in the example; they are recommendations, not tested behaviour.

## Atomic is not the same as idempotent

Atomicity means the batch is written fully or not at all. Idempotency is a property of the effect: applying the same operation twice leaves the system in the same state as applying it once. Atomicity does not give you the second property.

In this example an exact replay does leave the table unchanged. Every `line_id` already exists, the first `INSERT` fails and the transaction is rolled back. That comes from the primary key, not from the transaction: without the key, the same atomic write would store every row a second time.

What the caller is told is a separate question. The replay ends with `IntegrityError`, the same error a different batch gets when it reuses a stored `line_id`, so a retrying pipeline cannot tell "already applied" from "conflict" by the exception alone. Whether a replay should be acknowledged as a success is an interface decision. If it should, the writer needs a way to recognise the same batch, for example a batch identifier or a comparison with the stored rows; one way to build that is described in [A pipeline you can run twice]({{ '/blog/pipeline-idempotencja/' | relative_url }}). This example stops at an unchanged table and an explicit error.

## The advisory monitor

After a successful commit, `publish()` returns advisories from a simple distribution check: the median of `net_minor` in the batch is compared with a reference median, and anything outside a tenfold band in either direction is reported.

The reference value in the example, 4,550 cents, is invented for the demo. It is not a measured baseline or a benchmark, and the tenfold band is an arbitrary illustration. A real monitor would derive the reference from history, per segment, and would account for seasonality.

The important property is not the statistic but its place in the flow: here the advisory is computed *after* the commit and never blocks it. That is my recommendation for this example, not a universal rule. A tenfold band picked for a demo is not a contractual rule; turned into a gate, it would reject correct data for reasons nobody agreed on, and a large promotion could stop the pipeline. A distribution gate can be the right tool when its threshold has been validated against history and the cost of blocking correct data is lower than the cost of loading wrong data. It is then a deliberate rule with an owner, documented like the rest of the contract.

## The blind spot: a consistent lie

The last scenario is deliberately uncomfortable. The producer sends whole euros, declares cents in the envelope, and calculates tax and gross correctly from those wrong numbers. Every type is right, every relation holds, every key is unique. The contract accepts the batch and it is written to the table. The test for this case documents a gap; it does not claim a detection.

In this synthetic example the advisory monitor does flag the batch, because the median drops from 4,550 to 46. I would not rely on that. With a smaller scale difference, a mixed batch or a poor reference value, there may be no signal at all, and even a clear signal does not say which unit was meant. What actually catches a consistent lie is outside the batch: an agreement with the producer about what the fields mean, and a comparison with an independent source, such as control totals from the accounting system.

This is the honest version of the title. The contract narrows the space of possible errors and turns many of them into immediate, specific rejections. It does not give the data a meaning it never carried.

## What the row validator does not see

Some failures leave no trace in a single batch:

- **Freshness.** A batch from three days ago passes every row rule.
- **Missing batches.** A batch that never arrived cannot be validated at all.
- **Completeness across batches.** Ten valid batches can still miss one store or one day.

These need checks at the level of the pipeline: an expected schedule, a registry of received batches, and reconciliation against an independent count. They belong next to the contract, not inside it.

## Running the example

Download [data-contracts-validation.zip]({{ '/assets/downloads/data-contracts-validation.zip' | relative_url }}), or view the main modules directly: [invoice_contract.py]({{ '/assets/examples/data-contracts-validation/invoice_contract.py' | relative_url }}) and [invoice_store.py]({{ '/assets/examples/data-contracts-validation/invoice_store.py' | relative_url }}). The archive also contains the fixtures, the demo, the tests, `pyproject.toml`, `uv.lock` and a README. The modules need Python 3.11.x and nothing outside its standard library; the tests need pytest.

```
uv run --python 3.11 --locked pytest -q
uv run --python 3.11 --locked python demo.py
```

The demo publishes eight synthetic batches to one in-memory database:

```
REJECTED     not an envelope
    - envelope: object required, got list
REJECTED     declared unit change
    - envelope.unit: expected 'minor', got 'major'
REJECTED     truncated tax
    - rows[2].tax_minor: 227 != rounded_tax 228
REJECTED     duplicate key, one copy malformed
    - rows[1].net_minor: int required, got float
    - rows[1].line_id: duplicate of rows[0]
PUBLISHED    canonical batch: 3 rows
ROLLED BACK  replay of canonical batch: UNIQUE constraint failed: invoice_lines.line_id
ROLLED BACK  new key + existing key: UNIQUE constraint failed: invoice_lines.line_id
PUBLISHED    consistent unit lie: 3 rows
    advisory: median net_minor 46 outside advisory band [455, 45500] around baseline 4550
rows in table: 6
0 mismatch(es)
```

The test suite has 86 test cases. They cover the envelope and row rules above, the half-up rounding, bounds on values beyond 64 bits, duplicates next to malformed rows, the refusal of other Python versions, of autocommit connections, of an open caller transaction and of an out-of-range baseline before anything is written, rollback after a key conflict, the replay case, the `typeof` constraint and what it still accepts, and the documented blind spot. They do not test concurrent writers, a process crash during the commit, or a real network source.

**A note on reproduction.** The 19 October 2023 date is the editorial date of this article. The commands above and their output come from a run on 12 October 2026 with Python 3.11.12, SQLite library 3.47.1 and pytest 7.4.2 on macOS. uv is modern tooling used to reproduce the environment; it was not part of a 2023 setup. The lock file is restricted to packages uploaded before 19 October 2023, and pytest 7.4.2 was released on 7 September 2023. The code was not run on Python 3.11.0 itself or on the oldest SQLite version that Python 3.11 supports. As a negative check, the runtime guard was also run on Python 3.14.4 with an `autocommit=True` connection: `publish()` and `connect()` refused before writing and the table stayed empty. That run shows the refusal works; it is not a claim that the code supports Python 3.14.

## Takeaways

- A model inherits the units and meaning of its features and targets. Without independent information it cannot reliably recover a meaning that was never recorded, and neither can imputation. Statistics can flag a change, not prove what was meant.
- Declare meaning in the data: currency, unit, amount basis and contract version travel with the batch, and a mismatch rejects it.
- Check exact shapes and exact types. `bool` is an `int` in Python, and an `INTEGER` column in SQLite accepts text.
- Bound numbers before arithmetic and before the database driver.
- Validate the whole batch in memory, then write it in one transaction on a connection and a runtime whose transaction behaviour you have checked.
- Check every argument before the write, including configuration such as a baseline, so that nothing after the commit can fail on an accepted input.
- Atomicity prevents partial writes; idempotency means a repeated application leaves the same state. How a replay is reported to the caller is a separate decision.
- Contracts block, monitors inform, performance metrics arrive late. They answer different questions; which ones you need, and whether a monitor should ever block, depends on the risk.
- A consistent lie passes every row rule. Name that limit and close it with ownership and independent reconciliation.

### Sources

- [Python 3.11.0 release page](https://www.python.org/downloads/release/python-3110/) — release date 24 October 2022.
- [Python 3.11 `sqlite3`: transaction control](https://docs.python.org/3.11/library/sqlite3.html#sqlite3-controlling-transactions) — implicit transactions before `INSERT`, `UPDATE`, `DELETE` and `REPLACE` when `isolation_level` is not `None`; minimum SQLite 3.7.15 is stated at the top of the same page.
- [Python 3.11 `sqlite3`: the connection context manager](https://docs.python.org/3.11/library/sqlite3.html#sqlite3-connection-context-manager) — commits or rolls back, but neither opens a transaction nor closes the connection.
- [Python 3.11: numeric types](https://docs.python.org/3.11/library/stdtypes.html#typesnumeric) — Booleans are a subtype of integers; integers have unlimited precision.
- [SQLite: Datatypes, storage classes](https://www.sqlite.org/datatype3.html#storage_classes_and_datatypes) and [type affinity](https://www.sqlite.org/datatype3.html#type_affinity) — integers up to 8 bytes; text that is not a well-formed number is stored as text in a numeric column.
- [SQLite: STRICT tables](https://www.sqlite.org/stricttables.html) — available since 3.37.0 (2021-11-27); not used here.
