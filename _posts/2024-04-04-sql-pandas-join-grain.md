---
layout: post
lang: en
title: "SQL and pandas: joins, grain, and unexpected row multiplication"
date: 2024-04-04 09:00:00 +0200
permalink: /blog/sql-pandas-join-grain/
author: Rafał Prońko
categories: [data-engineering]
description: "Why joins multiply rows, how lost and duplicated rows can cancel out, and how to check grain with pandas validate/indicator and SQLite GROUP BY, HAVING and constraints."
image: /assets/images/sql-pandas-join-grain.svg
---

# SQL and pandas: joins, grain, and unexpected row multiplication

![Diagram: order A has two lines and three payments. Joining them on order_id gives six rows and inflated totals. Aggregating each table to one row per order first, then joining one to one, gives a single correct row.]({{ '/assets/images/sql-pandas-join-grain.svg' | relative_url }})

<time datetime="2024-04-04">4 April 2024</time>

A join almost never fails loudly. If the data breaks the assumption I had about it, the join does not raise an error. It quietly returns a different answer: more rows, fewer rows, or the same number of rows with different content.

Revenue that is three times too high is easy to spot. Revenue that is 15% too high, in a report that also lost a few orders, is not.

So before I write a join I want to answer three questions. What does one row mean on each side? Which columns do I compare? How many rows can each key have on each side? Then I check those answers in code instead of trusting them.

## Grain, join key and cardinality are three different things

These three terms are often used as if they were one. They are not.

**Grain** is what a single row represents. In an `order_lines` table one row is one line of an order. In `payments` one row is one payment. In an `orders` table one row should be one order. The grain is usually defined by a set of columns that is unique in the table: `(order_id, line_no)` for lines, `payment_id` for payments, `order_id` for orders.

**Join key** is what I compare in the join condition. In the main example below it is `order_id`.

**Relationship cardinality** is how many rows can exist for one key value on each side. Lines and orders are many-to-one: many lines per order, one order per line. Lines and payments, joined on `order_id`, are many-to-many: an order can have several lines and several payments, and neither belongs to the other.

The join key is the same in both cases. The grain of the tables and the cardinality of the relationship are different, and that is what decides the size of the result.

The rule is simple: for every key value, an inner join returns *left rows × right rows* for that key. One times one is one. Two times three is six.

## The example data

I use three small synthetic tables. Amounts are integer cents, so there is no floating-point arithmetic in the totals.

- `LINES` — one row per order line. Order A has two lines, B and C have one each. Total: 5200 cents.
- `PAYMENTS` — one row per payment. Order A was paid in three instalments of 1000 cents. Order D has a payment but no lines. Total: 5000 cents.
- `ORDERS` — should be one row per order, but this extract is broken in two ways: order B arrived twice with two *different* customers, and order C is missing.

The examples use Python 3.11, pandas 2.2.1 and numpy 1.26.4. The SQL dialect is SQLite, through the `sqlite3` module from the Python standard library. The complete pandas example can be saved as `join_grain.py`:

```python
"""Grain, join keys and row multiplication in pandas (synthetic data)."""
import pandas as pd
from pandas.errors import MergeError

# Grain: one row per order line. Amounts are integer cents.
LINES = pd.DataFrame({
    "order_id": ["A", "A", "B", "C"],
    "line_no": [1, 2, 1, 1],
    "line_cents": [1000, 2000, 1500, 700],
})
# Grain: one row per payment. Order A was paid in three instalments.
PAYMENTS = pd.DataFrame({
    "order_id": ["A", "A", "A", "B", "D"],
    "payment_id": ["P1", "P2", "P3", "P4", "P5"],
    "paid_cents": [1000, 1000, 1000, 1500, 500],
})
# Intended grain: one row per order. This extract breaks it: order B
# arrived twice with conflicting customers, and order C is missing.
ORDERS = pd.DataFrame({
    "order_id": ["A", "B", "B", "D"],
    "customer_id": ["K1", "K2", "K3", "K4"],
})


def naive_line_payment_join() -> pd.DataFrame:
    """Join two independent child tables directly: the fan-out trap."""
    return LINES.merge(PAYMENTS, on="order_id", how="inner")


def order_balance() -> pd.DataFrame:
    """Aggregate both sides to one row per order, then join 1:1."""
    lines = LINES.groupby("order_id", as_index=False)["line_cents"].sum()
    paid = PAYMENTS.groupby("order_id", as_index=False)["paid_cents"].sum()
    result = lines.merge(paid, on="order_id", how="outer",
                         validate="one_to_one", indicator="coverage")
    amounts = ["line_cents", "paid_cents"]
    result[amounts] = result[amounts].fillna(0).astype("int64")
    result["open_cents"] = result["line_cents"] - result["paid_cents"]
    return result


def enrich_lines(how: str) -> pd.DataFrame:
    """Attach the customer to each line; the output grain should stay a line."""
    return LINES.merge(ORDERS, on="order_id", how=how, indicator=True)


def validate_message(mode: str) -> str:
    """Return pandas' verdict on a declared relationship for LINES x ORDERS."""
    try:
        LINES.merge(ORDERS, on="order_id", how="left", validate=mode)
    except MergeError as exc:
        return f"MergeError: {exc}"
    return "passed"


def main() -> None:
    fanout = naive_line_payment_join()
    print("rows per order, lines x payments:",
          fanout.groupby("order_id").size().to_dict())
    print("line cents: true", LINES["line_cents"].sum(),
          "| after join", fanout["line_cents"].sum())
    print("paid cents: true", PAYMENTS["paid_cents"].sum(),
          "| after join", fanout["paid_cents"].sum())

    print("\norder balance at order grain:")
    print(order_balance().to_string(index=False))

    inner, left = enrich_lines("inner"), enrich_lines("left")
    print("\nrows: lines", len(LINES), "| inner", len(inner),
          "| left", len(left))
    print("left coverage:", left["_merge"].value_counts().to_dict())
    print("rows per order, inner:", inner.groupby("order_id").size().to_dict())
    print("line cents after inner join:", inner["line_cents"].sum())

    for mode in ("many_to_one", "one_to_one"):
        print(f"validate={mode!r}:", validate_message(mode))


if __name__ == "__main__":
    main()
```

Running it prints:

```text
rows per order, lines x payments: {'A': 6, 'B': 1}
line cents: true 5200 | after join 10500
paid cents: true 5000 | after join 7500

order balance at order grain:
order_id  line_cents  paid_cents   coverage  open_cents
       A        3000        3000       both           0
       B        1500        1500       both           0
       C         700           0  left_only         700
       D           0         500 right_only        -500

rows: lines 4 | inner 4 | left 5
left coverage: {'both': 4, 'left_only': 1, 'right_only': 0}
rows per order, inner: {'A': 2, 'B': 2}
line cents after inner join: 6000
validate='many_to_one': MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
validate='one_to_one': MergeError: Merge keys are not unique in either left or right dataset; not a one-to-one merge
```

I'll go through it piece by piece.

## Two lines times three payments

The first join connects lines with payments on `order_id`. For order A there are two lines and three payments, so the result has 2 × 3 = 6 rows for A. Every line appears three times, every payment twice.

Nothing in that result is technically wrong. Each row is a valid pair (line, payment) for the same order. The problem is that this pair is not a business object. There is no rule saying that line 1 was paid by payment P2. The result has a grain I never intended: "a combination of a line and a payment of the same order".

Summing measures over that grain multiplies them. Line totals went from 5200 to 10500 cents, payments from 5000 to 7500. The multipliers differ per order — A's lines are counted three times, B's once — so I cannot "divide it back" with a single factor.

At the same time, the inner join silently dropped order C (lines, no payment) and order D (payment, no lines). The total is both inflated and incomplete.

## When loss and multiplication cancel out

The second join is the more dangerous one. I want to attach the customer to each line. That is a legitimate many-to-one relationship, and the output grain should stay *one row per line*.

With the broken `ORDERS` extract the inner join returns 4 rows. `LINES` also has 4 rows. A typical sanity check — "the row count did not change after the join" — passes.

But the rows are not the same. Order C's line was lost because C is missing from `ORDERS`. Order B's line appears twice because B is duplicated. One loss plus one extra row gives the same count. The line total after the join is 6000 cents instead of 5200.

A left join does not hide the loss: it returns 5 rows, and the extra row is the line of B again. A left join from lines to a dimension should return *exactly* the number of lines. More rows means the right side is not unique on the key. Fewer is impossible with a left join; missing matches show up as rows without values from the right side.

So I don't compare counts after an inner join and stop there. I check two things separately: uniqueness of the key on the "one" side, and coverage of keys between the two sides.

## Aggregate first, to the grain you want

For lines and payments the right fix is not a different join type. It is a different order of operations.

Lines and payments are independent facts about an order. If the question is "how much was ordered and how much was paid, per order", the intended grain of the answer is one row per order. So I bring *both* sides to that grain first, and only then join them:

```python
lines = LINES.groupby("order_id", as_index=False)["line_cents"].sum()
paid = PAYMENTS.groupby("order_id", as_index=False)["paid_cents"].sum()
result = lines.merge(paid, on="order_id", how="outer",
                     validate="one_to_one", indicator="coverage")
```

Now each side has one row per order, the join is one-to-one, and the totals are preserved: 5200 cents of lines and 5000 cents of payments, distributed over four orders.

I use an outer join because both kinds of mismatch are interesting here. C has lines but no payment, so its open balance is 700. D has a payment without lines, so the balance is -500. In a real pipeline D is probably a data problem — a payment for an order that never reached the lines table — and I want to see it, not lose it.

The outer join introduces missing values for the unmatched side. In this run the cent columns became `float64` with `NaN` before the cast; I fill them with 0 and cast back to `int64`. Filling with 0 is a business decision: "no payment rows" really does mean 0 cents paid. For other measures, missing and zero are not the same thing, and the `coverage` column keeps the distinction anyway.

Aggregating only one side is not enough. If I sum payments per order but join them to *lines*, both lines of A get the full 3000 cents paid, and summing `paid_cents` over lines gives 6000. That join even passes `validate="many_to_one"`, because it *is* many-to-one. The check is correct; the question asked of the result is not.

## Declaring the relationship with `validate`

pandas can check part of my assumptions for me. The `validate` argument of `merge` declares the expected relationship:

- `"one_to_one"` — keys must be unique on both sides,
- `"one_to_many"` — keys must be unique on the left,
- `"many_to_one"` — keys must be unique on the right,
- `"many_to_many"` — allowed, no check.

In the pandas 2.2.1 source this is a uniqueness test of the join keys on each side, which raises `MergeError` when it fails.

For lines → orders the correct declaration is `"many_to_one"`. With the broken extract it fails with *"Merge keys are not unique in right dataset; not a many-to-one merge"*. That message points directly at the duplicated order.

`"one_to_one"` would be the wrong declaration here. It fails too, but with *"not unique in either left or right dataset"*, because lines legitimately repeat `order_id`. The tests in the example project also check the clean case: against a correct orders table `"many_to_one"` passes and `"one_to_one"` still fails, this time only on the left side. Declaring a stricter relationship than the real one does not make the data safer. It just produces errors I will learn to ignore.

`validate` checks uniqueness, not coverage. It said nothing about order C, which is missing.

## Coverage with `indicator`

That is what `indicator` is for. With `indicator=True` pandas adds a categorical column `_merge` with the values `left_only`, `right_only` and `both`. A string argument changes the column name, as with `coverage` in `order_balance`.

For the left join from lines to orders I got `{'both': 4, 'left_only': 1, 'right_only': 0}`. The `left_only` row is the line of order C. A left join cannot produce `right_only` rows, so that category shows 0.

In a pipeline I turn these counts into an explicit rule: for example, "every line must find its order", or "at most N payments may be unmatched and they go to a quarantine table". The important part is that the rule exists and fails visibly, not that the threshold is zero.

## Null keys: pandas and SQL disagree

Missing keys behave differently in the two tools.

The pandas 2.2.1 documentation of `merge` says it directly: if both key columns contain null values, those rows are matched against each other, unlike in usual SQL join behaviour. In `null_keys.py` I join two tiny tables where each side has a row with a missing `order_id`:

```text
pandas inner merge:
order_id  line_cents customer_id
       A        1000          K1
    None         700          K9
pandas rows, object keys with None: 2
pandas rows, float keys with NaN: 2
SQLite ON l.order_id = r.order_id: [('A', 1000, 'K1')]
SQLite ON l.order_id IS r.order_id: [(None, 700, 'K9'), ('A', 1000, 'K1')]
SQLite SELECT NULL = NULL, NULL IS NULL: (None, 1)
```

pandas paired the line without an order with the customer without an order. That is almost never what I want: two unrelated "unknown" records become one.

In SQLite, `NULL = NULL` evaluates to NULL, not true, and the join keeps only rows for which the `ON` expression is true. The row with the missing key disappears from the inner join. `IS` treats two NULLs as equal and reproduces the pandas behaviour, which is a good way to see the difference, not a recommendation.

The practical consequence: when I port logic between SQL and pandas, I handle null keys explicitly. I count them before the join, decide whether they are an error, and filter or route them before calling `merge`.

## Why `drop_duplicates` and `DISTINCT` are not a repair

When a join returns too many rows, the quickest "fix" is to remove duplicates afterwards. It hides the symptom and often makes the data wrong in a less visible way.

In the orders extract, the two rows for B differ in `customer_id`. `ORDERS.drop_duplicates()` removes nothing, because the rows are not identical. `drop_duplicates("order_id")` keeps one of them, and which one depends on row order: `keep="first"` gives customer K2, `keep="last"` gives K3. The source contradicts itself, and the code just picked a winner. Order C is still missing.

`DISTINCT` after a join has the same problem in SQL. In `sql_grain.py` I tried it on the line × payment fan-out:

```sql
SELECT SUM(paid_cents) FROM (
    SELECT DISTINCT p.order_id, p.paid_cents
    FROM lines AS l JOIN payments AS p ON l.order_id = p.order_id)
```

The result was 2500 cents. Orders A and B were actually paid 3000 + 1500 = 4500. A's three payments of 1000 cents are three real payments; once the payment identifier is not in the projection, `DISTINCT` cannot tell them apart from copies produced by the join. Equal values are not the same as the same fact.

Deduplication is correct only when I can name the rule: which key identifies a record, and which version wins (for example, the highest version number or the latest change timestamp). Then it belongs *before* the join, on the table whose grain it restores, and it is a documented decision rather than a reaction to a row count.

## The same checks in SQL

In SQL I use the same three steps: check uniqueness, check coverage, and let the database enforce the grain. The queries below run on SQLite from the Python standard library; `sql_grain.py` loads the same synthetic data into an in-memory database.

The uniqueness diagnostic is a `GROUP BY` on the intended key with `HAVING`:

```sql
SELECT order_id, COUNT(*) AS n_rows,
       COUNT(DISTINCT customer_id) AS n_customers
FROM orders_extract
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY order_id
```

It returned `[('B', 2, 2)]`: two rows for B, and two different customers. The second count tells me whether I'm looking at exact copies or at conflicting data, which need very different handling.

One detail: for grouping, SQLite treats NULL values as equal. All rows with a missing key fall into a single group. So I count `WHERE order_id IS NULL` separately rather than reading it from this query.

Coverage is a left join with a filter on the missing side:

```sql
SELECT l.order_id, l.line_no
FROM lines AS l LEFT JOIN orders_extract AS o ON o.order_id = l.order_id
WHERE o.order_id IS NULL
```

It returned `[('C', 1)]` — the same line that pandas marked as `left_only`.

Diagnostics find problems; constraints prevent them. In the target tables I declare the grain:

```sql
CREATE TABLE orders (
    order_id TEXT NOT NULL PRIMARY KEY,
    customer_id TEXT NOT NULL);
CREATE TABLE order_lines (
    order_id TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    line_cents INTEGER NOT NULL CHECK (line_cents >= 0),
    PRIMARY KEY (order_id, line_no));
```

Loading the broken extract with `INSERT INTO orders SELECT * FROM orders_extract` failed with `IntegrityError: UNIQUE constraint failed: orders.order_id`, and the `orders` table stayed empty. I prefer this to a successful load that makes every later join unreliable.

`NOT NULL` is not decoration here. The SQLite documentation states that, for historical reasons, a `PRIMARY KEY` column that is not an `INTEGER PRIMARY KEY` (and not in a `STRICT` or `WITHOUT ROWID` table) accepts NULL unless declared `NOT NULL`. In the tests a `TEXT PRIMARY KEY` without `NOT NULL` accepted a NULL, and a `UNIQUE` column accepted two NULLs, because for `UNIQUE` NULLs are distinct from each other. Other databases have their own rules, so I check them instead of assuming.

## When multiplication is correct

Row multiplication is not always a bug. It is a bug when the result grain differs from the grain I intended.

**Legitimate many-to-many.** If the question is "which promotions applied to which order lines", the answer has the grain of the pair (line, promotion). One line with two promotions should produce two rows. What I must not do is sum the line amount over those rows, because the amount belongs to the line, not to the pair. If I need amounts, I either aggregate first or use an explicit allocation rule.

**Composite keys.** Sometimes `order_id` alone is not the key. Lines are identified by `(order_id, line_no)`; data from two source systems may need `(source_system, order_id)`. If the real key is composite and I join on only part of it, I get a many-to-many join that looks like a many-to-one. In pandas I pass all key columns to `on=`; `validate` then checks uniqueness of the whole combination. In the tests, a line-level table joined on `["order_id", "line_no"]` passed `validate="one_to_one"`, the same join on `order_id` alone failed, and a repeated `('A', 1)` pair failed again. In SQL I declare a composite `PRIMARY KEY`, like `(order_id, line_no)` above — the test inserting line `('A', 1)` twice was rejected with a `UNIQUE constraint` error.

**Temporal dimensions.** A customer's segment can change over time. A history table then has the grain *customer and validity interval*, with a key like `(customer_id, valid_from)`. Joining an order on `customer_id` alone matches every version. In the example, customer K1 has two segment versions and order A is dated 2024-03-15:

```text
temporal join on key only: [('A', 'retail'), ('A', 'business')]
temporal join on key + interval: [('A', 'business')]
```

The correct join has a range condition: `order_date >= valid_from AND order_date < valid_to`. Here `validate` in pandas does not help. It checks equality keys, and a correct history legitimately has many rows per `customer_id`. The property I need — validity intervals that do not overlap for one customer — is something I have to test separately. Checking that every order matches exactly one version is useful, but it only covers the dates that orders happen to have. The direct test is a self-join of the history table that looks for two versions of the same customer whose intervals overlap; it should return no rows.

## What these checks do not cover

`validate` and constraints confirm that a key is unique. They don't confirm that it is the *right* key, or that the data behind it is correct.

Coverage counts tell me that some keys have no match. They don't tell me whether the missing data is late, lost, or correctly absent. That needs a rule from the business process, not from pandas.

None of the examples involve performance or large data. I don't measure the cost of the checks here; on large tables, the uniqueness test is itself a hash or sort pass over the key, and it's worth knowing where it runs.

## How I approach a new join

Before the join, I write down the grain of each input and of the result I want. If the result grain is coarser than an input — one row per order from lines and payments — I aggregate that input first.

I declare the relationship (`validate="many_to_one"` and its relatives in pandas, keys and constraints in SQL). I measure coverage separately (`indicator`, or a left join with `IS NULL`). I decide what null keys mean before the join, not after.

And when the number of rows surprises me, I look for the key that has more matches than I expected, instead of reaching for `drop_duplicates` or `DISTINCT`.

## Running the examples

The example project for this article can be downloaded as [sql-pandas-join-grain.zip]({{ '/assets/downloads/sql-pandas-join-grain.zip' | relative_url }}). It pins its dependencies for uv — Python 3.11, `pandas==2.2.1`, `numpy==1.26.4`, `pytest==8.1.1`, with exact versions in `uv.lock` — and contains `join_grain.py`, `sql_grain.py`, `null_keys.py` and 18 tests asserting the results quoted above. From the unpacked project directory:

```bash
uv run --frozen python join_grain.py
uv run --frozen python sql_grain.py
uv run --frozen python null_keys.py
uv run --frozen pytest -v
```

uv is the tooling I use to reproduce the environment today; any Python 3.11 environment with the same pinned packages will run the code. The outputs in this article were recorded on 12 October 2026 with Python 3.11.12, whose `sqlite3` module reported SQLite 3.47.1, together with pandas 2.2.1 and numpy 1.26.4. The SQL used here relies on long-standing SQLite behaviour, but the exact library version depends on how your Python was built.

### Sources

- [pandas 2.2.1 release notes](https://pandas.pydata.org/docs/whatsnew/v2.2.1.html) — release dated February 22, 2024.
- [pandas 2.2.1 source: `pandas/core/reshape/merge.py`](https://raw.githubusercontent.com/pandas-dev/pandas/v2.2.1/pandas/core/reshape/merge.py) — `validate` implemented as a key-uniqueness check raising `MergeError`; `indicator` values `left_only`, `right_only`, `both`. The `merge` docstring, including the note on null keys, is defined in [`pandas/core/frame.py` at v2.2.1](https://raw.githubusercontent.com/pandas-dev/pandas/v2.2.1/pandas/core/frame.py).
- [SQLite: SELECT](https://www.sqlite.org/lang_select.html) — section 2.1 (joins as a filtered cartesian product; only rows where `ON` is true are kept), section 2.3 (rows are excluded when the condition is false or NULL), and the aggregate-query rules where NULLs are considered equal for `GROUP BY`.
- [SQLite: expressions](https://www.sqlite.org/lang_expr.html) — `=` with a NULL operand yields NULL; `IS` treats two NULLs as equal.
- [SQLite: CREATE TABLE](https://www.sqlite.org/lang_createtable.html) — NULLs in `UNIQUE` constraints and in non-`INTEGER` `PRIMARY KEY` columns.
- [Python 3.11.0 release](https://www.python.org/downloads/release/python-3110/) — released October 24, 2022.
- [numpy 1.26.4 on PyPI](https://pypi.org/project/numpy/1.26.4/) — files uploaded February 5–6, 2024 (UTC); [release notes](https://numpy.org/doc/stable/release/1.26.4-notes.html).
- [pytest 8.1.1 changelog](https://docs.pytest.org/en/stable/changelog.html) — "pytest 8.1.1 (2024-03-08)"; [PyPI](https://pypi.org/project/pytest/8.1.1/) files uploaded March 9, 2024 (UTC).
