---
layout: post
lang: en
title: "dbt staging, intermediate and marts: a layered project you can run and break"
date: 2024-04-11 09:00:00 +0200
permalink: /blog/dbt-staging-intermediate-marts/
author: Rafał Prońko
categories: [data-engineering]
description: "A runnable dbt and DuckDB project on synthetic data: grain at every layer, DECIMAL money, canceled orders, fan-out guards, reconciliation tests and a deliberate failure."
image: /assets/images/dbt-staging-intermediate-marts.svg
---

# dbt staging, intermediate and marts: a layered project you can run and break

![Diagram of a dbt project in four layers: seeds loaded as text, three staging views that keep the source grain, one intermediate view that sums order lines to orders, and two mart tables, orders and customers, with the tests that guard each layer.]({{ '/assets/images/dbt-staging-intermediate-marts.svg' | relative_url }})

<time datetime="2024-04-11">11 April 2024</time>

Folders named `staging`, `intermediate` and `marts` are easy to create. They don't protect anything on their own.

What I want from the layers is a set of promises I can check. Every model has a stated grain — what one row means. Only specific models are allowed to change that grain. Money keeps an exact type. A customer who never ordered doesn't disappear on the way to the mart. A canceled order doesn't quietly inflate revenue.

In this article I build a small dbt project that keeps those promises, test it, run it twice and then break it on purpose. All data is synthetic. Everything runs locally against one DuckDB file.

The complete project is available as a download: [dbt-layers.zip]({{ '/assets/examples/dbt-layers.zip' | relative_url }}). The article shows all the SQL and YAML that matter, so you can also rebuild it from the text.

## About the date and the versions

The date on this post is an editorial label. I ran the example and recorded its output on 12 October 2026, not in April 2024.

What I did check is that the code fits the date. The pinned packages were all published on PyPI before 11 April 2024, according to PyPI's JSON API:

| Package | Version | First file uploaded (UTC) |
|---|---|---|
| dbt-core | 1.7.11 | 2024-03-28 |
| dbt-duckdb | 1.7.3 | 2024-03-07 |
| duckdb | 0.10.1 | 2024-03-18 |

I use Python 3.11. One consequence of choosing dbt 1.7: data tests are declared under `tests:` in YAML. The `data_tests:` key arrived with dbt 1.8, which renamed it when unit tests were added. On 1.8 and later, `tests:` still works for backward compatibility.

The tooling around the project is newer than the date. I use uv to create the environment and `uv.lock` to pin all transitive dependencies. uv's project commands (`uv lock`, `uv sync`) became stable in uv 0.3.0, released in August 2024. To keep the lock file historically consistent anyway, `pyproject.toml` tells uv to ignore anything published after the editorial date:

```toml
# pyproject.toml (abridged: description and comments omitted)
[project]
name = "dbt-layers"
version = "0.1.0"
requires-python = "==3.11.*"
dependencies = ["dbt-core==1.7.11", "dbt-duckdb==1.7.3", "duckdb==0.10.1"]

[tool.uv]
exclude-newer = "2024-04-11T00:00:00Z"
```

If you prefer tooling from April 2024, `pip install dbt-core==1.7.11 dbt-duckdb==1.7.3 duckdb==0.10.1` in a Python 3.11 virtual environment asks for the same three pins, but transitive versions then depend on what pip resolves on the day you run it. I did not run that route; every output below comes from the uv lock file.

## What we build

Three CSV files play the role of raw tables: customers, orders and order lines. From them dbt builds three layers:

| Layer | Model | One row per | Materialization |
|---|---|---|---|
| staging | `stg_shop__customers` | customer | view |
| staging | `stg_shop__orders` | order, canceled included | view |
| staging | `stg_shop__order_items` | order line | view |
| intermediate | `int_order_items_summed_to_orders` | order that has at least one line | view |
| marts | `orders` | order in staging, canceled and line-less included | table |
| marts | `customers` | customer in staging, customers without orders included | table |

This table is the most important part of the design. Each grain statement becomes a test later.

Note the intermediate row. It contains only orders that have lines. That is a correct grain for an aggregate of lines, but it is not "every order". If a mart started from this model, an order without lines would vanish. So the `orders` mart starts from staged orders and joins the aggregate to them, not the other way round.

The layer names and the `stg_[source]__[entity]` and `int_[entity]_[verb]` file names follow dbt's own guide on project structure. The guide suggests intermediate models as ephemeral or as views in a separate schema. I use plain views in the same schema to keep the profile minimal and to make them easy to query during the demo.

## Project setup

The profile points at one local DuckDB file. No credentials, no server:

```yaml
# profiles.yml
config:
  send_anonymous_usage_stats: false
  use_colors: false

dbt_layers:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: dbt_layers.duckdb
      schema: main
      threads: 1
```

`threads: 1` keeps the log order deterministic, which helps when I compare two runs. It isn't required for correctness here.

The project file fixes the materialization per folder and — this is the part that is easy to miss — the column types of the seeds:

```yaml
# dbt_project.yml (abridged: default paths omitted)
name: dbt_layers
version: "1.0.0"
config-version: 2
profile: dbt_layers
require-dbt-version: [">=1.7.0", "<1.8.0"]

seeds:
  dbt_layers:
    # Raw landing data arrives as text. Staging owns every cast.
    raw_customers:
      +column_types: {customer_id: varchar, full_name: varchar, email: varchar, signup_date: varchar}
    raw_orders:
      +column_types: {order_id: varchar, customer_id: varchar, ordered_at: varchar, status: varchar}
    raw_order_items:
      +column_types: {order_item_id: varchar, order_id: varchar, sku: varchar, quantity: varchar, unit_price: varchar}

models:
  dbt_layers:
    staging:
      +materialized: view
    intermediate:
      +materialized: view
    marts:
      +materialized: table
```

Without `column_types`, dbt infers seed types from the CSV content. In dbt-core 1.7.11 the default inference maps any number with decimal places to `float8`, and dbt-duckdb 1.7.3 doesn't override it. A column like `12.50` would land as a floating-point number before my code ever saw it. Loading everything as text means the only place where a type is decided is the staging model, where I can read it.

## Seeds, `ref` and `source`

The synthetic data is small enough to read in full. It contains the edge cases I want the marts to handle: C004 has no orders, C005 has only a canceled order, O1007 has no lines yet, and one status is capitalized differently.

```text
# seeds/raw_customers.csv
customer_id,full_name,email,signup_date
C001,Ada Example,ADA@example.com,2024-01-03
C002,Ben Sample,ben@example.org,2024-01-15
C003,Cleo Test,cleo@example.net,2024-02-01
C004,Dev Placeholder,dev@example.com,2024-02-20
C005,Eve Fixture,eve@example.org,2024-03-05

# seeds/raw_orders.csv
order_id,customer_id,ordered_at,status
O1001,C001,2024-02-01 10:15:00,completed
O1002,C001,2024-02-14 18:40:00,Completed
O1003,C002,2024-02-20 09:05:00,shipped
O1004,C002,2024-03-01 12:00:00,canceled
O1005,C003,2024-03-03 16:30:00,placed
O1006,C005,2024-03-10 11:11:00,canceled
O1007,C003,2024-03-12 08:00:00,placed

# seeds/raw_order_items.csv
order_item_id,order_id,sku,quantity,unit_price
L01,O1001,SKU-TEA,2,12.50
L02,O1001,SKU-MUG,1,8.99
L03,O1002,SKU-TEA,1,12.50
L04,O1003,SKU-KETTLE,1,49.90
L05,O1003,SKU-MUG,2,8.99
L06,O1004,SKU-KETTLE,1,49.90
L07,O1005,SKU-TEA,3,12.50
L08,O1005,SKU-FILTER,4,0.10
L09,O1006,SKU-MUG,1,8.99
```

Staging models read these with `{% raw %}{{ ref('raw_orders') }}{% endraw %}`. That is deliberate, and it is the one place where this project differs from a real one.

`ref()` points at another node of the dbt project — a model, a seed or a snapshot. dbt resolves it to the right relation for the current target and records the dependency, which is how it knows to load the seed before building the view that reads it.

`source()` points at a table that dbt does not build: something an ingestion tool, a replication job or another team loads. You declare it in YAML under `sources:`, and dbt records the dependency for lineage and node selection, and can check its freshness.

Seeds are dbt nodes, so `ref` is the correct way to read them. They also make the example self-contained: anyone who downloads it gets the same input. With real ingestion, the raw tables would be declared as sources and only the `from` line of each staging model would change:

```sql
-- with real ingestion
from {% raw %}{{ source('shop', 'orders') }}{% endraw %}
```

The rest of the model stays the same. dbt's documentation describes seeds as best suited to static data that changes infrequently, such as a mapping of country codes to country names. Using them as a stand-in for production tables is a teaching shortcut, not a loading strategy.

## Staging: rename, cast, keep the grain

A staging model has one job: make a raw table pleasant and safe to use, without changing what a row means. One row in, one row out. No joins, no aggregation, no filtering of business events.

```sql
-- models/staging/stg_shop__orders.sql
-- Grain: one row per order_id, exactly as delivered by raw_orders.
select
    cast(order_id as varchar)       as order_id,
    cast(customer_id as varchar)    as customer_id,
    cast(ordered_at as timestamp)   as ordered_at,
    lower(trim(status))             as order_status,
    lower(trim(status)) = 'canceled' as is_canceled
from {% raw %}{{ ref('raw_orders') }}{% endraw %}
```

Canceled orders stay. Removing them here would be a business decision hidden in the most generic layer, and every later model would inherit it without knowing. Instead I expose `is_canceled` and let the marts apply the policy.

```sql
-- models/staging/stg_shop__order_items.sql
-- Grain: one row per order_item_id (an order line), exactly as delivered.
select
    cast(order_item_id as varchar)          as order_item_id,
    cast(order_id as varchar)               as order_id,
    cast(sku as varchar)                    as sku,
    cast(quantity as integer)               as quantity,
    cast(unit_price as decimal(18, 2))      as unit_price,
    cast(cast(quantity as integer) * cast(unit_price as decimal(18, 2))
         as decimal(18, 2))                 as line_amount
from {% raw %}{{ ref('raw_order_items') }}{% endraw %}
```

Money becomes `DECIMAL(18,2)` here and stays decimal until the end. DuckDB's documentation describes `DECIMAL` as an exact fixed-point type: addition, subtraction and multiplication are exact. `DOUBLE` is a binary floating-point type on which the same operations are approximate. It also recommends a width of 18 or below for performance. Computing `line_amount` in staging is a small, row-local calculation; it doesn't change the grain.

`stg_shop__customers` follows the same pattern: it renames `full_name` to `customer_name`, lowercases and trims `email` and casts `signup_date` to a date as `signed_up_on`.

The tests for this layer state the grain and the links between tables:

```yaml
# models/staging/_stg_shop__models.yml (excerpt)
version: 2

models:
  - name: stg_shop__orders
    description: One row per order, canceled orders included.
    columns:
      - name: order_id
        tests: [unique, not_null]
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_shop__customers')
              field: customer_id
      - name: ordered_at
        tests: [not_null]
      - name: order_status
        tests:
          - not_null
          - accepted_values:
              values: [placed, shipped, completed, canceled]

  - name: stg_shop__order_items
    description: One row per order line. Money is DECIMAL(18,2), never DOUBLE.
    columns:
      - name: order_item_id
        tests: [unique, not_null]
      - name: order_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_shop__orders')
              field: order_id
      - name: quantity
        tests: [not_null]
      - name: unit_price
        tests: [not_null]
      - name: line_amount
        tests: [not_null]
```

`not_null` also covers status, order time, quantity, price and line amount. `accepted_values` alone permits NULL; a SQL predicate such as `quantity <= 0` also returns unknown for NULL, which a `where` clause does not select. Both the column tests and the business predicates therefore reject missing required values explicitly.

`unique` plus `not_null` on `order_item_id` is the grain statement "one row per order line", written in a form dbt can check. `relationships` checks that every line points at an existing order and every order at an existing customer. `stg_shop__customers` gets the same `unique`/`not_null` pair on `customer_id` and `not_null` on `signed_up_on`; the full file is in the download.

## Intermediate: from lines to orders

Customers have orders and orders have lines. As soon as a query touches both "one row per order" and "one row per line", it has to decide at which grain it adds things up. I roll lines up to order grain in this model. The customer mart later makes the second grain change, from orders to customers:

```sql
-- models/intermediate/int_order_items_summed_to_orders.sql
-- Grain change: order line -> order. One row per order_id that has at least one line.
-- Aggregating here, before any join, is what keeps the marts free of fan-out.
select
    order_id,
    count(*)                                as line_count,
    sum(quantity)                           as unit_count,
    cast(sum(line_amount) as decimal(18, 2)) as order_amount
from {% raw %}{{ ref('stg_shop__order_items') }}{% endraw %}
group by order_id
```

The model's name says what it does. After it, nothing downstream needs to know that lines exist. Its tests confirm the new grain: `order_id` is `unique` and `not_null` and has a `relationships` test to `stg_shop__orders`; `order_amount` is `not_null`.

## Marts: every order, every customer

The `orders` mart starts from staged orders, so its grain is "every order", and joins the line totals to it:

```sql
-- models/marts/orders.sql
-- Grain: one row per order in stg_shop__orders, including canceled orders
-- and orders that have no lines yet. Policy: canceled orders keep their
-- order_amount for audit, but recognize 0.00.
with orders as (
    select * from {% raw %}{{ ref('stg_shop__orders') }}{% endraw %}
),

order_totals as (
    select * from {% raw %}{{ ref('int_order_items_summed_to_orders') }}{% endraw %}
)

select
    orders.order_id,
    orders.customer_id,
    orders.ordered_at,
    orders.order_status,
    orders.is_canceled,
    coalesce(order_totals.line_count, 0)                         as line_count,
    coalesce(order_totals.unit_count, 0)                         as unit_count,
    cast(coalesce(order_totals.order_amount, 0) as decimal(18, 2)) as order_amount,
    cast(case when orders.is_canceled then 0
              else coalesce(order_totals.order_amount, 0) end
         as decimal(18, 2))                                      as recognized_amount
from orders
left join order_totals
    on orders.order_id = order_totals.order_id
```

The join is safe because both sides have the same grain: one row per order. A left join from the side that defines the grain can't add rows when the right side is unique on the join key — and that uniqueness is tested in the intermediate layer.

The canceled-order policy is explicit and lives in one column. A canceled order remains visible with what was ordered (`order_amount`), so nobody has to ask where it went, but it contributes `0.00` to `recognized_amount`. Your business may define this differently — for example by also tracking refunds. The point is that the rule is stated once, in a mart, and tested.

The `customers` mart follows the same idea one level up. It rolls orders up to customer grain first and only then joins:

```sql
-- models/marts/customers.sql
-- Grain: one row per customer in stg_shop__customers, including customers
-- with no orders. Orders are rolled up to customer grain before the join.
with customers as (
    select * from {% raw %}{{ ref('stg_shop__customers') }}{% endraw %}
),

customer_orders as (
    select
        customer_id,
        count(*) filter (where not is_canceled)            as order_count,
        count(*) filter (where is_canceled)                as canceled_order_count,
        cast(sum(recognized_amount) as decimal(18, 2))     as lifetime_value,
        min(ordered_at) filter (where not is_canceled)     as first_order_at,
        max(ordered_at) filter (where not is_canceled)     as last_order_at
    from {% raw %}{{ ref('orders') }}{% endraw %}
    group by customer_id
)

select
    customers.customer_id,
    customers.customer_name,
    customers.email,
    customers.signed_up_on,
    coalesce(customer_orders.order_count, 0)                         as order_count,
    coalesce(customer_orders.canceled_order_count, 0)                as canceled_order_count,
    cast(coalesce(customer_orders.lifetime_value, 0) as decimal(18, 2)) as lifetime_value,
    customer_orders.first_order_at,
    customer_orders.last_order_at
from customers
left join customer_orders
    on customers.customer_id = customer_orders.customer_id
```

Two decisions are visible here. Starting from `customers` with a left join keeps C004, who never ordered. `coalesce` turns their missing totals into `0` and `0.00`; their first and last order dates stay `NULL`, because there is no honest value to put there.

The second decision is that `customers` reads from the `orders` mart rather than repeating its logic. The canceled-order rule exists once. If it changes, both marts change together.

## What fan-out looks like

To see what the intermediate step prevents, I ran a deliberately wrong query against the built database. It joins orders to their lines and then sums an order-level amount:

```sql
select o.customer_id,
       count(o.order_id)        as order_rows_seen,
       sum(o.recognized_amount) as lifetime_value
from main.orders o
left join main.stg_shop__order_items i on i.order_id = o.order_id
group by o.customer_id
order by o.customer_id
```

Observed output, next to the mart:

```text
== naive join, then sum (total lifetime_value = 292.04)
C001  order_rows_seen=3  lifetime_value=80.48
C002  order_rows_seen=3  lifetime_value=135.76
C003  order_rows_seen=3  lifetime_value=75.80
C005  order_rows_seen=1  lifetime_value=0.00

== customers mart (customers with at least one order) (total lifetime_value = 152.27)
C001  order_rows_seen=2  lifetime_value=46.49
C002  order_rows_seen=2  lifetime_value=67.88
C003  order_rows_seen=2  lifetime_value=37.90
C005  order_rows_seen=1  lifetime_value=0.00
```

Orders with lines appear once per line, repeating their order-level amount. O1001's 33.99 is counted twice. C002's amount doubles from 67.88 to 135.76 because O1003 has two lines; its row count rises from 2 to 3. C003's 37.90 also doubles because O1005 has two lines.

Canceled O1004 contributes one zero-amount row, and line-less O1007 contributes one null-extended zero-amount row. Both remain in the displayed counts; neither adds recognized revenue. The comparison filters out customers without orders, so C004 is absent on both sides. Nothing errors; the query returns plausible-looking, wrong totals.

Two things in the project guard against this. The structural guard is that lines are aggregated in one place before any join. The test guard is `unique` on each mart's key: if someone later adds a join that multiplies rows, `unique_orders_order_id` or `unique_customers_customer_id` fails. For aggregates that don't keep a key, the reconciliation tests below catch the inflated sum.

## Tests that describe the business, not just the columns

The marts get the usual key tests:

```yaml
# models/marts/_marts.yml (descriptions omitted)
version: 2
models:
  - name: orders
    columns:
      - name: order_id
        tests: [unique, not_null]     # unique on the mart key is the fan-out alarm
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('customers')
              field: customer_id
      - name: order_amount
        tests: [not_null]
      - name: recognized_amount
        tests: [not_null]

  - name: customers
    columns:
      - name: customer_id
        tests: [unique, not_null]
      - name: order_count
        tests: [not_null]
      - name: lifetime_value
        tests: [not_null]
```

`unique` catches a repeated key, but these built-in key tests alone can't show that the mart keeps exactly the staged set of orders and customers with correct per-entity totals: "no order was lost and none was counted twice". For that I write singular tests: SQL files in `tests/` that return rows only when something is wrong. dbt treats any returned row as a failure.

The first reconstructs each order directly from staging. A full outer join compares key sets in both directions, so dropping an order, inventing one or swapping a line-less order ID cannot hide behind an unchanged grand total. For matching keys it compares attributes, line and unit counts, gross amounts and recognized amounts:

```sql
-- tests/assert_orders_reconcile_to_order_lines.sql
-- Compare both key sets and the expected counts and amounts of EACH order.
with line_totals as (
    select order_id,
           count(*) as line_count,
           coalesce(sum(quantity), cast(0 as bigint)) as unit_count,
           coalesce(cast(sum(line_amount) as decimal(18, 2)),
                    cast(0 as decimal(18, 2))) as order_amount
    from {% raw %}{{ ref('stg_shop__order_items') }}{% endraw %}
    group by order_id
),
expected as (
    select o.order_id, o.customer_id, o.ordered_at, o.order_status, o.is_canceled,
           coalesce(l.line_count, cast(0 as bigint)) as line_count,
           coalesce(l.unit_count, cast(0 as bigint)) as unit_count,
           coalesce(l.order_amount, cast(0 as decimal(18, 2))) as order_amount,
           case when o.is_canceled then cast(0 as decimal(18, 2))
                else coalesce(l.order_amount, cast(0 as decimal(18, 2))) end as recognized_amount
    from {% raw %}{{ ref('stg_shop__orders') }}{% endraw %} o
    left join line_totals l on o.order_id = l.order_id
),
actual as (
    select * from {% raw %}{{ ref('orders') }}{% endraw %}
)
select e.order_id as expected_order_id, a.order_id as actual_order_id,
       e.order_amount as expected_amount, a.order_amount as actual_amount,
       e.line_count as expected_lines, a.line_count as actual_lines,
       e.unit_count as expected_units, a.unit_count as actual_units
from expected e
full outer join actual a on e.order_id = a.order_id
where e.order_id is null or a.order_id is null
   or a.customer_id is distinct from e.customer_id
   or a.ordered_at is distinct from e.ordered_at
   or a.order_status is distinct from e.order_status
   or a.is_canceled is distinct from e.is_canceled
   or a.line_count is distinct from e.line_count
   or a.unit_count is distinct from e.unit_count
   or a.order_amount is distinct from e.order_amount
   or a.recognized_amount is distinct from e.recognized_amount
```

The second checks every staged customer against the customer mart, including customers without orders. It compares non-canceled and canceled counts separately, money and first/last non-canceled order dates for each customer. Moving revenue or counts between customers now fails even when the grand totals stay the same:

```sql
-- tests/assert_customers_reconcile_to_orders.sql
-- Compare both customer key sets and totals for EACH staged customer.
with order_totals as (
    select customer_id,
           count(*) filter (where not is_canceled) as order_count,
           count(*) filter (where is_canceled) as canceled_order_count,
           coalesce(cast(sum(recognized_amount) as decimal(18, 2)),
                    cast(0 as decimal(18, 2))) as lifetime_value,
           min(ordered_at) filter (where not is_canceled) as first_order_at,
           max(ordered_at) filter (where not is_canceled) as last_order_at
    from {% raw %}{{ ref('orders') }}{% endraw %}
    group by customer_id
),
expected as (
    select c.customer_id,
           coalesce(o.order_count, cast(0 as bigint)) as order_count,
           coalesce(o.canceled_order_count, cast(0 as bigint)) as canceled_order_count,
           coalesce(o.lifetime_value, cast(0 as decimal(18, 2))) as lifetime_value,
           o.first_order_at, o.last_order_at
    from {% raw %}{{ ref('stg_shop__customers') }}{% endraw %} c
    left join order_totals o on c.customer_id = o.customer_id
),
actual as (
    select * from {% raw %}{{ ref('customers') }}{% endraw %}
)
select e.customer_id as expected_customer_id, a.customer_id as actual_customer_id,
       e.lifetime_value as expected_value, a.lifetime_value as actual_value,
       e.order_count as expected_orders, a.order_count as actual_orders,
       e.canceled_order_count as expected_canceled, a.canceled_order_count as actual_canceled
from expected e
full outer join actual a on e.customer_id = a.customer_id
where e.customer_id is null or a.customer_id is null
   or a.order_count is distinct from e.order_count
   or a.canceled_order_count is distinct from e.canceled_order_count
   or a.lifetime_value is distinct from e.lifetime_value
   or a.first_order_at is distinct from e.first_order_at
   or a.last_order_at is distinct from e.last_order_at
```

`is distinct from` is a NULL-safe comparison: a missing actual value differs from the expected value. Legitimately missing aggregates become a typed zero: an integer zero for counts and `DECIMAL(18,2)` for money. The count columns keep the type their aggregate produces in DuckDB, `BIGINT` for `count(*)` and `HUGEINT` for `sum()` of an integer column such as `quantity`; the zero only fills the gap. Customers with no orders are valid, as are three entirely empty inputs; empty key sets match and have no failing rows. Required input fields still have their own `not_null` tests, because `sum()` ignores NULL inputs. Decimal money needs no tolerance here.

Two shorter tests pin the policy and basic sanity:

```sql
-- tests/assert_canceled_orders_recognize_nothing.sql
-- Business policy: canceled orders retain their gross amount but recognize 0.00.
select order_id, order_status, order_amount, recognized_amount
from {% raw %}{{ ref('orders') }}{% endraw %}
where is_canceled is null
   or order_amount is null
   or recognized_amount is distinct from
      case when is_canceled then cast(0 as decimal(18, 2)) else order_amount end

-- tests/assert_order_lines_are_positive.sql
-- Quantity must be positive; a zero price is a valid free item, not missing data.
select order_item_id, quantity, unit_price, line_amount
from {% raw %}{{ ref('stg_shop__order_items') }}{% endraw %}
where quantity is null or quantity <= 0
   or unit_price is null or unit_price < 0
   or line_amount is null
```

A zero unit price means a free item in this example; a missing price is invalid. Quantity must be positive. The test name `assert_order_lines_are_positive` refers to quantity, while price is non-negative. In total the project has 35 tests: 31 generic ones from YAML and 4 singular ones.

## Running it

From the project directory:

```bash
uv sync --frozen --python 3.11
uv run dbt build --profiles-dir . --threads 1
uv run python scripts/show_marts.py
```

`dbt build` runs seeds, models and tests according to their dependencies. Tests wait for all relations they read, and failed tests block eligible downstream nodes. This is not a guarantee that every test finishes before every downstream model: the relationship test from `orders.customer_id` to `customers.customer_id` needs both marts, so it runs after `customers` has been built. The run ended with:

```text
Finished running 3 seeds, 4 view models, 35 tests, 2 table models in 0 hours 0 minutes and 5.74 seconds (5.74s).

Completed successfully

Done. PASS=44 WARN=0 ERROR=0 SKIP=0 TOTAL=44
```

The duration depends on the machine; the counts don't. `show_marts.py` reads both marts in read-only mode and prints them:

```text
== orders: 7 rows
types: {'order_amount': 'DECIMAL(18,2)', 'recognized_amount': 'DECIMAL(18,2)'}
order_id | customer_id | ordered_at | order_status | is_canceled | line_count | unit_count | order_amount | recognized_amount
O1001 | C001 | 2024-02-01 10:15:00 | completed | False | 2 | 3 | 33.99 | 33.99
O1002 | C001 | 2024-02-14 18:40:00 | completed | False | 1 | 1 | 12.50 | 12.50
O1003 | C002 | 2024-02-20 09:05:00 | shipped | False | 2 | 3 | 67.88 | 67.88
O1004 | C002 | 2024-03-01 12:00:00 | canceled | True | 1 | 1 | 49.90 | 0.00
O1005 | C003 | 2024-03-03 16:30:00 | placed | False | 2 | 7 | 37.90 | 37.90
O1006 | C005 | 2024-03-10 11:11:00 | canceled | True | 1 | 1 | 8.99 | 0.00
O1007 | C003 | 2024-03-12 08:00:00 | placed | False | 0 | 0 | 0.00 | 0.00

== customers: 5 rows
types: {'lifetime_value': 'DECIMAL(18,2)'}
customer_id | customer_name | email | signed_up_on | order_count | canceled_order_count | lifetime_value | first_order_at | last_order_at
C001 | Ada Example | ada@example.com | 2024-01-03 | 2 | 0 | 46.49 | 2024-02-01 10:15:00 | 2024-02-14 18:40:00
C002 | Ben Sample | ben@example.org | 2024-01-15 | 1 | 1 | 67.88 | 2024-02-20 09:05:00 | 2024-02-20 09:05:00
C003 | Cleo Test | cleo@example.net | 2024-02-01 | 2 | 0 | 37.90 | 2024-03-03 16:30:00 | 2024-03-12 08:00:00
C004 | Dev Placeholder | dev@example.com | 2024-02-20 | 0 | 0 | 0.00 | NULL | NULL
C005 | Eve Fixture | eve@example.org | 2024-03-05 | 0 | 1 | 0.00 | NULL | NULL

order_amount total: 211.16  recognized total: 152.27
marts fingerprint: 4451240e1e7629a6795df6d4f9a6fb2cc51d67b4df30714d00632de8d7b10c21
```

I check the output against expectations I can derive by hand from the seeds:

- 7 staged orders, 7 mart rows. 5 staged customers, 5 mart rows.
- Line amounts add up to 211.16; that is the `order_amount` total.
- The two canceled orders are 49.90 and 8.99. 211.16 − 58.89 = 152.27, the recognized total and the sum of `lifetime_value`.
- C004 is present with zeros and `NULL` dates. C005 has one canceled order and recognizes nothing.
- O1007 is in the mart with no lines and `0.00`. C003's `order_count` is 2, because O1007 is a placed, non-canceled order. Whether a line-less order should count is a policy choice too; here it does.
- Money columns are `DECIMAL(18,2)`, with no `DOUBLE` in sight.

### Running it again

Then I ran `dbt build` a second time, with the same seeds, and `show_marts.py` again. The second build also finished with `PASS=44 WARN=0 ERROR=0 SKIP=0 TOTAL=44`, and the fingerprint — a SHA-256 over all mart rows in a fixed order — was the same: `4451240e…7b10c21`.

This works because every node here is rebuilt from scratch: seeds are reloaded, views and tables are replaced. With incremental models, a second run is exactly where duplicates and gaps appear, and this check would matter much more.

## Breaking it on purpose

A green build only means something if a broken input turns it red. `scripts/bad_data_demo.py` simulates a common delivery problem: the same order line arrives twice. It appends one row to the seed, runs the build, inspects the database, then restores the original file from a backup in a `finally` block, verifies it byte for byte and builds again.

```text
L01,O1001,SKU-TEA,2,12.50   # appended: same order_item_id, same content
```

The first build failed, as it should (abridged; passing lines removed):

```text
2 of 44 OK loaded seed file main.raw_order_items ............................... [INSERT 10 in 0.04s]
5 of 44 OK created sql view model main.stg_shop__order_items ................... [OK in 0.04s]
16 of 44 FAIL 1 unique_stg_shop__order_items_order_item_id ..................... [FAIL 1 in 0.03s]
25 of 44 SKIP relation main.int_order_items_summed_to_orders ................... [SKIP]
30 of 44 SKIP relation main.orders ............................................. [SKIP]
38 of 44 SKIP relation main.customers .......................................... [SKIP]

Failure in test unique_stg_shop__order_items_order_item_id (models/staging/_stg_shop__models.yml)
  Got 1 result, configured to fail if != 0

Done. PASS=23 WARN=0 ERROR=1 SKIP=20 TOTAL=44
exit code: 1
```

The grain test caught the duplicate in the layer where it entered. dbt then skipped everything downstream: the intermediate view, both marts and all of their tests. The exit code was 1, so a scheduler would mark the run as failed.

The more instructive part is the state of the database right after the failure:

```text
-- state after failed build
stg_shop__order_items rows (view): 10
int_order_items_summed_to_orders O1001 (view): 58.99
orders O1001 (table): 33.99
```

The seed had already been reloaded with ten rows, and the staging view was recreated on top of it. The intermediate model was skipped — but it is a view that still exists from the previous build, and a view runs its query whenever someone reads it. It already showed O1001 at 58.99, with the duplicated line counted twice. The `orders` table, also skipped, still held 33.99 from the last good build.

So "skipped" didn't mean "unchanged". The views downstream of the failure showed new, wrong data. The table showed old, correct data. Anyone querying both would have seen them disagree.

After the script restored the seed (`seed restored byte-for-byte: True`), the next build passed again with `PASS=44 WARN=0 ERROR=0 SKIP=0 TOTAL=44`, the view and the table both showed 33.99 for O1001, and `show_marts.py` printed the same fingerprint as before the failure.

## Testing the repairs, including empty inputs

I also ran `uv run python scripts/repair_regression.py`. This script writes the synthetic seeds and temporarily corrupts mart tables. It restores every seed in `finally`, rebuilds and asserts both byte-for-byte seed recovery and the original mart content fingerprint. Its corruption checks execute the actual singular SQL compiled by dbt, rather than a second implementation of the assertions.

Before adding the guards, the script exited 1: 15 of its 26 checks were unsatisfied. The old tests accepted missing price, quantity and order time; swapped keys and redistributed totals also escaped. Missing status already failed, but the new `not_null` test now rejects it at staging with a direct diagnosis.

After the repairs, the observed builds were:

| Input | Build result | Exit |
|---|---|---|
| missing price | PASS=21, ERROR=3, SKIP=20, TOTAL=44 | 1 |
| missing quantity | PASS=21, ERROR=3, SKIP=20, TOTAL=44 | 1 |
| missing status | PASS=28, ERROR=1, SKIP=15, TOTAL=44 | 1 |
| missing order time | PASS=28, ERROR=1, SKIP=15, TOTAL=44 | 1 |
| five customers, no orders or lines | PASS=44, ERROR=0, SKIP=0, TOTAL=44 | 0 |
| all three inputs empty | PASS=44, ERROR=0, SKIP=0, TOTAL=44 | 0 |

Each missing-price/quantity build fails the input's `not_null`, `line_amount`'s `not_null` and the line sanity test. The no-orders build keeps all five customers with zero counts, 0.00 lifetime value and NULL order dates. The all-empty build produces empty marts. All these builds reported WARN=0.

The compiled reconciliation tests returned failure rows for all 16 corruptions: missing, invented and replaced keys at both mart grains; redistributed order amounts; wrong line/unit counts; redistributed customer amounts and both kinds of order counts; NULL order/customer amounts; and a nonzero amount or NULL count for a customer when there were no orders. Replacing an ID and moving totals preserved the old grand totals, which is precisely why per-entity checks matter.

The final script output was:

```text
REGRESSION checks=26 satisfied=26 failed=0
failed checks: []
```

Its exit code was 0. Every recovery build passed, and the subsequent mart inspection reproduced the original fingerprint. I also extracted the final downloadable ZIP into a fresh directory, installed its locked dependencies into a new `.venv`, and reran the builds and both mutation demos there.

## Where the guarantees end

**Tests are not constraints.** dbt tests are queries run after their dependencies are built. Once a table is replaced, its data is readable before its tests run and after a test fails. dbt does not roll it back. dbt does support model contracts with constraints, but enforcement depends on the platform, and its documentation notes that on some platforms constraints are only definable, not enforced. I didn't use contracts here; the project relies on tests and states that honestly.

**`dbt build` is not an atomic deployment.** Each node is committed as it's built. A failure in the middle leaves earlier nodes updated and later ones not rebuilt; existing downstream views still read updated data, as the demo showed. If consumers must never see a half-updated set of models, that needs a different design: building into a separate schema and swapping it in once all tests pass, for example. That is outside this article.

**Views and tables age differently.** A view is a stored query: it always shows the current state of what it reads, including data that just failed a test. A table stores the result of the last time that model was built, even if its tests then failed. Mixing the two in one DAG is normal, but after a failure they may disagree, and a reader should know which one they are looking at.

**Intentionally omitted:** source declarations and freshness checks, incremental models, snapshots, schema evolution, backfills, late-arriving data and retries of failed runs. Each of them changes what "running it again" means, and each deserves its own example rather than a paragraph here.

**Synthetic and small.** Nine order lines prove that the logic and the tests behave as designed. They say nothing about performance, and the build durations above are not benchmarks.

## What I keep from this example

Before I write the SQL of a model, I write down its grain. I change the grain in as few places as possible — here, line → order in intermediate and order → customer in the customer mart. I keep money exact, keep inconvenient rows instead of filtering them away, and put business rules into marts where they are visible and tested. Then I check that the tests actually fail when the data is wrong.

The full project — seeds, models, tests, scripts, `uv.lock` and a README with the run commands — is in [dbt-layers.zip]({{ '/assets/examples/dbt-layers.zip' | relative_url }}).

### Sources

- [dbt: How we structure our dbt projects — staging](https://docs.getdbt.com/best-practices/how-we-structure/2-staging), [intermediate](https://docs.getdbt.com/best-practices/how-we-structure/3-intermediate) and [marts](https://docs.getdbt.com/best-practices/how-we-structure/4-marts) — layer purposes and naming.
- [dbt: `ref`](https://docs.getdbt.com/reference/dbt-jinja-functions/ref) and [`source`](https://docs.getdbt.com/reference/dbt-jinja-functions/source) — references and dependencies.
- [dbt: Sources](https://docs.getdbt.com/docs/build/sources) — declaring raw tables and checking their freshness.
- [dbt: Seeds](https://docs.getdbt.com/docs/build/seeds) — what seeds are for.
- [dbt: Data tests](https://docs.getdbt.com/docs/build/data-tests) — generic and singular tests.
- [dbt: `build` command](https://docs.getdbt.com/reference/commands/build) — DAG order and skipping after test failures.
- [dbt: Constraints](https://docs.getdbt.com/reference/resource-properties/constraints) — definable versus enforced constraints.
- [dbt: Upgrading to v1.8](https://docs.getdbt.com/docs/dbt-versions/dbt-upgrade/upgrading-to-v1.8) — `tests:` renamed to `data_tests:`.
- [dbt-duckdb 1.7.3 README](https://github.com/duckdb/dbt-duckdb/blob/1.7.3/README.md) — profile configuration.
- [DuckDB 0.10: Numeric types](https://github.com/duckdb/duckdb-web/blob/main/docs/0.10/sql/data_types/numeric.md) — exact `DECIMAL` versus floating-point types.
- PyPI JSON API for [dbt-core 1.7.11](https://pypi.org/pypi/dbt-core/1.7.11/json), [dbt-duckdb 1.7.3](https://pypi.org/pypi/dbt-duckdb/1.7.3/json), [duckdb 0.10.1](https://pypi.org/pypi/duckdb/0.10.1/json) and [uv 0.3.0](https://pypi.org/pypi/uv/0.3.0/json) — release dates.
