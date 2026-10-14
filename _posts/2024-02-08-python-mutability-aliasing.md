---
layout: post
lang: en
title: "Mutability and aliasing: the data bug your types won't catch"
date: 2024-02-08 09:00:00 +0100
permalink: /blog/python-mutability-aliasing/
author: Rafał Prońko
categories: [python, data-engineering]
description: "Shallow copies, shared nested lists and frozen-but-mutable data in Python: why a correctly typed pipeline step can still corrupt its input, and how to rebuild what you own."
image: /assets/images/python-mutability-aliasing.svg
---

# Mutability and aliasing: the data bug your types won't catch

![Two panels. On the left, after a shallow copy, the input order and two results are separate dictionaries whose tags keys all point to one shared list. On the right, after explicit reconstruction, each dictionary points to its own list.]({{ '/assets/images/python-mutability-aliasing.svg' | relative_url }})

<time datetime="2024-02-08">8 February 2024 — editorial series date</time>

*The date is an editorial label, not a record of historical publication. This article and its example were prepared and verified on 12 October 2026.*

A pipeline step takes a batch of records and returns a batch of records. The signature says `list[Order] -> list[Order]`. The type checker is happy, and the unit test that compares the output with the expected values passes.

Then a later step reads the original batch and finds tags it never added. Or two outputs built from the same input start changing together.

Nothing here is a type error. Every value has the right type at every moment. The problem is **who else holds a reference to the same object**, and type annotations don't describe that.

I want to show the mechanism on one small example, then fix it in a way that I can test.

## Assignment binds, it does not copy

In Python, a name or a container slot refers to an object. Assignment creates a new binding to an existing object; it does not copy it. The `copy` module documentation says this directly.

Three operations are easy to blur:

- **Binding** — `b = a`. Now two names refer to one object.
- **Mutation** — `a.append(x)`, `a.sort()`, `d["k"] = v`. The object changes; every binding sees the change.
- **Rebinding** — `a = a + [x]`. A new object is created and only this name is moved to it.

Two comparison operators answer different questions. `==` asks whether two values are equal. `is` asks whether two expressions refer to the same object. Two lists can be `==` and still be independent. If `a is b`, they can never be independent: there is only one object.

For data pipelines the second question is the one that bites.

## One example: tagging a batch of orders

The example uses only the standard library and runs on Python 3.12. Orders are plain dictionaries, described with a `TypedDict`. Amounts are integer cents. The complete [downloadable example](#download-and-run-the-complete-example) includes `aliasing.py`, whose `main()` runs both versions and prints the results. The blocks below are fragments: continue with the definitions above, or use the complete file. The frozen-dataclass fragment deliberately raises `FrozenInstanceError`.

```python
from typing import TypedDict


class Order(TypedDict):
    order_id: str
    line_cents: list[int]
    tags: list[str]


def load_batch() -> list[Order]:
    """Synthetic input: two orders, each with its own lists."""
    return [
        Order(order_id="A-1", line_cents=[1200, 300], tags=["raw"]),
        Order(order_id="A-2", line_cents=[500], tags=["raw"]),
    ]


def tag_orders_shallow(batch: list[Order], tag: str) -> list[Order]:
    """BUG: new outer dicts, but the nested lists are the caller's."""
    result: list[Order] = []
    for order in batch:
        row = order.copy()  # shallow: row["tags"] is order["tags"]
        row["line_cents"].sort()  # mutates the input list in place
        row["tags"].append(tag)  # ...and so does this
        result.append(row)
    return result
```

The step looks careful. It doesn't modify `order`; it copies it first.

But `dict.copy()` is a **shallow copy**. It creates a new outer dictionary and inserts references to the same objects that the original held. `row is not order` is true. `row["tags"] is order["tags"]` is also true.

So `.sort()` and `.append()` modify the caller's lists. I run the step twice on the same input, once for an `"eu"` result and once for `"vip"`. This fragment uses the definitions above:

```python
batch = load_batch()
eu = tag_orders_shallow(batch, "eu")
vip = tag_orders_shallow(batch, "vip")
```

The output of `aliasing.py`:

```text
shallow: input unchanged: False
shallow: input A-1: {'order_id': 'A-1', 'line_cents': [300, 1200], 'tags': ['raw', 'eu', 'vip']}
shallow: eu A-1 tags: ['raw', 'eu', 'vip'] | same list as vip: True
```

The input was reordered and tagged. The `eu` result carries the `vip` tag, because `eu[0]["tags"]`, `vip[0]["tags"]` and `batch[0]["tags"]` are one list.

Types did not change at any point. Before and after, `batch` is a `list` of dicts whose `tags` is a `list[str]`. The Python runtime does not enforce annotations at all, and a static checker reads `list[str]` as "a list of strings" — not "a list of strings that nobody else holds". `mypy --strict` reports no issues for this module, buggy function included.

## Mutation versus rebinding, one operator apart

The same trap hides in augmented assignment. For a list, `+=` extends the list in place where possible, and only then assigns the result back to the target. This fragment uses `load_batch` from above:

```python
order = load_batch()[0]
row = order.copy()
row["tags"] = row["tags"] + ["eu"]  # new list; input untouched

row = order.copy()
row["tags"] += ["eu"]  # extends the shared list; input changed
```

The two lines read almost the same and produce the same `row`. Only one of them corrupts the input. The language reference describes this in-place behaviour of augmented assignment explicitly; I would not rely on reviewers spotting it.

## The fix: rebuild what you own

I want a rule I can check: **every mutable object in the result is created by the function that returns it**. Immutable parts, such as `str`, can be shared freely, because nobody can change them.

```python
def tag_orders(batch: list[Order], tag: str) -> list[Order]:
    """Fixed: every mutable part of the result is built here."""
    return [
        Order(
            order_id=order["order_id"],  # immutable str: safe to share
            line_cents=sorted(order["line_cents"]),  # new list
            tags=[*order["tags"], tag],  # new list
        )
        for order in batch
    ]
```

`sorted()` returns a new list instead of sorting in place. `[*order["tags"], tag]` builds a new list instead of appending. The signature is identical to the buggy version — which is exactly the point.

This fragment uses the fixed function above and mutates one result on purpose:

```python
batch = load_batch()
eu = tag_orders(batch, "eu")
vip = tag_orders(batch, "vip")
eu[0]["tags"].append("manual-review")
```

```text
owned:   input unchanged: True
owned:   eu A-1 tags: ['raw', 'eu', 'manual-review'] | vip A-1 tags: ['raw', 'vip']
```

The input is unchanged and the sibling result is unaffected.

Explicit reconstruction has a cost: when I add a mutable field to `Order`, I must also decide how `tag_orders` builds it. I see that as a feature. The ownership decision lives in the code, next to the field, instead of being implied by a generic copy.

## Why not just `deepcopy`?

`copy.deepcopy()` recursively copies nested objects, so it would fix this example. I use it in tests as a snapshot tool. In pipeline code I prefer explicit reconstruction, for three reasons.

**It preserves internal aliases.** `deepcopy` keeps a memo of objects already copied during one pass. If two orders in the input share one `tags` list, the copies will share one new `tags` list. This fragment uses `Order` from above:

```python
import copy

shared = ["raw"]
batch = [
    Order(order_id="A-1", line_cents=[1], tags=shared),
    Order(order_id="A-2", line_cents=[2], tags=shared),
]
cloned = copy.deepcopy(batch)
assert cloned[0]["tags"] is not shared  # detached from the original
assert cloned[0]["tags"] is cloned[1]["tags"]  # internal alias survives
```

That is the correct behaviour for a faithful copy of an object graph. It's not what I want when each output record must own its data. `tag_orders` gives each record a new list, so the alias in the input does not survive.

**Not everything can be copied.** The `copy` module does not copy types such as modules, files, sockets or stack frames. On Python 3.12.1, deep-copying a dataclass that holds an open text file raises `TypeError: cannot pickle 'TextIOWrapper' instances`. Functions and classes are "copied" by returning the original object. A record that carries a file handle, a socket or another external resource is not something `deepcopy` can duplicate in general.

**It copies too much.** The documentation itself names this risk: data intended to be shared between copies is copied anyway. Lookup tables or configuration attached to a record get duplicated without anyone deciding that they should be.

Classes can customise this with `__copy__()` and `__deepcopy__()`. That's useful for library types. For a pipeline step, I'd rather read the ownership rule in the function body.

## Frozen dataclasses and tuples are only shallowly immutable

A common reaction is "make it immutable". It helps, but only one level deep.

A tuple is immutable: the identities of the objects it contains cannot change. If one of those objects is a list, the list can still change, and the tuple's value changes with it. The data model reference states this for immutable containers.

A frozen dataclass behaves the same way. `frozen=True` emulates immutability by adding `__setattr__()` and `__delattr__()` that raise `FrozenInstanceError`. This fragment uses `Order` and `load_batch` from above; its final line intentionally raises that exception:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenBatch:
    source: str
    orders: tuple[Order, ...]


frozen = FrozenBatch(source="eu", orders=tuple(load_batch()))
frozen.orders[0]["tags"].append("eu")  # nested mutation works
frozen.orders = ()  # raises FrozenInstanceError intentionally
```

`hash(frozen)` raises `TypeError: unhashable type: 'dict'` here because the generated hash reaches the dictionaries in `FrozenBatch.orders`. This is a property of these fields, not a general immutability test: immutable custom types can be unhashable, and mutable objects can have identity-based hashes.

## `default_factory` does not protect caller-supplied objects

Dataclasses reject unhashable defaults such as `x: list = []`, and `field(default_factory=list)` creates a new list each time a default is needed. That covers only the default. This standalone fragment shows caller-supplied sharing:

```python
from dataclasses import dataclass, field


@dataclass
class RunReport:
    warnings: list[str] = field(default_factory=list)


shared: list[str] = []
a, b = RunReport(shared), RunReport(shared)
a.warnings.append("late file")
# b.warnings == ["late file"] and shared == ["late file"]
```

When the caller passes a list, the generated `__init__()` stores that object. Two reports built from the same list share it. If the class must own the list, it has to take a copy itself, for example `self.warnings = list(self.warnings)` in `__post_init__()`. That is again a shallow copy, which is sufficient here only because `str` is immutable.

## Tests that check ownership, not just values

The full downloadable test file has 18 tests, including a hashability counterexample; these two carry the main regression and run with `pytest` beside `aliasing.py`:

```python
from aliasing import load_batch, tag_orders


def test_fixed_version_leaves_input_unchanged() -> None:
    batch = load_batch()
    tag_orders(batch, "eu")

    assert batch == load_batch()


def test_fixed_version_keeps_sibling_results_independent() -> None:
    batch = load_batch()
    eu = tag_orders(batch, "eu")
    vip = tag_orders(batch, "vip")

    eu[0]["tags"].append("manual-review")
    eu[0]["line_cents"].append(1)

    assert vip[0] == {
        "order_id": "A-1",
        "line_cents": [300, 1200],
        "tags": ["raw", "vip"],
    }
    assert batch == load_batch()
```

The first checks the input after the call. The second mutates one result and checks the other, which is the operation a later pipeline step would perform. A further test passes an input whose two orders share one `tags` list and checks that the outputs don't.

To check the tests themselves, I swapped `tag_orders_shallow` in for `tag_orders` and ran the tests for the fixed version. Exactly four ownership tests failed, the one value-only test passed, and pytest returned its test-failure status (1). The mutation checker requires those exact test IDs and outcomes; skips or setup/teardown errors make the check fail. A value-only test cannot see aliasing; it has to mutate something and look elsewhere.

## Download and run the complete example

Download all five files into one directory: [aliasing.py]({{ '/assets/examples/python-mutability-aliasing/aliasing.py' | relative_url }}), [test_aliasing.py]({{ '/assets/examples/python-mutability-aliasing/test_aliasing.py' | relative_url }}), [mutation_check.py]({{ '/assets/examples/python-mutability-aliasing/mutation_check.py' | relative_url }}), [pyproject.toml]({{ '/assets/examples/python-mutability-aliasing/pyproject.toml' | relative_url }}) and [uv.lock]({{ '/assets/examples/python-mutability-aliasing/uv.lock' | relative_url }}). A [ZIP containing those files and a README]({{ '/assets/examples/python-mutability-aliasing.zip' | relative_url }}) is also available.

In a POSIX shell, this downloads the individual files from this site's example directory:

```sh
mkdir -p python-mutability-aliasing
cd python-mutability-aliasing
example_base="{{ '/assets/examples/python-mutability-aliasing' | absolute_url }}"
for file in aliasing.py test_aliasing.py mutation_check.py pyproject.toml uv.lock
do
    curl --fail --location --output "$file" "$example_base/$file" || exit 1
done
```

With modern `uv` installed and available on `PATH`, run from that directory (or the extracted ZIP directory):

```sh
uv run --isolated --locked --python 3.12.1 python aliasing.py
uv run --isolated --locked --python 3.12.1 pytest -v
uv run --isolated --locked --python 3.12.1 mypy --strict --cache-dir /dev/null aliasing.py test_aliasing.py mutation_check.py
uv run --isolated --locked --python 3.12.1 python mutation_check.py
```

uv selects CPython 3.12.1 and installs the pinned development dependencies in an isolated environment. The lock must stay unchanged. The example prints the five lines shown above; pytest reports **18 passed**, and mypy reports no issues in three source files. The mutation run intentionally prints **4 failed, 1 passed** with pytest status **1**, then exits **0** only if the exact expected outcomes match. These commands use modern verification tooling, not tooling claimed to have existed on the editorial date.

## Where this guarantee ends

The guarantee is: `tag_orders` does not mutate its input, and each call returns records whose lists belong only to that result. The tests check it for the shapes in this example.

It doesn't make the input immutable. Another function can still mutate `batch`. It doesn't protect objects the caller keeps after the call, either; if the caller hands the same result to two consumers, those consumers share it.

It also doesn't scale automatically. When `Order` gains a nested dict or a list of dicts, the reconstruction must go one level deeper, and the ownership tests should grow with it.

In a pipeline I treat this as a contract of each step: either "I return new objects" or "I mutate in place, and the signature and name say so" — for example, `sort_in_place(batch) -> None`. Mixing the two behind the same `list[Order] -> list[Order]` signature is how input corruption ends up looking like a valid result.

### Sources

- [Python 3.12.1: `copy` — Shallow and deep copy operations](https://docs.python.org/release/3.12.1/library/copy.html) — assignment creates bindings, shallow copy inserts references, deep copy recursion, memo, uncopyable types, `__copy__`/`__deepcopy__`.
- [Python 3.12.1: Data model — Objects, values and types](https://docs.python.org/release/3.12.1/reference/datamodel.html#objects-values-and-types) — identity, mutability, immutable containers holding mutable objects.
- [Python 3.12.1: Identity comparisons](https://docs.python.org/release/3.12.1/reference/expressions.html#is-not) and [augmented assignment](https://docs.python.org/release/3.12.1/reference/simple_stmts.html#augmented-assignment-statements) — `is` and in-place `+=`.
- [Python 3.12.1: `dataclasses` — frozen instances](https://docs.python.org/release/3.12.1/library/dataclasses.html#frozen-instances), [default factory functions](https://docs.python.org/release/3.12.1/library/dataclasses.html#default-factory-functions) and [mutable default values](https://docs.python.org/release/3.12.1/library/dataclasses.html#mutable-default-values).
- [Python 3.12.1: `typing`](https://docs.python.org/release/3.12.1/library/typing.html) — the runtime does not enforce annotations.

*Verification note: the example, its 18-test pytest suite, exact mutation check and the `mypy --strict` check were executed on 12 October 2026 on CPython 3.12.1 (released 8 December 2023) with pytest 8.0.0 (released 27 January 2024) and mypy 1.8.0 (released 21 December 2023). uv 0.7.8 served only as a modern verification harness; it was not available in February 2024. The application code uses only the standard library.*
