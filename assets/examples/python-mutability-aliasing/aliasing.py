"""Shallow copies, aliasing and owned reconstruction in a small step.

Synthetic data, standard library only, written for Python 3.12.
Run: python aliasing.py
"""
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


def main() -> None:
    batch = load_batch()
    eu = tag_orders_shallow(batch, "eu")
    vip = tag_orders_shallow(batch, "vip")
    print("shallow: input unchanged:", batch == load_batch())
    print("shallow: input A-1:", batch[0])
    print(
        "shallow: eu A-1 tags:", eu[0]["tags"],
        "| same list as vip:", eu[0]["tags"] is vip[0]["tags"],
    )

    batch = load_batch()
    eu = tag_orders(batch, "eu")
    vip = tag_orders(batch, "vip")
    eu[0]["tags"].append("manual-review")
    print("owned:   input unchanged:", batch == load_batch())
    print(
        "owned:   eu A-1 tags:", eu[0]["tags"],
        "| vip A-1 tags:", vip[0]["tags"],
    )


if __name__ == "__main__":
    main()
