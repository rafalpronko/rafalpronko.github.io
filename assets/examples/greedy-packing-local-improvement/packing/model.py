"""Instance model, validation, invariants and the trivial lower bound."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    size: int


Bins = tuple[tuple[Item, ...], ...]


class PackingError(ValueError):
    """A packing breaks an invariant."""


def positive_int(value, name):
    # bool is a subclass of int; True must not silently mean 1.
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_instance(capacity, items):
    """Return the items as a tuple or raise TypeError/ValueError."""
    positive_int(capacity, "capacity")
    items = tuple(items)
    for item in items:
        if not isinstance(item, Item):
            raise TypeError(f"expected Item, got {type(item).__name__}")
        positive_int(item.size, f"size of {item.id!r}")
        if item.size > capacity:
            raise ValueError(f"{item.id!r} ({item.size}) exceeds capacity")
    repeated = [i for i, n in Counter(it.id for it in items).items() if n > 1]
    if repeated:
        raise ValueError(f"duplicate item ids: {repeated}")
    return items


def load(bin_):
    return sum(item.size for item in bin_)


def lower_bound(capacity, items):
    """ceil(total size / capacity): no packing can use fewer bins."""
    total = sum(item.size for item in validate_instance(capacity, items))
    return -(-total // capacity)


def check_packing(capacity, items, bins):
    """Every item exactly once, no empty or overfull bin."""
    for index, bin_ in enumerate(bins):
        if not bin_:
            raise PackingError(f"bin {index} is empty")
        if load(bin_) > capacity:
            raise PackingError(f"bin {index} exceeds capacity {capacity}")
    packed = Counter(item for bin_ in bins for item in bin_)
    if packed != Counter(items):
        raise PackingError("packed items differ from instance items")
