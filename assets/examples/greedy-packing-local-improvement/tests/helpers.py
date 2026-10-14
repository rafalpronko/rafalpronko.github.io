"""Small builders shared by the tests."""
from packing.model import Item


def items_of(*sizes, prefix="x"):
    """Distinct ids even for duplicate sizes: x0, x1, ..."""
    return [Item(f"{prefix}{i}", size) for i, size in enumerate(sizes)]


def sizes_of(bins):
    return [[item.size for item in b] for b in bins]


def ids_of(bins):
    return sorted(item.id for b in bins for item in b)
