"""First-fit decreasing baseline."""
from packing.model import validate_instance


def first_fit_decreasing(capacity, items):
    """Largest item first; each goes into the first open bin with room.

    sorted() is stable, so equal sizes keep their input order and the
    result is deterministic even when sizes repeat.
    """
    order = sorted(validate_instance(capacity, items), key=lambda it: -it.size)
    bins, free = [], []
    for item in order:
        for index, room in enumerate(free):
            if item.size <= room:
                bins[index].append(item)
                free[index] -= item.size
                break
        else:
            bins.append([item])
            free.append(capacity - item.size)
    return tuple(tuple(bin_) for bin_ in bins)
