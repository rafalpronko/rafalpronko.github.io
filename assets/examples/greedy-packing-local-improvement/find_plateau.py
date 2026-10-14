"""Deterministic search for a small instance where FFD + repair is stuck.

Run: uv run --python 3.11 python find_plateau.py
Prints the first few instances (in search order) where the repair stops
with no improving move although an exact search finds fewer bins. This
is how the plateau test fixture was selected; it is an illustration,
not evidence about how often plateaus happen.
"""
import random

from packing.ffd import first_fit_decreasing
from packing.model import Item, lower_bound
from packing.repair import Budget, pack_exact, repair


def optimum(capacity, items):
    bins = lower_bound(capacity, items)
    while pack_exact(capacity, items, bins, Budget(10**7)) is None:
        bins += 1
    return bins


def main(seed=1, trials=200_000, show=3):
    rng, found = random.Random(seed), 0
    for trial in range(trials):
        capacity = 20
        sizes = [rng.randint(1, capacity) for _ in range(rng.randint(8, 12))]
        items = [Item(f"x{i}", s) for i, s in enumerate(sizes)]
        ffd = first_fit_decreasing(capacity, items)
        result = repair(capacity, ffd, max_work=100_000)
        if result.stop_reason != "no_improving_move":
            continue
        best = optimum(capacity, items)
        if best < len(result.bins):
            found += 1
            print(f"trial {trial}: sizes={sorted(sizes, reverse=True)} "
                  f"ffd={len(ffd)} repaired={len(result.bins)} "
                  f"optimum={best} lower_bound={lower_bound(capacity, items)}")
            if found == show:
                break
    print(f"searched {trial + 1} trials, seed={seed}")


if __name__ == "__main__":
    main()
