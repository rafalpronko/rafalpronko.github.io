"""Two small fixtures: one the repair fixes, one where it gets stuck.

Run: uv run --python 3.11 python demo.py
"""
from packing.ffd import first_fit_decreasing
from packing.model import Item, load, lower_bound
from packing.repair import repair


def show(label, capacity, sizes):
    items = [Item(f"x{i}", s) for i, s in enumerate(sizes)]
    ffd = first_fit_decreasing(capacity, items)
    result = repair(capacity, ffd, max_work=10_000)
    fmt = lambda bins: " ".join(
        "[" + ",".join(str(it.size) for it in b) + f"]={load(b)}" for b in bins)
    print(f"{label}: capacity={capacity} lower_bound={lower_bound(capacity, items)}")
    print(f"  ffd    {len(ffd)} bins  {fmt(ffd)}")
    print(f"  repair {len(result.bins)} bins  {fmt(result.bins)}")
    print(f"  stop={result.stop_reason} windows={result.windows} "
          f"placements={result.placements}")


show("manufactured", 10, [5, 4, 4, 3, 2, 2])
show("plateau", 20, [13, 12, 9, 9, 5, 4, 3, 3, 1])
