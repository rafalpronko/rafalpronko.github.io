"""Fixed-seed synthetic corpus with a uniform and an adversarial family.

random.Random(seed) is reproducible here, but the docs only guarantee
that random() keeps its sequence across Python versions, not randint()
or shuffle(). So the generated corpus is persisted to JSON and the
benchmark reads the file, never the generator.
"""
import json
import random
from dataclasses import dataclass

from packing.model import Item


@dataclass(frozen=True)
class Instance:
    id: str
    family: str
    capacity: int
    items: tuple


def uniform_sizes(rng, n=60):
    # Sizes uniform in [20, 100], capacity 150.
    return 150, [rng.randint(20, 100) for _ in range(n)]


def triplet_sizes(rng, triplets=20):
    # Each triplet sums exactly to the capacity, so the optimum equals
    # the lower bound: n / 3 bins, every bin full. Order is shuffled.
    capacity, sizes = 1000, []
    for _ in range(triplets):
        first = rng.randint(380, 490)
        second = rng.randint(250, capacity - first - 250)
        sizes += [first, second, capacity - first - second]
    rng.shuffle(sizes)
    return capacity, sizes


FAMILIES = {"uniform": uniform_sizes, "triplet": triplet_sizes}


def make_corpus(seed, per_family):
    rng = random.Random(seed)
    corpus = []
    for family, generate in FAMILIES.items():
        for index in range(per_family):
            capacity, sizes = generate(rng)
            name = f"{family}-{index:02d}"
            corpus.append({
                "id": name, "family": family, "capacity": capacity,
                "items": [{"id": f"{name}-i{i:02d}", "size": s}
                          for i, s in enumerate(sizes)],
            })
    return corpus


def load_corpus(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        Instance(raw["id"], raw["family"], raw["capacity"],
                 tuple(Item(it["id"], it["size"]) for it in raw["items"]))
        for raw in data["instances"]
    ]
