# Greedy packing versus local improvement

Companion code for the article "Greedy packing versus local improvement:
measure, don't declare victory" (blog path /blog/greedy-packing-local-improvement/).

Synthetic one-dimensional bin packing with integer sizes. Standard library
only; pytest 8.3.3 is a test-only dependency.

| File | Purpose |
|---|---|
| packing/model.py | Item, validation, lower bound, invariant check |
| packing/ffd.py | First-fit decreasing baseline |
| packing/repair.py | Bounded, strictly accepted local repair |
| packing/corpus.py | Corpus generator and loader |
| packing/evaluate.py | Paired quality evaluation (no timing) |
| data/corpus.json | Persisted corpus used by the benchmark (seed 20240926) |
| demo.py | Two small fixtures: one fixed, one stuck |
| bench.py | Paired benchmark: quality, work counters, timing |
| make_corpus.py | Regenerates data/corpus.json |
| find_plateau.py | Search that selected the plateau test fixture |
| tests/ | pytest suite |

With uv (Python 3.11):

    uv run --python 3.11 --extra test pytest -q
    uv run --python 3.11 python demo.py
    uv run --python 3.11 python bench.py

Without uv, in a Python 3.11 virtual environment:

    python -m pip install pytest==8.3.3
    python -m pytest -q
    python demo.py
    python bench.py

Do not regenerate the corpus to compare results: Python only guarantees
that random.random() keeps its sequence across versions, so bench.py
reads data/corpus.json instead.
