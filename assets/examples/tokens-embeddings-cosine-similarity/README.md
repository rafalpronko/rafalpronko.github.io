# cosine-lab (Article 09)

Standard-library cosine similarity with explicit edge cases, plus a walk-through on
hand-chosen synthetic vectors. No model, no download, no network at runtime.

```bash
uv sync --locked                 # Python 3.11, pytest 7.4.3 (dev group, test tooling only)
uv run pytest -v                 # 45 tests
uv run python demo.py            # synthetic walk-through
```

`cosine_lab.py` and `demo.py` need only the Python 3.11 standard library.
The vectors in `demo.py` are hand-chosen; they are not encoder output and the
rankings are not a retrieval benchmark.

Public copies of `cosine_lab.py`, `demo.py`, `tests/test_cosine_lab.py`,
`pyproject.toml` and `uv.lock` live in
`../public/assets/examples/tokens-embeddings-cosine-similarity/` (byte-identical,
see `../metadata.json`).

Normalization scales by the largest absolute component before taking the norm,
including subnormal inputs and finite vectors with unrepresentable raw lengths.
Pooling validates every row and scales each column to avoid raw-sum overflow.
Tiny relative components and cancellation remain subject to float rounding.
`check_same_recipe` covers the symmetric toy only; real retrieval must validate
a compatible query/document contract, including any distinct role recipes.
