# Lab: train/eval, padding, checkpoints and seeds

Companion code for the article
"Train/eval, padding, checkpoints and seeds: four experiment traps".

- Python 3.11, `torch==2.2.2`, `numpy<2`, `pytest>=8.0,<8.2`
- CPU only, synthetic data, tiny model; no GPU, no dataset download
- `lab.py` and the tests call `torch.set_num_threads(1)`

```bash
uv run python lab.py   # PASS lines and a per-seed training-loss trace
uv run pytest -q       # separate tests for each property
```

Files: `pyproject.toml`, `lab.py`, `tests/test_lab.py`.

Dates and tooling: the article's 16 May 2024 date is an editorial label, not
proof of historical publication. The lab was actually run in October 2026
with the historical `torch==2.2.2` pin but with modern uv: the
`[dependency-groups]` project layout, the `uv run` commands and the
transitive dependencies in `uv.lock` are as resolved in October 2026.
Don't read them as a setup that was available in May 2024.

The equality checks (resume, reload parity) were observed on CPU in one
environment with the pinned versions. They are not a statement about CUDA,
other PyTorch versions, DataLoader resume, or crash-safe checkpoint writing.
Printed losses are a sanity trace, not a benchmark.

Trust warning: `torch.load(..., weights_only=True)` restricts what a
checkpoint may contain; it does not make a file from an unknown source safe.
Load only checkpoints you trust.
