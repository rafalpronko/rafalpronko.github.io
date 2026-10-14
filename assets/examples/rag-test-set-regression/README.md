# RAG test set regression report (offline example)

Companion code for "Building a RAG Test Set: Questions, References, and Regression Reports".

This is an **offline** evaluator. It does not call a model or a retriever. It
reads a synthetic corpus, a versioned test set and two files of *stored*
responses (`baseline.json`, `candidate.json`), scores each case with
deterministic checks, pairs the runs case by case and applies a release gate.
All data is synthetic; the product "Tidepool" does not exist.

The checks are proxies. A required fact string appearing in the answer and
cited passage IDs covering a reference evidence set do not prove that the
answer is faithful to that evidence.

## Run

Use the complete unpacked ZIP archive (`rag-test-set-regression.zip`); the
package needs every file, including `rageval/__init__.py` and `rageval/__main__.py`.

Requires uv and an installed Python 3.11 interpreter (`uv venv --python 3.11`
looks for it). No other runtime dependencies (standard library only).

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp
.venv/bin/python -m rageval                                  # exit 1: gate fails on the demo data
.venv/bin/python -m rageval --candidate data/baseline.json   # exit 0
```

`--basetemp=.pytest-tmp` keeps test files inside this directory instead of
pytest's shared temporary directory.

These commands were executed on macOS (arm64) with uv 0.7.8 and CPython 3.11.12.

`requirements-dev.txt` pins pytest 8.2.2, the version available on 6 June 2024,
to keep the article's historical claim checkable. It is not a recommendation:
PyPI lists a known vulnerability for that release. For real use, install a
current pytest.

Exit codes: `0` gate passed, `1` gate failed, `2` invalid or incompatible input
(including a missing or unreadable file, an empty test set and an invalid policy).

## Files

- `data/corpus.json` – passages with stable IDs, pinned by `corpus_version`
- `data/dataset.json` – cases with split, group, kind, criticality, expected behaviour, facts and evidence sets
- `data/baseline.json`, `data/candidate.json` – stored responses for every case
- `data/gate_policy.json` – illustrative thresholds, not recommendations;
  `max_slice_drop` is an absolute drop in pass rate (0.25 = 25 percentage points)
- `rageval/` – loading, validation, scoring, comparison, gate, report, CLI
- `tests/` – pytest suite
