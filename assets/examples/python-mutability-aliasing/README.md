# Mutability and aliasing example

Keep `aliasing.py`, `test_aliasing.py`, `mutation_check.py`,
`pyproject.toml` and `uv.lock` together in this directory.
The example uses synthetic data and only Python's standard library.

Install modern uv and make it available on `PATH`. In a terminal, change
to this directory, then run:

```sh
uv run --isolated --locked --python 3.12.1 python aliasing.py
uv run --isolated --locked --python 3.12.1 pytest -v
uv run --isolated --locked --python 3.12.1 mypy --strict --cache-dir /dev/null aliasing.py test_aliasing.py mutation_check.py
uv run --isolated --locked --python 3.12.1 python mutation_check.py
```

uv selects CPython 3.12.1 and installs the development dependencies in an
isolated environment. Initial setup may need internet access to download
Python and packages. `--locked` rejects changes to the supplied lock.
The commands were verified with uv 0.7.8, pytest 8.0.0 and mypy 1.8.0.

Expected results:

- `aliasing.py` prints these five lines:

  ```text
  shallow: input unchanged: False
  shallow: input A-1: {'order_id': 'A-1', 'line_cents': [300, 1200], 'tags': ['raw', 'eu', 'vip']}
  shallow: eu A-1 tags: ['raw', 'eu', 'vip'] | same list as vip: True
  owned:   input unchanged: True
  owned:   eu A-1 tags: ['raw', 'eu', 'manual-review'] | vip A-1 tags: ['raw', 'vip']
  ```

- pytest: **18 passed**.
- mypy: **Success: no issues found in 3 source files**.
- The mutation checker substitutes the shallow function for the fixed
  function. It requires exactly these four failed tests:

  ```text
  test_aliasing.py::test_fixed_version_leaves_input_unchanged
  test_aliasing.py::test_fixed_version_keeps_sibling_results_independent
  test_aliasing.py::test_fixed_version_owns_every_mutable_component
  test_aliasing.py::test_fixed_version_separates_aliases_present_in_the_input
  ```

  Only `test_aliasing.py::test_fixed_version_returns_expected_values`
  must pass. The inner pytest run must return test-failure status **1**.
  The checker itself returns **0** only for that exact result; skips,
  unexpected test IDs and setup/teardown errors are rejected. The four
  displayed assertion failures are intentional.

Prepared and verified on 12 October 2026. The article's 8 February 2024
date is an editorial series label, not a historical publication date.
uv is modern verification tooling, not claimed available in February
2024. Python 3.12.1, pytest 8.0.0 and mypy 1.8.0 predate that label;
the dependency resolution excludes uploads after 8 February 2024 UTC.
