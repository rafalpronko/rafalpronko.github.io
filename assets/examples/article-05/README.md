# Group leakage demo: row split versus grouped train/validation/test

Companion code for the article "Train, validation and test: how group
leakage inflates your score". Synthetic data only; not a benchmark.

## Files

- `group_leakage.py` — the example (data generator, both splits, model
  selection on validation, one final test score, 20-seed summary).
- `test_group_leakage.py` — pytest suite (16 tests).
- `pyproject.toml` — dependencies pinned to exact versions.
- `uv.lock` — lock file with hashes for every distribution.
- `.python-version` — Python 3.11.
- `SHA256SUMS` — SHA-256 checksums of the files above.

## Run it with uv

Requires [uv](https://docs.astral.sh/uv/). The commands were executed with
uv 0.7.8. By default, uv downloads CPython 3.11 if you do not have it.

```sh
uv sync --frozen
uv run --frozen python group_leakage.py
uv run --frozen pytest -v
```

`--frozen` installs exactly what `uv.lock` records and refuses to
re-resolve, so you get numpy 1.26.1, scipy 1.11.3, scikit-learn 1.3.2,
joblib 1.3.2, threadpoolctl 3.2.0 and pytest 7.4.3. uv creates `.venv/`
next to the files; delete it when you are done.

## Without uv

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install numpy==1.26.1 scipy==1.11.3 scikit-learn==1.3.2 \
    joblib==1.3.2 threadpoolctl==3.2.0 pytest==7.4.3 \
    iniconfig==2.0.0 pluggy==1.3.0 packaging==23.2
python group_leakage.py
pytest -v
```

The last line pins pytest's own dependencies to the versions in `uv.lock`.
On Windows, pytest also needs colorama; `uv.lock` records 0.4.6, so add
`colorama==0.4.6` there (and activate with `.venv\Scripts\activate`).

This path does not check the hashes in `uv.lock`; the uv path does. Only
the uv commands above were executed for the article. The pip path is an
untested alternative: a modern pip and Python 3.11 patch release may
behave differently from the run recorded in the article.

## Archive timestamps

The files in the ZIP carry the timestamp 2023-11-16 09:00. That is the
article's editorial date label, set so the archive is byte-reproducible.
It is not provenance: the recorded runs were executed on 2026-10-12 with
the tools named above.

## What to expect

The output should match the article's results block. The 20-seed summary
reports binary balanced accuracy only for test sets that contain both
classes. With the group split, seed 11 draws a test set with no positive
group, so its score is reported as undefined and the mean, min and max
cover the other 19 seeds. A validation set with a single class would stop
the run with `SingleClassError` instead of tuning on an undefined score;
none of the 20 seeds triggers it.
