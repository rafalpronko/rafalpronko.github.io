# Optuna without tuning on the test set

Companion code for the article "Tuning with Optuna without tuning on the test set".

A synthetic binary classification problem (1000 rows from `make_classification`, about 20% positives) is split once into 750 development rows and 250 test rows. Optuna searches the regularisation of a `StandardScaler` + `LogisticRegression` pipeline using the mean ROC AUC over three fixed stratified folds of the development rows. A baseline is scored on the same folds. A rule fixed in advance keeps the tuned parameters only if they beat the baseline by at least 0.005 ROC AUC. The chosen model is refitted on all development rows and scored once on the test rows.

No external data, services or GPUs are needed.

## Files

- `tuning.py` – the whole example; run it as a script or import its functions.
- `tests/test_tuning.py` – twelve pytest checks: the split, the folds, fold-local scaling, the search interfaces, the rows seen by cross-validation during the search, the metric and direction, parameter application, the sampler's startup trials and the trial budget, the selection rule, the exact rows `run()` passes to tuning, final fitting, prediction and evaluation, and the single direct call to the test metric.
- `pyproject.toml`, `uv.lock` – pinned environment.

## Versions

Python 3.11, optuna 3.4.0, scikit-learn 1.3.2, numpy 1.26.2, scipy 1.11.4, pytest 7.4.3.

`pyproject.toml` sets `exclude-newer = "2023-12-07T00:00:00Z"` for uv, so every transitive dependency in `uv.lock` was also released before that date.

## Run

With [uv](https://docs.astral.sh/uv/):

```bash
uv run --locked python tuning.py
uv run --locked pytest -v
```

Without uv, create a Python 3.11 virtual environment and install the versions listed above, then run `python tuning.py` and `pytest -v`. This route was not part of the recorded verification.

## Verification note

uv did not exist in 2023 and is not part of the article's historical setting; it is the tool used to run the example. The article's date (7 December 2023) is an editorial label, not a record of when this code was run. The code was actually run on 12 October 2026 (CPython 3.11.12, macOS arm64); the output below comes from that run.

```text
baseline cv_auc=0.8654
best     cv_auc=0.8689  {'penalty': 'l1', 'C': 0.2001368402568556}
chosen   {'penalty': 'l2', 'C': 1.0}
held-out test roc_auc=0.8949 (reported once, not used to choose)
```

The best trial improved on the baseline by about 0.0035, below the 0.005 margin, so the rule kept the baseline. Other platforms or library builds may produce different trials; the numbers say nothing about logistic regression or Optuna in general.
