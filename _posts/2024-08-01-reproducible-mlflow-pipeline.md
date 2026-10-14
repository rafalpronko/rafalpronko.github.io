---
layout: post
lang: en
title: "A reproducible ML pipeline with MLflow and a regression test"
date: 2024-08-01 09:00:00 +0200
permalink: /blog/reproducible-mlflow-pipeline/
author: Rafał Prońko
categories: [mlops]
description: "Scaler + Ridge with MLflow 2.14 and pytest: fingerprinted data and split, logged artifacts, a reload check and a gate that separates reproducibility from model quality."
image: /assets/images/reproducible-mlflow-pipeline.svg
---

# A reproducible ML pipeline with MLflow and a regression test

![Diagram: committed data, code and uv.lock feed one MLflow run; the logged model is reloaded; a pytest gate compares the run with an approved fixture in two checks, reproducibility and quality.]({{ '/assets/images/reproducible-mlflow-pipeline.svg' | relative_url }})

<time datetime="2024-08-01">1 August 2024</time> — editorial date label, not a record of historical publication. The example pins releases available before that date, but it was written and run in October 2026 (details under "Versions and how to run it").

A model is retrained and the test RMSE moves in the fourth decimal place. Is that a bug, a dependency update, a changed split, or just floating-point noise? And if someone changes a hyperparameter on purpose, how do I know the new model is still good enough?

These are two different questions, and I want two different answers:

1. **Reproducibility:** with the same data, split, code, environment and configuration, do I get the approved predictions back?
2. **Quality:** a candidate is *allowed* to change predictions. Is it still not meaningfully worse than the approved model?

MLflow helps with the first question by recording what happened in a run. It does not answer either question by itself: a logged metric does not fail a build. The decision belongs to a test that compares the run with something a person has approved.

In this article I build a small, complete example: a `StandardScaler` + `Ridge` pipeline on synthetic data, tracked with MLflow 2.14.3, guarded by a pytest regression gate. Everything runs locally in a few seconds, without a tracking server and without paid services.

## What the example promises — and what it doesn't

"Reproducible" here means something narrow and testable:

- the training data is a **committed CSV**, identified by its SHA-256;
- the train/evaluation split is part of that file and has its own fingerprint;
- the environment is pinned by `uv.lock`;
- the code that ran is logged with the run;
- with all of the above unchanged, the predictions match the approved ones **within a stated tolerance**.

It does not mean "the same bytes on any machine". I'll come back to why that tolerance is not zero, because the example ran into exactly that problem.

## Versions and how to run it

The direct dependencies are pinned to releases that were available on PyPI before the article date. I checked the upload times in the PyPI JSON API:

| Package | Version | First file uploaded to PyPI |
|---|---|---|
| mlflow | 2.14.3 | 2024-07-12 |
| scikit-learn | 1.5.1 | 2024-07-03 |
| numpy | 1.26.4 | 2024-02-05 |
| pytest | 8.3.2 | 2024-07-25 |
| setuptools | 72.1.0 | 2024-07-29 |

`setuptools` is in the list for a practical reason: in this environment MLflow 2.14.3 imports `pkg_resources` (you can see the `DeprecationWarning` from `mlflow/utils/requirements_utils.py` in the test output), but `pkg_resources` comes from setuptools, which MLflow does not declare as a dependency. A clean uv environment does not include it, so I pin it explicitly.

To be clear about what is historical and what is not: the pins are historical, but the tooling around them is current. The `pyproject.toml` uses `[dependency-groups]` and `[tool.uv] exclude-newer = "2024-08-01T00:00:00Z"`, and the lock file was generated and executed with uv 0.7.8 in October 2026, on macOS arm64 with CPython 3.11.12 that uv installed. That interpreter is a 2025 release, used as the contemporary reproduction interpreter; it was not available in 2024. `exclude-newer` makes uv ignore every file uploaded after 1 August 2024 when resolving transitive dependencies; the newest file in the resulting lock was uploaded on 2024-07-29. I make no claim that these uv features, or that lock format, existed in 2024 — they are how the example is reproduced today.

The example is six files. Download them, keeping the directory layout:

- [`repro_pipeline.py`]({{ '/assets/code/reproducible-mlflow-pipeline/repro_pipeline.py' | relative_url }})
- [`tests/test_repro_pipeline.py`]({{ '/assets/code/reproducible-mlflow-pipeline/tests/test_repro_pipeline.py' | relative_url }})
- [`data/synthetic_regression.csv`]({{ '/assets/code/reproducible-mlflow-pipeline/data/synthetic_regression.csv' | relative_url }})
- [`baseline/approved_baseline.json`]({{ '/assets/code/reproducible-mlflow-pipeline/baseline/approved_baseline.json' | relative_url }})
- [`pyproject.toml`]({{ '/assets/code/reproducible-mlflow-pipeline/pyproject.toml' | relative_url }})
- [`uv.lock`]({{ '/assets/code/reproducible-mlflow-pipeline/uv.lock' | relative_url }})

Keep the files byte for byte, including the LF (`\n`) line endings of the CSV. The data fingerprint is computed over the raw bytes, so a copy converted to CRLF — for example by Git with `core.autocrlf=true` — fails the fingerprint test. If you commit the files to Git, a `.gitattributes` line such as `*.csv -text` keeps the CSV unchanged.

Then, in that directory:

```bash
uv run --locked pytest
```

`--locked` makes uv fail instead of silently re-resolving if the lock file no longer matches `pyproject.toml`. I ran it with uv 0.7.8; a much older uv may not read this lock file.

## Data: materialize once, then trust the file

The data is generated by `make_dataset` with `numpy.random.default_rng(20240801)`: 240 rows, four features (one on a much larger scale, so the scaler has something to do) and a linear target with Gaussian noise. But the seed is not the source of truth. The CSV written by that function is committed, and everything else reads the file.

The reason is in NumPy's own random-number policy (NEP 19): exact bit-for-bit results require the exact NumPy version, and even the same version built on another platform, or built differently, may change the stream. A seed is a recipe; the file is the result. One test checks that regenerating the data in the pinned environment still gives identical bytes. It did on the machine I used — useful to know, but not a promise for other platforms, and if it ever fails, the committed CSV still wins.

The split is stored in the file as a `split` column and decided only by the row id:

```python
            row_id = f"r{i:04d}"
            # Split depends only on the row id, not on the RNG or row order.
            split = "test" if int(sha256_bytes(row_id.encode()), 16) % 5 == 0 else "train"
```

Reordering rows or changing the random generator cannot move a row to the other side. That gives 197 training rows and 43 evaluation rows.

When the file is loaded, `load_dataset` rejects unknown split labels and any non-finite value (`nan`, `inf`, `-inf`), and computes two fingerprints: `data_sha256` over the raw bytes, and `split_sha256` over the row ids assigned to each split. The second one tells me something the first cannot: whether two runs were evaluated on *the same rows*. A metric computed on a different split is not comparable, however close the number looks.

`split_sha256` hashes the *ordered* lists of row ids, so it captures row order as well as membership. If the rows of the file are permuted, the membership is the same but the fingerprint changes, and the gate rejects the run as "not the approved test split". That is deliberately conservative: the reproducibility check compares predictions position by position, and a permuted split would pair each prediction with the wrong approved value.

## Training: the scaler sees training rows only

```python
def select_features(X: np.ndarray, config: TrainConfig) -> np.ndarray:
    # Fancy indexing returns a Fortran-ordered array. With the same values, BLAS may then
    # sum in another order and change the last bits, so always hand the model C order.
    return np.ascontiguousarray(X[:, [FEATURES.index(name) for name in config.features]])


def train(dataset: Dataset, config: TrainConfig) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=config.alpha))])
    # The scaler sees training rows only; Pipeline.fit never touches the test split.
    return model.fit(select_features(dataset.X_train, config), dataset.y_train)
```

Putting the scaler inside the `Pipeline` means its mean and variance are learned in `fit`, from the rows passed to `fit`, and only applied in `predict`. The scikit-learn documentation describes exactly this as the way to avoid leaking statistics from evaluation data into preprocessing. I don't rely on the description alone: a test checks that `model["scaler"].mean_` equals the mean of the training rows and differs from the mean of all rows.

### The comment about Fortran order is not decoration

The first version selected the feature columns with plain fancy indexing, `X[:, indices]`, which here returns a Fortran-ordered array. While building the example I compared the reloaded model's predictions on a C-ordered matrix with the predictions computed during the run: 7 of 43 values differed, by at most 1.78e-15, although the coefficients and scaler statistics were equal. Predicting on a C-ordered and on a Fortran-ordered copy of the same 43 rows reproduced exactly that difference. (The pinned NumPy wheel uses its bundled OpenBLAS 0.3.23.dev on this machine.) 1.78e-15 is the size of one unit in the last place for the largest predictions in this set, values between 8 and 16.

Nothing was wrong with the model. Matrix multiplication can add the same numbers in a different order depending on memory layout, and floating-point addition is not associative. So I made the layout explicit (C order) — and I drew a conclusion for the gate: exact equality is the wrong contract once code, BLAS or CPU can change.

The approved fixture in this example was created before that fix, when both the training and the evaluation matrix were Fortran-ordered. The current code differs from it in 26 of 43 predictions, by at most 2.66e-15. These are not the same numbers as above, and they are not supposed to be: the 7 of 43 came from changing the layout of the *prediction* input only, with one fitted model. Against the fixture, the layout of the *training* input changed too, which moves the last bits of the scaler statistics and the Ridge coefficients, and then the prediction input changes as well. A re-run of both experiments in the correction pass gave the same figures (prediction only: 7 of 43, at most 1.78e-15; fitting and prediction: 26 of 43, at most 2.66e-15). More values differ, by up to about one and a half units in the last place; it is the same kind of last-bit noise, not a numerical inconsistency, and the gate tolerates it.

## Logging the run with MLflow

`run_experiment` loads the data, trains, evaluates and logs everything inside one `mlflow.start_run`:

```python
    mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
    mlflow.set_experiment("article-42-reproducible-pipeline")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"alpha": config.alpha, "features": ",".join(config.features),
                           "data_seed": DATA_SEED, "model": "StandardScaler+Ridge"})
        mlflow.set_tags({"data_sha256": dataset.data_sha256,
                         "split_sha256": dataset.split_sha256,
                         "source_sha256": source_sha256})
        mlflow.log_artifact(str(source), artifact_path="source")
        mlflow.log_artifact(str(data_path), artifact_path="data")
        mlflow.log_dict(environment_snapshot(), "environment/environment.json")
        for lock in lock_files():
            mlflow.log_artifact(str(lock), artifact_path="environment")

        model = train(dataset, config)
        test_predictions = model.predict(X_test)
        naive = np.full_like(dataset.y_test, dataset.y_train.mean())
        metrics = {
            "train_rmse": regression_metrics(dataset.y_train, model.predict(X_train))["rmse"],
            **{f"test_{k}": v for k, v in regression_metrics(dataset.y_test, test_predictions).items()},
            "naive_test_rmse": regression_metrics(dataset.y_test, naive)["rmse"],
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            pip_requirements=[f"{name}=={version(name)}" for name in PINNED])
        model_uri = f"runs:/{run.info.run_id}/model"

        # Reload what MLflow stored and require identical predictions in this environment.
        reloaded = mlflow.sklearn.load_model(model_uri)
        reload_max_abs_diff = float(np.max(np.abs(reloaded.predict(X_test) - test_predictions)))
        mlflow.log_metric("reload_max_abs_diff", reload_max_abs_diff)
        if reload_max_abs_diff != 0.0:
            raise RuntimeError(f"reloaded model differs: {reload_max_abs_diff}")
```

A few decisions worth spelling out:

- **The tracking URI is an explicit local directory** (`file://.../mlruns`). Without configuration MLflow 2.14 also writes to a local `mlruns` directory, but the tests pass a temporary directory so they never mix their runs with yours.
- **Fingerprints are tags, files are artifacts.** The tags make runs searchable and comparable; the artifacts (`source/repro_pipeline.py`, `data/synthetic_regression.csv`, `environment/environment.json`, `environment/uv.lock`) let me see later what actually ran. `environment.json` records the Python version, platform, the installed versions of the pinned packages and the SHA-256 of the lock file. The source and environment fingerprints are there for diagnosis — when a rerun fails, they show what changed — but the gate does not assert them.
- **There is a naive baseline in every run.** `naive_test_rmse` is the error of always predicting the training mean. A model that does not beat it has learned nothing useful, whatever its absolute RMSE.
- **`pip_requirements` is explicit.** MLflow records these requirements next to the model. In a uv environment without pip, MLflow logs a warning that it cannot resolve the installed pip version; it is harmless here, but you will see it in the output.
- **I don't use `mlflow.sklearn.autolog()`.** The MLflow 2.14.3 documentation lists autologging as known to be compatible with scikit-learn up to 1.4.2, and this example uses 1.5.1. Explicit calls also make it obvious what is logged.

The reload check is deliberately strict: inside one process, with one environment, a model that went through MLflow serialization and back must give **exactly** the predictions computed before logging. Here tolerance would only hide a real problem.

It is worth being precise about what the reload proves. The `mlflow.sklearn` flavor stores the estimator with pickle (cloudpickle by default). Loading a pickle can execute arbitrary code, so a model artifact is part of your trust boundary: load only models you produced or whose origin you control. scikit-learn's persistence guide also says that persisted models are expected to be portable assuming the same versions of dependencies and Python are used — which is one more reason to keep the lock file next to the model.

## The approved fixture

The reference the gate compares against is a JSON file, `baseline/approved_baseline.json`. It contains:

- `status: "APPROVED"` — the loader refuses anything else;
- the approved config (`alpha=1.0`, all four features);
- `data_sha256`, `split_sha256`, `n_train=197`, `n_test=43`;
- the approved metrics: test RMSE 0.4510, train RMSE 0.5685, train-mean predictor 2.1661;
- all 43 approved evaluation predictions;
- the tolerances, each with a written reason.

**About the word "approved":** in this example, approval is an *example policy*, not a real organizational sign-off. The fixture's `approval` block records that the status was set by the AI assistant that generated this example (Claude Opus 5.5) — an illustrative approval, explicitly not an independent review — and what was checked: the model clearly beats the train-mean predictor, training RMSE is close to the noise level used to generate the data (0.5), and two consecutive runs gave identical metrics. In a team, this is where a named reviewer and a reviewed commit would go.

New fixtures are only ever *proposed*:

```bash
uv run --locked python repro_pipeline.py propose-baseline --output baseline/proposal.json
```

This trains, logs a run and writes a file with `status: "PENDING_REVIEW"`. It refuses to run without `--output` or when the output file already exists, so it cannot overwrite the approved fixture by accident:

```text
$ uv run --locked python repro_pipeline.py propose-baseline
propose-baseline needs --output pointing to a file that does not exist yet
(exit code 2)
```

I added that guard after writing three tests for it. In the first version of the example, `propose-baseline` wrote to the `--baseline` path, which defaults to the approved fixture: the first new test failed because the command returned 0 and replaced the test's copy of the approved fixture with a pending proposal. Promoting a proposal is a human step done in a separate change, not a side effect of a CLI command.

## The gate: two checks, two kinds of failure

```python
def check_reproducibility(result: RunResult, baseline: dict) -> list:
    """Same data, split and config must give the approved predictions."""
    failures = same_evaluation_data(result, baseline)
    approved = TrainConfig(**baseline["config"])
    if result.config != approved:
        failures.append(f"config {result.config} differs from approved {approved}")
    expected = np.array(baseline["test_predictions"])
    actual = np.asarray(result.test_predictions)
    atol = baseline["tolerances"]["prediction_atol"]
    if actual.shape != expected.shape:
        failures.append(f"predictions shape {actual.shape} != {expected.shape}")
    elif not np.allclose(actual, expected, rtol=0.0, atol=atol):
        worst = float(np.max(np.abs(actual - expected)))
        failures.append(f"test predictions differ by up to {worst:.3g} (atol {atol:g})")
    return failures


def check_quality(result: RunResult, baseline: dict) -> list:
    """Any candidate must not be meaningfully worse on the evaluation fixture.

    The "test" rows are a reusable regression fixture, not an untouched holdout: every
    gate run looks at them. Generalization needs a separate final test set.
    """
    failures = same_evaluation_data(result, baseline)
    rmse, naive = result.metrics["test_rmse"], result.metrics["naive_test_rmse"]
    limit = baseline["metrics"]["test_rmse"] * baseline["tolerances"]["max_test_rmse_ratio"]
    # NaN, infinity or a negative value is a broken run, not a good model.
    for name, value in (("test_rmse", rmse), ("naive_test_rmse", naive)):
        if not (math.isfinite(value) and value >= 0):
            failures.append(f"{name} {value} is not a finite non-negative RMSE")
    # Written as "not (x <= limit)" so that NaN fails instead of slipping through.
    if not rmse <= limit:
        failures.append(f"test_rmse {rmse:.4f} exceeds limit {limit:.4f}")
    if not rmse < naive:
        failures.append("test_rmse does not beat the train-mean predictor")
    return failures
```

Both checks start with `same_evaluation_data`: if the data or split fingerprint differs, the comparison is meaningless and both checks fail.

To be exact about the scope: the gate compares the data and split fingerprints, the config, the predictions and the quality metrics. It does not compare the source-code hash or the lock-file hash with approved values. Those are logged with every run for diagnosis, not enforced; a code change that keeps predictions within tolerance passes.

### Why the prediction tolerance is 1e-9

The approved predictions lie between about 1.6 and 9.5. One float64 unit in the last place at 9.5 is about 1.8e-15, and the layout noise above was one to one and a half of those units (1.78e-15 and 2.66e-15). A tolerance of `1e-9` leaves more than five orders of magnitude for that kind of noise. (The fixture's own `tolerance_reasons` text, written when it was approved, says "about five"; the ratio is 1e-9 / 1.78e-15 ≈ 10^5.7.)

At the same time it is far below a real change. Changing `alpha` from 1.0 to 1.01 — a 1% change of one hyperparameter — moved predictions by up to 2.1e-4. One test checks the boundary directly: adding 1e-12 to every prediction passes the reproducibility check, adding 1e-6 fails it. `rtol=0` keeps the rule simple: the same absolute tolerance for every prediction.

### Why the quality limit is 5%

`max_test_rmse_ratio = 1.05` means a deliberate change may cost at most 5% evaluation RMSE. That is a product decision, not a statistical test. With 43 evaluation rows the gate can catch clear degradation; it cannot distinguish a 1% improvement from noise, and I don't pretend it can. Before any comparison, both RMSE values must be finite and non-negative, so a `NaN`, an infinite or a negative metric fails instead of slipping through. The limit check is also written as `not rmse <= limit`, so a `NaN` fails there too: every comparison with `NaN` is false. Tests cover a `NaN` candidate RMSE, a negative one, and an infinite or `NaN` train-mean RMSE. For the valid runs in this article the extra domain check changes nothing.

### Who decides: one acceptance policy

```python
def accepted(report: GateReport, config: TrainConfig, baseline: dict) -> bool:
    """The acceptance policy, shared by the CLI exit code and the CI test."""
    if config == TrainConfig(**baseline["config"]):
        return report.passed  # a rerun must reproduce and keep quality
    # A new candidate is expected to change predictions; it only has to keep quality.
    return not report.quality_failures
```

A rerun of the approved config must pass both checks. A candidate with a different config is *expected* to fail reproducibility — the report still prints it, as information — and only quality decides.

The CLI returns `0 if accepted(report, config, baseline) else 1`, and the CI test below asserts the same function. The first version of the test checked only `report.quality_failures`. A probe in the correction pass showed what that missed: an approved-config rerun whose predictions were shifted by 1e-6, with the metrics recomputed from the shifted predictions, kept its quality (RMSE 0.45097), so the test passed while the CLI returned 1. Now a parametrized test runs three cases through both paths — that drifted rerun, `alpha=1.01` and `alpha=1000` — and requires the same answer from `accepted` and from the CLI exit code: reject, accept, reject.

## What happened when I ran it

All numbers below are copied from runs on 13 October 2026 in the locked environment described above.

| Command | Reproducibility | Quality | Exit code |
|---|---|---|---|
| `gate` (approved config, `alpha=1.0`) | OK | OK — RMSE 0.45097 | 0 |
| `gate --alpha 1.01` | FAIL — predictions differ by up to 0.000211 | OK — RMSE 0.45096 | 0 |
| `gate --alpha 1000` | FAIL — predictions differ by up to 3.32 | FAIL — RMSE 1.8273 > limit 0.4735 | 1 |

The full output of the degraded candidate:

```text
$ uv run --locked python repro_pipeline.py gate --tracking-dir mlruns --alpha 1000
{
  "run_id": "08a7d7f13f56428dbae52aa7b5672065",
  "metrics": {
    "train_rmse": 2.145788832628335,
    "test_rmse": 1.8272972197594555,
    "test_mae": 1.530716857691987,
    "naive_test_rmse": 2.1660715355372946
  }
}
reproducibility: FAIL
  - config TrainConfig(alpha=1000.0, features=('x0', 'x1', 'x2', 'x3')) differs from approved TrainConfig(alpha=1.0, features=('x0', 'x1', 'x2', 'x3'))
  - test predictions differ by up to 3.32 (atol 1e-09)
quality: FAIL
  - test_rmse 1.8273 exceeds limit 0.4735
```

The middle row is the interesting one. `alpha=1.01` is not reproducible against the approved fixture — it is a different model — yet its quality is fine. If I had a single "metric within tolerance" check, I would not see that the model changed at all. If I had a single "predictions identical" check, I would reject a perfectly acceptable candidate. Two checks give two honest answers.

The rerun of the approved config matched the approved test RMSE except in the last printed digit: 0.45096857685897573 against 0.4509685768589758 in the fixture. The difference comes from the fitting-and-prediction layout change described in the section on Fortran order: the fixture was produced by the older, Fortran-ordered code.

### The test that is supposed to fail

The last test in the suite is the gate a CI job would run. Its candidate config comes from the environment and defaults to the approved one:

```python
def test_candidate_gate(tracking_dir, baseline):
    """Candidate config comes from the environment; defaults to the approved one.

    CANDIDATE_ALPHA=1000 makes this test fail on purpose: that is the gate. It applies the
    same acceptance policy as the CLI, so a drifting rerun of the approved config fails too.
    """
    config = rp.TrainConfig(**baseline["config"])
    if "CANDIDATE_ALPHA" in os.environ:
        config = dataclasses.replace(config, alpha=float(os.environ["CANDIDATE_ALPHA"]))
    result = rp.run_experiment(DATA, tracking_dir, config, run_name="candidate")
    report = rp.gate(result, baseline)
    assert rp.accepted(report, config, baseline), report.summary()
```

```text
$ uv run --locked env CANDIDATE_ALPHA=1000 pytest -q -p no:warnings -k candidate_gate
E       AssertionError: reproducibility: FAIL
E           - config TrainConfig(alpha=1000.0, features=('x0', 'x1', 'x2', 'x3')) differs from approved TrainConfig(alpha=1.0, features=('x0', 'x1', 'x2', 'x3'))
E           - test predictions differ by up to 3.32 (atol 1e-09)
E         quality: FAIL
E           - test_rmse 1.8273 exceeds limit 0.4735
1 failed, 30 deselected
```

With `CANDIDATE_ALPHA=1.01`, or without the variable, the same command reports `1 passed, 30 deselected`. Because the test uses `accepted`, running only `-k candidate_gate` with the approved config still enforces reproducibility, not just quality.

A red test here is the correct outcome. I don't mark it `xfail`: a gate that is expected to fail and reported as "expected failure" stops being a gate.

## The "test" rows are a regression fixture, not a holdout

The CSV calls the 43 evaluation rows `test`, and the metrics are named `test_rmse`. I kept those names because changing them would change the committed data and the approved fixture. But the right name for these rows is a **reusable regression/evaluation fixture**.

Every gate run looks at them. Every candidate is scored on them. If I tried ten values of `alpha` and kept the one that passed the gate with the best `test_rmse`, I would be tuning on these rows, and the number would become optimistic. That is why the example never selects hyperparameters with the gate, and why the approved fixture records that the 43 rows were not used for choosing `alpha` or features before approval. After approval they are reused on purpose, which is fine for a regression check and wrong for an estimate of generalization.

So the gate answers "did this candidate get worse on a fixed, known set of rows?", not "how well will this model do on new data?". For the second question you need a **separate, untouched final test set**: data that is not in this CSV, not seen by the gate, and evaluated once, when you are done choosing. Model selection belongs to cross-validation or a validation split inside the training data.

## What the tests cover

`uv run --locked pytest` runs 31 tests:

- **Data and split:** the committed CSV matches the approved fingerprints and sizes; regenerating it in the pinned environment gives identical bytes; non-finite values (`nan`, `inf`, `-inf`) and unknown split labels are rejected.
- **Leakage and layout:** scaler statistics come from training rows only; `select_features` returns C-ordered columns in config order; memory-layout noise stays inside the prediction tolerance.
- **Prediction input:** the reloaded model rejects `NaN` at prediction time with scikit-learn's own `ValueError`.
- **MLflow metadata:** params, metrics, fingerprints and the source, data, environment (`environment.json`, `uv.lock`) and model artifacts are in the run; the reloaded model reproduces the in-run predictions exactly when given the run's own prepared input, `select_features(X_test, config)`, for the approved config and for a reordered and a reduced feature list.
- **Fixture handling:** the fixture is `APPROVED` and every tolerance has a reason; `propose-baseline` never overwrites the approved fixture, refuses an existing output file and writes a `PENDING_REVIEW` proposal to a new path.
- **Gate behaviour:** an approved rerun passes both checks; 1e-12 noise passes and a 1e-6 shift fails reproducibility; `alpha=1.01` fails reproducibility but not quality; `alpha=1000` fails quality; `NaN`, negative or infinite RMSE values never pass; a different split fingerprint makes the run not comparable; the CLI exit code and the CI test agree on the drifted rerun, `alpha=1.01` and `alpha=1000`.

Some tests call the CLI, which points MLflow's global tracking URI at their own temporary store. An autouse fixture resets the URI to the session store before every test, so the session-scoped approved run is found whatever the order. I checked that by running the order a reviewer suggested (approved run, proposal test, then the metadata and reload tests), which failed with "Run … not found" before the fixture and passes now, and by running the whole file in reverse order.

The code was written test-first. I kept the logs of the red runs: the suite before the module existed, the layout test before `select_features` existed, the three proposal-guard tests before the guard, and, in a correction pass after review, the shared-policy, metric-domain, feature-order reload and test-order cases before the fixes.

`repro_pipeline.py` is 346 lines. I kept it in one file on purpose: it is a single download, the tests import one module, and the file is split into four labelled sections — data, training and evaluation, approved baseline and gate, command line — that follow the order of this article. Splitting it into a package would add import plumbing without teaching anything new.

## Limitations

- **Byte equality is not portable.** A seed does not guarantee the same numbers across NumPy versions, and the same code can produce different last bits on another BLAS, CPU or memory layout. The gate uses a tolerance; the committed CSV, not the seed, is the data.
- **Retention is your job.** The fingerprints are only useful if the data, code and lock file they point to still exist. MLflow artifacts in a local directory are not an archive; decide where data, code and environment are kept and for how long.
- **Pickle is a trust boundary.** Loading an MLflow scikit-learn model unpickles it. Treat model artifacts like code.
- **No production claims.** There is no tracking server, authentication, model registry, serving or monitoring here. The example shows how to record a run and gate a candidate, not how to operate a model.
- **The evaluation rows are reused.** As above: the gate measures regression on a fixed set, not generalization. No repeated tuning against it.
- **43 rows is small.** The 5% quality limit catches clear degradation only.
- **MLflow records; it does not enforce.** Nothing in MLflow stops a worse model. The decision is in pytest and in the CLI exit code, and whoever wires that exit code into CI owns it.

## Where I start

Not with the tracking server. I start by writing down what "the same run" means — data bytes, split, code, lock file, config — and which tolerance separates noise from change. Then I log exactly those things, and I write the test that fails when the ones I enforce — here data, split, config, predictions and quality — don't hold; the code and lock-file hashes stay in the run to explain a failure. MLflow makes the evidence easy to find; the test is what turns it into a decision.

### Sources

- [MLflow 2.14.3 release](https://github.com/mlflow/mlflow/releases/tag/v2.14.3) — release published 12 July 2024.
- [MLflow 2.14.3: Tracking](https://mlflow.org/docs/2.14.3/tracking.html) — runs, parameters, metrics, artifacts; default local `mlruns` directory.
- [MLflow 2.14.3: `mlflow` Python API](https://mlflow.org/docs/2.14.3/python_api/mlflow.html) — `set_tracking_uri`, `log_params`, `log_metrics`, `set_tags`, `log_artifact`, `log_dict`.
- [MLflow 2.14.3: `mlflow.sklearn`](https://mlflow.org/docs/2.14.3/python_api/mlflow.sklearn.html) — `log_model`, `load_model`, pickle/cloudpickle formats, autologging compatibility range.
- [MLflow 2.14.3: `mlflow.models`](https://mlflow.org/docs/2.14.3/python_api/mlflow.models.html) — `infer_signature`.
- [scikit-learn 1.5.1: Pipelines and composite estimators](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/doc/modules/compose.rst) — pipelines avoid leaking statistics from test data.
- [scikit-learn 1.5.1: Common pitfalls](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/doc/common_pitfalls.rst) — never include test data in `fit`.
- [scikit-learn 1.5.1: Model persistence](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/doc/model_persistence.rst) — trust in the model source; same dependency and Python versions.
- [Python 3.11: `pickle`](https://docs.python.org/3.11/library/pickle.html) — "The pickle module is not secure."
- [NumPy NEP 19: Random number generator policy](https://numpy.org/neps/nep-0019-rng-policy.html) — exact results require the exact NumPy version.
- [uv: Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) — `--locked` (current documentation, not a 2024 source).
- PyPI JSON API for [mlflow 2.14.3](https://pypi.org/pypi/mlflow/2.14.3/json), [scikit-learn 1.5.1](https://pypi.org/pypi/scikit-learn/1.5.1/json), [numpy 1.26.4](https://pypi.org/pypi/numpy/1.26.4/json), [pytest 8.3.2](https://pypi.org/pypi/pytest/8.3.2/json), [setuptools 72.1.0](https://pypi.org/pypi/setuptools/72.1.0/json) — upload dates.
