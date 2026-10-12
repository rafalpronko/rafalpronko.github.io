---
layout: post
lang: en
title: "Tuning with Optuna without tuning on the test set"
date: 2023-12-07 09:00:00 +0100
permalink: /blog/optuna-without-test-leakage/
author: Rafał Prońko
categories: [machine-learning]
description: "Hyperparameter search with Optuna and scikit-learn that never looks at the test set: fixed folds, a Pipeline, a baseline rule written in advance and one final evaluation."
image: /assets/images/optuna-without-test-leakage.svg
---

# Tuning with Optuna without tuning on the test set

![Diagram of a hyperparameter search that never looks at the test set: split once, search with Optuna on three fixed folds of development data, compare with a baseline using a rule written in advance, refit, then evaluate once on the locked test set.]({{ '/assets/images/optuna-without-test-leakage.svg' | relative_url }})

Adding Optuna to a training script takes a few lines. You write an objective, call `study.optimize`, read `study.best_params`. The trouble starts with one question: what does the objective return?

If it returns a score on the test set, the search is tuning on the test set. It doesn't matter that no model was ever fitted on those rows. A hundred trials that each look at the test score and keep the best one are a hundred small decisions made with test data.

In this article I build a small, complete tuning setup where the test set is used exactly once, at the end. The model is deliberately boring — a `StandardScaler` and a `LogisticRegression` — because I want to focus on the decisions around the search, not on the model.

## Why a test-guided search leaks

A test set has one job: to estimate how the final model will do on data that played no part in building it.

Every hyperparameter search is a selection process. Optuna proposes settings, measures each one and steers towards the higher scores. If that measurement uses the test set, the settings that win are the ones that happen to fit those particular test rows well — partly because they are genuinely better, partly by chance.

The scikit-learn documentation describes this directly: when settings are tweaked until the estimator performs optimally, “knowledge about the test set can ‘leak’ into the model and evaluation metrics no longer report on generalization performance”.

The leak does not need an automated loop. Looking at the test score, changing the search space and running again does the same thing by hand. So the rule I follow is simple: **nothing that comes out of the test set may change any decision about the model.**

## Two kinds of parameters

It helps to keep two layers apart.

**Learned parameters** are fitted by `fit`: the logistic regression coefficients and intercept, and the means and standard deviations stored by `StandardScaler`. They come from the training rows of whatever split is being used.

**Hyperparameters** are choices made *before* `fit`: the inverse regularisation strength `C` (smaller values mean a stronger penalty), the penalty type, the elastic-net mixing ratio. `fit` does not touch them. Optuna's objective is the function that scores them.

Both layers can leak. Scaling statistics computed on all rows before splitting leak into every fold. A search that scores hyperparameters on the test set leaks into the choice of the model. The example below deals with both.

## Decide before the first trial

Before any score exists I fix the things that a search otherwise tempts me to adjust along the way:

- **The data split.** 1000 synthetic rows, 750 for development, 250 for test, stratified by class.
- **The validation scheme.** Three stratified folds of the development rows, created once and reused by every trial and by the baseline.
- **The metric and its direction.** Mean ROC AUC across the folds, maximised.
- **The search space.** Penalty L2, L1 or elastic net; `C` between 0.001 and 100 on a log scale; the elastic-net ratio only when the penalty is elastic net.
- **The budget.** 16 sequential trials, the first 6 sampled at random.
- **The baseline and the selection rule.** The baseline is `C=1` with an L2 penalty. The tuned parameters replace it only if their mean CV score is at least 0.005 higher.

Writing these down as constants is not bureaucracy. It's the difference between “I ran a search” and “I ran searches until I liked one”. If I later want a larger budget or a different space, that's a new experiment, and it still never looks at the test set.

## The example

I use Python 3.11, Optuna 3.4.0, scikit-learn 1.3.2, NumPy 1.26.2 and SciPy 1.11.4. The data is synthetic, generated with `make_classification`: 20 features, of which 5 are informative and 5 are redundant combinations of them, about 20% positive class, and each label randomly reassigned with probability 0.03 (`flip_y=0.03`), so some reassigned labels keep their original class.

The complete example — `tuning.py`, the tests, `pyproject.toml`, a lock file and a README — can be downloaded as [one zip archive]({{ '/assets/examples/optuna-without-test-leakage/optuna-without-test-leakage.zip' | relative_url }}). The [README]({{ '/assets/examples/optuna-without-test-leakage/README.txt' | relative_url }}) lists the files and the commands. Everything below is quoted from [`tuning.py`]({{ '/assets/examples/optuna-without-test-leakage/tuning.py' | relative_url }}); the blocks are fragments of that one file.

The decisions from the previous section live at the top of the module:

```python
# Decisions fixed before any score is computed.
SEED = 7
N_TRIALS = 16
N_STARTUP_TRIALS = 6
N_FOLDS = 3
METRIC = "roc_auc"
MIN_GAIN = 0.005  # tuned params must beat the baseline mean CV score by this much
BASELINE = {"penalty": "l2", "C": 1.0}
```

### Split once, at the start

The test rows are set aside before anything else happens:

```python
def split_holdout(y):
    rows = np.arange(len(y))
    return train_test_split(rows, test_size=0.25, stratify=y, random_state=SEED)
```

The function returns row indices rather than arrays. That makes it easy to check in a test that the two parts are disjoint and together cover every row.

### Fixed, stratified folds on development data

```python
def make_folds(y_dev):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y_dev)), y_dev))
```

The folds are materialised into a list once. Every trial and the baseline are scored on exactly the same three partitions. That way, a difference between two trials comes from their hyperparameters, not from one of them drawing a luckier split.

Stratification keeps the share of positives roughly the same in each fold. With only about 150 positives among the development rows, an unlucky fold could otherwise contain noticeably fewer of them than the others.

### Scaling inside the fold

```python
def build_model(params):
    clf = LogisticRegression(solver="saga", max_iter=5000, random_state=SEED, **params)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])
```

Because the scaler is a step of the pipeline, `cross_val_score` fits it on the training part of each fold and only applies it to the validation part. Had I scaled all development rows first and then split them, every validation fold would have contributed to the means and standard deviations used to transform it.

For standardisation on 750 well-behaved synthetic rows the effect is probably small. For target encoding, feature selection or imputation it can be large. A pipeline removes the question instead of making me estimate the effect each time.

I use the `saga` solver because it supports all three penalties, including elastic net. `random_state` fixes the data shuffling `saga` does during fitting.

### The search space

```python
def suggest_params(trial):
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    params = {"penalty": penalty, "C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    return params
```

Two details matter here.

`C` is sampled on a log scale. Its useful values span several orders of magnitude, and the difference between 0.001 and 0.01 matters as much as the difference between 10 and 100. With a linear scale, about 90% of the trials would land between 10 and 100.

`l1_ratio` exists only when the penalty is elastic net. Optuna builds the search space while the objective runs, so a conditional parameter is just an `if`. I don't have to invent a dummy value for trials where the parameter means nothing.

The dictionary keys match `LogisticRegression` arguments, so the same dictionary goes into the pipeline, into `study.best_params` and into the selection rule.

### An objective with explicit development inputs

```python
def cv_score(params, X_dev, y_dev, folds):
    scores = cross_val_score(build_model(params), X_dev, y_dev, cv=folds, scoring=METRIC)
    return float(np.mean(scores))


def make_objective(X_dev, y_dev, folds):
    def objective(trial):
        return cv_score(suggest_params(trial), X_dev, y_dev, folds)
    return objective
```

The objective is built from development data and folds only. It doesn't receive the full dataset with a promise not to touch the test rows; its inputs say explicitly which data it works on. That isn't a security boundary — Python code can always call `make_data()` again or reach a global — but it makes the intended data flow visible in the signature and easy to check in a test, which a comment saying “don't use the test rows” is not.

### Baseline, search and the selection rule

```python
def select_final(baseline_cv, best_cv, best_params):
    return dict(best_params) if best_cv >= baseline_cv + MIN_GAIN else dict(BASELINE)


def tune(X_dev, y_dev):
    folds = make_folds(y_dev)
    baseline_cv = cv_score(BASELINE, X_dev, y_dev, folds)
    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_dev, y_dev, folds), n_trials=N_TRIALS)
    return study, baseline_cv, select_final(baseline_cv, study.best_value, study.best_params)
```

`direction="maximize"` is not a detail. Without a direction, `create_study` minimises, and a search for the *lowest* ROC AUC runs without any error.

`TPESampler` uses random sampling until `n_startup_trials` trials have finished; only then does it start modelling which regions look promising. The default in Optuna 3.4.0 is 10. With a budget of 16 that would leave only 6 guided trials, so I set 6 random trials and 10 guided ones. That's a judgement for this budget, not a universal setting.

The baseline is scored on the same folds as every trial. `select_final` then decides between the two using only CV scores. The margin protects against switching for a small, uninteresting difference. The value 0.005 is a pragmatic judgement for this example, fixed before the search; it isn't derived from the spread of the fold scores and it isn't a statistical guarantee that a larger gain is real. The best of 16 trials is the maximum of 16 noisy estimates, so it's expected to look a bit better than it is.

### Refit, then look at the test set once

```python
def run():
    X, y = make_data()
    dev_idx, test_idx = split_holdout(y)
    study, baseline_cv, chosen = tune(X[dev_idx], y[dev_idx])
    # Selection is finished; the test rows are used once, for reporting only.
    final = build_model(chosen).fit(X[dev_idx], y[dev_idx])
    test_auc = roc_auc_score(y[test_idx], final.predict_proba(X[test_idx])[:, 1])
```

After the selection, I refit the chosen configuration on all 750 development rows. During CV each model saw only two thirds of them, so the final model gets more data than any model in the search did. That's the usual practice, and it means the final model was never scored by CV in exactly this form. The test set is what measures it.

The function goes on to return the scores and the study. The script prints them.

## What the run produced

The post's date, 7 December 2023, is an editorial label; the library versions were chosen to match it. The code itself was actually run on 12 October 2026, with the locked environment from the example (CPython 3.11.12 on macOS arm64), using `uv run --locked python tuning.py`. uv is the tool used for that run and is not part of the article's 2023 setting. The output, shortened to the summary lines:

```text
baseline cv_auc=0.8654
best     cv_auc=0.8689  {'penalty': 'l1', 'C': 0.2001368402568556}
chosen   {'penalty': 'l2', 'C': 1.0}
held-out test roc_auc=0.8949 (reported once, not used to choose)
```

The best trial, an L1 penalty with `C≈0.2`, beat the baseline by about 0.0035 in mean CV ROC AUC. That's below the 0.005 margin, so the rule kept the baseline. On this data the search did not find anything worth changing the model for.

That isn't a failure of the search. It's an answer to the question the search was asking. I'd rather get that answer from the development data than discover, after five rounds of “just one more run”, that a configuration I'd been chasing was within noise.

Two trials scored exactly 0.5000: L1 at `C≈0.001` and elastic net at `C≈0.0046`. I checked the models fitted inside each of the three CV folds (500 training rows each). For both trials the penalty removed every coefficient in every fold, so all validation rows got the same score, producing a ROC AUC of 0.5. (The converse doesn't hold: an AUC of 0.5 doesn't by itself mean constant scores.) Refitted on all 750 development rows, the elastic-net configuration keeps one non-zero coefficient, so the CV score describes the fold models, not that refit. Such points are legitimate parts of a log-scale space; they also show why the first random trials are useful — they map the edges of the space.

The test ROC AUC is higher than the CV scores. I don't read anything into that. The test set has 250 rows and only 51 positives, so a single estimate from it moves noticeably with the particular rows that ended up there. More importantly, if the test score had been *lower*, I would not go back and change the search. Its role is to report, not to steer.

These numbers describe one synthetic dataset and one run. They don't say that logistic regression, L1, L2 or Optuna is better or worse in general.

## What the tests check

The example has twelve pytest tests. They check that:

1. The holdout split is disjoint, covers all 1000 rows and keeps the class balance.
2. The folds partition the development rows and are identical when regenerated.
3. In each fold the fitted scaler's means equal the means of that fold's training rows, not of all development rows.
4. Suggested parameters, including the conditional `l1_ratio`, reach the `LogisticRegression` step.
5. `cv_score` equals the mean ROC AUC computed manually over the same folds.
6. `make_objective` and `tune` accept only development data, by signature.
7. During a full search, every array passed to cross-validation has the development size and contains no test row.
8. The study maximises, uses a `TPESampler` that received the declared 6 startup trials, and completes exactly the declared 16 trials.
9. The selection rule switches exactly at the margin.
10. The chosen parameters follow from the CV results via that rule.
11. `run()` passes exactly the development rows to the search and to the final fit, exactly the test rows to `predict_proba`, and exactly the test labels with those predictions to `roc_auc_score`.
12. A full run calls `roc_auc_score` directly exactly once, on the 250 test rows.

The recorded `uv run --locked pytest -v -p no:cacheprovider` run on 12 October 2026 passed all twelve tests. The tests don't prove that the chosen hyperparameters are good, and they can't catch every possible use of the test rows — for example code that bypasses these functions. They protect the structure: which rows each step of `run()` receives, which number is maximised, and how many times the test metric is computed.

## Where the guarantees end

### The best CV score is optimistic

`study.best_value` is the maximum over all trials. Each trial's score includes noise from the particular folds, and the maximum favours trials whose noise happened to be positive. So the winning CV score is a biased estimate of how the winning configuration will perform.

The holdout test solves this for the final number: it was not used for the choice. It doesn't fix the choice itself — with a different random split or more trials, a different configuration might win. That's why I compare against a baseline with a margin rather than taking the top trial at face value.

### Nested cross-validation

A single 750/250 split gives one test estimate, and on small datasets that estimate can be noisy. Nested cross-validation is the alternative: an outer loop of splits, and inside each outer training part a complete search with its own inner folds. The outer scores estimate how the *whole procedure* — search included — generalises.

The scikit-learn documentation shows the bias it removes: choosing parameters that maximise a non-nested CV score “biases the model to the dataset, yielding an overly-optimistic score”.

Nested CV costs a full search per outer fold, and it evaluates a procedure rather than one final model. For a small, cheap model like this one it's affordable. For a large model I'd usually settle for a single holdout and accept a wider uncertainty.

### Groups and time

Use independent rows for this evaluation scheme — a random holdout and shuffled folds. If several rows come from the same customer, patient or device, a random split puts related rows on both sides and the CV score rewards memorising them. `GroupKFold` (or `StratifiedGroupKFold`) keeps each group on one side, and the initial holdout must respect the same groups.

If the data has a time order and the model will predict the future, the folds and the holdout must respect that order too — for example `TimeSeriesSplit` for the search and the most recent period as the test set. Shuffled folds on such data measure something the model will never be asked to do.

The pattern from this article stays the same; only the splitter changes. But it has to change in both places, in the folds and in the holdout.

### Threshold and calibration are separate decisions

ROC AUC doesn't depend on a decision threshold, which is one reason I use it here. A real system usually needs a threshold, and sometimes calibrated probabilities.

Both are fitted decisions. If I choose the threshold that looks best on the test set, I've tuned on it again. The threshold belongs on development data — for example on out-of-fold predictions — and must be fixed before the test evaluation. The same applies to calibration: scikit-learn's documentation points out that the data used to fit the calibrator must be disjoint from the data used to fit the classifier.

I keep them as separate, explicit steps rather than hiding them inside the objective. That makes it obvious which data each one used.

### Pruning

Optuna can stop unpromising trials early. To use it, the objective calls `trial.report` with intermediate values and `trial.should_prune` to ask whether to stop.

I don't use pruning here, for three reasons. Each trial takes three short fits, so there's little to save. With cross-validation, the natural intermediate value is the score after one or two folds — a noisier number than the full mean, and dependent on fold order. And a pruned trial's last reported value was measured on fewer folds than a complete trial's score, so the two are not directly comparable. Pruning pays off when a single trial is long and iterative, such as training a neural network for many epochs. Even then it decides which trials to *finish*; it doesn't change which data the decision uses.

### Seeds are not full determinism

`TPESampler(seed=SEED)` makes the sequence of suggestions repeatable, as long as the objective returns the same values. The Optuna FAQ names the limits: parallel or distributed optimisation is inherently non-deterministic, and a non-deterministic objective can't be reproduced by fixing the sampler seed.

In this example the split, the folds and `saga` all receive a fixed seed, and the trials run sequentially. Two consecutive runs on the same machine produced identical trial parameters, identical CV scores and an identical test score. That's evidence for that machine and those builds, not a promise for another operating system, another BLAS library or another version of any of the packages.

### Storage, resuming and appending trials

The study here lives in memory and disappears when the script ends. Optuna can store studies in a database, for example SQLite, and `create_study(..., load_if_exists=True)` continues an existing study.

That's useful for long searches that may be interrupted. It's also an easy way to mix experiments. If I change the search space, the data, the folds or the metric and then resume the same study name, the study now contains trials scored under different conditions. `best_value` compares them as if they were alike. The same goes for running the study “a bit longer” after seeing the test score: the budget was a decision, and extending it because of the test result is a test-guided decision.

So I treat a stored study as belonging to one fixed configuration. A changed configuration gets a new study name, and I record the configuration next to it. The Optuna documentation adds a detail worth knowing: storage doesn't save the sampler's state, so resuming a seeded study reproducibly requires saving and restoring the sampler separately.

## Summary

Optuna does not decide what to optimise. The objective does.

If I take one thing from this example, it's the order of operations: split first, fix the search and the selection rule, run the search on development data, compare with a baseline on the same folds, refit, and only then look at the test set — once.

In this run the result was “keep the baseline”. I'd want any tuning setup to be able to give that answer.

### Sources

- [scikit-learn 1.3: Cross-validation](https://scikit-learn.org/1.3/modules/cross_validation.html) — overfitting on the test set when tuning, holding out a test set with CV, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, preprocessing inside folds.
- [scikit-learn 1.3: Common pitfalls](https://scikit-learn.org/1.3/common_pitfalls.html) — data leakage during preprocessing and `Pipeline`.
- [scikit-learn 1.3: Nested versus non-nested cross-validation](https://scikit-learn.org/1.3/auto_examples/model_selection/plot_nested_cross_validation_iris.html) — optimism of non-nested CV.
- [scikit-learn 1.3: LogisticRegression](https://scikit-learn.org/1.3/modules/generated/sklearn.linear_model.LogisticRegression.html) — penalties, solver compatibility, `C`, `l1_ratio`.
- [scikit-learn 1.3: cross_val_score](https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.cross_val_score.html) — `cv` as an iterable of train/test index splits.
- [scikit-learn 1.3: Probability calibration](https://scikit-learn.org/1.3/modules/calibration.html) — disjoint data for the classifier and the calibrator.
- [Optuna v3.4.0: `TPESampler` source](https://raw.githubusercontent.com/optuna/optuna/v3.4.0/optuna/samplers/_tpe/sampler.py) — `n_startup_trials` (default 10) and `seed`.
- [Optuna v3.4.0: `create_study` source](https://raw.githubusercontent.com/optuna/optuna/v3.4.0/optuna/study/study.py) — `direction` defaults to minimise; `load_if_exists`.
- [Optuna v3.4.0: FAQ](https://raw.githubusercontent.com/optuna/optuna/v3.4.0/docs/source/faq.rst) — reproducibility with a seeded sampler and its limits.
- [Optuna v3.4.0 tutorial: efficient optimisation algorithms](https://raw.githubusercontent.com/optuna/optuna/v3.4.0/tutorial/10_key_features/003_efficient_optimization_algorithms.py) — pruning with `report` and `should_prune`.
- [Optuna v3.4.0 tutorial: RDB storage](https://raw.githubusercontent.com/optuna/optuna/v3.4.0/tutorial/20_recipes/001_rdb.py) — saving and resuming studies; sampler state is not stored.
