---
layout: post
lang: en
title: "Train, validation and test: how group leakage inflates your score"
date: 2023-11-16 09:00:00 +0100
permalink: /blog/train-validation-test-group-leakage/
author: Rafał Prońko
categories: [machine-learning]
description: "A random row split puts the same entities in train and test. A small synthetic experiment with scikit-learn shows how much that inflates a score, and how a grouped three-way split fixes it."
image: /assets/images/train-validation-test-group-leakage.svg
---

# Train, validation and test: how group leakage inflates your score

![Diagram comparing a row split, where rows of the same entities appear in train, validation and test, with a group split, where every entity sits in exactly one role and the test set is locked.]({{ '/assets/images/train-validation-test-group-leakage.svg' | relative_url }})

<time datetime="2023-11-16">16 November 2023</time>

A random split of a table into train, validation and test looks like the most neutral decision in a project. Every row has the same chance, the proportions add up, and the code is one line.

The trouble is that a row is rarely the unit of independence. If the same patient, customer or device has rows on both sides of the split, the model is partly scored on recognising someone it has already seen.

In this post I build a small synthetic example in which that recognition is the *only* thing a model can learn. Then I evaluate the same classifier twice: once with a row split, once with a group split. The difference between the two scores shows what leakage does in this setup. It is a demonstration, not an exact measurement of "the leakage effect": the two splits score different test rows, validation may pick a different `k` for each, and a single run with twelve test groups is noisy.

## First the question, then the split

Before choosing a splitting method I write down what the evaluation is supposed to estimate. In practice it is usually one of two questions:

- **New entities.** Will the model work for patients, customers or devices it has never seen? A new hospital, a new user signing up next week.
- **Known entities, later.** Will the model work on future rows of entities that are already in the training data? The next transaction of an existing customer, the next reading of a sensor that has been running for a year.

Both are legitimate. They need different splits.

For new entities every group has to sit entirely in one role. For future rows of known entities it is *correct* for the same group to appear in training and in test — that is exactly the deployment situation — but now time becomes the boundary: the test rows have to come after the training rows, and the features have to be computed as they would have been at prediction time.

So "the same group on both sides" is not leakage by definition. It becomes leakage when the evaluation answers the known-entities question while the product asks the new-entities one. A row split silently answers the known-entities question, and because it ignores time, it answers even that one only loosely: test rows can be older than training rows.

Dropping the ID column does not fix it. A model does not need an explicit identifier if a combination of other features works as a fingerprint: a patient's age and chronic conditions, a sensor's noise signature, near-duplicate versions of one document.

## Three roles, three different jobs

Each role answers a different question:

- **Train** fits the model's parameters — and the parameters of every preprocessing step: imputer, scaler, feature selection, vocabulary.
- **Validation** is where I make choices: model family, hyperparameters, decision threshold. Those choices are fitted to the validation set, so its score ends up slightly optimistic.
- **Test** is locked until all choices are made and then used once, to score decisions that are already final. If I go back and tune after seeing the test score, the test set has become a second validation set and I no longer have an unbiased estimate.

The scikit-learn guide puts it directly: test data should never be used to make choices about the model, and the general rule is to never call `fit` on the test data.

For the new-entities question all three roles need disjoint groups. Disjointness only between train and test is not enough: if the validation set shares groups with train, model selection rewards memorisation, and I pick the wrong model before the test set even gets a say.

## A synthetic experiment where only memorisation can score

This is a small synthetic demonstration, **not a benchmark**. No real data is involved, and the numbers say nothing about any real model or dataset.

The data generator does three things:

1. It creates 60 groups (entities) with 3–15 rows each.
2. Each group gets a random 8-dimensional "fingerprint" — a centre point. Its rows are that centre plus a little noise (standard deviation 0.3 against a spread of 1.0 between centres), so rows of one group cluster tightly.
3. Each group gets a label drawn **independently of its features**: positive with probability 0.3. All rows of a group share that label.

Because the label is independent of the features, nothing about the features predicts the label of a group the model has not seen. On new groups, the expected balanced accuracy of any classifier is 0.5. A single run can still land well above or below it by luck: twelve test groups are a small sample. Under a row split, scores near 1.0 in every one of twenty runs are not plausibly luck; in this design the explanation is that the model recognises groups it saw in training.

The classifier is a `StandardScaler` followed by k-nearest neighbours. KNN makes the mechanism easy to see: with `k=1` the prediction for a row is the label of its single closest training row, which, under a row split, is almost always a row of the same group.

Both splits use the same proportions:

- **Row split:** `train_test_split` twice over row indices — 60% train, then the remaining 40% halved into validation and test.
- **Group split:** `GroupShuffleSplit` first locks 20% of the groups as test. A second `GroupShuffleSplit`, run only on the remaining development rows, takes 25% of those groups as validation. That gives 36 train, 12 validation and 12 test groups.

In scikit-learn 1.3 `test_size` in `GroupShuffleSplit` refers to groups, not rows, and a float is rounded up. That is why the role sizes in rows are uneven: the test set gets whichever 12 groups were drawn, large or small.

The validation step is real but deliberately small: for `k` in 1, 5 and 15 the pipeline is fitted on train and scored on validation by balanced accuracy, and the best `k` wins (ties go to the smaller `k`). The chosen model is then fitted on train and scored once on test. The test set plays no part in the choice. A `DummyClassifier(strategy="most_frequent")` fitted on train gives the baseline. Balanced accuracy for two classes is the mean of the recall on positives and the recall on negatives, so it needs both classes in the scored set. The script checks that before every score: a validation set with one class stops the run, a test set with one class is reported as undefined (more on both below).

## The code

The whole example is a single file, `group_leakage.py`, with a pytest file next to it. It needs Python 3.11, numpy 1.26.1 and scikit-learn 1.3.2 (with scipy 1.11.3, joblib 1.3.2 and threadpoolctl 3.2.0 underneath); the tests need pytest 7.4.3.

**Download:** [article-05.zip]({{ '/assets/examples/article-05.zip' | relative_url }}) contains the script, the tests, `pyproject.toml`, `uv.lock`, `.python-version`, a README with the commands and a `SHA256SUMS` file. Unpack it and build a clean environment from the lock file with [uv](https://docs.astral.sh/uv/):

```sh
cd article-05
uv sync --frozen
uv run --frozen python group_leakage.py
uv run --frozen pytest -v
```

`--frozen` installs exactly the versions and hashes recorded in `uv.lock` and refuses to resolve new ones. The README also gives a plain `pip` alternative that pins the same direct dependencies and pytest's dependencies, without hash checking; that path was not executed for this article.

```python
"""Synthetic demo: how a row split leaks groups and inflates a test score.

Not a benchmark. Every group gets a feature "fingerprint" and a label drawn
independently of its features, so the features carry no information about
the label of a group the model has not seen. A row split can still score
high, because the model recognises groups it met during training.

Binary balanced accuracy needs both classes in the scored labels. Policy:
a validation set with one class stops the run (no tuning on an undefined
score); a test set with one class keeps its seed, its score is reported
as undefined, and the summary states how many splits it covers.

Run: uv run --frozen python group_leakage.py
"""
from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]
Parts = dict[str, IntArray]  # role -> row indices

K_CANDIDATES = (1, 5, 15)
ROLES = ("train", "validation", "test")


class SingleClassError(ValueError):
    """Binary balanced accuracy is undefined: y_true holds one class."""


class Summary(NamedTuple):
    n_splits: int
    undefined: list[int]  # positions (seeds) with an undefined score
    mean: float | None  # mean/low/high: only over the defined scores
    low: float | None
    high: float | None


def make_data(seed: int, n_groups: int = 60, n_features: int = 8,
              noise: float = 0.3, pos_rate: float = 0.3,
              ) -> tuple[FloatArray, IntArray, IntArray]:
    """Rows of one group = shared fingerprint + small noise, shared label."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 16, size=n_groups)  # 3..15 rows per group
    centers = rng.normal(size=(n_groups, n_features))
    group_label = (rng.random(n_groups) < pos_rate).astype(int)
    groups = np.repeat(np.arange(n_groups), sizes)
    X = centers[groups] + noise * rng.normal(size=(len(groups), n_features))
    return X, group_label[groups], groups


def row_split(n_rows: int, seed: int) -> Parts:
    """60/20/20 split of row indices, ignoring groups."""
    idx = np.arange(n_rows)
    train, rest = train_test_split(idx, test_size=0.4, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return {"train": train, "validation": val, "test": test}


def group_split(groups: IntArray, seed: int) -> Parts:
    """Lock 20% of groups as test, then take validation from the rest."""
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    dev, test = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, va = next(inner.split(dev, groups=groups[dev]))
    return {"train": dev[tr], "validation": dev[va], "test": test}


def overlap(groups: IntArray, seen: IntArray,
            scored: IntArray) -> tuple[int, int]:
    """(groups in both index sets, rows of `scored` from a shared group)."""
    shared = np.intersect1d(groups[seen], groups[scored])
    return len(shared), int(np.isin(groups[scored], shared).sum())


def make_model(k: int) -> Pipeline:
    # The scaler is part of the pipeline, so it is fitted on train rows only.
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def binary_balanced_accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    """Mean recall of classes 0 and 1; both must occur in y_true."""
    present = np.unique(y_true)
    if len(present) != 2:
        raise SingleClassError(
            f"binary balanced accuracy needs both classes, got {present}")
    return float(balanced_accuracy_score(y_true, y_pred))


def select_k(X: FloatArray, y: IntArray,
             parts: Parts) -> tuple[int, dict[int, float]]:
    """Fit each candidate on train, score on validation; never sees test."""
    tr, va = parts["train"], parts["validation"]
    scores = {}
    for k in K_CANDIDATES:
        model = make_model(k).fit(X[tr], y[tr])
        scores[k] = binary_balanced_accuracy(y[va], model.predict(X[va]))
    best = max(K_CANDIDATES, key=lambda k: scores[k])  # ties: smallest k
    return best, scores


def group_mean_accuracy(y_true: IntArray, y_pred: IntArray,
                        groups: IntArray) -> float:
    """Accuracy per group, then the plain mean; each entity counts once."""
    hits = y_true == y_pred
    per_group = [hits[groups == g].mean() for g in np.unique(groups)]
    return float(np.mean(per_group))


def evaluate(X: FloatArray, y: IntArray, groups: IntArray,
             parts: Parts) -> dict[str, Any]:
    tr, te = parts["train"], parts["test"]
    best_k, val_scores = select_k(X, y, parts)
    pred = make_model(best_k).fit(X[tr], y[tr]).predict(X[te])
    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    base = dummy.predict(X[te])
    two_classes = len(np.unique(y[te])) == 2  # else: balanced acc undefined
    return {
        "best_k": best_k,
        "val_bal_acc": val_scores,
        "test_acc": accuracy_score(y[te], pred),
        "test_bal_acc": (binary_balanced_accuracy(y[te], pred)
                         if two_classes else None),
        "test_group_mean_acc": group_mean_accuracy(y[te], pred, groups[te]),
        "baseline_acc": accuracy_score(y[te], base),
        "baseline_bal_acc": (binary_balanced_accuracy(y[te], base)
                             if two_classes else None),
    }


def summarize(scores: list[float | None]) -> Summary:
    """Keep every split; statistics are conditional on defined scores."""
    defined = [s for s in scores if s is not None]
    undefined = [i for i, s in enumerate(scores) if s is None]
    if not defined:
        return Summary(len(scores), undefined, None, None, None)
    return Summary(len(scores), undefined, float(np.mean(defined)),
                   min(defined), max(defined))


def fmt(score: float | None) -> str:
    return "undefined (one class)" if score is None else f"{score:.3f}"


def describe(y: IntArray, groups: IntArray, parts: Parts) -> None:
    for role in ROLES:
        idx = parts[role]
        g, first = np.unique(groups[idx], return_index=True)
        pos_groups = int(y[idx][first].sum())
        print(f"  {role:<10} rows={len(idx):3d} groups={len(g):2d} "
              f"positive rows={y[idx].mean():.2f} "
              f"positive groups={pos_groups}")
    for a, b in combinations(ROLES, 2):
        n_groups, n_rows = overlap(groups, parts[a], parts[b])
        print(f"  overlap {a}/{b}: {n_groups} shared groups, "
              f"{n_rows} of {len(parts[b])} {b} rows from a shared group")


def report(name: str, X: FloatArray, y: IntArray, groups: IntArray,
           parts: Parts) -> None:
    print(f"\n[{name}]")
    describe(y, groups, parts)
    r = evaluate(X, y, groups, parts)
    vals = ", ".join(f"k={k}: {s:.3f}" for k, s in r["val_bal_acc"].items())
    print(f"  validation balanced accuracy: {vals} -> chosen k={r['best_k']}")
    print(f"  test accuracy={r['test_acc']:.3f} "
          f"balanced={fmt(r['test_bal_acc'])} "
          f"group-mean={r['test_group_mean_acc']:.3f}")
    print(f"  majority baseline accuracy={r['baseline_acc']:.3f} "
          f"balanced={fmt(r['baseline_bal_acc'])}")


def repeat(n_seeds: int = 20) -> None:
    """All seeds 0..n_seeds-1 are prespecified; none is dropped or redrawn."""
    runs: dict[str, list[dict[str, Any]]] = {"row split": [],
                                             "group split": []}
    for seed in range(n_seeds):
        X, y, groups = make_data(seed)
        parts = row_split(len(y), seed)
        runs["row split"].append(evaluate(X, y, groups, parts))
        parts = group_split(groups, seed)
        runs["group split"].append(evaluate(X, y, groups, parts))
    print(f"\n[{n_seeds} prespecified seeds: new data and new split per seed]")
    for name, results in runs.items():
        s = summarize([r["test_bal_acc"] for r in results])
        ks = [r["best_k"] for r in results]
        n_def = s.n_splits - len(s.undefined)
        print(f"  {name}: test balanced accuracy defined on "
              f"{n_def}/{s.n_splits} splits; one-class test set in seeds "
              f"{s.undefined}")
        if s.mean is not None:
            print(f"    over the {n_def} defined splits: mean={s.mean:.3f} "
                  f"min={s.low:.3f} max={s.high:.3f}")
        print(f"    chosen k counts (all {s.n_splits} seeds)="
              f"{ {k: ks.count(k) for k in K_CANDIDATES} }")


def main() -> None:
    X, y, groups = make_data(seed=0)
    first = np.unique(groups, return_index=True)[1]
    print(f"synthetic data: {len(y)} rows, {len(first)} groups, "
          f"{X.shape[1]} features, positive groups={int(y[first].sum())}")
    report("row split", X, y, groups, row_split(len(y), seed=0))
    report("group split", X, y, groups, group_split(groups, seed=0))
    repeat()


if __name__ == "__main__":
    main()
```

## Results

I ran the script on 12 October 2026 with uv 0.7.8 and CPython 3.11.12. Both are modern tooling, used here only to install and run the pinned library versions exactly; nothing in this post was executed in 2023. This is the full, unedited output:

```text
synthetic data: 543 rows, 60 groups, 8 features, positive groups=20

[row split]
  train      rows=325 groups=60 positive rows=0.34 positive groups=20
  validation rows=109 groups=48 positive rows=0.37 positive groups=15
  test       rows=109 groups=48 positive rows=0.31 positive groups=14
  overlap train/validation: 48 shared groups, 109 of 109 validation rows from a shared group
  overlap train/test: 48 shared groups, 109 of 109 test rows from a shared group
  overlap validation/test: 39 shared groups, 93 of 109 test rows from a shared group
  validation balanced accuracy: k=1: 1.000, k=5: 0.973, k=15: 0.891 -> chosen k=1
  test accuracy=1.000 balanced=1.000 group-mean=1.000
  majority baseline accuracy=0.688 balanced=0.500

[group split]
  train      rows=316 groups=36 positive rows=0.34 positive groups=12
  validation rows=106 groups=12 positive rows=0.41 positive groups=5
  test       rows=121 groups=12 positive rows=0.26 positive groups=3
  overlap train/validation: 0 shared groups, 0 of 106 validation rows from a shared group
  overlap train/test: 0 shared groups, 0 of 121 test rows from a shared group
  overlap validation/test: 0 shared groups, 0 of 121 test rows from a shared group
  validation balanced accuracy: k=1: 0.629, k=5: 0.621, k=15: 0.579 -> chosen k=1
  test accuracy=0.521 balanced=0.444 group-mean=0.453
  majority baseline accuracy=0.736 balanced=0.500

[20 prespecified seeds: new data and new split per seed]
  row split: test balanced accuracy defined on 20/20 splits; one-class test set in seeds []
    over the 20 defined splits: mean=0.982 min=0.934 max=1.000
    chosen k counts (all 20 seeds)={1: 19, 5: 1, 15: 0}
  group split: test balanced accuracy defined on 19/20 splits; one-class test set in seeds [11]
    over the 19 defined splits: mean=0.483 min=0.341 max=0.591
    chosen k counts (all 20 seeds)={1: 9, 5: 4, 15: 7}
```

### Overlap counts

The row split puts all 60 groups in train, and **all 109 test rows** come from a group that also has rows in train. The same is true for validation: 109 of 109 rows. Between validation and test, 93 of 109 test rows share a group.

In the group split every overlap count is zero, in all three pairs of roles.

### Scores

With the row split, validation prefers `k=1` with a balanced accuracy of 1.000, and the test score is also 1.000. The model looks perfect. It is perfect at one thing only: finding the nearest training row and copying its group's label.

With the group split, the same pipeline gets a test balanced accuracy of **0.444** and plain accuracy of 0.521. The majority baseline gets 0.736 accuracy and 0.500 balanced accuracy. On new entities, in this run, the "perfect" model scores below the majority baseline on both measures. That is consistent with the data design — there is nothing to learn — though a single split with twelve test groups could also have landed somewhat above 0.5 by luck.

### Validation chose on noise

The group split's validation scores were 0.629, 0.621 and 0.579 for `k` = 1, 5 and 15. That looks like a meaningful ranking, and `k=1` won again. But the test score of the chosen model was 0.444. With 12 validation groups, differences of that size are what noise looks like; by construction, no candidate can be better on new groups.

The 20-seed summary shows the same thing at a larger scale. Under the row split, validation picked `k=1` in 19 of 20 runs: leakage made memorisation look like the best model. Under the group split the choices spread almost evenly — 9, 4 and 7 — because no candidate is actually better.

A validation set that shares groups with train does not only inflate the score. It also pushes model selection toward the model that memorises best.

### Twenty seeds, and one undefined score

I fixed the seeds 0 to 19 in advance and kept all of them. Under the row split every test set contained both classes, and test balanced accuracy averaged **0.982** (min 0.934, max 1.000).

Under the group split, one of the 20 test sets — seed 11 — contained **no positive group at all**: twelve negative groups. Two-class balanced accuracy averages the recall on positives and the recall on negatives. With no positives in the test set the first recall does not exist, so the score is undefined.

scikit-learn 1.3.2 does not refuse in that case. It warns that `y_pred` contains classes not in `y_true` and returns the recall of the one class that is present — 0.860 for seed 11. That is a different quantity, and averaging it with genuine two-class scores mixes two metrics. An earlier version of this script did exactly that and reported a group-split mean of 0.502 with a maximum of 0.860.

The script now checks the classes before scoring. Seed 11 stays in the run and in the report, marked as undefined, and the summary says that its mean, min and max cover the 19 test sets where the score is defined: **0.483** (min 0.341, max 0.591). That is a conditional summary: it describes the runs whose test set happened to contain both classes. It is still close to the 0.5 the data design implies. I did not swap seed 11 for another seed. Choosing seeds after seeing their results is a quiet way of tuning on the test set.

Validation gets a stricter rule. A one-class validation set would make every candidate's score undefined, and picking `k` from it would mean choosing on a different metric. `select_k` raises `SingleClassError` instead of falling back to something. The fix is a different split design — more groups, or a stratified grouped split — chosen before looking at test results. None of the 20 seeds triggers it.

Across the 20 seeds, the number of positive groups among the 12 test groups ranged from 0 to 6 (I checked this per seed with a few extra lines next to the example). That is the next point.

## Groups are not stratified

`GroupShuffleSplit` draws groups without looking at labels or sizes. With few groups, class balance in a role can drift far from the overall rate. In seed 0 the test set had 3 positive groups out of 12 (26% positive rows), validation had 5 of 12 (41%), train 12 of 36 (34%).

Three practical consequences:

- Check class counts **per role in groups and in rows** before trusting a score. A test set with one positive group says very little about recall on positives; with zero, two-class balanced accuracy is not defined at all.
- For cross-validation inside the development pool, scikit-learn 1.3 offers `StratifiedGroupKFold`, which tries to keep class proportions while keeping each group in one fold. The documentation notes it is greedy and may produce imbalanced splits even when perfect stratification is possible.
- With few groups, the uncertainty is about groups, not rows. Twelve test groups are twelve observations of "a new entity", however many rows they contain. Repeating the split with different seeds, or grouped cross-validation on the development pool, shows how much the score moves.

## Row metrics versus group-balanced metrics

The script prints two kinds of accuracy. Row accuracy weighs every row equally, so a group with 15 rows counts five times as much as a group with 3. The group mean computes accuracy per group and then averages those, so every entity counts once.

In seed 0 with the group split they differ: 0.521 row accuracy versus 0.453 group mean. The test of `group_mean_accuracy` shows the extreme case: four correct rows of one group and one wrong row of another give 0.8 row accuracy but 0.5 group mean.

Neither is "the right one". If the cost is per transaction, row metrics fit. If the question is "for how many customers does this work?", a per-group metric fits better. I choose the one that matches the deployment question and write the choice down next to the result.

## Preprocessing fits on train only

A disjoint split is not enough if preprocessing sees validation or test rows. A scaler fitted on the whole table has learned means and variances from the test set; feature selection fitted on everything has looked at test labels.

In the example the scaler sits inside the pipeline, so `fit` only ever sees training rows. One of the tests checks it: the fitted scaler's `mean_` equals the mean of the training rows and differs from the mean of all rows. In cross-validation the same pipeline is refitted in every fold.

If, after all choices are made, I want a final model trained on train plus validation, I refit the whole pipeline — preprocessing included — on that combined pool. The test set still only passes through `transform` and `predict`. The example does not do this final refit; it fits the chosen model on train and stops there.

## Time is a separate axis

Group disjointness says nothing about time. A random group split can still put rows from 2023 in training and rows from 2021 in test.

When the deployment question involves time — known entities later, or new entities joining later — I also need:

- **Chronology:** test after validation after train.
- **Point-in-time features:** computed from what was known at prediction time, not from the final state of the table.
- **Label delay:** if an outcome is known a month later, it cannot be in the training data for predictions made before it existed.
- **A gap, when the setup calls for one.** Overlapping rolling windows are not leakage in themselves: if a feature at prediction time legitimately uses the last 30 days, a test row's window will reach back into the training period, and that is what deployment looks like. A gap is needed when something would be unavailable at prediction time — a label that only matures after a horizon, a target that overlaps the next window, data that arrives late. Its length follows from those horizons and delays, not from a fixed rule. scikit-learn's `TimeSeriesSplit` has a `gap` parameter for it.

A correct design may need disjoint groups *and* chronology at the same time. scikit-learn's guide on time series data makes the same point about i.i.d. splitters: they assume independence, and on correlated data they give poor estimates of generalisation error.

## What the tests check

The sixteen tests in `test_group_leakage.py` check:

1. Every group has a single label (the data has the structure I claim).
2. Both splits assign every row to exactly one role.
3. The group split's three roles are non-empty and pairwise group-disjoint.
4. The row split shares groups between train and test.
5. The same seed reproduces the same split.
6. The scaler is fitted on training rows only.
7. `select_k` scores every candidate and returns the best one.
8. The group mean weighs groups equally.
9. `binary_balanced_accuracy` raises `SingleClassError` when the true labels hold one class.
10. On two classes it equals the mean of both recalls.
11. `summarize` keeps every split, lists the undefined ones and computes mean, min and max over the rest.
12. `summarize` returns no statistics when no score is defined.
13. In seed 11 the group split's test set has no positive group (the case above, pinned).
14. `select_k` refuses a validation set with one class.
15. `evaluate` reports an undefined balanced accuracy for a one-class test set, while plain accuracy stays defined.
16. For seed 0 the row split scores above 0.95 balanced accuracy, the group split below 0.75, and the gap is over 0.2.

Tests 9–15 were written before the single-class guard and run against the previous version of the script first: 14 and 15 failed on its behaviour; adding 9–12 then made the whole suite fail at collection with an ImportError until the new functions existed; and 13 passed because it pins a fact about the data, not new code. The last test pins a result for one seed under the pinned versions. It is a regression test for this demonstration, not a statement about other data.

## Limitations

- The data is synthetic and extreme by design: features carry no information about labels of new groups. Real data usually also has signal that generalises to new entities, which changes the picture, but it does not make the gap predictably smaller. Its size depends on how strongly rows of one group resemble each other, how much of the label is entity-specific, how flexible the model is and how many groups there are. Real data can show a smaller gap than this demonstration, or a comparable one; only a grouped evaluation on that data says which.
- KNN with small `k` is especially good at memorising. Other models memorise too, but by different amounts.
- Sixty groups is small. Twelve test groups make a noisy estimate: across the 19 group-split runs with a defined score, test balanced accuracy ranged from 0.341 to 0.591, and one run of 20 had no positive test group at all.
- The validation step chooses one hyperparameter from three values. It is a demonstration of where tuning belongs, not a full tuning procedure.
- The script does not cover time; the time section is conceptual.

## Where I start

Before I split, I write down whether the model will meet new entities or known entities later. For new entities I split by group, once, into three roles, lock the test set, fit preprocessing on train, choose on validation, and report class counts per role in groups and rows. For known entities later I split by time and check feature availability. Often I need both.

A correct split will not make a weak model better. It makes the test score answer the question I am actually asking.

### Sources

- [scikit-learn 1.3: Cross-validation](https://scikit-learn.org/1.3/modules/cross_validation.html) — 3.1.2.3 "Cross-validation iterators for grouped data" (GroupKFold, StratifiedGroupKFold, GroupShuffleSplit), 3.1.2.5 on using group splitters for train/test (`train_test_split` cannot account for groups), 3.1.2.6 on time series.
- [scikit-learn 1.3: `GroupShuffleSplit`](https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) — `test_size` and `train_size` refer to groups, not samples; a fractional `test_size` is rounded up.
- [scikit-learn 1.3: Common pitfalls and recommended practices](https://scikit-learn.org/1.3/common_pitfalls.html) — data leakage, never calling `fit` on test data, pipelines.
- [scikit-learn 1.3 release notes](https://scikit-learn.org/1.3/whats_new/v1.3.html) — version 1.3.2, October 2023.
