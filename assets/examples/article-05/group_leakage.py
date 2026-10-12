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
