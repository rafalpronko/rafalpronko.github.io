from itertools import combinations

import numpy as np
import pytest

from group_leakage import (
    K_CANDIDATES,
    ROLES,
    SingleClassError,
    binary_balanced_accuracy,
    evaluate,
    group_mean_accuracy,
    group_split,
    make_data,
    make_model,
    overlap,
    row_split,
    select_k,
    summarize,
)


@pytest.fixture(scope="module")
def data():
    return make_data(seed=0)


def all_rows(parts):
    return np.sort(np.concatenate([parts[r] for r in ROLES]))


def test_label_is_constant_within_each_group(data):
    _, y, groups = data
    for g in np.unique(groups):
        assert len(np.unique(y[groups == g])) == 1


def test_both_splits_use_every_row_exactly_once(data):
    _, y, groups = data
    expected = np.arange(len(y))
    assert np.array_equal(all_rows(row_split(len(y), seed=0)), expected)
    assert np.array_equal(all_rows(group_split(groups, seed=0)), expected)


def test_group_split_roles_are_pairwise_group_disjoint_and_non_empty(data):
    _, _, groups = data
    parts = group_split(groups, seed=0)
    assert all(len(parts[r]) > 0 for r in ROLES)
    for a, b in combinations(ROLES, 2):
        assert overlap(groups, parts[a], parts[b]) == (0, 0)


def test_row_split_shares_groups_between_train_and_test(data):
    _, y, groups = data
    parts = row_split(len(y), seed=0)
    n_groups, n_rows = overlap(groups, parts["train"], parts["test"])
    assert n_groups > 0 and n_rows > 0


def test_splits_are_deterministic_for_a_seed(data):
    _, y, groups = data
    for split in (lambda: row_split(len(y), 3),
                  lambda: group_split(groups, 3)):
        a, b = split(), split()
        assert all(np.array_equal(a[r], b[r]) for r in ROLES)


def test_scaler_is_fitted_on_training_rows_only(data):
    X, y, groups = data
    tr = group_split(groups, seed=0)["train"]
    scaler = make_model(1).fit(X[tr], y[tr])[0]
    assert np.allclose(scaler.mean_, X[tr].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_select_k_scores_every_candidate_and_returns_the_best(data):
    X, y, groups = data
    best, scores = select_k(X, y, group_split(groups, seed=0))
    assert set(scores) == set(K_CANDIDATES)
    assert scores[best] == max(scores.values())


def test_group_mean_accuracy_weighs_groups_equally():
    groups = np.array([0, 0, 0, 0, 1])
    y_true = np.array([1, 1, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1])
    # Row accuracy would be 0.8; each group counts once, so 0.5.
    assert group_mean_accuracy(y_true, y_pred, groups) == 0.5


def test_binary_balanced_accuracy_rejects_single_class_labels():
    with pytest.raises(SingleClassError):
        binary_balanced_accuracy(np.array([0, 0, 0]), np.array([0, 1, 0]))


def test_binary_balanced_accuracy_is_mean_of_both_recalls():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert binary_balanced_accuracy(y_true, y_pred) == pytest.approx(5 / 6)


def test_summarize_keeps_every_split_and_counts_undefined_ones():
    summary = summarize([0.4, None, 0.6, 0.5])
    assert summary.n_splits == 4
    assert summary.undefined == [1]
    assert summary.mean == pytest.approx(0.5)
    assert (summary.low, summary.high) == (0.4, 0.6)


def test_summarize_reports_no_statistics_when_nothing_is_defined():
    summary = summarize([None, None])
    assert summary.undefined == [0, 1]
    assert summary.mean is None and summary.low is None


def test_group_split_seed_11_test_set_has_no_positive_group():
    _, y, groups = make_data(seed=11)
    test = group_split(groups, seed=11)["test"]
    assert np.unique(y[test]).tolist() == [0]


def test_select_k_refuses_a_single_class_validation_set(data):
    X, y, groups = data
    parts = group_split(groups, seed=0)
    negatives = parts["validation"][y[parts["validation"]] == 0]
    with pytest.raises(ValueError, match="both classes"):
        select_k(X, y, {**parts, "validation": negatives})


def test_evaluate_marks_single_class_test_score_as_undefined():
    X, y, groups = make_data(seed=11)
    result = evaluate(X, y, groups, group_split(groups, seed=11))
    assert result["test_bal_acc"] is None
    assert result["baseline_bal_acc"] is None
    assert 0.0 <= result["test_acc"] <= 1.0  # plain accuracy stays defined


def test_row_split_inflates_score_relative_to_group_split(data):
    X, y, groups = data
    leaky = evaluate(X, y, groups, row_split(len(y), seed=0))
    honest = evaluate(X, y, groups, group_split(groups, seed=0))
    assert leaky["test_bal_acc"] > 0.95
    assert honest["test_bal_acc"] < 0.75
    assert leaky["test_bal_acc"] - honest["test_bal_acc"] > 0.2
