import itertools
import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from pr_calibration_demo import (
    brier_decomposition,
    confusion_at,
    cost_threshold,
    expected_random_ap,
    main,
    make_data,
    reliability_table,
    threshold_for_recall,
)


def test_synthetic_groups_have_exact_event_rates():
    y, p = make_data()
    assert y.size == 40 and y.sum() == 19
    for value in np.unique(p):
        assert y[p == value].mean() == pytest.approx(value)


def test_square_keeps_order_ties_and_ap():
    y, p = make_data()
    q = p**2
    assert np.array_equal(
        np.argsort(p, kind="stable"), np.argsort(q, kind="stable")
    )
    assert len(np.unique(q)) == len(np.unique(p))
    assert average_precision_score(y, q) == average_precision_score(y, p)


def test_constant_score_is_one_tie_and_ap_equals_prevalence():
    y, _ = make_data()
    constant = np.full(y.size, 0.5)
    assert average_precision_score(y, constant) == pytest.approx(y.mean())


def test_collapsing_two_groups_into_a_tie_changes_ap():
    y, p = make_data()
    pooled = np.where((p == 0.3) | (p == 0.6), 0.45, p)
    assert y[pooled == 0.45].mean() == pytest.approx(0.45)  # still calibrated
    assert average_precision_score(y, pooled) < average_precision_score(y, p)


def test_confusion_counts_and_ties_at_threshold():
    y, p = make_data()
    c = confusion_at(y, p, 0.6)  # `>=`: the whole 0.6 group is flagged
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (15, 5, 4, 16)
    assert c["precision"] == pytest.approx(0.75)
    assert c["recall"] == pytest.approx(15 / 19)


def test_threshold_above_every_score_leaves_precision_undefined():
    y, p = make_data()
    c = confusion_at(y, p, 1.01)
    assert c["tp"] == c["fp"] == 0
    assert c["precision"] is None and c["recall"] == 0.0


def test_threshold_at_lowest_score_flags_everything():
    y, p = make_data()
    c = confusion_at(y, p, p.min())
    assert c["recall"] == 1.0 and c["tn"] == 0
    assert c["precision"] == pytest.approx(y.mean())


def test_recall_is_undefined_without_positives():
    c = confusion_at([0, 0, 0], [0.2, 0.4, 0.9], 0.3)
    assert c["recall"] is None and c["precision"] == 0.0


def test_nan_threshold_is_rejected():
    with pytest.raises(ValueError):
        confusion_at([0, 1], [0.2, 0.8], math.nan)


@pytest.mark.parametrize(
    "y, s",
    [
        ([0, 2], [0.1, 0.9]),
        ([0, 1, 1], [0.1, 0.9]),
        ([], []),
        ([0, 1], [0.1, math.nan]),
        ([0, 1], [0.1, math.inf]),
        (["no", "yes"], [0.1, 0.9]),
    ],
)
def test_invalid_inputs_are_rejected(y, s):
    with pytest.raises(ValueError):
        confusion_at(y, s, 0.5)


def test_probability_functions_reject_scores_outside_unit_interval():
    with pytest.raises(ValueError):
        brier_decomposition([0, 1], [-0.1, 0.9])
    with pytest.raises(ValueError):
        reliability_table([0, 1], [0.1, 1.2], [0.0, 0.5, 1.0])


def test_cost_threshold():
    assert cost_threshold(1, 4) == pytest.approx(0.2)
    assert cost_threshold(5, 5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cost_threshold(0, 4)


@pytest.mark.parametrize(
    "cost_fp, cost_fn",
    [(math.inf, 4), (1, math.inf), (math.nan, 4), (1, math.nan)],
)
def test_cost_threshold_rejects_non_finite_costs(cost_fp, cost_fn):
    with pytest.raises(ValueError):
        cost_threshold(cost_fp, cost_fn)


@pytest.mark.parametrize(
    "cost_fp, cost_fn, expected",
    [(1e308, 1e308, 0.5), (1.5e308, 5e307, 0.75), (5e307, 1.5e308, 0.25)],
)
def test_cost_threshold_handles_huge_finite_costs(
    cost_fp, cost_fn, expected
):
    # cost_fp + cost_fn overflows to inf, but the ratio is well defined.
    assert cost_threshold(cost_fp, cost_fn) == pytest.approx(expected)


def test_threshold_for_recall_moves_whole_ties_and_follows_ranking():
    y, p = make_data()
    assert threshold_for_recall(y, p, 0.9) == pytest.approx(0.3)
    assert threshold_for_recall(y, p**2, 0.9) == pytest.approx(0.09)
    flagged_p = p >= threshold_for_recall(y, p, 0.9)
    flagged_q = p**2 >= threshold_for_recall(y, p**2, 0.9)
    assert np.array_equal(flagged_p, flagged_q)
    assert threshold_for_recall(y, p, 1.0) == pytest.approx(p.min())


@pytest.mark.parametrize("min_recall", [0.0, -0.1, 1.1, math.nan])
def test_threshold_for_recall_rejects_bad_target(min_recall):
    y, p = make_data()
    with pytest.raises(ValueError):
        threshold_for_recall(y, p, min_recall)


def test_threshold_for_recall_needs_positives():
    with pytest.raises(ValueError):
        threshold_for_recall([0, 0], [0.1, 0.2], 0.5)


@pytest.mark.parametrize("n, n_pos", [(6, 1), (6, 2), (6, 3), (7, 4)])
def test_expected_random_ap_matches_enumeration(n, n_pos):
    scores = np.arange(n, 0, -1, dtype=float)  # distinct scores, rank 1 first
    values = []
    for positions in itertools.combinations(range(n), n_pos):
        y = np.zeros(n, dtype=int)
        y[list(positions)] = 1
        values.append(average_precision_score(y, scores))
    assert expected_random_ap(n_pos, n) == pytest.approx(np.mean(values))


def test_expected_random_ap_edges_and_finite_sample_gap():
    assert expected_random_ap(3, 3) == pytest.approx(1.0)
    assert expected_random_ap(19, 40) > 19 / 40
    with pytest.raises(ValueError):
        expected_random_ap(0, 10)
    with pytest.raises(ValueError):
        expected_random_ap(11, 10)


def test_brier_decomposition_is_exact_and_separates_reliability():
    y, p = make_data()
    for scores in (p, p**2):
        d = brier_decomposition(y, scores)
        total = d["reliability"] - d["resolution"] + d["uncertainty"]
        assert total == pytest.approx(brier_score_loss(y, scores))
    dp, dq = brier_decomposition(y, p), brier_decomposition(y, p**2)
    assert dp["reliability"] == pytest.approx(0.0)
    assert dq["reliability"] > 0
    assert dq["resolution"] == pytest.approx(dp["resolution"])


def test_reliability_table_counts_empty_bins_and_intervals():
    y, p = make_data()
    rows = reliability_table(y, p**2, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert [r["n"] for r in rows] == [20, 10, 0, 10]
    assert rows[2]["observed"] is None
    for r in rows:
        if r["n"]:
            assert r["low"] <= r["observed"] <= r["high"]
    edge = reliability_table([1, 0], [1.0, 0.0], [0.0, 0.5, 1.0])
    assert [r["n"] for r in edge] == [1, 1]  # 1.0 belongs to the last bin
    assert [r["closed"] for r in rows] == [False, False, False, True]


@pytest.mark.parametrize(
    "edges",
    [
        [0.1, 1.0],
        [0.0, 0.9],
        [0.0, 0.5, 0.5, 1.0],
        [],
        [0.0],
        0.5,
        [[0.0, 1.0]],
        [0.0, math.nan, 1.0],
    ],
)
def test_reliability_table_rejects_bad_edges(edges):
    y, p = make_data()
    with pytest.raises(ValueError):
        reliability_table(y, p, edges)


def test_main_runs(capsys):
    main()
    out = capsys.readouterr().out
    assert "AP unchanged by p**2: True" in out
