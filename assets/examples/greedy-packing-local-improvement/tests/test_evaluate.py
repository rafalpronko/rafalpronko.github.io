from packing.corpus import Instance
from packing.evaluate import evaluate, summarize
from tests.helpers import items_of


def test_evaluate_reports_bins_gap_and_outcome():
    inst = Instance("demo", "fixture", 10, tuple(items_of(5, 4, 4, 3, 2, 2)))
    row = evaluate(inst, max_work=10_000)
    assert row["ffd_bins"] == 3
    assert row["repaired_bins"] == 2
    assert row["lower_bound"] == 2
    assert row["ffd_gap"] == 1 and row["repaired_gap"] == 0
    assert row["outcome"] == "improve"
    assert row["stop_reason"] == "lower_bound"
    assert row["work"] == row["windows"] + row["placements"] > 0


def test_evaluate_tie_when_no_move_found():
    inst = Instance("six", "fixture", 10, tuple(items_of(6, 6, 6)))
    row = evaluate(inst, max_work=10_000)
    assert row["outcome"] == "tie"
    assert row["repaired_gap"] == 1


def test_summarize_counts_outcomes_per_family():
    rows = [
        {"family": "a", "outcome": "improve", "ffd_gap": 1,
         "repaired_gap": 0, "work": 5, "stop_reason": "lower_bound"},
        {"family": "a", "outcome": "tie", "ffd_gap": 1,
         "repaired_gap": 1, "work": 9, "stop_reason": "budget"},
    ]
    assert summarize(rows) == {"a": {
        "instances": 2, "improve": 1, "tie": 1, "worse": 0,
        "ffd_gap_total": 2, "repaired_gap_total": 1,
        "at_lower_bound_ffd": 0, "at_lower_bound_repaired": 1,
        "work_total": 14, "work_max": 9, "budget_stops": 1,
    }}
