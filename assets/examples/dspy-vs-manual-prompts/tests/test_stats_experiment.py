from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import DATA

from fair_compare.candidates import load_candidates
from fair_compare.experiment import run_experiment
from fair_compare.freeze import HoldoutError
from fair_compare.runner import Outcome
from fair_compare.stats import group_sign_flip_exact, paired_compare


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(0, 0, 1.0), (1, 2, 1.0), (0, 5, 0.0625), (5, 0, 0.0625),
     (1, 9, 0.021484375)],
)
def test_group_sign_flip_exact(
    only_a: int, only_b: int, expected: float,
) -> None:
    # Every delta here represents a separate independent group, not a row.
    deltas = (Fraction(-1),) * only_a + (Fraction(1),) * only_b
    actual = group_sign_flip_exact(deltas or (Fraction(0),))
    assert actual == pytest.approx(expected)


def _outcomes(correct: list[bool], ids: str = "abcd") -> tuple[Outcome, ...]:
    return tuple(
        Outcome(i, f"g{n // 2}", "ok", "x", c, 1)
        for n, (i, c) in enumerate(zip(ids, correct, strict=True))
    )


def test_paired_compare_needs_identical_cases() -> None:
    with pytest.raises(ValueError, match="same cases"):
        paired_compare(_outcomes([True] * 4), _outcomes([True] * 4, "abce"))


def test_paired_compare_counts_discordant_pairs() -> None:
    a = _outcomes([True, True, False, False])
    b = _outcomes([True, False, True, True])
    result = paired_compare(a, b, resamples=500, seed=1)
    assert (result.only_a_correct, result.only_b_correct) == (1, 2)
    assert result.diff_ci95 == paired_compare(a, b, 500, 1).diff_ci95


def test_end_to_end_report(tmp_path: Path) -> None:
    report = run_experiment(DATA, tmp_path)
    assert report["frozen"]["selected"] == {
        "manual": "manual-v2", "dspy": "dspy-bootstrap-k4"}
    manual, dspy = report["arms"]["manual"], report["arms"]["dspy"]
    assert manual["optimization"]["billable_attempts"] == 37
    assert manual["optimization"]["cache_hits"] == 12
    assert "manual-v5 not evaluated" in manual["stop_reason"]
    assert dspy["optimization"]["billable_attempts"] == 60
    assert "dspy-bootstrap-k8 discarded" in dspy["stop_reason"]
    assert report["holdout"]["manual"]["correct"] == 7
    assert report["holdout"]["dspy"]["correct"] == 8
    paired = report["paired_dspy_minus_manual"]
    assert (paired["only_manual_correct"], paired["only_dspy_correct"]) == (
        1, 2)
    assert paired["group_sign_flip_exact_p"] == 1.0
    assert paired["groups"] == 5
    assert paired["nonzero_groups"] == 1
    manifest = json.loads((tmp_path / "freeze_manifest.json").read_text())
    assert manifest == report["frozen"]


def test_second_holdout_run_in_same_dir_is_refused(tmp_path: Path) -> None:
    run_experiment(DATA, tmp_path)
    with pytest.raises(HoldoutError, match="already used"):
        run_experiment(DATA, tmp_path)


def test_same_artifact_with_different_behaviour_is_rejected(
    tmp_path: Path,
) -> None:
    for name in ("cases.json", "experiment.json"):
        shutil.copy(DATA / name, tmp_path / name)
    raw = json.loads((DATA / "candidates.json").read_text())
    raw["arms"]["manual"][3]["overrides"] = {}  # v4 == v2 text, new output
    (tmp_path / "candidates.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="reuses the artifact"):
        load_candidates(tmp_path / "candidates.json")
