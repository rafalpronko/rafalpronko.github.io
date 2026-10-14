"""Protocol regressions: group dependence and preservation on reruns."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import DATA

from fair_compare.experiment import run_experiment
from fair_compare.freeze import HoldoutError
from fair_compare.runner import Outcome
from fair_compare.stats import paired_compare


def outcomes(correct: bool, groups: tuple[str, ...]) -> tuple[Outcome, ...]:
    return tuple(
        Outcome(str(i), group, "ok", "billing", correct, 1)
        for i, group in enumerate(groups)
    )


def test_six_discordant_rows_in_one_group_are_not_significant() -> None:
    groups = ("one-customer",) * 6
    result = paired_compare(outcomes(False, groups), outcomes(True, groups))
    assert result.group_sign_flip_p == 1.0
    assert result.groups == 1
    assert result.group_mean_diff == 1.0


def test_six_independent_discordant_groups_have_exact_tail() -> None:
    groups = tuple(f"g{i}" for i in range(6))
    result = paired_compare(outcomes(False, groups), outcomes(True, groups))
    assert result.group_sign_flip_p == 2 / 64


def test_null_effect_has_p_one() -> None:
    rows = outcomes(True, ("a", "b"))
    assert paired_compare(rows, rows).group_sign_flip_p == 1.0


def test_within_group_discordance_cancels() -> None:
    a = outcomes(False, ("g", "g"))
    b = (replace(a[0], correct=True), a[1])
    a = (a[0], replace(a[1], correct=True))
    assert paired_compare(a, b).group_sign_flip_p == 1.0


def test_cluster_endpoint_gives_each_group_equal_weight() -> None:
    a = outcomes(False, ("small", "large", "large", "large"))
    b = (replace(a[0], correct=True), *a[1:])
    result = paired_compare(a, b)
    assert result.accuracy_b - result.accuracy_a == 0.25
    assert result.group_mean_diff == 0.5


@pytest.mark.parametrize("a,b", [((), ()), (None, ()), ((), None)])
def test_empty_or_null_inputs_are_rejected(a, b) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_compare(a, b)


def test_mismatched_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="same groups"):
        paired_compare(outcomes(True, ("a",)), outcomes(False, ("b",)))


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same cases"):
        paired_compare(outcomes(True, ("a",)), outcomes(False, ("a", "b")))


def test_duplicate_case_ids_are_rejected() -> None:
    rows = outcomes(True, ("a", "b"))
    duplicate = (rows[0], replace(rows[1], case_id=rows[0].case_id))
    with pytest.raises(ValueError, match="unique"):
        paired_compare(duplicate, duplicate)


@pytest.mark.parametrize("resamples", [0, -1])
def test_nonpositive_bootstrap_count_is_rejected(resamples: int) -> None:
    rows = outcomes(True, ("a",))
    with pytest.raises(ValueError, match="resamples"):
        paired_compare(rows, rows, resamples=resamples)


def snapshot(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*") if p.is_file()}


def test_changed_input_rerun_preserves_every_artifact(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "run"
    shutil.copytree(DATA, data)
    report = run_experiment(data, out)
    (out / "report.json").write_text(json.dumps(report))
    before = snapshot(out)
    raw = json.loads((data / "candidates.json").read_text())
    raw["arms"]["dspy"][2]["artifact"] += " changed after freeze"
    (data / "candidates.json").write_text(json.dumps(raw))
    with pytest.raises(HoldoutError, match="already used"):
        run_experiment(data, out)
    assert snapshot(out) == before


@pytest.mark.parametrize("artifact", [
    "freeze_manifest.json", "holdout_receipt.txt", "report.json",
    "run_started.txt", "unexpected-partial-file",
])
def test_partial_run_is_rejected_before_loading_or_search(
    tmp_path: Path, artifact: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / artifact).write_bytes(b"keep this exactly\x00\xff")
    before = snapshot(tmp_path)

    def fail_if_loaded(*args):
        pytest.fail("existing run must be rejected before input loading")

    monkeypatch.setattr("fair_compare.experiment.load_config", fail_if_loaded)
    with pytest.raises(HoldoutError, match="already used"):
        run_experiment(DATA, tmp_path)
    assert snapshot(tmp_path) == before


def test_failure_leaves_in_progress_marker(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_experiment(tmp_path / "missing-data", out)
    before = snapshot(out)
    assert "run_started.txt" in before
    with pytest.raises(HoldoutError, match="already used"):
        run_experiment(DATA, out)
    assert snapshot(out) == before
