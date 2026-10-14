from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import LABELS, make_candidate, make_split

from fair_compare.budget import Ledger
from fair_compare.freeze import HoldoutError, HoldoutGuard, freeze
from fair_compare.search import LeakageError, search_arm, select_on_dev

TRAIN = make_split("train", 8, "t")
DEV = make_split("dev", 4, "d")
TEST = make_split("test", 4, "h")


def run(cands, cap=100, feedback_cap=10, dev=DEV):
    return search_arm("arm", tuple(cands), TRAIN, dev, LABELS,
                      Ledger("arm/opt", cap), feedback_cap, 2)


def test_search_refuses_the_test_split() -> None:
    with pytest.raises(ValueError, match="train\\+dev only"):
        run([make_candidate()], dev=TEST)


def test_demos_from_dev_are_leakage() -> None:
    with pytest.raises(LeakageError, match="d0"):
        run([make_candidate(demo_ids=("t0", "d0"))])


def test_feedback_cap_stops_further_dev_evaluations() -> None:
    log = run([make_candidate(f"v{i}") for i in range(3)], feedback_cap=2)
    assert [r.candidate.candidate_id for r in log.results] == ["v0", "v1"]
    assert "v2 not evaluated" in log.stop_reason


def test_budget_hit_during_compile_discards_the_candidate() -> None:
    cands = [make_candidate("a"),
             make_candidate("b", compile_attempts={"teacher": 10})]
    log = run(cands, cap=8)
    assert [r.candidate.candidate_id for r in log.results] == ["a"]
    assert "b discarded" in log.stop_reason
    assert log.ledger.billable == 8  # spent attempts stay spent


def test_budget_hit_mid_dev_discards_the_partial_score() -> None:
    log = run([make_candidate("a"), make_candidate("b")], cap=6)
    assert [r.candidate.candidate_id for r in log.results] == ["a"]
    assert log.ledger.billable == 6


def test_selection_uses_dev_and_breaks_ties_by_order() -> None:
    worse = make_candidate("worse", overrides={"d0": ["account"]})
    tie_a, tie_b = make_candidate("tie-a"), make_candidate("tie-b")
    log = run([worse, tie_a, tie_b])
    assert select_on_dev(log).candidate.candidate_id == "tie-a"


def test_select_with_no_complete_candidate_fails_loudly() -> None:
    log = run([make_candidate("a", compile_attempts={"teacher": 5})], cap=3)
    with pytest.raises(ValueError, match="no complete dev evaluation"):
        select_on_dev(log)


def _guard(tmp_path: Path) -> HoldoutGuard:
    return HoldoutGuard(TEST, tmp_path / "receipt.txt")


def test_holdout_opens_exactly_once(tmp_path: Path) -> None:
    chosen = {"manual": make_candidate("m"), "dspy": make_candidate("d")}
    guard = _guard(tmp_path)
    manifest = freeze(chosen, guard.test_ids_sha256, "fixture-sha")
    assert guard.open(manifest, chosen) is TEST
    with pytest.raises(HoldoutError, match="already used"):
        guard.open(manifest, chosen)
    with pytest.raises(HoldoutError, match="already used"):
        _guard(tmp_path).open(manifest, chosen)  # survives a restart


def test_artifact_edited_after_freeze_is_rejected(tmp_path: Path) -> None:
    chosen = {"manual": make_candidate("m", artifact="v2 prompt")}
    guard = _guard(tmp_path)
    manifest = freeze(chosen, guard.test_ids_sha256, "fixture-sha")
    edited = {"manual": make_candidate("m", artifact="v2 prompt, tweaked")}
    with pytest.raises(HoldoutError, match="artifact changed"):
        guard.open(manifest, edited)
    assert not (tmp_path / "receipt.txt").exists()


def test_changed_test_set_is_rejected(tmp_path: Path) -> None:
    chosen = {"manual": make_candidate("m")}
    manifest = freeze(chosen, "digest-of-another-test-set", "fixture-sha")
    with pytest.raises(HoldoutError, match="test split changed"):
        _guard(tmp_path).open(manifest, chosen)


def test_manifest_is_immutable() -> None:
    manifest = freeze({"manual": make_candidate("m")}, "x", "y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.test_ids_sha256 = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.selected["manual"] = "v9"  # type: ignore[index]


def test_guard_only_wraps_test() -> None:
    with pytest.raises(ValueError):
        HoldoutGuard(DEV, Path("unused"))
