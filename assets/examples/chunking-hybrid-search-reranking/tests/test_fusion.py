import pytest

from retrieval_lab.fusion import rrf


def test_rrf_uses_one_based_ranks_and_constant():
    fused = dict(rrf([["x", "y"], ["y"]], c=60, depth=5))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)


def test_item_absent_from_truncated_list_contributes_zero():
    fused = dict(rrf([["x"], []], c=60, depth=5))
    assert fused == {"x": pytest.approx(1 / 61)}


def test_output_sorted_truncated_and_tie_broken_by_id():
    fused = rrf([["b", "a"], ["a", "b"]], c=60, depth=1)
    assert [cid for cid, _ in fused] == ["a"]
