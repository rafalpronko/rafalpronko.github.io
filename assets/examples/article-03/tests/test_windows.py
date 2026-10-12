"""Extraction completeness: visibility delay, fixed upper bound, ties."""
import pytest

from conftest import OVERLAP, T, rows
from incremental_demo.model import Change
from incremental_demo.pipeline import run_backfill, run_incremental
from incremental_demo.source import pages_keyset, pages_timestamp_only
from incremental_demo.store import connect, watermark

SOURCE = [
    Change("A", T + 100, T + 100, "paid", 1_000),
    Change("B", T + 900, T + 1_150, "paid", 2_500),   # 250 s delay <= D
    Change("C", T + 1_000, T + 1_000, "new", 700),    # exactly on U1
    Change("D", T + 1_000, T + 1_000, "new", 800),    # same timestamp as C
    Change("E", T + 500, T + 1_600, "paid", 300),     # 1100 s delay > D
]


def run(con, upper, overlap=OVERLAP):
    return run_incremental(con, SOURCE, contract_version=1, upper=upper,
                           read_at=upper, overlap=overlap)


def test_rows_on_the_upper_bound_wait_for_the_next_window(con):
    assert run(con, T + 1_000) == ["A"]
    assert set(run(con, T + 2_000)) >= {"C", "D"}


def test_delay_within_overlap_is_caught_by_the_next_run(con):
    run(con, T + 1_000)
    assert "B" not in rows(con)
    assert "B" in run(con, T + 2_000)


def test_without_overlap_the_late_row_is_lost(con):
    run(con, T + 1_000, overlap=0)
    run(con, T + 2_000, overlap=0)
    assert "B" not in rows(con)


def test_delay_beyond_overlap_needs_explicit_backfill(con):
    run(con, T + 1_000)
    run(con, T + 2_000)
    assert "E" not in rows(con) and watermark(con) == T + 2_000
    got = run_backfill(con, SOURCE, contract_version=1, lo=T, hi=T + 1_000,
                       read_at=T + 2_000)
    assert "E" in got and "E" in rows(con)
    assert watermark(con) == T + 2_000


@pytest.mark.parametrize("upper, read_at", [
    (T, T + 10),            # not after the watermark
    (T + 20, T + 10),       # beyond what the snapshot could see
    (True, T + 10),         # bool is an int subclass, rejected explicitly
])
def test_invalid_upper_bound_is_rejected_before_reading(con, upper, read_at):
    with pytest.raises((TypeError, ValueError)):
        run_incremental(con, SOURCE, contract_version=1, upper=upper,
                        read_at=read_at, overlap=OVERLAP)
    assert watermark(con) == T and rows(con) == {}


@pytest.mark.parametrize("overlap, page_size", [
    (-500, 100),            # window starts after rows it should re-read
    (True, 100),            # bool again
    (OVERLAP, 0),           # zero pages, yet U would be committed
    (OVERLAP, -1),
    (OVERLAP, True),
])
def test_invalid_overlap_or_page_size_is_rejected_before_reading(
        con, overlap, page_size):
    with pytest.raises(ValueError):
        run_incremental(con, SOURCE, contract_version=1, upper=T + 1_000,
                        read_at=T + 1_000, overlap=overlap, page_size=page_size)
    assert watermark(con) == T and rows(con) == {}


def test_snapshot_converges_to_current_state_not_history(con):
    """Both versions of K are visible before the run; only the last loads."""
    paid = Change("K", T + 100, T + 100, "paid", 10)
    refunded = Change("K", T + 200, T + 200, "refunded", 10)
    assert run_incremental(con, [paid, refunded], contract_version=1,
                           upper=T + 1_000, read_at=T + 1_000,
                           overlap=OVERLAP) == ["K"]
    assert rows(con, "updated_at", "status") == {"K": (T + 200, "refunded")}


TIES = [Change(k, T + 5, T + 5, "new", 1) for k in ("X1", "X2", "X3")]


@pytest.mark.parametrize("pager", [pages_keyset, pages_timestamp_only])
@pytest.mark.parametrize("rows_in, page_size", [
    (TIES[::-1], 2),                  # not sorted
    ([TIES[0], TIES[0]], 2),          # duplicate cursor
    (TIES, 0),
])
def test_pagers_require_sorted_unique_input_and_positive_page_size(
        pager, rows_in, page_size):
    with pytest.raises(ValueError):
        list(pager(rows_in, page_size))


def test_timestamp_only_cursor_skips_ties_at_a_page_edge():
    seen = [r.order_id for p in pages_timestamp_only(TIES, 2) for r in p]
    assert seen == ["X1", "X2"]


def test_keyset_cursor_with_tie_breaker_keeps_every_row():
    seen = [r.order_id for p in pages_keyset(TIES, 2) for r in p]
    assert seen == ["X1", "X2", "X3"]


TIE_PAID = Change("A", T + 100, T + 200, "paid", 1_000)
TIE_REFUNDED = Change("A", T + 101, T + 200, "refunded", 1_000)


@pytest.mark.parametrize("source", [[TIE_PAID, TIE_REFUNDED],
                                    [TIE_REFUNDED, TIE_PAID]])
def test_visibility_tie_resolves_to_the_higher_version_in_any_order(
        con, source):
    """Both versions become visible at the same second; order must not matter."""
    for upper in (T + 1_000, T + 2_000):
        run_incremental(con, source, contract_version=1, upper=upper,
                        read_at=upper, overlap=OVERLAP)
    assert rows(con, "updated_at", "status") == {"A": (T + 101, "refunded")}


HISTORY = [Change("H", T - 5_000, T - 5_000, "paid", 40),
           Change("A", T + 100, T + 100, "paid", 1_000)]


def test_bootstrap_lower_bound_covering_all_history_loads_every_row():
    """First window [W0 - overlap, U) starts at or below the oldest row."""
    con = connect(T - 5_000)
    run_incremental(con, HISTORY, contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    assert set(rows(con)) == {"H", "A"}


def test_bootstrap_lower_bound_above_old_rows_misses_them_for_good(con):
    """connect(T) on an empty target: H at T - 5000 is never in any window."""
    for upper in (T + 1_000, T + 2_000, T + 3_000):
        run_incremental(con, HISTORY, contract_version=1, upper=upper,
                        read_at=upper, overlap=OVERLAP)
    assert set(rows(con)) == {"A"} and watermark(con) == T + 3_000
    run_backfill(con, HISTORY, contract_version=1, lo=T - 5_000, hi=T,
                 read_at=T + 3_000)          # preload what lies below W0
    assert set(rows(con)) == {"H", "A"} and watermark(con) == T + 3_000
