"""Merge semantics: replays, regressions, conflicts and deletes."""
import pytest

from conftest import OVERLAP, T, rows
from incremental_demo.model import Batch, Change, VersionConflict
from incremental_demo.pipeline import apply_batch, run_incremental
from incremental_demo.store import watermark

V1 = Change("A", T + 100, T + 100, "paid", 1_000)
V2 = Change("A", T + 200, T + 200, "refunded", 1_000)


def test_replaying_the_same_batch_leaves_identical_state(con):
    batch = Batch(1, (V1, Change("B", T + 150, T + 150, "new", 50)))
    apply_batch(con, batch)
    first = rows(con, "updated_at", "status", "amount_minor")
    apply_batch(con, batch)
    assert rows(con, "updated_at", "status", "amount_minor") == first


def test_older_version_does_not_regress_state(con):
    apply_batch(con, Batch(1, (V2,)))
    apply_batch(con, Batch(1, (V1,)))
    assert rows(con, "status") == {"A": ("refunded",)}


def test_equal_version_with_other_payload_aborts_run_and_keeps_checkpoint(con):
    first = Change("A", T + 800, T + 800, "paid", 1_000)
    run_incremental(con, [first], contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    # Same updated_at, other amount, re-read through the overlap.
    contradiction = Change("A", T + 800, T + 1_100, "paid", 999)
    other = Change("B", T + 1_200, T + 1_200, "new", 10)
    with pytest.raises(VersionConflict):
        run_incremental(con, [contradiction, other], contract_version=1,
                        upper=T + 2_000, read_at=T + 2_000, overlap=OVERLAP)
    assert rows(con, "amount_minor") == {"A": (1_000,)}
    assert watermark(con) == T + 1_000


def test_tombstone_is_a_versioned_change(con):
    apply_batch(con, Batch(1, (V1,)))
    tombstone = Change("A", T + 300, T + 300, "paid", 1_000, deleted=True)
    apply_batch(con, Batch(1, (tombstone,)))
    apply_batch(con, Batch(1, (V2,)))  # late, older non-deleted version
    assert rows(con, "deleted") == {"A": (1,)}


def test_hard_delete_is_invisible_to_a_timestamp_window(con):
    source = [V1]
    run_incremental(con, source, contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    source.clear()  # the row is physically removed at the source
    run_incremental(con, source, contract_version=1, upper=T + 2_000,
                    read_at=T + 2_000, overlap=OVERLAP)
    assert rows(con, "deleted") == {"A": (0,)}
