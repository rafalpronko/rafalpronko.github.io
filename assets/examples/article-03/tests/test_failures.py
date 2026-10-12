"""Partial failure and checkpoint atomicity."""
import sqlite3

import pytest

from conftest import OVERLAP, T, rows
from incremental_demo.model import Batch, Change, CheckpointMoved, ContractError
from incremental_demo.pipeline import apply_batch, run_incremental
from incremental_demo.store import add_currency_column, transaction, watermark

GOOD = Change("G", T + 100, T + 100, "paid", 1_000)
BAD = Change("H", T + 200, T + 200, "paid", "12.50")  # type: ignore[arg-type]


def test_bad_second_row_rolls_back_first_row_and_checkpoint(con):
    with pytest.raises(ContractError):
        run_incremental(con, [GOOD, BAD], contract_version=1,
                        upper=T + 1_000, read_at=T + 1_000, overlap=OVERLAP)
    assert rows(con) == {} and watermark(con) == T


@pytest.mark.parametrize("version", [3, True, "1", None])
def test_empty_batch_with_unsupported_contract_keeps_checkpoint(con, version):
    """Checked per batch, so zero rows cannot smuggle a watermark move."""
    add_currency_column(con)  # v3 must not fail on a missing column instead
    with pytest.raises(ContractError):
        apply_batch(con, Batch(version, ()), checkpoint=(T, T + 1_000))
    assert rows(con) == {} and watermark(con) == T


def test_valid_empty_batch_advances_the_watermark(con):
    """Nothing changed in [W - overlap, U): a supported contract moves W to U."""
    assert run_incremental(con, [], contract_version=1, upper=T + 1_000,
                           read_at=T + 1_000, overlap=OVERLAP) == []
    assert rows(con) == {} and watermark(con) == T + 1_000


def test_bool_contract_version_is_rejected_for_a_whole_run(con):
    with pytest.raises(ContractError):
        run_incremental(con, [GOOD], contract_version=True, upper=T + 1_000,
                        read_at=T + 1_000, overlap=OVERLAP)
    assert rows(con) == {} and watermark(con) == T


@pytest.mark.parametrize("order_id", ["", None, 7])
def test_order_id_must_be_a_non_empty_string(con, order_id):
    bad_key = Change(order_id, T + 200, T + 200, "paid", 1)  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        run_incremental(con, [GOOD, bad_key], contract_version=1,
                        upper=T + 1_000, read_at=T + 1_000, overlap=OVERLAP)
    assert rows(con) == {} and watermark(con) == T


def test_failed_commit_is_rolled_back_and_leaves_no_open_transaction(con):
    """SQLite keeps a transaction open when COMMIT fails on a deferred FK."""
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE child (parent_id INTEGER REFERENCES parent (id)"
                " DEFERRABLE INITIALLY DEFERRED)")
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(con):
            con.execute("INSERT INTO child VALUES (42)")
            con.execute("UPDATE checkpoint SET watermark = ?", (T + 1_000,))
    assert not con.in_transaction
    assert con.execute("SELECT count(*) FROM child").fetchone() == (0,)
    assert watermark(con) == T


def test_checkpoint_moved_by_someone_else_rolls_back_rows(con):
    with pytest.raises(CheckpointMoved):
        apply_batch(con, Batch(1, (GOOD,)), checkpoint=(T - 1, T + 1_000))
    assert rows(con) == {} and watermark(con) == T


def test_rerun_after_lost_acknowledgement_converges(con):
    """The commit happened, the caller never heard; the next run re-reads."""
    source = [GOOD, Change("I", T + 900, T + 900, "new", 5)]
    run_incremental(con, source, contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    before = rows(con, "updated_at", "status", "amount_minor")
    got = run_incremental(con, source, contract_version=1, upper=T + 1_100,
                          read_at=T + 1_100, overlap=OVERLAP)
    assert got == ["I"]  # re-read through the overlap, applied as a no-op
    assert rows(con, "updated_at", "status", "amount_minor") == before
