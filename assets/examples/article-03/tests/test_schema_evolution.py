"""Schema evolution: nullable currency, rejected batches, enrichment policy."""
import sqlite3

import pytest

from conftest import OVERLAP, T, rows
from incremental_demo.model import (Batch, Change, ContractError,
                                    MigrationRequired, VersionConflict)
from incremental_demo.pipeline import apply_batch, run_backfill, run_incremental
from incremental_demo.store import add_currency_column, watermark

OLD = Change("A", T + 100, T + 100, "paid", 1_000)
OLD_WITH_PLN = Change("A", T + 100, T + 100, "paid", 1_000, "PLN")


@pytest.fixture
def migrated(con: sqlite3.Connection) -> sqlite3.Connection:
    run_incremental(con, [OLD], contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    assert add_currency_column(con) is True
    return con


def test_added_column_leaves_old_rows_unknown_not_defaulted(migrated):
    assert rows(migrated, "currency") == {"A": (None,)}
    assert add_currency_column(migrated) is False  # idempotent migration


def test_not_null_default_would_invent_a_currency(con):
    apply_batch(con, Batch(1, (OLD,)))
    con.execute("ALTER TABLE orders ADD COLUMN currency TEXT "
                "NOT NULL DEFAULT 'PLN'")
    assert rows(con, "currency") == {"A": ("PLN",)}  # nobody said PLN


def test_not_null_without_default_is_refused_on_a_populated_table(con):
    apply_batch(con, Batch(1, (OLD,)))
    with pytest.raises(sqlite3.OperationalError):
        con.execute("ALTER TABLE orders ADD COLUMN currency TEXT NOT NULL")


def test_v2_batch_before_migration_is_rejected(con):
    with pytest.raises(MigrationRequired):
        apply_batch(con, Batch(2, (OLD_WITH_PLN,)))
    assert rows(con) == {}


def test_failed_validation_rolls_back_rows_and_checkpoint(migrated):
    good = Change("C", T + 1_100, T + 1_100, "new", 900, "EUR")
    bad = Change("D", T + 1_200, T + 1_200, "new", 400, "eur")
    with pytest.raises(ContractError):
        run_incremental(migrated, [good, bad], contract_version=2,
                        upper=T + 2_000, read_at=T + 2_000, overlap=OVERLAP)
    assert "C" not in rows(migrated)
    assert watermark(migrated) == T + 1_000


def test_v2_row_without_currency_is_a_contract_violation(migrated):
    missing = Change("C", T + 1_100, T + 1_100, "new", 900, None)
    with pytest.raises(ContractError):
        apply_batch(migrated, Batch(2, (missing,)))


def test_backfill_enriches_unknown_currency_at_the_same_version(migrated):
    run_backfill(migrated, [OLD_WITH_PLN], contract_version=2, lo=T,
                 hi=T + 1_000, read_at=T + 1_000)
    assert rows(migrated, "updated_at", "currency") == {"A": (T + 100, "PLN")}
    assert watermark(migrated) == T + 1_000


def test_same_known_currency_at_the_same_version_is_a_no_op(migrated):
    apply_batch(migrated, Batch(2, (OLD_WITH_PLN,)))
    apply_batch(migrated, Batch(2, (OLD_WITH_PLN,)))
    assert rows(migrated, "updated_at", "currency") == {"A": (T + 100, "PLN")}


def test_known_currency_is_not_replaced_at_the_same_version(migrated):
    apply_batch(migrated, Batch(2, (OLD_WITH_PLN,)))
    eur = Change("A", T + 100, T + 100, "paid", 1_000, "EUR")
    with pytest.raises(VersionConflict):
        apply_batch(migrated, Batch(2, (eur,)))
    assert rows(migrated, "currency") == {"A": ("PLN",)}


def test_replayed_v1_extract_does_not_erase_known_currency(migrated):
    apply_batch(migrated, Batch(2, (OLD_WITH_PLN,)))
    apply_batch(migrated, Batch(1, (OLD,)))
    assert rows(migrated, "currency") == {"A": ("PLN",)}


def test_newer_version_from_v1_producer_is_honestly_unknown(migrated):
    apply_batch(migrated, Batch(2, (OLD_WITH_PLN,)))
    newer = Change("A", T + 500, T + 500, "refunded", 1_000)
    apply_batch(migrated, Batch(1, (newer,)))
    assert rows(migrated, "status", "currency") == {"A": ("refunded", None)}


def test_undeclared_contract_version_is_rejected(migrated):
    whole_units = Change("Z", T + 1_100, T + 1_100, "new", 9, "EUR")
    with pytest.raises(ContractError):
        apply_batch(migrated, Batch(3, (whole_units,)))
    assert "Z" not in rows(migrated)


def test_unit_change_inside_the_same_contract_is_not_detected(migrated):
    """Limit of type checks: 12 (euros) passes where 1200 (cents) was meant."""
    whole_units = Change("Z", T + 1_100, T + 1_100, "new", 12, "EUR")
    apply_batch(migrated, Batch(2, (whole_units,)))
    assert rows(migrated, "amount_minor")["Z"] == (12,)
