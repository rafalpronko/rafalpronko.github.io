"""Incremental run, explicit backfill and the one-transaction batch apply."""
import sqlite3
from collections.abc import Iterable

from incremental_demo.merge import check_contract_version, merge_row, validate
from incremental_demo.model import Batch, Change, MigrationRequired
from incremental_demo.source import extract, pages_keyset
from incremental_demo.store import (advance_checkpoint, columns, transaction,
                                    watermark)


def apply_batch(con: sqlite3.Connection, batch: Batch,
                checkpoint: tuple[int, int] | None = None) -> None:
    """Validate and merge every row, then optionally move the checkpoint.

    Everything happens in one transaction: an unsupported contract version
    (checked even for an empty batch), one bad row, a version conflict or a
    moved checkpoint rolls back all rows of the batch and the watermark.
    """
    with transaction(con):
        check_contract_version(batch.contract_version)
        has_currency = "currency" in columns(con)
        if batch.contract_version >= 2 and not has_currency:
            raise MigrationRequired("contract v2 needs the currency column")
        for row in batch.rows:
            validate(row, batch.contract_version)
            merge_row(con, row, has_currency)
        if checkpoint is not None:
            advance_checkpoint(con, *checkpoint)


def check_bounds(current: int, upper: object, read_at: object) -> None:
    """The upper bound is fixed once per run: W < U <= read time."""
    if type(upper) is not int or type(read_at) is not int:
        raise TypeError("upper and read_at must be int seconds")
    if not current < upper <= read_at:
        raise ValueError(f"need watermark {current} < upper {upper} <= read_at")


def check_run_params(overlap: object, page_size: object) -> None:
    """A negative overlap misses late rows; page_size 0 reads nothing."""
    if type(overlap) is not int or overlap < 0:
        raise ValueError(f"overlap {overlap!r}: int >= 0 required")
    if type(page_size) is not int or page_size < 1:
        raise ValueError(f"page_size {page_size!r}: int >= 1 required")


def run_incremental(con: sqlite3.Connection, changes: Iterable[Change], *,
                    contract_version: int, upper: int, read_at: int,
                    overlap: int, page_size: int = 100) -> list[str]:
    """Read ``[W - overlap, U)`` page by page; commit rows and W := U at once."""
    current = watermark(con)
    check_bounds(current, upper, read_at)
    check_run_params(overlap, page_size)
    window = extract(changes, read_at, current - overlap, upper)
    rows = [row for page in pages_keyset(window, page_size) for row in page]
    apply_batch(con, Batch(contract_version, tuple(rows)),
                checkpoint=(current, upper))
    return [row.order_id for row in rows]


def run_backfill(con: sqlite3.Connection, changes: Iterable[Change], *,
                 contract_version: int, lo: int, hi: int,
                 read_at: int) -> list[str]:
    """Explicit re-read of an already committed range. The checkpoint stays."""
    if type(lo) is not int or type(hi) is not int:
        raise TypeError("lo and hi must be int seconds")
    if not lo < hi <= watermark(con):
        raise ValueError("backfill range must lie below the watermark")
    rows = extract(changes, read_at, lo, hi)
    apply_batch(con, Batch(contract_version, tuple(rows)))
    return [row.order_id for row in rows]
