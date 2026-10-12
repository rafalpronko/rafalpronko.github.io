"""A simulated mutable source table: snapshots, windows and pagination."""
from collections.abc import Iterable, Iterator

from incremental_demo.model import Change

Cursor = tuple[int, str]


def sort_key(row: Change) -> Cursor:
    """Total order for extraction: timestamp first, primary key as tie-breaker."""
    return (row.updated_at, row.order_id)


def snapshot(changes: Iterable[Change], read_at: int) -> list[Change]:
    """Latest committed version of every row that is visible at ``read_at``.

    Like a real table, a snapshot holds current state, not history: a version
    replaced before anyone read it is gone. One row per key, sorted by
    ``sort_key``, which is the input the pagers below require.

    "Latest" is decided first by ``visible_at``: of a key's visible
    versions, the one that became visible last wins. Only when two versions
    of a key become visible in the same second does ``updated_at`` break
    the tie, because the contract orders versions by ``updated_at``. Either
    way the result does not depend on the order of ``changes``.
    """
    latest: dict[str, Change] = {}
    for change in changes:
        if change.visible_at > read_at:
            continue
        current = latest.get(change.order_id)
        if current is None or ((change.visible_at, change.updated_at)
                               > (current.visible_at, current.updated_at)):
            latest[change.order_id] = change
    return sorted(latest.values(), key=sort_key)


def extract(changes: Iterable[Change], read_at: int,
            lo: int, hi: int) -> list[Change]:
    """Half-open window ``[lo, hi)`` on ``updated_at`` of one snapshot."""
    return [r for r in snapshot(changes, read_at) if lo <= r.updated_at < hi]


def check_page_input(rows: list[Change], page_size: int) -> None:
    """Pagers take one snapshot, sorted and unique by ``sort_key``."""
    if type(page_size) is not int or page_size < 1:
        raise ValueError(f"page_size {page_size!r}: int >= 1 required")
    keys = [sort_key(r) for r in rows]
    if any(a >= b for a, b in zip(keys, keys[1:])):
        raise ValueError("rows must be sorted and unique by sort_key")


def pages_keyset(rows: list[Change], page_size: int) -> Iterator[list[Change]]:
    """Keyset pagination on ``(updated_at, order_id)``.

    The composite cursor is a total order, so rows sharing a timestamp
    cannot fall between two pages.
    """
    check_page_input(rows, page_size)
    cursor: Cursor | None = None
    while True:
        page = [r for r in rows if cursor is None or sort_key(r) > cursor]
        page = page[:page_size]
        if not page:
            return
        yield page
        cursor = sort_key(page[-1])


def pages_timestamp_only(rows: list[Change],
                         page_size: int) -> Iterator[list[Change]]:
    """Broken on purpose: ``updated_at > last`` skips ties at a page edge."""
    check_page_input(rows, page_size)
    cursor: int | None = None
    while True:
        page = [r for r in rows if cursor is None or r.updated_at > cursor]
        page = page[:page_size]
        if not page:
            return
        yield page
        cursor = page[-1].updated_at
