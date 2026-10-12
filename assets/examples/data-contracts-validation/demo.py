"""Walk through the contract on synthetic batches and print each outcome.

Run:  python demo.py   (exit code 0 = every outcome matched expectations)
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from typing import Any

from fixtures import (SYNTHETIC_BASELINE_MEDIAN_NET, canonical_batch,
                      consistent_unit_lie_batch, line)
from invoice_contract import ContractViolation
from invoice_store import connect, count_rows, publish


def changed(mutate: Any) -> dict[str, Any]:
    batch = canonical_batch()
    mutate(batch)
    return batch


def scenarios() -> list[tuple[str, object, str]]:
    def malformed_duplicate(b: dict[str, Any]) -> None:
        b["rows"][1].update(line_id="L-1", net_minor=19.99)

    def collides_with_table(b: dict[str, Any]) -> None:
        b["rows"] = [line("L-9", 100, 0, 0, 100), line("L-1", 100, 0, 0, 100)]

    return [
        ("not an envelope", ["L-1", 10_000], "rejected"),
        ("declared unit change", changed(lambda b: b.update(unit="major")),
         "rejected"),
        ("truncated tax", changed(lambda b: b["rows"][2].update(
            tax_minor=227, gross_minor=4_777)), "rejected"),
        ("duplicate key, one copy malformed", changed(malformed_duplicate),
         "rejected"),
        ("canonical batch", canonical_batch(), "published"),
        ("replay of canonical batch", canonical_batch(), "rolled back"),
        ("new key + existing key", changed(collides_with_table),
         "rolled back"),
        ("consistent unit lie", consistent_unit_lie_batch(), "published"),
    ]


def run(con: sqlite3.Connection, name: str, batch: object) -> str:
    try:
        result = publish(con, batch, SYNTHETIC_BASELINE_MEDIAN_NET)
    except ContractViolation as exc:
        print(f"REJECTED     {name}")
        for error in exc.errors:
            print(f"    - {error}")
        return "rejected"
    except sqlite3.IntegrityError as exc:
        print(f"ROLLED BACK  {name}: {exc}")
        return "rolled back"
    print(f"PUBLISHED    {name}: {result.rows_written} rows")
    for advisory in result.advisories:
        print(f"    advisory: {advisory}")
    return "published"


def main() -> int:
    mismatches = 0
    with closing(connect()) as con:
        for name, batch, expected in scenarios():
            outcome = run(con, name, batch)
            if outcome != expected:
                print(f"    MISMATCH: expected {expected}, got {outcome}")
                mismatches += 1
        print(f"rows in table: {count_rows(con)}")
    print(f"{mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
