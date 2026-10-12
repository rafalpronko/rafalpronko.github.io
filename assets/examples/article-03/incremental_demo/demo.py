"""Walk through one synthetic timeline and print the target after each step.

Run from the example directory:  python -m incremental_demo.demo
"""
import sqlite3
import sys

from incremental_demo.model import Change, ContractError
from incremental_demo.pipeline import run_backfill, run_incremental
from incremental_demo.store import add_currency_column, connect, watermark

T = 1_696_118_400  # 2023-10-01T00:00:00Z, synthetic
OVERLAP = 300      # source contract: visibility delay <= 300 s


def show(con: sqlite3.Connection, label: str) -> None:
    cols = "order_id, updated_at - ?, status, amount_minor"
    has_currency = any(r[1] == "currency"
                       for r in con.execute("PRAGMA table_info(orders)"))
    if has_currency:
        cols += ", currency"
    rows = con.execute(f"SELECT {cols} FROM orders ORDER BY order_id",
                       (T,)).fetchall()
    print(f"{label}: watermark=T+{watermark(con) - T} rows={rows}")


def main() -> int:
    source = [
        Change("A", T + 100, T + 100, "paid", 1_000),
        Change("B", T + 900, T + 1_150, "paid", 2_500),   # visible 250 s late
        Change("C", T + 1_000, T + 1_000, "new", 700),    # exactly on U1
        Change("E", T + 500, T + 1_600, "paid", 300),     # visible 1100 s late
    ]
    con = connect(T)
    print(f"python {sys.version.split()[0]}, sqlite {sqlite3.sqlite_version}")
    run_incremental(con, source, contract_version=1, upper=T + 1_000,
                    read_at=T + 1_000, overlap=OVERLAP)
    show(con, "run 1 [T-300, T+1000)")
    run_incremental(con, source, contract_version=1, upper=T + 2_000,
                    read_at=T + 2_000, overlap=OVERLAP)
    show(con, "run 2 [T+700, T+2000)")
    run_backfill(con, source, contract_version=1, lo=T, hi=T + 1_000,
                 read_at=T + 2_000)
    show(con, "backfill [T, T+1000)")
    add_currency_column(con)
    show(con, "after ADD COLUMN currency")
    v2 = [Change("A", T + 100, T + 100, "paid", 1_000, "PLN"),
          Change("D", T + 2_100, T + 2_100, "new", 900, "eur")]
    try:
        run_incremental(con, v2, contract_version=2, upper=T + 3_000,
                        read_at=T + 3_000, overlap=OVERLAP)
    except ContractError as exc:
        print(f"run 3 rejected: {exc}")
    show(con, "after rejected run 3")
    run_backfill(con, v2[:1], contract_version=2, lo=T, hi=T + 1_000,
                 read_at=T + 3_000)
    show(con, "enrichment backfill of A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
