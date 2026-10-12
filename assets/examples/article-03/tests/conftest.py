import sqlite3

import pytest

from incremental_demo.store import connect

T = 1_696_118_400  # 2023-10-01T00:00:00Z, synthetic
OVERLAP = 300      # source contract: visibility delay <= 300 s


def rows(con: sqlite3.Connection, *cols: str) -> dict[str, tuple]:
    """Target state as {order_id: (cols...)} for whole-state comparisons."""
    select = ", ".join(("order_id",) + cols)
    return {r[0]: tuple(r[1:]) for r in
            con.execute(f"SELECT {select} FROM orders ORDER BY order_id")}


@pytest.fixture
def con() -> sqlite3.Connection:
    connection = connect(T)
    yield connection
    connection.close()
