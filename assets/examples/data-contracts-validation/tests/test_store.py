"""Publishing into SQLite: validate first, write the whole batch or nothing."""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing
from typing import Any, Callable

import pytest

from fixtures import (SYNTHETIC_BASELINE_MEDIAN_NET, canonical_batch,
                      consistent_unit_lie_batch, line)
from invoice_contract import AMOUNT_MAX, EXPECTED_ENVELOPE, ContractViolation
from invoice_store import (connect, count_rows, distribution_advisories,
                           publish)

BASELINE = SYNTHETIC_BASELINE_MEDIAN_NET
NET_MAX = AMOUNT_MAX["net_minor"]
MODERN_RUNTIME = (3, 14, 4, "final", 0)


@pytest.fixture
def con() -> Iterator[sqlite3.Connection]:
    with closing(connect()) as connection:
        yield connection


def test_canonical_batch_is_published(con: sqlite3.Connection) -> None:
    result = publish(con, canonical_batch(), BASELINE)
    assert result.rows_written == 3
    assert result.advisories == []
    assert count_rows(con) == 3
    assert not con.in_transaction


@pytest.mark.parametrize("mutate", [
    lambda b: b.update(unit="major"),
    lambda b: b["rows"][1].update(net_minor=19.99),       # bad second row
    lambda b: b["rows"][2].update(tax_minor=227, gross_minor=4_777),
    lambda b: b["rows"][0].update(net_minor=2**63),       # would overflow
    lambda b: b["rows"][1].update(line_id="L-1"),
])
def test_contract_violation_writes_nothing(
        con: sqlite3.Connection, mutate: Callable[[dict[str, Any]], None]) -> None:
    batch = canonical_batch()
    mutate(batch)
    with pytest.raises(ContractViolation):
        publish(con, batch, BASELINE)
    assert count_rows(con) == 0


def test_non_dict_envelope_raises_contract_violation(
        con: sqlite3.Connection) -> None:
    with pytest.raises(ContractViolation) as exc:
        publish(con, ["not", "an", "envelope"], BASELINE)
    assert any("envelope: object required" in e for e in exc.value.errors)


def test_target_key_conflict_rolls_back_whole_batch(
        con: sqlite3.Connection) -> None:
    publish(con, canonical_batch(), BASELINE)
    retry = {**EXPECTED_ENVELOPE, "rows": [line("L-9", 100, 0, 0, 100),
                                           line("L-1", 100, 0, 0, 100)]}
    with pytest.raises(sqlite3.IntegrityError):
        publish(con, retry, BASELINE)
    l9 = con.execute(
        "SELECT count(*) FROM invoice_lines WHERE line_id = 'L-9'").fetchone()[0]
    assert l9 == 0 and count_rows(con) == 3


def test_exact_replay_leaves_table_unchanged_but_raises(con: sqlite3.Connection) -> None:
    publish(con, canonical_batch(), BASELINE)
    with pytest.raises(sqlite3.IntegrityError):
        publish(con, canonical_batch(), BASELINE)
    assert count_rows(con) == 3


def test_autocommit_connection_is_refused(con: sqlite3.Connection) -> None:
    con.isolation_level = None
    with pytest.raises(ValueError, match="isolation_level"):
        publish(con, canonical_batch(), BASELINE)
    assert count_rows(con) == 0


def test_other_runtime_is_refused_before_writes(
        con: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", MODERN_RUNTIME)
    with pytest.raises(RuntimeError, match="Python 3.11.x required"):
        publish(con, canonical_batch(), BASELINE)
    monkeypatch.undo()
    assert count_rows(con) == 0


def test_connect_refuses_other_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", MODERN_RUNTIME)
    with pytest.raises(RuntimeError, match="Python 3.11.x required"):
        connect()


@pytest.mark.parametrize("baseline", [
    0, -1, True, 4550.0, "4550", NET_MAX + 1,
    pytest.param(10**400, id="10**400")])
def test_invalid_baseline_is_refused_before_writes(
        con: sqlite3.Connection, baseline: object) -> None:
    with pytest.raises(ValueError, match="baseline_median_net"):
        publish(con, canonical_batch(), baseline)  # type: ignore[arg-type]
    assert count_rows(con) == 0


@pytest.mark.parametrize(("baseline", "band"), [
    (1, "[0.1, 10]"), (NET_MAX, "[100000000000, 10000000000000]")])
def test_boundary_baseline_returns_after_commit(
        con: sqlite3.Connection, baseline: int, band: str) -> None:
    result = publish(con, canonical_batch(), baseline)
    assert result.rows_written == 3 and count_rows(con) == 3
    assert len(result.advisories) == 1 and band in result.advisories[0]


@pytest.mark.parametrize(("nets", "baseline"), [
    ([0], 1), ([NET_MAX], 1), ([0, NET_MAX], NET_MAX), ([NET_MAX] * 2, 1)])
def test_advisories_never_raise_on_accepted_inputs(
        nets: list[int], baseline: int) -> None:
    assert len(distribution_advisories(
        [{"net_minor": n} for n in nets], baseline)) <= 1


def test_advisory_reports_even_row_median() -> None:
    advisories = distribution_advisories(
        [{"net_minor": 1}, {"net_minor": 2}], 100)
    assert advisories == ["median net_minor 1.5 outside advisory band "
                          "[10, 1000] around baseline 100"]


@pytest.mark.parametrize("baseline", [
    0, NET_MAX + 1, pytest.param(10**400, id="10**400")])
def test_advisory_helper_enforces_its_own_baseline_contract(
        baseline: int) -> None:
    with pytest.raises(ValueError, match="baseline_median_net"):
        distribution_advisories([{"net_minor": 1}], baseline)


def test_open_caller_transaction_is_refused(con: sqlite3.Connection) -> None:
    con.execute("INSERT INTO invoice_lines VALUES "
                "('X-1', 'INV-X', 100, 0, 0, 100, 'invoice_lines.v2')")
    assert con.in_transaction
    with pytest.raises(RuntimeError, match="open transaction"):
        publish(con, canonical_batch(), BASELINE)
    con.rollback()
    assert count_rows(con) == 0


def test_limit_consistent_lie_is_published_with_advisory(
        con: sqlite3.Connection) -> None:
    result = publish(con, consistent_unit_lie_batch(), BASELINE)
    assert count_rows(con) == 3               # not blocked
    assert len(result.advisories) == 1        # only a signal
    assert "median net_minor 46" in result.advisories[0]


def test_integer_column_alone_stores_text() -> None:
    with closing(sqlite3.connect(":memory:")) as loose:
        loose.execute("CREATE TABLE t (x INTEGER)")
        loose.execute("INSERT INTO t VALUES ('12.5 EUR')")
        assert loose.execute("SELECT typeof(x) FROM t").fetchone()[0] == "text"


def test_typeof_check_blocks_text_amount(con: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO invoice_lines VALUES "
                        "('X-2', 'INV-X', '12.5', 0, 0, '12.5', 'v')")
    assert count_rows(con) == 0


def test_integer_affinity_converts_lossless_text(
        con: sqlite3.Connection) -> None:
    # The CHECK refuses text that cannot become an integer, not every string.
    with con:
        con.execute("INSERT INTO invoice_lines VALUES "
                    "('X-3', 'INV-X', '12', 0, 0, '12.0', 'v')")
    assert con.execute("SELECT typeof(net_minor), typeof(gross_minor) "
                       "FROM invoice_lines").fetchone() == ("integer", "integer")


@pytest.mark.parametrize(("line_id", "invoice_id"), [
    (" X-4", "INV-X"), ("X-4", " INV-X"), ("X-4", "")])
def test_schema_rejects_untrimmed_or_empty_ids(
        con: sqlite3.Connection, line_id: str, invoice_id: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO invoice_lines VALUES "
                        "(?, ?, 100, 0, 0, 100, 'v')", (line_id, invoice_id))
    assert count_rows(con) == 0


def test_python_int_beyond_64_bits_cannot_be_bound() -> None:
    # Why the validator bounds integers before they reach sqlite3.
    with closing(sqlite3.connect(":memory:")) as raw:
        with pytest.raises(OverflowError):
            raw.execute("SELECT ?", (2**63,))
