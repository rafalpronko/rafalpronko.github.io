"""Row- and envelope-level contract checks (no database involved)."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from fixtures import canonical_batch, consistent_unit_lie_batch
from invoice_contract import rounded_tax, validate_batch

Mutation = Callable[[dict[str, Any]], None]


def errors_after(mutate: Mutation) -> list[str]:
    batch = canonical_batch()
    mutate(batch)
    return validate_batch(batch)


def has(errors: list[str], needle: str) -> bool:
    return any(needle in e for e in errors)


def test_canonical_batch_is_valid() -> None:
    assert validate_batch(canonical_batch()) == []


@pytest.mark.parametrize(("net", "bp", "expected"), [
    (4_550, 500, 228),    # 227.5 rounds half up
    (1_999, 1000, 200),   # 199.9
    (10_000, 2000, 2_000),
    (0, 2000, 0),
])
def test_rounded_tax_is_half_up(net: int, bp: int, expected: int) -> None:
    assert rounded_tax(net, bp) == expected


@pytest.mark.parametrize("envelope", [None, [], "rows", 42, ("a", 1)])
def test_non_dict_envelope_is_rejected_without_exception(envelope: object) -> None:
    errors = validate_batch(envelope)
    assert has(errors, "envelope: object required")


def test_missing_envelope_field_is_rejected() -> None:
    assert has(errors_after(lambda b: b.pop("amount_basis")),
               "envelope.amount_basis: missing")


def test_unexpected_envelope_field_is_rejected() -> None:
    assert has(errors_after(lambda b: b.update(unit_note="cents, mostly")),
               "envelope.unit_note: unexpected field")


def test_declared_unit_change_is_rejected() -> None:
    assert has(errors_after(lambda b: b.update(unit="major")), "envelope.unit")


@pytest.mark.parametrize("rows", [None, {}, "L-1", ()])
def test_rows_must_be_a_list(rows: object) -> None:
    assert has(errors_after(lambda b: b.update(rows=rows)), "envelope.rows")


def test_empty_batch_is_rejected() -> None:
    assert has(errors_after(lambda b: b.update(rows=[])),
               "envelope.rows: at least one row")


def test_non_dict_row_is_rejected() -> None:
    assert has(errors_after(lambda b: b["rows"].append(["L-9"])),
               "rows[3]: object required")


@pytest.mark.parametrize("mutate", [
    lambda b: b["rows"][0].pop("tax_minor"),
    lambda b: b["rows"][0].update(discount_minor=0),
])
def test_row_fields_must_be_exact(mutate: Mutation) -> None:
    assert has(errors_after(mutate), "rows[0]: fields must be exactly")


@pytest.mark.parametrize("bad_id", ["", "   ", " L-1", "L-1\n", 17, None])
def test_identifiers_must_be_clean_nonempty_strings(bad_id: object) -> None:
    errors = errors_after(lambda b: b["rows"][0].update(line_id=bad_id))
    assert has(errors, "rows[0].line_id")


@pytest.mark.parametrize(("value", "needle"), [
    (True, "got bool"),
    (20.0, "got float"),
    ("2000", "got str"),
])
def test_amounts_must_be_exact_int(value: object, needle: str) -> None:
    errors = errors_after(lambda b: b["rows"][0].update(tax_bp=value))
    assert has(errors, needle)


@pytest.mark.parametrize(("field", "value"), [
    ("net_minor", -1),
    ("net_minor", 10**30),
    ("tax_minor", 2**63),
    ("gross_minor", -(2**70)),
])
def test_amounts_are_bounded_before_any_arithmetic(field: str, value: int) -> None:
    errors = errors_after(lambda b: b["rows"][0].update({field: value}))
    assert has(errors, f"rows[0].{field}: out of range")


def test_false_rate_in_zero_rate_row_is_rejected_only_as_bool() -> None:
    # With isinstance(), this row would satisfy every rule as a 0% rate.
    errors = errors_after(lambda b: b["rows"][0].update(
        tax_bp=False, tax_minor=0, gross_minor=10_000))
    assert errors == ["rows[0].tax_bp: int required, got bool"]


@pytest.mark.parametrize(("length", "valid"), [(64, True), (65, False)])
def test_identifier_length_boundary(length: int, valid: bool) -> None:
    errors = errors_after(lambda b: b["rows"][0].update(line_id="L" * length))
    assert (errors == []) is valid


@pytest.mark.parametrize("bad_id", ["", " INV-1", "INV-1\t", 7])
def test_invoice_id_follows_identifier_rules(bad_id: object) -> None:
    errors = errors_after(lambda b: b["rows"][0].update(invoice_id=bad_id))
    assert has(errors, "rows[0].invoice_id")


def test_disallowed_tax_rate_is_rejected() -> None:
    assert has(errors_after(lambda b: b["rows"][0].update(tax_bp=2300)),
               "rows[0].tax_bp: 2300 not allowed")


def test_truncated_tax_is_rejected() -> None:
    errors = errors_after(
        lambda b: b["rows"][2].update(tax_minor=227, gross_minor=4_777))
    assert has(errors, "rows[2].tax_minor: 227 != rounded_tax 228")


def test_net_sent_as_gross_is_rejected() -> None:
    errors = errors_after(lambda b: b["rows"][0].update(gross_minor=10_000))
    assert has(errors, "rows[0].gross_minor")


def test_duplicate_keys_between_valid_rows() -> None:
    errors = errors_after(lambda b: b["rows"][1].update(line_id="L-1"))
    assert has(errors, "rows[1].line_id: duplicate of rows[0]")


def test_duplicate_is_reported_even_if_one_duplicate_is_malformed() -> None:
    def mutate(b: dict[str, Any]) -> None:
        b["rows"][1].update(line_id="L-1", net_minor=19.99)

    errors = errors_after(mutate)
    assert has(errors, "rows[1].net_minor")
    assert has(errors, "rows[1].line_id: duplicate of rows[0]")


def test_all_row_errors_are_collected() -> None:
    def mutate(b: dict[str, Any]) -> None:
        b["rows"][0].update(tax_bp=True)
        b["rows"][2].update(tax_minor=227, gross_minor=4_777)

    errors = errors_after(mutate)
    assert has(errors, "rows[0].tax_bp") and has(errors, "rows[2].tax_minor")


def test_limit_consistent_unit_lie_passes_hard_checks() -> None:
    # Documents a blind spot; this is not a detection.
    assert validate_batch(consistent_unit_lie_batch()) == []
