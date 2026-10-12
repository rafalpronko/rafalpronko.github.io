"""Hard data contract for synthetic invoice lines (invoice_lines.v2).

Written for Python 3.11.x, standard library only. validate_batch() never
raises on malformed input: it returns human-readable errors, and an empty
list means the batch satisfies every hard rule of this contract.

The tax rates and the half-up rounding rule are choices of this synthetic
contract. They are not a statement about any jurisdiction's tax law.
"""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = "invoice_lines.v2"
EXPECTED_ENVELOPE: dict[str, str] = {
    "contract_version": CONTRACT_VERSION,
    "currency": "EUR",
    "unit": "minor",  # integer cents
    "amount_basis": "net_tax_gross",
}
ENVELOPE_FIELDS = frozenset({*EXPECTED_ENVELOPE, "rows"})
ID_FIELDS = ("line_id", "invoice_id")
AMOUNT_FIELDS = ("net_minor", "tax_bp", "tax_minor", "gross_minor")
ROW_FIELDS = frozenset({*ID_FIELDS, *AMOUNT_FIELDS})
ALLOWED_TAX_BP = frozenset({0, 500, 1000, 2000})  # basis points: 2000 = 20%
MAX_ID_LENGTH = 64
# Inclusive bounds. Every amount is non-negative in this contract version,
# and every bound sits far inside SQLite's signed 64-bit INTEGER range.
AMOUNT_MAX: dict[str, int] = {
    "net_minor": 10**12,
    "tax_bp": 10_000,
    "tax_minor": 10**12,
    "gross_minor": 2 * 10**12,
}


class ContractViolation(Exception):
    """The batch broke at least one hard rule; nothing may be written."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"{len(errors)} contract violation(s)")
        self.errors: tuple[str, ...] = tuple(errors)


def rounded_tax(net_minor: int, tax_bp: int) -> int:
    """Tax in minor units, rounded half up. Defined for net_minor >= 0 only."""
    return (net_minor * tax_bp + 5_000) // 10_000


def validate_batch(envelope: object) -> list[str]:
    """Return the violations that can be checked independently.

    Envelope errors stop before the rows are read; within a row, malformed
    values skip the relation checks that depend on them. Empty list = valid.
    """
    if not isinstance(envelope, dict):
        return [f"envelope: object required, got {type(envelope).__name__}"]
    errors = check_envelope(envelope)
    if errors:
        return errors
    rows: list[object] = envelope["rows"]
    for i, row in enumerate(rows):
        errors.extend(check_row(i, row))
    errors.extend(duplicate_key_errors(rows))
    return errors


def check_envelope(envelope: dict[Any, Any]) -> list[str]:
    keys = set(envelope)
    errors = [f"envelope.{k}: missing" for k in sorted(ENVELOPE_FIELDS - keys)]
    errors += [f"envelope.{k}: unexpected field"
               for k in sorted(keys - ENVELOPE_FIELDS, key=repr)]
    for key, expected in EXPECTED_ENVELOPE.items():
        if key in envelope and envelope[key] != expected:
            errors.append(f"envelope.{key}: expected {expected!r}, "
                          f"got {envelope[key]!r}")
    if "rows" in envelope:
        rows = envelope["rows"]
        if type(rows) is not list:
            errors.append(f"envelope.rows: list required, "
                          f"got {type(rows).__name__}")
        elif not rows:
            errors.append("envelope.rows: at least one row required")
    return errors


def check_row(i: int, row: object) -> list[str]:
    where = f"rows[{i}]"
    if not isinstance(row, dict):
        return [f"{where}: object required, got {type(row).__name__}"]
    if set(row) != ROW_FIELDS:
        return [f"{where}: fields must be exactly {sorted(ROW_FIELDS)}"]
    errors = [error for key in ID_FIELDS
              if (error := identifier_error(f"{where}.{key}", row[key]))]
    errors += amount_errors(where, row)
    if errors:  # relations are meaningless on malformed values
        return errors
    return relation_errors(where, row)


def identifier_error(where: str, value: object) -> str | None:
    if type(value) is not str:
        return f"{where}: str required, got {type(value).__name__}"
    if not value.strip():
        return f"{where}: empty or whitespace-only"
    if value != value.strip():
        return f"{where}: leading or trailing whitespace"
    if len(value) > MAX_ID_LENGTH:
        return f"{where}: longer than {MAX_ID_LENGTH} characters"
    return None


def amount_errors(where: str, row: dict[str, Any]) -> list[str]:
    errors = []
    for key in AMOUNT_FIELDS:
        value = row[key]
        if type(value) is not int:  # rejects bool (an int subclass) and float
            errors.append(f"{where}.{key}: int required, "
                          f"got {type(value).__name__}")
        elif not 0 <= value <= AMOUNT_MAX[key]:
            errors.append(f"{where}.{key}: out of range "
                          f"[0, {AMOUNT_MAX[key]}]")
    return errors


def relation_errors(where: str, row: dict[str, Any]) -> list[str]:
    net, bp, tax = row["net_minor"], row["tax_bp"], row["tax_minor"]
    errors = []
    if bp not in ALLOWED_TAX_BP:
        errors.append(f"{where}.tax_bp: {bp} not allowed")
    elif tax != rounded_tax(net, bp):
        errors.append(f"{where}.tax_minor: {tax} != rounded_tax "
                      f"{rounded_tax(net, bp)}")
    if row["gross_minor"] != net + tax:
        errors.append(f"{where}.gross_minor: != net_minor + tax_minor")
    return errors


def duplicate_key_errors(rows: list[object]) -> list[str]:
    """Report repeated line_id values, even when the row is otherwise bad."""
    first_seen: dict[str, int] = {}
    errors = []
    for i, row in enumerate(rows):
        key = row.get("line_id") if isinstance(row, dict) else None
        if type(key) is not str or identifier_error("", key):
            continue  # already reported by check_row
        if key in first_seen:
            errors.append(f"rows[{i}].line_id: duplicate of "
                          f"rows[{first_seen[key]}]")
        else:
            first_seen[key] = i
    return errors
