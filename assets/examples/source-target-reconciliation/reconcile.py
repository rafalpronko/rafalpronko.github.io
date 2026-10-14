"""Source-to-target reconciliation for one closed event-time window.

Rows are compared as multisets with collections.Counter, keys are
checked for uniqueness on both sides and the publish decision fails
closed. Python 3.11, standard library only, amounts in integer cents.
Timestamps and window bounds must use canonical timezone.utc.

Run:   python3.11 reconcile.py
Tests: python3.11 -m unittest -v
"""
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

UTC = timezone.utc


class Row(NamedTuple):
    row_id: str
    event_at: datetime  # event time, not load time; canonical timezone.utc
    segment: str
    currency: str  # compared as-is, never converted
    amount_cents: int


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    duplicate_reference_keys: dict[str, int] = field(default_factory=dict)
    duplicate_target_keys: dict[str, int] = field(default_factory=dict)
    only_reference: list[Row] = field(default_factory=list)
    only_target: list[Row] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    publish: bool
    reasons: tuple[str, ...]
    report: Report


def is_utc(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is timezone.utc


def check_window(start: datetime, end: datetime) -> None:
    if not (is_utc(start) and is_utc(end)):
        raise ValueError("window bounds must be UTC datetimes")
    if not start < end:
        raise ValueError("window must satisfy start < end")


def find_problems(rows: object, side: str, start: datetime, end: datetime,
                  currencies: frozenset[str]) -> list[str]:
    if type(rows) is not list:  # a fixed, materialized snapshot
        return [f"{side}: expected a list of rows, got {type(rows).__name__}"]
    found = []
    for i, row in enumerate(rows):
        where = f"{side}[{i}]"
        if not isinstance(row, tuple) or len(row) != len(Row._fields):
            found.append(f"{where}: expected a {len(Row._fields)}-field row")
            continue
        row_id, event_at, segment, currency, amount = row
        for name, value in (("row_id", row_id), ("segment", segment)):
            if type(value) is not str or not value or value != value.strip():
                found.append(f"{where}: {name} must be a trimmed, non-empty "
                             "str")
        if not is_utc(event_at):
            found.append(f"{where}: event_at must be a UTC datetime")
        elif not start <= event_at < end:
            found.append(f"{where}: event_at outside [start, end)")
        if type(currency) is not str or currency not in currencies:
            found.append(f"{where}: currency {currency!r} not allowed")
        if type(amount) is not int:  # rejects bool, float, Decimal, None
            found.append(f"{where}: amount_cents must be int")
    return found


def duplicates(keys: Counter[str]) -> dict[str, int]:
    return {key: n for key, n in sorted(keys.items()) if n > 1}


def counted(keys: dict[str, int]) -> list[str]:
    return [f"{key} x{n}" for key, n in keys.items()]


def reconcile(reference: list[Row], target: list[Row], start: datetime,
              end: datetime, currencies: frozenset[str]) -> Report:
    check_window(start, end)
    problems = (find_problems(reference, "reference", start, end, currencies)
                + find_problems(target, "target", start, end, currencies))
    if problems:
        return Report(problems=problems)
    ref_rows = Counter(Row._make(row) for row in reference)
    tgt_rows = Counter(Row._make(row) for row in target)
    ref_keys = Counter(row[0] for row in reference)
    tgt_keys = Counter(row[0] for row in target)
    return Report(
        missing_keys=sorted(ref_keys.keys() - tgt_keys.keys()),
        extra_keys=sorted(tgt_keys.keys() - ref_keys.keys()),
        duplicate_reference_keys=duplicates(ref_keys),
        duplicate_target_keys=duplicates(tgt_keys),
        only_reference=sorted((ref_rows - tgt_rows).elements()),
        only_target=sorted((tgt_rows - ref_rows).elements()),
    )


def gate(reference: list[Row], target: list[Row], start: datetime,
         end: datetime, *, currencies: frozenset[str] = frozenset({"EUR"}),
         allow_empty: bool = False) -> Decision:
    report = reconcile(reference, target, start, end, currencies)
    reasons = list(report.problems)
    if not report.problems and not reference and not allow_empty:
        reasons.append("reference window is empty and allow_empty is False")
    key_checks = [
        ("missing from target", report.missing_keys),
        ("unexpected in target", report.extra_keys),
        ("duplicated in reference", counted(report.duplicate_reference_keys)),
        ("duplicated in target", counted(report.duplicate_target_keys)),
    ]
    reasons += [f"keys {text}: {', '.join(keys[:5])}"
                for text, keys in key_checks if keys]
    row_checks = [
        ("reference row(s) without exact match", report.only_reference),
        ("target row(s) without exact match", report.only_target),
    ]
    reasons += [f"{len(rows)} {text}" for text, rows in row_checks if rows]
    return Decision(not reasons, tuple(reasons), report)


def control_totals(rows: list[Row]) -> dict[tuple[str, str], tuple[int, int]]:
    """The weak check: row count and cents per (segment, currency)."""
    totals: dict[tuple[str, str], tuple[int, int]] = {}
    for _, _, segment, currency, cents in rows:
        count, total = totals.get((segment, currency), (0, 0))
        totals[(segment, currency)] = (count + 1, total + cents)
    return totals


START = datetime(2024, 4, 1, tzinfo=UTC)
END = START + timedelta(days=1)


def at(hour: int, minute: int = 0) -> datetime:
    return START + timedelta(hours=hour, minutes=minute)


REFERENCE = [
    Row("t1", at(0, 5), "retail", "EUR", 4900),
    Row("t2", at(3, 10), "retail", "EUR", 4900),
    Row("t3", at(7, 45), "retail", "EUR", 12550),
    Row("t4", at(9, 0), "retail", "EUR", 4900),
    Row("t5", at(12, 30), "business", "EUR", 120000),
    Row("t6", at(18, 15), "business", "EUR", 30000),
]
t1, t2, t3, t4, t5, t6 = REFERENCE

SCENARIOS = [  # (name, reference, target, allow_empty, should_publish)
    ("same rows, reordered", REFERENCE, REFERENCE[::-1], False, True),
    ("t4 lost, t2 loaded twice", REFERENCE,
     [t1, t2, t2, t3, t5, t6], False, False),
    ("offsetting errors in one segment", REFERENCE,
     [t1, t2, t3, t4, t5._replace(amount_cents=119000),
      t6._replace(amount_cents=31000)], False, False),
    ("t3 reloaded under key t9", REFERENCE,
     [t1, t2, t3._replace(row_id="t9"), t4, t5, t6], False, False),
    ("amount arrives as float", REFERENCE,
     [t1, t2, t3, t4, t5, t6._replace(amount_cents=30000.0)], False, False),
    ("empty window, default policy", [], [], False, False),
    ("empty window, allow_empty=True", [], [], True, True),
]


def show(row: Row) -> str:
    return f"{row.row_id} {row.segment} {row.currency} {row.amount_cents}"


def main() -> int:
    print(f"window [{START.isoformat()}, {END.isoformat()}) event time, UTC")
    unexpected = 0
    for name, reference, target, allow_empty, should_publish in SCENARIOS:
        decision = gate(reference, target, START, END, allow_empty=allow_empty)
        weak = control_totals(reference) == control_totals(target)
        verdict = "PUBLISH" if decision.publish else "BLOCK"
        print(f"\n{name}\n  count+sum per segment equal: {weak}"
              f"\n  gate: {verdict}")
        for reason in decision.reasons:
            print(f"  - {reason}")
        report = decision.report
        for row in report.only_reference:
            print(f"    only in reference: {show(row)}")
        for row in report.only_target:
            print(f"    only in target:    {show(row)}")
        unexpected += decision.publish != should_publish
    print(f"\nunexpected decisions: {unexpected}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
