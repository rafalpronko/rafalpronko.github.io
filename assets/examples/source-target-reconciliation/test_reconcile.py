"""Tests for reconcile.py. Run: python3.11 -m unittest -v"""
import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reconcile import (END, REFERENCE, START, Row, control_totals, gate,
                       is_utc, main, t1, t2, t3, t4, t5, t6)


class WeakChecksMissDifferentRows(unittest.TestCase):
    def assert_blocked_although_totals_match(self, target):
        self.assertEqual(len(target), len(REFERENCE))
        self.assertEqual(control_totals(target), control_totals(REFERENCE))
        decision = gate(REFERENCE, target, START, END)
        self.assertFalse(decision.publish)
        return decision.report

    def test_reordered_rows_publish(self):
        decision = gate(REFERENCE, REFERENCE[::-1], START, END)
        self.assertTrue(decision.publish)
        self.assertEqual(decision.reasons, ())

    def test_loss_plus_duplication(self):
        target = [t1, t2, t2, t3, t5, t6]
        report = self.assert_blocked_although_totals_match(target)
        reasons = gate(REFERENCE, target, START, END).reasons
        self.assertIn("keys missing from target: t4", reasons)
        self.assertIn("keys duplicated in target: t2 x2", reasons)
        self.assertEqual(report.missing_keys, ["t4"])
        self.assertEqual(report.duplicate_target_keys, {"t2": 2})
        self.assertEqual(report.only_reference, [t4])
        self.assertEqual(report.only_target, [t2])

    def test_offsetting_errors_in_one_segment(self):
        target = [t1, t2, t3, t4, t5._replace(amount_cents=119000),
                  t6._replace(amount_cents=31000)]
        report = self.assert_blocked_although_totals_match(target)
        self.assertEqual(report.missing_keys, [])
        self.assertEqual(report.extra_keys, [])
        self.assertEqual(report.only_reference, [t5, t6])
        self.assertEqual(len(report.only_target), 2)

    def test_key_reassignment(self):
        moved = t3._replace(row_id="t9")
        report = self.assert_blocked_although_totals_match(
            [t1, t2, moved, t4, t5, t6])
        self.assertEqual(report.missing_keys, ["t3"])
        self.assertEqual(report.extra_keys, ["t9"])
        self.assertEqual(report.only_target, [moved])

    def test_counter_keeps_multiplicity(self):
        reference = [t1, t1]  # same row twice, e.g. a grain mismatch
        decision = gate(reference, [t1], START, END)
        self.assertFalse(decision.publish)
        self.assertEqual(decision.report.duplicate_reference_keys, {"t1": 2})
        self.assertEqual(decision.report.only_reference, [t1])

    def test_duplicate_reference_key_blocks_even_if_sides_agree(self):
        clash = t2._replace(row_id="t1")
        decision = gate([t1, clash], [clash, t1], START, END)
        self.assertFalse(decision.publish)
        self.assertEqual(decision.report.duplicate_reference_keys, {"t1": 2})
        self.assertEqual(decision.report.duplicate_target_keys, {"t1": 2})


class EmptyWindowPolicy(unittest.TestCase):
    def test_empty_window_blocks_by_default(self):
        decision = gate([], [], START, END)
        self.assertFalse(decision.publish)
        self.assertIn("allow_empty", decision.reasons[0])

    def test_empty_window_publishes_only_when_allowed(self):
        self.assertTrue(gate([], [], START, END, allow_empty=True).publish)

    def test_allow_empty_does_not_hide_extra_target_rows(self):
        decision = gate([], [t1], START, END, allow_empty=True)
        self.assertFalse(decision.publish)
        self.assertEqual(decision.report.extra_keys, ["t1"])

    def test_empty_target_is_a_loss(self):
        decision = gate(REFERENCE, [], START, END, allow_empty=True)
        self.assertFalse(decision.publish)
        self.assertEqual(len(decision.report.missing_keys), 6)


class InvalidInputFailsClosed(unittest.TestCase):
    def assert_invalid(self, target, fragment):
        decision = gate(REFERENCE, target, START, END)
        self.assertFalse(decision.publish)
        self.assertTrue(any(fragment in p for p in decision.report.problems),
                        decision.report.problems)

    def replace_t1(self, **changes):
        return [t1._replace(**changes), t2, t3, t4, t5, t6]

    def test_bad_amounts(self):
        for amount in (True, 4900.0, None, "4900"):
            with self.subTest(amount=amount):
                self.assert_invalid(self.replace_t1(amount_cents=amount),
                                    "amount_cents must be int")

    def test_bad_keys_and_segments(self):
        for changes in ({"row_id": ""}, {"row_id": " t1"}, {"row_id": 1},
                        {"segment": None}):
            with self.subTest(changes=changes):
                self.assert_invalid(self.replace_t1(**changes),
                                    "must be a trimmed, non-empty str")

    def test_unknown_or_unhashable_currency(self):
        for currency in ("PLN", "eur", None, ["EUR"]):
            with self.subTest(currency=currency):
                self.assert_invalid(self.replace_t1(currency=currency),
                                    "not allowed")

    def test_naive_or_non_utc_timestamps(self):
        naive = t1.event_at.replace(tzinfo=None)
        plus_two = t1.event_at.astimezone(timezone(timedelta(hours=2)))
        for event_at in (naive, plus_two, "2024-04-01T00:05:00Z"):
            with self.subTest(event_at=event_at):
                self.assert_invalid(self.replace_t1(event_at=event_at),
                                    "UTC datetime")

    def test_wrong_shape_and_container(self):
        self.assert_invalid([("t1", 4900)], "5-field row")
        self.assert_invalid(iter(REFERENCE), "expected a list")

    def test_window_is_half_open(self):
        self.assertTrue(gate([t1._replace(event_at=START)],
                             [t1._replace(event_at=START)],
                             START, END).publish)
        self.assert_invalid(self.replace_t1(event_at=END), "outside")
        self.assert_invalid(
            self.replace_t1(event_at=START - timedelta(microseconds=1)),
            "outside")

    def test_bad_window_raises(self):
        naive = datetime(2024, 4, 1)
        for start, end in ((END, START), (START, START), (naive, END)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    gate(REFERENCE, REFERENCE, start, end)


class CanonicalUTCContract(unittest.TestCase):
    def london(self):
        try:
            return ZoneInfo("Europe/London")
        except ZoneInfoNotFoundError:
            self.skipTest("Europe/London tzdata unavailable; DST regression "
                          "not executed (no dependencies installed)")

    def fold_row(self):
        canonical = datetime(2023, 10, 29, 1, 30, tzinfo=timezone.utc)
        raw = canonical.astimezone(self.london())
        self.assertEqual(raw.fold, 1)
        self.assertEqual(raw.utcoffset(), timedelta(0))
        self.assertEqual(raw.timestamp(), canonical.timestamp())
        row = t1._replace(event_at=canonical)
        start = datetime(2023, 10, 29, tzinfo=timezone.utc)
        return row, raw, start, start + timedelta(days=1)

    def fold_bounds(self):
        london = self.london()
        start = datetime(2023, 10, 29, 1, tzinfo=london, fold=1)
        end = datetime(2023, 10, 29, 2, tzinfo=london)
        self.assertEqual(start.utcoffset(), timedelta(0))
        self.assertEqual(end.utcoffset(), timedelta(0))
        return start, end

    def test_requires_exact_datetime_and_utc_singleton(self):
        class DateTimeSubclass(datetime):
            pass

        named_zero = timezone(timedelta(0), "GMT")
        self.assertIsNot(named_zero, timezone.utc)
        self.assertTrue(is_utc(t1.event_at))
        for value in (t1.event_at.replace(tzinfo=named_zero),
                      t1.event_at.replace(tzinfo=None),
                      DateTimeSubclass(2024, 4, 1, tzinfo=timezone.utc),
                      "2024-04-01T00:05:00Z", None):
            with self.subTest(value=value):
                self.assertFalse(is_utc(value))

    def test_london_autumn_fold_timestamp_is_rejected(self):
        row, raw, start, end = self.fold_row()
        self.assertFalse(is_utc(raw))
        for reference, target, side in (
                ([row], [row._replace(event_at=raw)], "target"),
                ([row._replace(event_at=raw)], [row], "reference")):
            with self.subTest(side=side):
                decision = gate(reference, target, start, end)
                self.assertFalse(decision.publish)
                self.assertEqual(decision.report.problems,
                                 [f"{side}[0]: event_at must be a UTC datetime"])
                self.assertEqual(decision.report.only_reference, [])
                self.assertEqual(decision.report.only_target, [])

    def test_normalized_london_autumn_fold_timestamp_matches(self):
        row, raw, start, end = self.fold_row()
        normalized = raw.astimezone(timezone.utc)
        self.assertIs(normalized.tzinfo, timezone.utc)
        self.assertEqual(normalized, row.event_at)
        self.assertEqual(hash(normalized), hash(row.event_at))
        decision = gate([row], [row._replace(event_at=normalized)], start, end)
        self.assertTrue(decision.publish)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(decision.report.only_reference, [])
        self.assertEqual(decision.report.only_target, [])

    def test_london_autumn_fold_raw_bounds_are_rejected(self):
        row, _, _, _ = self.fold_row()
        raw_start, raw_end = self.fold_bounds()
        start = raw_start.astimezone(timezone.utc)
        end = raw_end.astimezone(timezone.utc)
        for bounds in ((raw_start, end), (start, raw_end),
                       (raw_start, raw_end)):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "UTC datetimes"):
                    gate([row], [row], *bounds)

    def test_normalized_london_autumn_fold_bounds_publish(self):
        row, _, _, _ = self.fold_row()
        raw_start, raw_end = self.fold_bounds()
        start = raw_start.astimezone(timezone.utc)
        end = raw_end.astimezone(timezone.utc)
        self.assertIs(start.tzinfo, timezone.utc)
        self.assertIs(end.tzinfo, timezone.utc)
        decision = gate([row], [row], start, end)
        self.assertTrue(decision.publish)
        self.assertEqual(decision.reasons, ())


class Demo(unittest.TestCase):
    def test_every_scenario_gets_the_expected_decision(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(), 0)
        self.assertIn("unexpected decisions: 0", out.getvalue())

    def test_row_is_plain_tuple_compatible(self):
        as_tuples = [tuple(row) for row in REFERENCE]
        self.assertTrue(gate(as_tuples, REFERENCE, START, END).publish)
        self.assertEqual(Row._make(as_tuples[0]), t1)


if __name__ == "__main__":
    unittest.main()
