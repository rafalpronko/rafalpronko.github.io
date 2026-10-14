import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import replay_eval as ev

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def record(case_id="c1", **overrides):
    base = {"case_id": case_id, "slice": "s", "expected": "x", "status": "ok",
            "output": "x", "elapsed_ms": 100, "input_tokens": 10,
            "output_tokens": 5}
    base.update(overrides)
    return base


class TempLogs(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def write(self, name, rows):
        path = self.dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path


class ValidationTests(TempLogs):
    def assert_rejected(self, row, fragment):
        with self.assertRaisesRegex(ev.LogError, fragment):
            ev.load_log(self.write("bad", [row]))

    def test_duplicate_case_id_is_rejected(self):
        path = self.write("dup", [record("c1"), record("c1")])
        with self.assertRaisesRegex(ev.LogError, "duplicate case_id"):
            ev.load_log(path)

    def test_latency_must_be_finite_positive_number(self):
        for bad in (0, -5, True, "100"):
            with self.subTest(bad=bad):
                self.assert_rejected(record(elapsed_ms=bad), "elapsed_ms")

    def test_non_finite_latency_is_rejected(self):
        # json.dumps writes Infinity/NaN, which json.loads accepts by default.
        for bad in (math.inf, math.nan):
            with self.subTest(bad=bad):
                self.assert_rejected(record(elapsed_ms=bad), "finite")

    def test_unknown_status_is_rejected(self):
        self.assert_rejected(record(status="error"), "unknown status")

    def test_ok_requires_string_output(self):
        self.assert_rejected(record(output=None), "requires a string output")

    def test_missing_field_and_bad_tokens(self):
        row = record()
        del row["slice"]
        self.assert_rejected(row, "missing field")
        self.assert_rejected(record(output_tokens=-1), "token counts")

    def test_prices_are_validated_and_note_keys_skipped(self):
        path = self.dir / "prices.json"
        path.write_text(json.dumps({"_note": "x", "a": {"input_per_1k": 1,
                                                       "output_per_1k": 2}}))
        self.assertEqual(ev.load_prices(path), {"a": ev.Price(1.0, 2.0)})
        path.write_text(json.dumps({"a": {"input_per_1k": -1,
                                          "output_per_1k": 2}}))
        with self.assertRaisesRegex(ev.LogError, "finite and >= 0"):
            ev.load_prices(path)

    def test_invalid_json_and_empty_log(self):
        path = self.dir / "broken.jsonl"
        path.write_text('{"case_id": \n')
        with self.assertRaisesRegex(ev.LogError, "invalid JSON"):
            ev.load_log(path)
        with self.assertRaisesRegex(ev.LogError, "empty"):
            ev.load_log(self.write("empty", []))


class PairingTests(TempLogs):
    def test_case_sets_must_match(self):
        a = ev.load_log(self.write("a", [record("c1"), record("c2")]))
        b = ev.load_log(self.write("b", [record("c1"), record("c3")]))
        with self.assertRaisesRegex(ev.LogError, r"missing=\['c2'\] extra=\['c3'\]"):
            ev.check_paired({"a": a, "b": b})

    def test_gold_labels_must_match(self):
        a = ev.load_log(self.write("a", [record("c1", expected="x")]))
        b = ev.load_log(self.write("b", [record("c1", expected="y", output="y")]))
        with self.assertRaisesRegex(ev.LogError, "expected/slice differ"):
            ev.check_paired({"a": a, "b": b})

    def test_single_log_is_not_a_comparison(self):
        a = ev.load_log(self.write("a", [record("c1")]))
        with self.assertRaisesRegex(ev.LogError, "at least two"):
            ev.check_paired({"a": a})


class MetricTests(unittest.TestCase):
    def test_nearest_rank_definition(self):
        values = list(range(1, 21))  # 1..20
        self.assertEqual(ev.nearest_rank(values, 50), 10)  # rank ceil(10) = 10
        self.assertEqual(ev.nearest_rank(values, 95), 19)  # rank ceil(19) = 19
        self.assertEqual(ev.nearest_rank(values, 100), 20)
        # With 16 values, p95 is rank ceil(15.2) = 16: simply the maximum.
        self.assertEqual(ev.nearest_rank(list(range(1, 17)), 95), 16)
        # Always an observed value, unlike the interpolated median of 2 and 4.
        self.assertEqual(ev.nearest_rank([4, 2], 50), 2)

    def test_nearest_rank_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ev.nearest_rank([], 50)
        with self.assertRaises(ValueError):
            ev.nearest_rank([1.0], 0)

    def test_wilson_interval_stays_in_bounds(self):
        lo, hi = ev.wilson_interval(0, 5)
        self.assertEqual(lo, 0.0)
        self.assertLess(hi, 1.0)
        lo, hi = ev.wilson_interval(13, 16)
        self.assertLess(lo, 13 / 16)
        self.assertGreater(hi, 13 / 16)

    def test_exact_mcnemar(self):
        self.assertEqual(ev.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(ev.exact_mcnemar(5, 2), 58 / 128)
        self.assertAlmostEqual(ev.exact_mcnemar(6, 0), 2 / 64)


class FixtureReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = {name: ev.load_log(FIXTURES / f"{name}.jsonl")
                    for name in ("candidate_a", "candidate_b")}
        cls.prices = ev.load_prices(FIXTURES / "prices.json")
        cls.a = ev.summarize("candidate_a", cls.logs["candidate_a"],
                             cls.prices["candidate_a"])
        cls.b = ev.summarize("candidate_b", cls.logs["candidate_b"],
                             cls.prices["candidate_b"])

    def test_quality_denominator_is_all_attempts(self):
        self.assertEqual((self.a.useful, self.a.attempts), (13, 16))
        self.assertEqual((self.b.useful, self.b.attempts), (10, 16))
        self.assertEqual(self.b.counts["timeout"], 2)
        self.assertEqual(self.b.counts["malformed"], 1)

    def test_abstention_scoring(self):
        # c14/c15 correct abstentions for A; B answers c15 and c16 instead.
        self.assertEqual(self.a.by_slice["out_of_scope"], (2, 3))
        self.assertEqual(self.b.by_slice["out_of_scope"], (1, 3))
        self.assertEqual(self.a.wrong_abstentions, 0)

    def test_valid_response_latency_hides_timeouts(self):
        self.assertEqual(len(self.b.valid_latencies), 13)
        self.assertEqual(max(self.b.valid_latencies), 720)
        self.assertEqual(self.b.p95_all, 4000)
        self.assertEqual(self.a.p95_all, 1880)

    def test_cost_counts_failed_attempts(self):
        tokens_in = sum(x.input_tokens for x in self.logs["candidate_b"].values())
        tokens_out = sum(x.output_tokens for x in self.logs["candidate_b"].values())
        expected = (tokens_in * 0.1 + tokens_out * 0.4) / 1000
        self.assertAlmostEqual(self.b.cost_units, expected)
        self.assertAlmostEqual(self.b.cost_per_useful, expected / 10)

    def test_constraints_then_pareto(self):
        strict = ev.Constraints(0.70, 0.10, 3000)
        self.assertEqual(ev.violations(self.a, strict), [])
        self.assertEqual(len(ev.violations(self.b, strict)), 2)
        loose = ev.Constraints(0.50, 0.10, 5000)
        front = ev.pareto_front([self.a, self.b])
        self.assertEqual({s.name for s in front}, {"candidate_a", "candidate_b"})
        self.assertEqual(ev.violations(self.b, loose), [])

    def test_dominated_candidate_leaves_front(self):
        worse = ev.Summary("worse", 16, 9, self.b.counts, 0, 0, {}, [1.0],
                           [5000.0], 999.0, 20.0)
        front = ev.pareto_front([self.a, worse])
        self.assertEqual([s.name for s in front], ["candidate_a"])


class CliTests(TempLogs):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ev.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_fixture_report_labels_latency_source(self):
        code, out, _ = self.run_cli(str(FIXTURES / "candidate_a.jsonl"),
                                    str(FIXTURES / "candidate_b.jsonl"),
                                    "--fixture-provenance",
                                    "--prices", str(FIXTURES / "prices.json"))
        self.assertEqual(code, 0)
        self.assertIn("FIXTURE elapsed_ms", out)
        self.assertIn("not model latency", out)
        self.assertIn("candidate_b: REJECT", out)
        self.assertIn("pareto front among passing candidates: candidate_a", out)
        self.assertIn("WARNING: mean output tokens differ", out)

    def test_unpaired_logs_exit_with_error(self):
        a = self.write("a", [record("c1")])
        b = self.write("b", [record("c2")])
        code, out, err = self.run_cli(str(a), str(b))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("not paired", err)


if __name__ == "__main__":
    unittest.main()
