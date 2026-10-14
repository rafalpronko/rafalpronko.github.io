"""Correctness tests. No assertions about which variant is faster."""

from __future__ import annotations

import argparse
import copy
import json
import random
import unittest

from harness import correctness_gate, paired_verdict, run_case
from run_benchmark import build_report
from workload import (BASELINE, CANDIDATE, QUERY_ONLY, VARIANTS, Case, CaseSpec,
                      make_case)


def case_from(items: list[int], queries: list[int], loops: int = 1) -> Case:
    spec = CaseSpec("manual", len(items), len(queries), 0.0, loops)
    return Case(spec, items, queries, frozenset(items))


class VariantAgreement(unittest.TestCase):
    def assert_all(self, case: Case, expected: list[bool]) -> None:
        for name, fn in VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(fn(case), expected)

    def test_empty_items(self):
        self.assert_all(case_from([], [1, 2]), [False, False])

    def test_empty_queries(self):
        self.assert_all(case_from([1, 2], []), [])

    def test_duplicates_in_items_and_queries(self):
        self.assert_all(case_from([5, 5, 7, 5], [5, 5, 6, 7]), [True, True, False, True])

    def test_hits_and_misses(self):
        self.assert_all(case_from([0, 10, 20], [10, -10, 20, 21]), [True, False, True, False])

    def test_inputs_not_mutated(self):
        items, queries = [3, 1, 3, 2], [3, 4]
        case = case_from(items, queries)
        before = (copy.copy(items), copy.copy(queries))
        for fn in VARIANTS.values():
            fn(case)
        self.assertEqual((items, queries), before)


class SyntheticData(unittest.TestCase):
    spec = CaseSpec("t", 500, 200, 0.25, 1)

    def test_same_seed_same_case(self):
        a, b = make_case(self.spec, 1), make_case(self.spec, 1)
        self.assertEqual((a.items, a.queries), (b.items, b.queries))

    def test_different_seed_different_case(self):
        self.assertNotEqual(make_case(self.spec, 1).items, make_case(self.spec, 2).items)

    def test_exact_hit_ratio(self):
        case = make_case(self.spec, 3)
        self.assertEqual(len(case.queries), 200)
        self.assertEqual(sum(VARIANTS[BASELINE](case)), 50)

    def test_empty_items_produce_only_misses(self):
        case = make_case(CaseSpec("e", 0, 10, 0.5, 1), 1)
        self.assertEqual(VARIANTS[BASELINE](case), [False] * 10)

    def test_rejects_bad_hit_ratio(self):
        with self.assertRaises(ValueError):
            make_case(CaseSpec("bad", 10, 10, 1.5, 1), 1)


class GateAndVerdict(unittest.TestCase):
    def test_gate_excludes_wrong_and_crashing_variants(self):
        variants = dict(VARIANTS)
        variants["wrong"] = lambda case: [True] * len(case.queries)
        variants["crash"] = lambda case: 1 / 0
        gate = correctness_gate(case_from([1], [1, 2]), variants)
        self.assertEqual({f["variant"] for f in gate["failures"]}, {"wrong", "crash"})
        result = run_case(case_from([1], [1, 2]), variants, 2, 0, random.Random(0))
        self.assertNotIn("wrong", result["samples_ns"])

    def test_verdict_rules(self):
        self.assertEqual(paired_verdict([10, 10], [5, 5])["verdict"], "candidate faster")
        self.assertEqual(paired_verdict([10, 10], [20, 20])["verdict"], "baseline faster")
        self.assertEqual(paired_verdict([10, 10], [5, 20])["verdict"], "inconclusive")

    def test_tie_is_not_a_baseline_win(self):
        verdict = paired_verdict([10, 10, 10], [10, 20, 20])
        self.assertEqual(verdict["verdict"], "inconclusive")


class ReportSchema(unittest.TestCase):
    def test_report_structure_and_nonnegative_timings(self):
        args = argparse.Namespace(data_seed=1, order_seed=2, repeats=3, warmups=1)
        report = json.loads(json.dumps(build_report(args)))
        self.assertEqual(report["schema_version"], 1)
        for key in ("python_version", "os", "cpu_model", "perf_counter"):
            self.assertIn(key, report["environment"])
        self.assertEqual(len(report["cases"]), 4)
        for case in report["cases"]:
            self.assertEqual(case["correctness"]["failures"], [])
            self.assertEqual(len(case["orders"]), 3)
            for name in (BASELINE, CANDIDATE, QUERY_ONLY):
                samples = case["samples_ns"][name]
                self.assertEqual(len(samples), 3)
                self.assertTrue(all(isinstance(s, int) and s >= 0 for s in samples))
            self.assertIn(case["end_to_end"]["verdict"],
                          {"candidate faster", "baseline faster", "inconclusive"})


if __name__ == "__main__":
    unittest.main()
