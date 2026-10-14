import copy
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import demo  # noqa: E402
import harness as h  # noqa: E402

FIXTURES = ROOT / "fixtures"


def answer(decision, limit, citations):
    return json.dumps({"decision": decision, "limit_eur": limit, "citations": citations})


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.data = h.load_fixtures(FIXTURES)
        cls.cases = {c["id"]: c for c in cls.data["cases"]}


class RetrievalTests(Base):
    def test_deterministic_ranking(self):
        q = self.cases["C01"]["question"]
        self.assertEqual(h.retrieve(q, self.corpus, 2), ["P-TRV-3", "P-TRV-2"])
        self.assertEqual(h.retrieve(q, self.corpus, 2), h.retrieve(q, self.corpus, 2))

    def test_vocabulary_mismatch_misses_gold(self):
        # "wine" and "customer" never appear in P-MEAL-1 ("alcoholic", "client").
        top = h.retrieve(self.cases["C03"]["question"], self.corpus, 2)
        self.assertNotIn("P-MEAL-1", top)

    def test_current_only_filter_drops_retired(self):
        q = self.cases["C01"]["question"]
        top = h.retrieve(q, self.corpus, 3, current_only=True)
        self.assertNotIn("P-TRV-2", top)

    def test_invalid_k(self):
        for bad in (0, -1, True, 2.0, "2"):
            with self.assertRaises(ValueError):
                h.retrieve("hotel", self.corpus, bad)
        with self.assertRaises(ValueError):
            h.retrieve("hotel", self.corpus, len(self.corpus) + 1)
        with self.assertRaises(ValueError):  # one passage is retired
            h.retrieve("hotel", self.corpus, len(self.corpus), current_only=True)


class ContractTests(Base):
    def check(self, raw, context, expected_fragment=None):
        obj, errors = h.parse_answer(raw, context, self.corpus)
        if expected_fragment is None:
            self.assertEqual(errors, [])
        else:
            self.assertTrue(any(expected_fragment in e for e in errors), errors)
        return obj

    def test_valid_answer(self):
        self.check(answer("allowed", 220, ["P-TRV-3"]), ["P-TRV-3"])

    def test_abstention_without_citation_is_valid(self):
        self.check(answer("insufficient_context", None, []), [])

    def test_rejections(self):
        self.assertIsNone(self.check("Yes, allowed.", [], "not valid JSON"))
        self.assertIsNone(self.check("[1, 2]", [], "JSON object"))
        self.check('{"decision": "allowed", "limit_eur": null}', ["P-RAIL-1"], "missing fields")
        self.check(answer("yes", None, ["P-RAIL-1"]), ["P-RAIL-1"], "not in")
        self.check(answer("allowed", True, ["P-RAIL-1"]), ["P-RAIL-1"], "limit_eur")
        self.check(answer("allowed", -5, ["P-RAIL-1"]), ["P-RAIL-1"], "limit_eur")
        self.check(answer("allowed", None, ["P-RAIL-1"]), [], "not in context")
        self.check(answer("allowed", 150, ["P-TRV-2"]), ["P-TRV-2"], "retired")
        self.check(answer("allowed", None, []), [], "at least one citation")
        self.check(answer("allowed", None, "P-RAIL-1"), ["P-RAIL-1"], "list of passage ids")
        raw = json.dumps({"decision": "allowed", "limit_eur": None, "citations": ["P-RAIL-1"], "note": "x"})
        self.check(raw, ["P-RAIL-1"], "unexpected fields")

    def test_bool_is_not_a_limit(self):
        r = h.grade(answer("allowed", False, ["P-RAIL-1"]),
                    {"decision": "allowed", "limit_eur": 0}, ["P-RAIL-1"], self.corpus)
        self.assertFalse(r.correct)

    def test_unparsed_content_is_not_graded_as_wrong(self):
        expected = {"decision": "allowed", "limit_eur": None}
        prose = h.grade("Yes. First class is allowed.", expected, ["P-RAIL-1"], self.corpus)
        self.assertFalse(prose.readable)
        self.assertFalse(prose.correct)
        no_decision = h.grade('{"limit_eur": null, "citations": ["P-RAIL-1"]}', expected, ["P-RAIL-1"], self.corpus)
        self.assertFalse(no_decision.readable)
        uncited = h.grade('{"decision": "allowed", "limit_eur": null}', expected, ["P-RAIL-1"], self.corpus)
        self.assertTrue(uncited.readable and uncited.correct)
        self.assertFalse(uncited.valid)


class DiagnosisTests(Base):
    EXPECTED = {
        "C01": h.CONTEXT_USE,
        "C02": h.PASS,
        "C03": h.RETRIEVAL_MISS,
        "C04": h.PASS,
        "C05": h.BEHAVIOR,
        "C06": h.PASS,
        "C07": h.CAPABILITY_SUSPECTED,
        "C08": h.OUTPUT_CONTRACT,
        "C09": h.PASS,
        "C10": h.CONTEXT_USE,
        "C11": h.PASS,
    }

    def test_labels_for_fixture_cases(self):
        got = {cid: h.diagnose(h.evaluate_case(c, self.corpus, 2)) for cid, c in self.cases.items()}
        self.assertEqual(got, self.EXPECTED)

    def test_oracle_failure_without_fewshot_run(self):
        case = copy.deepcopy(self.cases["C05"])
        del case["observations"]["oracle_fewshot"]
        self.assertEqual(h.diagnose(h.evaluate_case(case, self.corpus, 2)), h.GOLD_CONTEXT_FAILS)

    def test_larger_k_turns_retrieval_miss_into_context_use(self):
        # With k=9 everything is retrieved, so the C03 failure is no longer a retrieval miss.
        # The label logic changes; the hand-authored answer is not re-observed for k=9.
        r = h.evaluate_case(self.cases["C03"], self.corpus, 9)
        self.assertTrue(r.gold_in_context)
        self.assertEqual(h.diagnose(r), h.CONTEXT_USE)

    def label_with(self, case_id, **observations):
        case = copy.deepcopy(self.cases[case_id])
        case["observations"].update(observations)
        return h.diagnose(h.evaluate_case(case, self.corpus, 2))

    def test_fewshot_right_content_broken_contract_is_behavior(self):
        # Examples fixed the content; the missing citation is a contract error, not a capability limit.
        uncited_answers = (
            answer("needs_approval", 300, []),
            '{"decision": "needs_approval", "limit_eur": 300}',
        )
        for uncited in uncited_answers:
            with self.subTest(raw=uncited):
                case = copy.deepcopy(self.cases["C05"])
                case["observations"]["oracle_fewshot"] = uncited
                result = h.evaluate_case(case, self.corpus, 2)
                self.assertEqual(h.diagnose(result), h.BEHAVIOR)
                fewshot = result.conditions["oracle_fewshot"]
                self.assertTrue(fewshot.readable and fewshot.correct)
                self.assertFalse(fewshot.valid)
                self.assertTrue(fewshot.errors)

    def test_unreadable_fewshot_does_not_decide(self):
        self.assertEqual(self.label_with("C05", oracle_fewshot="Needs approval."), h.GOLD_CONTEXT_FAILS)

    def test_unreadable_oracle_is_output_contract(self):
        self.assertEqual(self.label_with("C07", oracle="Not on the training budget."), h.OUTPUT_CONTRACT)

    def test_end_to_end_right_content_broken_contract(self):
        # Right decision and limit, but cites the retired policy.
        self.assertEqual(self.label_with("C02", retrieved=answer("allowed", 180, ["P-TRV-2"])), h.OUTPUT_CONTRACT)

    def test_abstention_case_has_no_gold_to_find(self):
        r = h.evaluate_case(self.cases["C10"], self.corpus, 2)
        self.assertFalse(r.answerable)
        self.assertFalse(r.gold_in_context)
        self.assertEqual(h.diagnose(r), h.CONTEXT_USE)

    def test_closed_book_correct_but_uncited(self):
        r = h.evaluate_case(self.cases["C09"], self.corpus, 2)
        cb = r.conditions["closed_book"]
        self.assertTrue(cb.correct)
        self.assertFalse(cb.valid)
        self.assertFalse(r.needs_external_knowledge)

    def test_needs_external_knowledge(self):
        r = h.evaluate_case(self.cases["C01"], self.corpus, 2)
        self.assertTrue(r.needs_external_knowledge)
        case = copy.deepcopy(self.cases["C01"])
        case["observations"]["closed_book"] = "I do not know."
        self.assertFalse(h.evaluate_case(case, self.corpus, 2).needs_external_knowledge)


class SplitTests(Base):
    def test_grouped_split_has_no_leak(self):
        dev, holdout = h.grouped_split(self.data["cases"], self.data["holdout_groups"])
        self.assertEqual(h.leaking_groups(dev, holdout), set())
        self.assertEqual(len(dev) + len(holdout), len(self.data["cases"]))
        h.check_exemplars(self.data["fewshot_exemplars"], dev, holdout)

    def test_row_split_leaks_paraphrases(self):
        cases = self.data["cases"]
        dev = [c for i, c in enumerate(cases) if i % 2 == 0]
        holdout = [c for i, c in enumerate(cases) if i % 2 == 1]
        self.assertIn("hotel", h.leaking_groups(dev, holdout))

    def test_exemplar_from_holdout_rejected(self):
        dev, holdout = h.grouped_split(self.data["cases"], self.data["holdout_groups"])
        with self.assertRaisesRegex(ValueError, "holdout group 'equipment'"):
            h.check_exemplars(["C02", "C06"], dev, holdout)
        with self.assertRaisesRegex(ValueError, "not a known dev case"):
            h.check_exemplars(["C99"], dev, holdout)

    def test_bad_holdout_groups(self):
        cases = self.data["cases"]
        with self.assertRaises(ValueError):
            h.grouped_split(cases, [])
        with self.assertRaises(ValueError):
            h.grouped_split(cases, ["nope"])
        with self.assertRaises(ValueError):
            h.grouped_split(cases, sorted({c["group"] for c in cases}))


class FixtureValidationTests(Base):
    def mutate(self, fn, fragment):
        data = copy.deepcopy(self.data)
        fn(data)
        with self.assertRaisesRegex(ValueError, fragment):
            h.validate_cases(data, self.corpus)

    def test_case_validation(self):
        self.mutate(lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])), "duplicate case id")
        self.mutate(lambda d: d["cases"][0].update(gold_ids=["P-NOPE"]), "unknown gold")
        self.mutate(lambda d: d["cases"][0].update(gold_ids=[]), "empty gold_ids")
        self.mutate(lambda d: d["cases"][0].update(gold_ids=["P-TRV-2"]), "must be current")
        self.mutate(lambda d: d["cases"][0]["observations"].pop("oracle"), "missing observations")
        self.mutate(lambda d: d["cases"][0]["observations"].update(fine_tuned="{}"), "unknown observation")
        self.mutate(lambda d: d["cases"][0]["observations"].update(oracle={"decision": "allowed"}), "raw answer string")
        self.mutate(lambda d: d["cases"][0]["expected"].update(decision="maybe"), "expected decision")
        self.mutate(lambda d: d["cases"][0]["expected"].update(limit_eur="220"), "limit_eur")
        self.mutate(lambda d: d["cases"][0].update(question=" "), "question")
        self.mutate(lambda d: d.update(cases=[]), "non-empty list")

    def test_corpus_validation(self):
        good = h.load_json(FIXTURES / "corpus.json")
        dup = copy.deepcopy(good)
        dup["passages"].append(copy.deepcopy(dup["passages"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate passage"):
            h.validate_corpus(dup)
        bad = copy.deepcopy(good)
        bad["passages"][0]["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "status"):
            h.validate_corpus(bad)
        with self.assertRaises(ValueError):
            h.validate_corpus({"passages": []})

    def test_load_json_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            with self.assertRaisesRegex(ValueError, "not found"):
                h.load_json(path)
            path.write_text("{oops", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                h.load_json(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                h.load_json(path)


class DemoTests(unittest.TestCase):
    def run_demo(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = demo.main(list(argv))
            except SystemExit as exc:  # argparse rejects bad arguments
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_report_runs_and_labels_fixtures(self):
        code, out, _ = self.run_demo()
        self.assertEqual(code, 0)
        self.assertIn("HAND-AUTHORED FIXTURES", out)
        self.assertIn("None is a model output", out)
        self.assertIn("C03  dev     meals", out)
        self.assertIn("RETRIEVAL_MISS", out)
        self.assertRegex(out, r"C10 .* n/a ")
        self.assertRegex(out, r"C08 .* unparsed OUTPUT_CONTRACT")
        self.assertIn("gold recall@2 over answerable cases: 9/10", out)

    def test_largest_k_for_full_corpus(self):
        # Nine passages, eight of them current: the current-only comparison is capped at 8.
        code, out, err = self.run_demo("--k", "9")
        self.assertEqual(code, 0, err)
        self.assertIn("k=8 of 8 current passages", out)
        # Long retrieved lists must not run into the next column.
        rows = [line.split() for line in out.splitlines() if re.match(r"C\d\d ", line)]
        self.assertEqual(len(rows), 11)
        for row in rows:
            self.assertEqual(len(row), 10, row)
            self.assertIn(row[4], ("yes", "no", "n/a"))

    def test_bad_arguments(self):
        self.assertEqual(self.run_demo("--k", "0")[0], 2)
        self.assertEqual(self.run_demo("--k", "two")[0], 2)
        code, _, err = self.run_demo("--k", "50")
        self.assertEqual(code, 2)
        self.assertIn("exceeds", err)
        code, _, err = self.run_demo("--fixtures", "/nonexistent/dir")
        self.assertEqual(code, 2)
        self.assertIn("not found", err)


if __name__ == "__main__":
    unittest.main()
