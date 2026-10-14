import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from release_gate import evaluate, load_json, sha256_file

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
DIGEST = sha256_file(FIX / "candidate.bin")


def fixture(name):
    return copy.deepcopy(load_json(FIX / name))


def gate(policy=None, candidate=None, baseline=None, digest=DIGEST):
    return evaluate(fixture("policy.json") if policy is None else policy,
                    fixture("candidate_ok.json") if candidate is None else candidate,
                    fixture("baseline.json") if baseline is None else baseline, digest)


class EligibleRelease(unittest.TestCase):
    def test_candidate_meeting_every_rule_is_eligible(self):
        self.assertEqual(gate(), [])


class QualityRules(unittest.TestCase):
    def test_aggregate_passes_but_critical_slice_blocks(self):
        reasons = gate(candidate=fixture("candidate_slice_fail.json"))
        self.assertFalse(any(r.startswith("overall") for r in reasons))
        self.assertTrue(any("channel=phone" in r and "floor" in r for r in reasons))
        self.assertTrue(any("channel=phone" in r and "regressed" in r for r in reasons))

    def test_regression_on_non_critical_slice_blocks(self):
        candidate = fixture("candidate_ok.json")
        candidate["slices"]["lang=en"] = {"correct": 500, "total": 600}
        self.assertEqual(gate(candidate=candidate),
                         ["lang=en: 0.8333 regressed from baseline 0.8667"])

    def test_too_small_slice_blocks(self):
        candidate = fixture("candidate_ok.json")
        candidate["slices"]["channel=phone"] = {"correct": 45, "total": 50}
        self.assertIn("channel=phone: 50 examples, policy needs 100",
                      gate(candidate=candidate))


class EvidenceBinding(unittest.TestCase):
    def test_wrong_artifact_digest_blocks(self):
        self.assertEqual(gate(digest="0" * 64),
                         ["report does not describe this artifact digest"])

    def test_wrong_corpus_version_blocks(self):
        candidate = fixture("candidate_ok.json")
        candidate["eval_corpus"]["version"] = "2"
        self.assertTrue(gate(candidate=candidate)[0].startswith("candidate evaluated on"))

    def test_wrong_corpus_id_blocks(self):
        baseline = fixture("baseline.json")
        baseline["eval_corpus"]["id"] = "support-intents-dev"
        self.assertTrue(gate(baseline=baseline)[0].startswith("baseline evaluated on"))

    def test_wrong_policy_version_blocks(self):
        candidate = fixture("candidate_ok.json")
        candidate["policy_version"] = "support-intent-release/2"
        self.assertEqual(gate(candidate=candidate),
                         ["report was produced for another policy version"])


class IdentifierValidation(unittest.TestCase):
    def inputs(self):
        return {"policy": fixture("policy.json"),
                "candidate": fixture("candidate_ok.json"),
                "baseline": fixture("baseline.json")}

    def test_empty_or_blank_corpus_identifiers_rejected(self):
        for target in ("policy", "candidate", "baseline"):
            for field in ("id", "version"):
                for value in ("", " ", "\t\n"):
                    with self.subTest(target=target, field=field, value=value):
                        inputs = self.inputs()
                        inputs[target]["eval_corpus"][field] = value
                        with self.assertRaisesRegex(ValueError, "eval_corpus"):
                            gate(**inputs)

    def test_missing_corpus_identity_rejected(self):
        for target in ("policy", "candidate", "baseline"):
            for field in ("eval_corpus", "id", "version"):
                with self.subTest(target=target, field=field):
                    inputs = self.inputs()
                    if field == "eval_corpus":
                        del inputs[target][field]
                    else:
                        del inputs[target]["eval_corpus"][field]
                    with self.assertRaisesRegex(ValueError, "eval_corpus"):
                        gate(**inputs)

    def test_malformed_corpus_object_rejected(self):
        for target in ("policy", "candidate", "baseline"):
            for value in (None, [], True, False, "corpus/3", 3):
                with self.subTest(target=target, value=value):
                    inputs = self.inputs()
                    inputs[target]["eval_corpus"] = value
                    with self.assertRaisesRegex(ValueError, "eval_corpus"):
                        gate(**inputs)

    def test_non_string_corpus_identifiers_rejected(self):
        for target in ("policy", "candidate", "baseline"):
            for field in ("id", "version"):
                for value in (None, True, False, 3, 3.0, [], {}):
                    with self.subTest(target=target, field=field, value=value):
                        inputs = self.inputs()
                        inputs[target]["eval_corpus"][field] = value
                        with self.assertRaisesRegex(ValueError, "eval_corpus"):
                            gate(**inputs)

    def test_empty_or_blank_policy_versions_rejected(self):
        for target in ("policy", "candidate"):
            for value in ("", " ", "\t\n"):
                with self.subTest(target=target, value=value):
                    inputs = self.inputs()
                    inputs[target]["policy_version"] = value
                    with self.assertRaisesRegex(ValueError, "policy_version"):
                        gate(**inputs)

    def test_missing_or_non_string_policy_versions_rejected(self):
        for target in ("policy", "candidate"):
            for value in ("missing", None, True, False, 3, 3.0, [], {}):
                with self.subTest(target=target, value=value):
                    inputs = self.inputs()
                    if value == "missing":
                        del inputs[target]["policy_version"]
                    else:
                        inputs[target]["policy_version"] = value
                    with self.assertRaisesRegex(ValueError, "policy_version"):
                        gate(**inputs)

    def test_matching_invalid_corpus_identities_rejected(self):
        # Equality alone cannot bind evidence when all three identities are invalid.
        for corpus in ({}, {"id": "support-intents-synthetic"}, {"version": "3"},
                       {"id": "", "version": "3"}, {"id": "ok", "version": True},
                       {"id": [], "version": "3"}):
            with self.subTest(corpus=corpus):
                inputs = self.inputs()
                for report in inputs.values():
                    report["eval_corpus"] = copy.deepcopy(corpus)
                with self.assertRaisesRegex(ValueError, "eval_corpus"):
                    gate(**inputs)

    def test_matching_empty_policy_versions_rejected(self):
        for value in ("", " "):
            with self.subTest(value=value):
                inputs = self.inputs()
                inputs["policy"]["policy_version"] = value
                inputs["candidate"]["policy_version"] = value
                with self.assertRaisesRegex(ValueError, "policy_version"):
                    gate(**inputs)


class MissingEvidence(unittest.TestCase):
    def test_missing_digest_field_blocks(self):
        candidate = fixture("candidate_ok.json")
        del candidate["artifact_sha256"]
        self.assertIn("report does not describe this artifact digest",
                      gate(candidate=candidate))

    def test_missing_critical_slice_blocks(self):
        candidate = fixture("candidate_ok.json")
        del candidate["slices"]["channel=phone"]
        self.assertEqual(gate(candidate=candidate),
                         ["channel=phone: missing from candidate report"])

    def test_critical_slice_without_baseline_blocks(self):
        baseline = fixture("baseline.json")
        del baseline["slices"]["lang=pl"]
        self.assertEqual(gate(baseline=baseline),
                         ["lang=pl: no baseline to compare against"])

    def test_missing_slices_raise(self):
        with self.assertRaises(ValueError):
            gate(candidate={"artifact_sha256": DIGEST})

    def test_missing_policy_field_raises(self):
        policy = fixture("policy.json")
        del policy["critical_slices"]
        with self.assertRaises(ValueError):
            gate(policy=policy)


class MalformedValues(unittest.TestCase):
    def write(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with tmp:
            tmp.write(text)
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_nan_constant_rejected_while_parsing(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_json(self.write('{"overall_floor": NaN}'))

    def test_infinity_constant_rejected_while_parsing(self):
        with self.assertRaisesRegex(ValueError, "Infinity"):
            load_json(self.write('{"max_regression": -Infinity}'))

    def test_overflowing_number_rejected_as_non_finite(self):
        # 1e999 is not a NaN/Infinity constant: it parses as float("inf").
        policy = fixture("policy.json")
        policy["overall_floor"] = load_json(self.write("1e999"))
        with self.assertRaisesRegex(ValueError, "overall_floor"):
            gate(policy=policy)

    def test_nan_count_rejected(self):
        candidate = fixture("candidate_ok.json")
        candidate["slices"]["overall"]["correct"] = float("nan")
        with self.assertRaises(ValueError):
            gate(candidate=candidate)

    def test_boolean_floor_rejected(self):
        policy = fixture("policy.json")
        policy["critical_floor"] = True  # would compare like 1.0
        with self.assertRaisesRegex(ValueError, "critical_floor"):
            gate(policy=policy)

    def test_boolean_count_rejected(self):
        candidate = fixture("candidate_ok.json")
        candidate["slices"]["channel=phone"] = {"correct": True, "total": 1}
        with self.assertRaises(ValueError):
            gate(candidate=candidate)

    def test_more_correct_than_total_rejected(self):
        candidate = fixture("candidate_ok.json")
        candidate["slices"]["overall"] = {"correct": 1001, "total": 1000}
        with self.assertRaises(ValueError):
            gate(candidate=candidate)

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_json(self.write('{"correct": 10, "correct": 900}'))


class CommandLine(unittest.TestCase):
    def run_gate(self, candidate):
        return subprocess.run(
            [sys.executable, str(HERE / "release_gate.py"),
             "--policy", str(FIX / "policy.json"),
             "--candidate", str(FIX / candidate),
             "--baseline", str(FIX / "baseline.json"),
             "--artifact", str(FIX / "candidate.bin")],
            capture_output=True, text=True)

    def test_exit_zero_for_eligible_candidate(self):
        result = self.run_gate("candidate_ok.json")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stdout.startswith("ELIGIBLE"))

    def test_exit_one_for_blocked_candidate(self):
        result = self.run_gate("candidate_slice_fail.json")
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_exit_two_for_missing_report(self):
        result = self.run_gate("does_not_exist.json")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cannot evaluate gate", result.stdout)

    def test_exit_two_for_invalid_identifiers(self):
        for target, field in (("policy", "id"), ("candidate", "version"),
                              ("baseline", "id"), ("policy", "policy_version"),
                              ("candidate", "policy_version")):
            for value in ("", "missing", [], True):
                with self.subTest(target=target, field=field, value=value):
                    inputs = IdentifierValidation().inputs()
                    obj = (inputs[target] if field == "policy_version"
                           else inputs[target]["eval_corpus"])
                    if value == "missing":
                        del obj[field]
                    else:
                        obj[field] = value
                    with tempfile.TemporaryDirectory() as tmp:
                        paths = {}
                        for label, report in inputs.items():
                            paths[label] = Path(tmp) / (label + ".json")
                            paths[label].write_text(json.dumps(report), encoding="utf-8")
                        result = subprocess.run(
                            [sys.executable, str(HERE / "release_gate.py"),
                             "--policy", str(paths["policy"]),
                             "--candidate", str(paths["candidate"]),
                             "--baseline", str(paths["baseline"]),
                             "--artifact", str(FIX / "candidate.bin")],
                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stdout)
                    self.assertIn("BLOCKED: cannot evaluate gate:", result.stdout)
                    self.assertNotIn("ELIGIBLE", result.stdout)


if __name__ == "__main__":
    unittest.main()
