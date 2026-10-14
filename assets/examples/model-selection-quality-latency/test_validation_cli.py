"""Regression cases from both independent FAIL reviews."""
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import replay_eval as ev
from test_replay_eval import FIXTURES, TempLogs, record


class ValidationRegressionTests(TempLogs):
    def rejected(self, row):
        with self.assertRaises(ev.LogError):
            ev.load_log(self.write('bad', [row]))

    def test_status_type_before_membership(self):
        for value in ([], {}, 1, None, True):
            with self.subTest(value=value):
                self.rejected(record(status=value))

    def test_output_type_for_every_status(self):
        for status in ('ok', 'abstain', 'malformed', 'timeout'):
            for value in ([], {}, 123, True):
                with self.subTest(status=status, value=value):
                    self.rejected(record(status=status, output=value))

    def test_abstention_requires_null_output(self):
        self.rejected(record(expected=None, status='abstain', output='x'))

    def test_huge_elapsed_and_tokens(self):
        for field in ('elapsed_ms', 'input_tokens', 'output_tokens'):
            with self.subTest(field=field):
                self.rejected(record(**{field: 10 ** 400}))

    def test_extreme_json_integer_is_log_error(self):
        path = self.dir / 'huge.jsonl'
        path.write_text(json.dumps(record()).replace('"elapsed_ms": 100',
                                                    '"elapsed_ms": ' + '9' * 5000))
        with self.assertRaises(ev.LogError):
            ev.load_log(path)

    def test_invalid_utf8_log_and_prices_are_log_errors(self):
        path = self.dir / 'bad'
        path.write_bytes(b'\xff\xfe\n')
        for loader in (ev.load_log, ev.load_prices):
            with self.subTest(loader=loader.__name__):
                with self.assertRaises(ev.LogError):
                    loader(path)

    def test_huge_prices_and_cost_overflow(self):
        path = self.dir / 'prices.json'
        path.write_text(json.dumps({'a': {'input_per_1k': 10 ** 400,
                                         'output_per_1k': 1}}))
        with self.assertRaises(ev.LogError):
            ev.load_prices(path)
        with self.assertRaises(ev.LogError):
            ev.Price(1e308, 1e308).cost(10000, 10000)

    def test_incorrect_abstention_is_not_useful(self):
        rows = ev.load_log(self.write('a', [record(status='abstain', output=None)]))
        summary = ev.summarize('a', rows, ev.Price(1, 1))
        self.assertFalse(rows['c1'].correct)
        self.assertEqual((summary.useful, summary.wrong_abstentions), (0, 1))
        self.assertEqual(summary.cost_per_useful, math.inf)


class CliRegressionTests(TempLogs):
    def cli(self, *argv):
        return subprocess.run([sys.executable, str(Path(ev.__file__)), *map(str, argv)],
                              capture_output=True, text=True)

    def assert_exit2(self, *argv):
        result = self.cli(*argv)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('error:', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_invalid_thresholds_exit2_without_report(self):
        a, b = self.write('a', [record()]), self.write('b', [record()])
        for flag, values in [('--min-quality', ['nan', 'inf', '-0.1', '1.1']),
                             ('--max-malformed-rate', ['nan', 'inf', '-0.1', '1.1']),
                             ('--max-p95-ms', ['nan', 'inf', '0', '-1'])]:
            for value in values:
                with self.subTest(flag=flag, value=value):
                    self.assert_exit2(a, b, flag, value)

    def test_rate_boundaries_and_positive_latency_are_accepted(self):
        a, b = self.write('a', [record()]), self.write('b', [record()])
        for rate in ('0', '1'):
            with self.subTest(rate=rate):
                result = self.cli(a, b, '--min-quality', rate,
                                  '--max-malformed-rate', rate, '--max-p95-ms', '0.1')
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_log_types_exit2_without_traceback(self):
        a = self.write('a', [record()])
        for fields in ({'status': []}, {'status': {}},
                       {'status': 'timeout', 'output': 123},
                       {'status': 'abstain', 'expected': None, 'output': {}},
                       {'status': 'abstain', 'output': 'x'},
                       {'elapsed_ms': 10 ** 400}, {'output_tokens': 10 ** 400}):
            with self.subTest(fields=fields):
                self.assert_exit2(a, self.write('b', [record(**fields)]))

    def test_utf8_and_extreme_json_cli_exit2(self):
        a = self.write('a', [record()])
        b = self.dir / 'b.jsonl'
        b.write_bytes(b'\xff\n')
        self.assert_exit2(a, b)
        b.write_text(json.dumps(record()).replace('"elapsed_ms": 100',
                                                 '"elapsed_ms": ' + '9' * 5000))
        self.assert_exit2(a, b)
        b = self.write('b', [record()])
        prices = self.dir / 'prices.json'
        prices.write_bytes(b'\xff\n')
        self.assert_exit2(a, b, '--prices', prices)

    def test_partial_and_empty_prices_exit2(self):
        a, b = self.write('a', [record()]), self.write('b', [record()])
        prices = self.dir / 'prices.json'
        for entries in ({}, {'a': {'input_per_1k': 1, 'output_per_1k': 1}}):
            with self.subTest(entries=entries):
                prices.write_text(json.dumps(entries))
                self.assert_exit2(a, b, '--prices', prices)

    def test_overflow_prices_and_cost_cli_exit2(self):
        a, b = self.write('a', [record()]), self.write('b', [record()])
        prices = self.dir / 'prices.json'
        for value in (10 ** 400, 1e308):
            with self.subTest(value=value):
                prices.write_text(json.dumps({name: {'input_per_1k': value,
                                                     'output_per_1k': value}
                                              for name in ('a', 'b')}))
                self.assert_exit2(a, b, '--prices', prices)

    def test_external_provenance_default_is_neutral(self):
        result = self.cli(self.write('a', [record()]), self.write('b', [record()]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('provenance not verified', result.stdout)
        self.assertNotIn('FIXTURE', result.stdout)
        self.assertIn('quality + p95 attempt duration (two objectives)', result.stdout)

    def test_fixture_flag_and_timeout_rate_are_reported(self):
        result = self.cli(FIXTURES / 'candidate_a.jsonl', FIXTURES / 'candidate_b.jsonl',
                          '--fixture-provenance', '--prices', FIXTURES / 'prices.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FIXTURE elapsed_ms', result.stdout)
        self.assertIn('timeout rate: 0.125', result.stdout)
        self.assertIn('attempt duration to completion or cancellation', result.stdout)
        self.assertIn('formula demonstration', result.stdout)


class TextAndNameTests(TempLogs):
    cli = CliRegressionTests.cli
    assert_exit2 = CliRegressionTests.assert_exit2
    rejected = ValidationRegressionTests.rejected

    def test_unpaired_surrogates_are_rejected_in_every_text_field(self):
        for field in ('case_id', 'slice', 'expected', 'output'):
            with self.subTest(field=field):
                self.rejected(record(**{field: 'x\ud800'}))

    def test_line_breaking_controls_are_rejected_in_identifiers(self):
        for field in ('case_id', 'slice'):
            for char in ('\n', '\r', '\t', '\x00', '\x1b', '\x85', '\u2028'):
                with self.subTest(field=field, char=repr(char)):
                    self.rejected(record(**{field: f'a{char}b'}))

    def test_valid_unicode_and_multiline_outputs_are_accepted(self):
        row = record(case_id='é-1', slice='płatności 👩\u200d💻',
                     expected='a\nb', output='a\nb')
        self.assertTrue(ev.load_log(self.write('a', [row]))['é-1'].correct)

    def test_surrogate_or_newline_slice_exit2_without_report(self):
        for value in ('\ud800', 's\nempirical hard constraints: fake'):
            with self.subTest(value=repr(value)):
                rows = [record(slice=value)]
                self.assert_exit2(self.write('a', rows), self.write('b', rows))

    def test_underscore_candidate_stem_is_rejected_early(self):
        a, b = self.write('_a', [record()]), self.write('b', [record()])
        prices = self.dir / 'prices.json'
        prices.write_text(json.dumps({'_a': {'input_per_1k': 1, 'output_per_1k': 1},
                                      'b': {'input_per_1k': 1, 'output_per_1k': 1}}))
        for extra in ((), ('--prices', prices)):
            with self.subTest(extra=extra):
                self.assert_exit2(a, b, *extra)
                self.assertIn("must not start with '_'", self.cli(a, b, *extra).stderr)

    def test_control_character_candidate_stem_exit2(self):
        self.assert_exit2(self.write('a\nb', [record()]), self.write('b', [record()]))

    def test_unicode_report_needs_an_encodable_stdout(self):
        rows = [record(slice='płatności')]
        a, b = self.write('a', rows), self.write('b', rows)
        result = self.cli(a, b)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('płatności=1/1', result.stdout)
        ascii_run = subprocess.run([sys.executable, str(Path(ev.__file__)), str(a), str(b)],
                                   capture_output=True,
                                   env={**os.environ, 'PYTHONIOENCODING': 'ascii'})
        self.assertEqual(ascii_run.returncode, 2, ascii_run.stderr)
        self.assertEqual(ascii_run.stdout, b'')
        self.assertNotIn(b'Traceback', ascii_run.stderr)


if __name__ == '__main__':
    unittest.main()
