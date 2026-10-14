"""Selection must use the same objectives for every candidate."""
import math
import unittest
from dataclasses import replace

import replay_eval as ev
from test_replay_eval import FIXTURES


class SelectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = {name: ev.load_log(FIXTURES / f'{name}.jsonl')
                    for name in ('candidate_a', 'candidate_b')}
        cls.prices = ev.load_prices(FIXTURES / 'prices.json')
        cls.a, cls.b = [ev.summarize(name, rows, cls.prices[name])
                        for name, rows in cls.logs.items()]

    def test_mixed_price_availability_is_rejected_globally(self):
        unpriced = replace(self.b, cost_units=None)
        with self.assertRaises(ev.LogError):
            ev.pareto_front([self.a, unpriced])

    def test_unpriced_mode_has_two_objectives(self):
        a, b = replace(self.a, cost_units=None), replace(self.b, cost_units=None)
        self.assertEqual([s.name for s in ev.pareto_front([a, b])], ['candidate_a'])

    def test_zero_useful_priced_cost_is_infinity_even_when_free(self):
        for cost in (0, 1):
            s = replace(self.b, useful=0, cost_units=cost)
            self.assertEqual(s.cost_per_useful, math.inf)

    def test_zero_useful_does_not_drop_cost_objective(self):
        empty = replace(self.b, name='empty', useful=0, all_latencies=[1], cost_units=1)
        slow = replace(self.a, name='slow', all_latencies=[2], cost_units=999)
        self.assertEqual({s.name for s in ev.pareto_front([empty, slow])}, {'empty', 'slow'})

    def test_constraints_are_applied_before_pareto_in_report(self):
        # The cheaper B is nondominated under loose gates; strict gates must remove it.
        strict = '\n'.join(ev.render(self.logs, self.prices, ev.Constraints(.7, .1, 3000)))
        loose = '\n'.join(ev.render(self.logs, self.prices, ev.Constraints(.5, .1, 5000)))
        self.assertEqual(strict.splitlines()[-1], 'pareto front among passing candidates: candidate_a')
        self.assertEqual(loose.splitlines()[-1], 'pareto front among passing candidates: candidate_a, candidate_b')

    def gate_lines(self, constraints):
        lines = ev.render(self.logs, self.prices, constraints)
        start = next(i for i, line in enumerate(lines) if line.startswith('empirical'))
        return lines[start:start + 3]

    def test_default_gate_lines_are_unchanged(self):
        self.assertEqual(self.gate_lines(ev.Constraints(.7, .1, 3000)), [
            'empirical hard constraints: quality >= 0.70, malformed rate <= 0.10, '
            'p95 attempt duration <= 3000 ms',
            '  candidate_a: PASS',
            '  candidate_b: REJECT: quality 0.625 < 0.700; '
            'p95 attempt duration 4000 ms > 3000 ms'])

    def test_fractional_budget_is_not_rounded_to_zero(self):
        header, a, _ = self.gate_lines(ev.Constraints(.5, .1, 0.4))
        self.assertTrue(header.endswith('p95 attempt duration <= 0.4 ms'), header)
        self.assertEqual(a, '  candidate_a: REJECT: p95 attempt duration 1880 ms > 0.4 ms')

    def test_near_threshold_values_do_not_look_equal(self):
        header, a, b = self.gate_lines(ev.Constraints(0.62500001, 0.0624, 5000))
        self.assertIn('quality >= 0.62500001, malformed rate <= 0.0624,', header)
        self.assertEqual(a, '  candidate_a: REJECT: malformed rate 0.0625 > 0.0624')
        self.assertEqual(b, '  candidate_b: REJECT: quality 0.625 < 0.62500001; '
                            'malformed rate 0.0625 > 0.0624')

    def test_wilson_known_values(self):
        lo, hi = ev.wilson_interval(13, 16)
        self.assertAlmostEqual(lo, 0.569911, places=5)
        self.assertAlmostEqual(hi, 0.934084, places=5)


if __name__ == '__main__':
    unittest.main()
