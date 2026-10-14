import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from simulate import (
    BATCH_SIZES,
    SCENARIOS,
    Config,
    arrivals,
    format_table,
    main,
    nearest_rank,
    run_matrix,
    simulate,
    summarize,
)


def run(config: Config):
    requests = simulate(config)
    return requests, summarize(config, requests)


class ArrivalTest(unittest.TestCase):
    def test_partial_interval_includes_every_arrival_inside_window(self):
        config = Config(rate_rps=3, max_batch_size=1, injection_us=500_000)
        self.assertEqual(arrivals(config), [0, 333_333])

    def test_exact_boundary_excludes_arrival_at_window_end(self):
        config = Config(rate_rps=4, max_batch_size=1, injection_us=500_000)
        schedule = arrivals(config)
        self.assertEqual(schedule, [0, 250_000])
        self.assertNotIn(500_000, schedule)

    def test_tiny_window_includes_arrival_at_zero(self):
        config = Config(rate_rps=1, max_batch_size=1, injection_us=1)
        self.assertEqual(arrivals(config), [0])


class NearestRankTest(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(nearest_rank([], 0.5))

    def test_nonempty_uses_ceiling_rank_without_interpolation(self):
        values = [5, 1, 3, 2, 4]
        self.assertEqual(nearest_rank(values, 0.2), 1)
        self.assertEqual(nearest_rank(values, 0.5), 3)
        self.assertEqual(nearest_rank(values, 1.0), 5)
        hundred = list(range(1, 101))
        self.assertEqual(nearest_rank(hundred, 0.07), 7)
        self.assertEqual(nearest_rank(hundred, 0.95), 95)
        self.assertEqual(nearest_rank(hundred, 0.99), 99)

    def test_invalid_quantile_raises(self):
        for q in (0, -0.1, 1.01):
            with self.subTest(q=q), self.assertRaises(ValueError):
                nearest_rank([1, 2, 3], q)
        with self.assertRaises(ValueError):
            nearest_rank([], 0)


class SimulationTest(unittest.TestCase):
    def test_counts_are_conserved_in_every_scenario(self):
        for rate in SCENARIOS.values():
            for size in BATCH_SIZES:
                with self.subTest(rate=rate, size=size):
                    _, s = run(Config(rate_rps=rate, max_batch_size=size))
                    self.assertEqual(s["offered"], rate * 10)
                    self.assertEqual(s["offered"], s["admitted"] + s["rejected"])
                    self.assertEqual(s["admitted"], s["completed"])
                    self.assertEqual(
                        s["completed"], s["within_deadline"] + s["deadline_missed"]
                    )

    def test_batches_never_exceed_max_size(self):
        requests, s = run(Config(rate_rps=600, max_batch_size=8))
        sizes = [r.batch_size for r in requests if r.finish_us is not None]
        self.assertTrue(all(1 <= size <= 8 for size in sizes))
        self.assertEqual(s["max_batch_observed"], 8)

    def test_overload_rejects_and_keeps_rejections_out_of_latency(self):
        requests, s = run(Config(rate_rps=600, max_batch_size=1))
        self.assertGreater(s["rejected"], 0)
        rejected = [r for r in requests if r.outcome == "rejected"]
        self.assertEqual(len(rejected), s["rejected"])
        self.assertTrue(all(r.start_us is None and r.finish_us is None for r in rejected))

    def test_timings_are_ordered_and_nonnegative(self):
        config = Config(rate_rps=380, max_batch_size=8)
        for r in simulate(config):
            if r.finish_us is None:
                continue
            self.assertGreaterEqual(r.arrival_us, 0)
            self.assertGreaterEqual(r.start_us - r.arrival_us, 0)
            service = config.overhead_us + config.per_item_us * r.batch_size
            self.assertEqual(r.finish_us - r.start_us, service)

    def test_low_load_waits_full_max_wait_and_sends_singletons(self):
        for rate in (10, 20):
            with self.subTest(rate=rate):
                config = Config(rate_rps=rate, max_batch_size=8, max_wait_us=5_000)
                requests, s = run(config)
                for r in requests:
                    self.assertEqual(r.batch_size, 1)
                    self.assertEqual(r.start_us - r.arrival_us, 5_000)
                self.assertEqual(s["deadline_missed"], 0)
                self.assertEqual(s["rejected"], 0)

    def test_batch_size_one_dispatches_without_waiting_at_low_load(self):
        for r in simulate(Config(rate_rps=10, max_batch_size=1)):
            self.assertEqual(r.start_us, r.arrival_us)

    def test_overload_misses_deadlines_but_still_completes_work(self):
        config = Config(rate_rps=600, max_batch_size=8)
        requests, s = run(config)
        self.assertGreater(s["deadline_missed"], 0)
        self.assertLess(s["goodput_rps"], s["completed_throughput_rps"])
        late = [r for r in requests if r.outcome == "deadline_missed"]
        self.assertTrue(all(r.finish_us is not None for r in late))
        self.assertTrue(all(r.finish_us - r.arrival_us > config.deadline_us for r in late))

    def test_runs_are_deterministic(self):
        config = Config(rate_rps=380, max_batch_size=8)
        self.assertEqual(summarize(config, simulate(config)), summarize(config, simulate(config)))

    def test_invalid_configuration_is_rejected(self):
        bad = [
            {"rate_rps": 0, "max_batch_size": 1},
            {"rate_rps": 10, "max_batch_size": 0},
            {"rate_rps": 10, "max_batch_size": 8, "queue_capacity": 4},
            {"rate_rps": 10, "max_batch_size": 1, "max_wait_us": -1},
            {"rate_rps": 10, "max_batch_size": 1, "overhead_us": 0, "per_item_us": 0},
            {"rate_rps": 10, "max_batch_size": 1, "deadline_us": 0},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                simulate(Config(**kwargs))


class CliTest(unittest.TestCase):
    def test_matrix_covers_every_scenario_and_batch_size(self):
        rows = run_matrix()
        self.assertEqual(len(rows), len(SCENARIOS) * len(BATCH_SIZES))
        self.assertEqual(len(format_table(rows).splitlines()), len(rows) + 1)

    def test_json_output_parses(self):
        out = StringIO()
        with redirect_stdout(out):
            main(["--json"])
        rows = json.loads(out.getvalue())
        self.assertEqual({row["scenario"] for row in rows}, set(SCENARIOS))


if __name__ == "__main__":
    unittest.main()
