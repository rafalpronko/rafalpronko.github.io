"""Compare candidate models from paired replay logs: quality, duration and cost.

No model calls. Bundled fixtures have hand-designed durations and outcomes.
Keep replay_validation.py, replay_metrics.py and replay_report.py beside this CLI.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Re-export the small example's API for tests and interactive use.
from replay_validation import Attempt, Constraints, LogError, Price, candidate_name, check_paired, load_log, load_prices
from replay_metrics import Summary, dominates, exact_mcnemar, nearest_rank, pareto_front, summarize, violations, wilson_interval
from replay_report import render


def _emit(lines: list[str], elapsed: float) -> int:
    """Print only a report stdout can encode completely; never a partial one."""
    report = "\n".join(lines) + (f"\n\nmeasured: evaluator wall time {elapsed:.1f} ms via "
                                 "time.perf_counter() (this machine's parsing/report time, "
                                 "not model latency)")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        report.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        print(f"error: stdout encoding {encoding!r} cannot represent the report; "
              "use UTF-8 (for example PYTHONIOENCODING=utf-8)", file=sys.stderr)
        return 2
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path,
                        help="JSONL replay logs; candidate name = file stem, not starting with '_'")
    parser.add_argument("--prices", type=Path,
                        help="JSON prices for every candidate; absent = two Pareto objectives")
    parser.add_argument("--fixture-provenance", action="store_true",
                        help="explicitly declare hand-designed synthetic fixtures")
    parser.add_argument("--min-quality", type=float, default=0.70)
    parser.add_argument("--max-malformed-rate", type=float, default=0.10)
    parser.add_argument("--max-p95-ms", type=float, default=3000.0,
                        help="p95 attempt duration to completion or cancellation limit")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        constraints = Constraints(args.min_quality, args.max_malformed_rate, args.max_p95_ms)
        names = [candidate_name(path) for path in args.logs]
        if len(set(names)) != len(names):
            raise LogError("candidate names (file stems) must be unique")
        logs = {name: load_log(path) for name, path in zip(names, args.logs)}
        prices = load_prices(args.prices) if args.prices else None
        lines = render(logs, prices, constraints, fixture_provenance=args.fixture_provenance)
    except (LogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _emit(lines, (time.perf_counter() - started) * 1000)


if __name__ == "__main__":
    sys.exit(main())
