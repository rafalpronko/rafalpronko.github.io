"""Run the membership benchmark and write raw samples to JSON.

Usage:
    python run_benchmark.py --out results/membership.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from env_info import describe_environment
from harness import LOSE_RATIO, WIN_RATIO, run_case
from workload import BASELINE, CANDIDATE, CASE_SPECS, QUERY_ONLY, VARIANTS, make_case

SCHEMA_VERSION = 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--data-seed", type=int, default=20240919)
    parser.add_argument("--order-seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    order_rng = random.Random(args.order_seed)
    cases = [
        run_case(make_case(spec, args.data_seed), VARIANTS,
                 args.repeats, args.warmups, order_rng)
        for spec in CASE_SPECS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "data_seed": args.data_seed,
            "order_seed": args.order_seed,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "baseline": BASELINE,
            "candidate": CANDIDATE,
            "query_only_view": QUERY_ONLY,
            "win_ratio": WIN_RATIO,
            "lose_ratio": LOSE_RATIO,
        },
        "environment": describe_environment(),
        "cases": cases,
    }


def format_table(report: dict) -> str:
    lines = [
        "Per-batch elapsed time in ms (min / median of the raw samples in the JSON).",
        "These are batch timings on one machine, not per-request latency percentiles.",
        "",
        f"{'case':<26} {'variant':<20} {'min ms':>10} {'median ms':>10}",
    ]
    for case in report["cases"]:
        name = case["spec"]["name"]
        for variant, stats in case.get("summary", {}).items():
            lines.append(f"{name:<26} {variant:<20} "
                         f"{stats['min_ms']:>10.4f} {stats['median_ms']:>10.4f}")
        for failure in case["correctness"]["failures"]:
            lines.append(f"{name:<26} {failure['variant']:<20} FAILED: {failure['reason']}")
    lines += ["", "End-to-end decision (candidate time / baseline time, paired):"]
    for case in report["cases"]:
        e2e = case.get("end_to_end")
        if e2e:
            lines.append(
                f"{case['spec']['name']:<26} median {e2e['median_ratio']:.3f} "
                f"[{e2e['min_ratio']:.3f}..{e2e['max_ratio']:.3f}] "
                f"wins {e2e['candidate_wins']}/{e2e['pairs']} -> {e2e['verdict']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_table(report))
    print(f"\nRaw samples written to {args.out}")
    failed = any(case["correctness"]["failures"] for case in report["cases"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
