"""Correctness gate, paired timing and summaries. Standard library only."""

from __future__ import annotations

import random
import statistics
import time
from typing import Mapping

from workload import BASELINE, CANDIDATE, Case, Variant

# Decision contract for the end-to-end comparison (see README). Both thresholds
# apply to the median of the paired candidate/baseline ratios, and the direction
# must hold strictly in every pair. 1.10 is chosen on its own, not as 1 / 0.90.
WIN_RATIO = 0.90   # candidate faster: every ratio < 1 and median ratio <= 0.90
LOSE_RATIO = 1.10  # baseline faster: every ratio > 1 and median ratio >= 1.10
# Anything else, including ties, is inconclusive.


def correctness_gate(case: Case, variants: Mapping[str, Variant]) -> dict:
    """Run every variant once and compare its output with the baseline.

    Variants that disagree or raise are reported and excluded from timing.
    """
    try:
        expected = variants[BASELINE](case)
    except Exception as exc:
        failure = {"variant": BASELINE, "reason": f"raised {exc!r}"}
        return {"expected_hits": None, "passed": [], "failures": [failure]}
    passed, failures = [], []
    for name, fn in variants.items():
        try:
            got = fn(case)
        except Exception as exc:  # recorded, never silently dropped
            failures.append({"variant": name, "reason": f"raised {exc!r}"})
            continue
        if got != expected:
            failures.append({"variant": name, "reason": "output differs from baseline"})
        else:
            passed.append(name)
    return {"expected_hits": sum(expected), "passed": passed, "failures": failures}


def time_once(fn: Variant, case: Case, loops: int) -> int:
    """Elapsed nanoseconds for ``loops`` consecutive batches."""
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(case)
    return time.perf_counter_ns() - start


def measure(case: Case, variants: Mapping[str, Variant], names: list[str],
            repeats: int, warmups: int, rng: random.Random) -> dict:
    """Paired measurement: every repetition runs all variants on the same case.

    The order inside a repetition is shuffled with a seeded RNG, so no variant
    always benefits from running first or last.
    """
    loops = case.spec.loops
    for _ in range(warmups):
        for name in names:
            variants[name](case)
    samples: dict[str, list[int]] = {name: [] for name in names}
    orders: list[list[str]] = []
    for _ in range(repeats):
        order = list(names)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            samples[name].append(time_once(variants[name], case, loops))
    return {"loops": loops, "samples_ns": samples, "orders": orders}


def summarize_samples(samples_ns: list[int], loops: int) -> dict:
    per_batch_ms = [s / loops / 1e6 for s in samples_ns]
    return {"min_ms": min(per_batch_ms), "median_ms": statistics.median(per_batch_ms)}


def paired_verdict(baseline_ns: list[int], candidate_ns: list[int]) -> dict:
    """Compare candidate with baseline repetition by repetition."""
    ratios = [c / b for b, c in zip(baseline_ns, candidate_ns)]
    wins = sum(r < 1.0 for r in ratios)
    median = statistics.median(ratios)
    if wins == len(ratios) and median <= WIN_RATIO:
        verdict = "candidate faster"
    elif all(r > 1.0 for r in ratios) and median >= LOSE_RATIO:
        verdict = "baseline faster"
    else:
        verdict = "inconclusive"
    return {
        "ratios": ratios,
        "median_ratio": median,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "candidate_wins": wins,
        "pairs": len(ratios),
        "verdict": verdict,
    }


def run_case(case: Case, variants: Mapping[str, Variant], repeats: int,
             warmups: int, rng: random.Random) -> dict:
    gate = correctness_gate(case, variants)
    result = {"spec": vars(case.spec), "correctness": gate}
    if BASELINE not in gate["passed"]:
        result["skipped"] = "baseline failed its own gate"
        return result
    names = [name for name in variants if name in gate["passed"]]
    timing = measure(case, variants, names, repeats, warmups, rng)
    result.update(timing)
    result["summary"] = {
        name: summarize_samples(timing["samples_ns"][name], timing["loops"])
        for name in names
    }
    if CANDIDATE in names:
        result["end_to_end"] = paired_verdict(
            timing["samples_ns"][BASELINE], timing["samples_ns"][CANDIDATE])
    return result
