"""Paired group inference; related rows are never independent trials."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from fair_compare.runner import Outcome


@dataclass(frozen=True)
class PairedResult:
    n: int
    groups: int
    nonzero_groups: int
    accuracy_a: float
    accuracy_b: float
    only_a_correct: int
    only_b_correct: int
    group_mean_diff: float
    group_sign_flip_p: float
    diff_ci95: tuple[float, float]


def group_sign_flip_exact(deltas: tuple[Fraction, ...]) -> float:
    """Two-sided tail of |sum(s_g * delta_g)|, over all cluster signs.

    Under the null, independent clusters must have exchangeable arm labels
    (equivalently, independent uniform signs conditional on magnitudes).
    Equal expected accuracy alone is not sufficient. Zero deltas need no
    enumeration: their duplicated assignments cancel from the ratio.
    """
    if not deltas:
        raise ValueError("sign-flip test needs non-empty group deltas")
    nonzero = tuple(d for d in deltas if d)
    if len(nonzero) > 20:
        raise ValueError("exact enumeration allows at most 20 nonzero groups")
    observed = abs(sum(nonzero, Fraction()))
    hits = sum(
        abs(sum((s * d for s, d in zip(signs, nonzero, strict=True)),
                Fraction())) >= observed
        for signs in product((-1, 1), repeat=len(nonzero))
    )
    return hits / float(2 ** len(nonzero))


def _group_deltas(
    a: tuple[Outcome, ...], b: tuple[Outcome, ...],
) -> tuple[Fraction, ...]:
    if [o.case_id for o in a] != [o.case_id for o in b]:
        raise ValueError("paired comparison needs the same cases, same order")
    if len({o.case_id for o in a}) != len(a):
        raise ValueError("paired comparison needs unique case ids")
    if [o.group for o in a] != [o.group for o in b]:
        raise ValueError("paired comparison needs the same groups per case")
    groups: dict[str, list[int]] = {}
    for x, y in zip(a, b, strict=True):
        if not x.group:
            raise ValueError("paired comparison needs non-empty group ids")
        groups.setdefault(x.group, []).append(int(y.correct) - int(x.correct))
    return tuple(
        Fraction(sum(groups[g]), len(groups[g])) for g in sorted(groups)
    )


def group_bootstrap_ci(
    deltas: tuple[Fraction, ...], resamples: int, seed: int,
) -> tuple[float, float]:
    """Percentile interval for the equally weighted mean cluster delta."""
    if not deltas or resamples < 1:
        raise ValueError("non-empty groups and positive resamples required")
    rng = random.Random(seed)
    values = [float(d) for d in deltas]
    diffs = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(resamples)
    )
    return (diffs[int(0.025 * (resamples - 1))],
            diffs[int(0.975 * (resamples - 1))])


def paired_compare(
    a: tuple[Outcome, ...] | None,
    b: tuple[Outcome, ...] | None,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedResult:
    if not a or not b:
        raise ValueError("paired comparison needs non-empty outcomes")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    deltas = _group_deltas(a, b)
    pairs = tuple(zip(a, b, strict=True))
    return PairedResult(
        n=len(a), groups=len(deltas),
        nonzero_groups=sum(bool(d) for d in deltas),
        accuracy_a=sum(o.correct for o in a) / len(a),
        accuracy_b=sum(o.correct for o in b) / len(b),
        only_a_correct=sum(x.correct and not y.correct for x, y in pairs),
        only_b_correct=sum(y.correct and not x.correct for x, y in pairs),
        group_mean_diff=float(sum(deltas, Fraction()) / len(deltas)),
        group_sign_flip_p=group_sign_flip_exact(deltas),
        diff_ci95=group_bootstrap_ci(deltas, resamples, seed),
    )
