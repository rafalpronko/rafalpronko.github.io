"""Quality, duration, formula demonstrations and globally consistent selection."""
from __future__ import annotations

import math
from dataclasses import dataclass

from replay_validation import Attempt, Constraints, LogError, Price, STATUSES, VALID_RESPONSE

Z_95 = 1.959964


def nearest_rank(values: list[float], percentile: int) -> float:
    """Nearest-rank percentile: the smallest value with at least
    `percentile`% of observations at or below it, i.e. the value at
    1-based rank ceil(percentile / 100 * n) of the sorted sample.
    No interpolation; the result is always an observed value."""
    if not values:
        raise ValueError("no values")
    if not 1 <= percentile <= 100:
        raise ValueError("percentile must be an integer in 1..100")
    n = len(values)
    rank = (percentile * n + 99) // 100  # integer ceil, no float rounding
    return sorted(values)[rank - 1]


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_mcnemar(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** n)


@dataclass(frozen=True)
class Summary:
    name: str
    attempts: int
    useful: int
    counts: dict[str, int]
    wrong_answers: int
    wrong_abstentions: int
    by_slice: dict[str, tuple[int, int]]
    valid_latencies: list[float]
    all_latencies: list[float]
    cost_units: float | None
    mean_output_tokens: float

    @property
    def quality(self) -> float:
        return self.useful / self.attempts

    def rate(self, status: str) -> float:
        return self.counts.get(status, 0) / self.attempts

    @property
    def p95_all(self) -> float:
        return nearest_rank(self.all_latencies, 95)

    @property
    def cost_per_useful(self) -> float | None:
        if self.cost_units is None:
            return None
        if self.useful == 0:
            return math.inf
        return self.cost_units / self.useful


def summarize(name: str, attempts: dict[str, Attempt],
              price: Price | None) -> Summary:
    rows = list(attempts.values())
    counts = {s: sum(a.status == s for a in rows) for s in sorted(STATUSES)}
    by_slice: dict[str, tuple[int, int]] = {}
    for a in rows:
        good, total = by_slice.get(a.slice, (0, 0))
        by_slice[a.slice] = (good + a.correct, total + 1)
    tokens_in = sum(a.input_tokens for a in rows)
    tokens_out = sum(a.output_tokens for a in rows)
    return Summary(
        name=name,
        attempts=len(rows),
        useful=sum(a.correct for a in rows),
        counts=counts,
        wrong_answers=sum(a.status == "ok" and not a.correct for a in rows),
        wrong_abstentions=sum(a.status == "abstain" and not a.correct
                              for a in rows),
        by_slice=dict(sorted(by_slice.items())),
        valid_latencies=[a.elapsed_ms for a in rows if a.status in VALID_RESPONSE],
        all_latencies=[a.elapsed_ms for a in rows],
        # Every logged token is costed. The harness must record billable usage
        # consistent with the provider contract, including failures where billed.
        cost_units=None if price is None else price.cost(tokens_in, tokens_out),
        mean_output_tokens=tokens_out / len(rows),
    )


def shown(value: float, decimals: int) -> str:
    """Fixed decimals when exact, otherwise the shortest exact repr (e.g. 0.4, not 0)."""
    text = f"{value:.{decimals}f}"
    return text if float(text) == value else repr(value)


def _breach(label: str, observed: float, op: str, limit: float, decimals: int,
            unit: str = "") -> str:
    """Round the observed value only while the printed comparison stays true."""
    text = f"{observed:.{decimals}f}"
    if not (float(text) < limit if op == "<" else float(text) > limit):
        text = repr(observed)
    return f"{label} {text}{unit} {op} {shown(limit, decimals)}{unit}"


def violations(s: Summary, c: Constraints) -> list[str]:
    found = []
    if s.quality < c.min_quality:
        found.append(_breach("quality", s.quality, "<", c.min_quality, 3))
    if s.rate("malformed") > c.max_malformed_rate:
        found.append(_breach("malformed rate", s.rate("malformed"), ">",
                             c.max_malformed_rate, 3))
    if s.p95_all > c.max_p95_all_ms:
        found.append(_breach("p95 attempt duration", s.p95_all, ">",
                             c.max_p95_all_ms, 0, " ms"))
    return found


def dominates(a: Summary, b: Summary, *, include_cost: bool) -> bool:
    """Use the same objective set as every other comparison in this front."""
    pairs = [(a.quality, b.quality), (-a.p95_all, -b.p95_all)]
    if include_cost:
        if a.cost_per_useful is None or b.cost_per_useful is None:
            raise LogError("cost objective requires prices for every candidate")
        pairs.append((-a.cost_per_useful, -b.cost_per_useful))
    return all(x >= y for x, y in pairs) and any(x > y for x, y in pairs)


def pareto_front(summaries: list[Summary]) -> list[Summary]:
    priced = [s.cost_units is not None for s in summaries]
    if any(priced) and not all(priced):
        raise LogError("mixed price availability: prices must cover every candidate")
    include_cost = bool(priced) and all(priced)
    return [s for s in summaries if not any(
        dominates(o, s, include_cost=include_cost) for o in summaries if o is not s)]
