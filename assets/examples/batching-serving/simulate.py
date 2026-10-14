"""Deterministic virtual-time simulation of single-worker dynamic batching.

All times are integer microseconds on a virtual clock. Nothing is measured:
service time is overhead_us + per_item_us * batch_size, so the output
describes this algorithm under these parameters, not any hardware.
"""

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction

US_PER_S = 1_000_000
SCENARIOS = {"low": 20, "near_b1": 100, "near_b8": 380, "overload": 600}
BATCH_SIZES = (1, 8)
QUANTILES = (0.5, 0.95, 0.99)
COLUMNS = (
    ("scenario", "scenario"),
    ("rate/s", "rate_rps"),
    ("batch", "max_batch_size"),
    ("offered", "offered"),
    ("rejected", "rejected"),
    ("completed", "completed"),
    ("missed", "deadline_missed"),
    ("goodput/s", "goodput_rps"),
    ("done/s", "completed_throughput_rps"),
    ("p50 ms", "latency_p50_ms"),
    ("p95 ms", "latency_p95_ms"),
    ("p99 ms", "latency_p99_ms"),
    ("queue p99 ms", "queue_delay_p99_ms"),
    ("mean batch", "mean_batch_size"),
)


@dataclass(frozen=True)
class Config:
    rate_rps: int
    max_batch_size: int
    max_wait_us: int = 5_000
    overhead_us: int = 8_000
    per_item_us: int = 1_500
    queue_capacity: int = 64
    deadline_us: int = 100_000
    injection_us: int = 10 * US_PER_S

    def validate(self) -> None:
        if self.rate_rps <= 0:
            raise ValueError("rate_rps must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.queue_capacity < self.max_batch_size:
            raise ValueError("queue_capacity must be >= max_batch_size")
        if min(self.max_wait_us, self.overhead_us, self.per_item_us) < 0:
            raise ValueError("wait and service times must be non-negative")
        if self.overhead_us + self.per_item_us == 0:
            raise ValueError("service time must be positive")
        if self.deadline_us <= 0 or self.injection_us <= 0:
            raise ValueError("deadline_us and injection_us must be positive")


@dataclass(slots=True)
class Request:
    arrival_us: int
    outcome: str = "pending"  # rejected | within_deadline | deadline_missed
    start_us: int | None = None
    finish_us: int | None = None
    batch_size: int = 0


def arrivals(config: Config) -> list[int]:
    """Open-loop schedule: fixed before the run, independent of completions."""
    count = (config.rate_rps * config.injection_us + US_PER_S - 1) // US_PER_S
    return [i * US_PER_S // config.rate_rps for i in range(count)]


def next_launch(queue: deque[Request], free_us: int, config: Config) -> int | None:
    """When the next batch leaves if no further requests arrive."""
    if not queue:
        return None
    trigger = queue[0].arrival_us + config.max_wait_us
    if len(queue) >= config.max_batch_size:
        trigger = min(trigger, queue[config.max_batch_size - 1].arrival_us)
    return max(free_us, trigger)


def dispatch(queue: deque[Request], launch_us: int, config: Config) -> int:
    """Start one batch at launch_us and return when the worker is free again."""
    size = min(len(queue), config.max_batch_size)
    finish_us = launch_us + config.overhead_us + config.per_item_us * size
    for _ in range(size):
        request = queue.popleft()
        request.start_us, request.finish_us = launch_us, finish_us
        request.batch_size = size
        late = finish_us - request.arrival_us > config.deadline_us
        request.outcome = "deadline_missed" if late else "within_deadline"
    return finish_us


def simulate(config: Config) -> list[Request]:
    """Run arrivals through a bounded queue and one batching worker, then drain."""
    config.validate()
    requests = [Request(t) for t in arrivals(config)]
    queue: deque[Request] = deque()
    free_us, index = 0, 0
    while index < len(requests) or queue:
        launch_us = next_launch(queue, free_us, config)
        upcoming = requests[index] if index < len(requests) else None
        if launch_us is not None and (upcoming is None or launch_us <= upcoming.arrival_us):
            free_us = dispatch(queue, launch_us, config)
        elif len(queue) >= config.queue_capacity:
            upcoming.outcome = "rejected"
            index += 1
        else:
            queue.append(upcoming)
            index += 1
    return requests


def nearest_rank(values: list[int], q: float) -> int | None:
    """Nearest-rank percentile: the ceil(q * n)-th smallest value, no interpolation."""
    if not 0 < q <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if not values:
        return None
    rank = math.ceil(Fraction(str(q)) * len(values))
    return sorted(values)[rank - 1]


def to_ms(value_us: int | None) -> float | None:
    return None if value_us is None else value_us / 1000


def per_second(count: int, window_us: int) -> float:
    return round(count * US_PER_S / window_us, 2)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def summarize(config: Config, requests: list[Request]) -> dict:
    """Counts, rates, and completed-request percentiles for one run."""
    done = [r for r in requests if r.finish_us is not None]
    rejected = sum(r.outcome == "rejected" for r in requests)
    within = sum(r.outcome == "within_deadline" for r in done)
    window_us = max([config.injection_us, *(r.finish_us for r in done)])
    batches = len({r.start_us for r in done})
    row = {
        **asdict(config),
        "offered": len(requests),
        "admitted": len(requests) - rejected,
        "rejected": rejected,
        "completed": len(done),
        "within_deadline": within,
        "deadline_missed": len(done) - within,
        "observation_window_s": window_us / US_PER_S,
        "offered_rps": per_second(len(requests), config.injection_us),
        "completed_throughput_rps": per_second(len(done), window_us),
        "goodput_rps": per_second(within, window_us),
        "rejection_rate": ratio(rejected, len(requests)),
        "deadline_miss_rate": ratio(len(done) - within, len(requests) - rejected),
        "batches": batches,
        "mean_batch_size": ratio(len(done), batches),
        "max_batch_observed": max((r.batch_size for r in done), default=0),
    }
    latency = [r.finish_us - r.arrival_us for r in done]
    queue_delay = [r.start_us - r.arrival_us for r in done]
    for q in QUANTILES:
        tag = f"p{round(q * 100)}"
        row[f"latency_{tag}_ms"] = to_ms(nearest_rank(latency, q))
        row[f"queue_delay_{tag}_ms"] = to_ms(nearest_rank(queue_delay, q))
    return row


def run_matrix() -> list[dict]:
    rows = []
    for name, rate in SCENARIOS.items():
        for size in BATCH_SIZES:
            config = Config(rate_rps=rate, max_batch_size=size)
            rows.append({"scenario": name, **summarize(config, simulate(config))})
    return rows


def format_table(rows: list[dict]) -> str:
    header = [label for label, _ in COLUMNS]
    body = [["-" if row[key] is None else str(row[key]) for _, key in COLUMNS] for row in rows]
    lines = [header, *body]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(line, widths)) for line in lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    args = parser.parse_args(argv)
    rows = run_matrix()
    print(json.dumps(rows, indent=2) if args.json else format_table(rows))


if __name__ == "__main__":
    main()
