# Membership benchmark: list scan versus set build plus lookup

Companion code for the article *Building a benchmark: baselines, constraints,
and fair comparisons*. Standard library only; Python 3.12 or newer.

## The decision contract

A batch of exact integer membership queries arrives together with a list of
integers that is new for that batch. For each batch we either:

- **baseline `list_scan`**: answer every query with `q in items` on the list, or
- **candidate `set_build_and_probe`**: build `set(items)` and then answer
  every query with `q in lookup`.

Building the set is paid once per batch, so it is **inside** the timed
region of the candidate. That is the end-to-end comparison and the only one
used for the verdict.

`set_probe_only` uses a set built before the timer starts. It is a separate,
labelled view of the case where the set can be reused across many batches
(amortized construction). It is not part of the end-to-end verdict.

### What the benchmark actually times

The benchmark is a **warm, in-memory proxy** for that decision. Each case is
generated once; every timed call then reuses the same lists, which are
already in memory and, after the warmups, likely in CPU caches. The candidate
rebuilds its set on every call, but generating or reading new input, creating
new input objects and cold-cache behaviour are **not** timed. Results describe
repeated work on warmed data, not first contact with fresh data.

### Verdict rule

For each case the harness computes, per paired repetition, the ratio
candidate time / baseline time, and then:

- `candidate faster`: every ratio is below 1.0 **and** the median of the
  ratios is at most 0.90;
- `baseline faster`: every ratio is above 1.0 **and** the median of the
  ratios is at least 1.10;
- otherwise (including ties) `inconclusive`.

The statistic is the median of the paired ratios, not the ratio of the two
medians. 1.10 is chosen independently and is not the exact reciprocal of 0.90.
The thresholds are a choice made before running, not a statistical law.

## Files

| File | Purpose |
|---|---|
| `workload.py` | deterministic synthetic cases and the three variants |
| `harness.py` | correctness gate, paired timing with `time.perf_counter_ns`, verdict |
| `env_info.py` | Python, OS, CPU, uv version and clock information for the report |
| `run_benchmark.py` | command-line entry point; writes raw samples to JSON |
| `test_benchmark.py` | correctness and schema tests (no speed assertions) |

## Run

From the directory containing these files:

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m unittest -v test_benchmark
.venv/bin/python run_benchmark.py --out results/membership.json
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.
Without uv, any Python 3.12+ interpreter works the same way:
`python -m unittest -v test_benchmark` and
`python run_benchmark.py --out results/membership.json`.

The published result file was produced from the parent directory with
equivalent commands that differ only in paths and the output destination:

```bash
examples/.venv/bin/python -m unittest discover -s examples -v
examples/.venv/bin/python examples/run_benchmark.py --out evidence/membership.json
```

Options: `--data-seed` (default 20240919), `--order-seed` (default 7),
`--repeats` (default 9), `--warmups` (default 2).

## What the output means

The table shows the minimum and median **per-batch elapsed time** in
milliseconds, computed from the raw samples saved in the JSON file. Each raw
sample stores the total elapsed nanoseconds for `loops` consecutive batches;
the per-batch summary divides that total by `loops` and converts it to
milliseconds. Tiny cases use more loops so the sample is well above the
clock resolution.

- A per-batch time here is an average over the loops of one sample. It is not
  an observation of a single request, so these numbers cannot be read as
  request-latency percentiles.
- The minimum is closer to "how fast this can run here"; the median is closer
  to "a typical run in this session". Look at the whole raw vector before
  drawing conclusions.
- Fixed seeds make the **inputs** and the **execution order** reproducible.
  They do not make the timings deterministic: other processes, CPU frequency
  and memory state still vary between runs.
- Garbage collection stays enabled during timing (unlike the default in
  `timeit`), and the report records that.

The script exits with code 1 if any variant fails the correctness gate.
