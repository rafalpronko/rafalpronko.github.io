---
layout: post
lang: en
title: "Building a benchmark: baselines, constraints, and fair comparisons"
date: 2024-09-19 09:00:00 +0200
permalink: /blog/building-a-benchmark/
author: Rafał Prońko
categories: [engineering]
description: "A decision contract, a competent baseline, a correctness gate and paired timing: a small Python benchmark that ends with a bounded conclusion instead of a winner."
image: /assets/images/building-a-benchmark.svg
---

# Building a benchmark: baselines, constraints, and fair comparisons

![Diagram of a benchmark workflow: decision contract, same cases, correctness gate with a failure log, paired measurement and a bounded conclusion.]({{ '/assets/images/building-a-benchmark.svg' | relative_url }})

<time datetime="2024-09-19">19 September 2024</time>

> **Dates.** 19 September 2024 is the editorial date of this article in the series, not a record of when anything was run. The code was written and executed on 12 October 2026 with CPython 3.12.11 and uv 0.7.8. Every Python API it uses already existed in Python 3.12.0, released on 2 October 2023.

"Which one is faster?" sounds like a question with one answer. It almost never is.

A comparison of two implementations, two models or two retrieval setups only makes sense for a particular input, a particular moment in the pipeline and a particular cost you care about. Change one of them and the answer can flip. A benchmark that hides those choices does not tell you which option is better. It tells you which option won under conditions nobody wrote down.

So I treat a benchmark as a small decision procedure. Before I time anything I want to know what I'm deciding, what I'm comparing against, which outputs count as correct and which part of the work sits inside the timer. Only then do the numbers mean something.

I'll go through those pieces in order and then put them together in a small, runnable Python example: answering exact integer membership queries by scanning a list versus building a set first. It's a deliberately simple problem. That makes it easier to see where a fair comparison can go wrong.

## Start with the decision, not the tools

The first thing I write down is the decision the benchmark is supposed to support. Not "is a set faster than a list" but something like:

> A batch of integer membership queries arrives together with a list of integers that is new for that batch. For each batch, should I answer the queries by scanning the list, or build a set first and look the queries up in it?

That sentence already fixes several things:

- **The unit of work** is one batch, not one query.
- **The list changes from batch to batch**, so building the set is a cost paid per batch. It can't be done once and forgotten.
- **The output is defined**: one boolean per query, in query order.

I call this the decision contract. It also says how I'll read the result. In the example I fixed the rule before running anything. For each paired repetition I compute the ratio of candidate time to baseline time. The candidate counts as faster for a case only if every one of those ratios is below 1.0 and the **median of the paired ratios** is at most 0.90. The baseline counts as faster only if every ratio is above 1.0 and the median of the paired ratios is at least 1.10. Everything else, including ties, is **inconclusive**.

Those thresholds aren't a statistical law. They are choices, and 1.10 was picked on its own rather than as the exact reciprocal of 0.90. The point is to make the choice before seeing the data, so I can't move the line afterwards to suit the result.

## The baseline has to be competent

A benchmark is only as honest as its baseline. If I compare a new approach with a deliberately weak version of the old one, I will always "win", and the result is useless.

In Python the obvious strawman for membership is a hand-written loop:

```python
def naive(items, q):
    for x in items:
        if x == q:
            return True
    return False
```

That loop runs one bytecode iteration and one comparison at a time in the interpreter. `q in items` performs the same linear scan inside the list's C implementation, so it does the same algorithmic work with much less interpreter overhead. That is the baseline I compare against.

The same rule applies elsewhere. A new retrieval method should be compared with a reasonably configured keyword search, not one with default settings nobody would ship. A new model should face a tuned simple model, not an untuned one. If the baseline needs work to be fair, that work is part of the benchmark.

## Correctness before speed

A fast wrong answer is not a result. It is a bug with a timing attached.

So every variant first goes through a **correctness gate**: it runs once on each case and its output is compared with the baseline's output. A variant that returns a different list or raises an exception is recorded as a failure and is not timed at all. It doesn't appear in the timing table, where someone might read it as a fast option.

This matters more than it seems. Optimisations often change behaviour at the edges: empty inputs, duplicates, values that are close but not equal, ordering. In the example the tests check exactly those situations: empty items, empty queries, duplicates in both, hits and misses, and that no variant mutates its input.

The gate only proves agreement on the cases I generated. It doesn't prove the implementations are equivalent in general. For this problem the semantics are simple enough that I'm comfortable with that. For anything with floating point, tokenisation or ranking, "equal output" needs a careful definition of its own.

## Same inputs, same resources

Each variant must see the same data. In the example the cases are synthetic and deterministic: a seed and a case name produce the same items and queries every time. I control the parts that obviously affect the result:

- **Input size**: 64 items or 20,000 items.
- **Query count**: 4 or 200 queries per batch.
- **Hit/miss ratio**: exactly half the queries are present. Misses are negative numbers, which can never appear among the non-negative items, so the expected result is known up front.

Membership by scanning is sensitive to the hit ratio. A hit stops the scan early. A miss walks the whole list. A benchmark with only hits or only misses would answer a different question.

Resources need the same care. Both variants run in the same process, on the same interpreter, with the same garbage-collector setting. If one option got more memory, more threads or a warmer cache by construction, the comparison would measure that difference rather than the algorithm.

## Where the timer starts: measurement boundaries

The `timeit` documentation is explicit that the execution time of *setup* is excluded from the timed run. That is useful, because it lets you measure a statement on its own. It is also a trap. If building an index or loading a model sits in setup, it disappears from the number.

In my contract the set has to be built for every batch, so building it **must** be inside the timer. This excerpt is from `workload.py`; the imports and the `Case` type are defined in the full file:

```python
def set_build_and_probe(case: Case) -> list[bool]:
    """Candidate, end-to-end: building the set is part of the timed work."""
    lookup = set(case.items)
    return [q in lookup for q in case.queries]
```

There is a different, legitimate question: what if the same list is reused across many batches, so the set is built once and amortised? That situation calls for a separate variant with a separate label (another excerpt from the same file):

```python
def set_probe_only(case: Case) -> list[bool]:
    """Amortized view: the set was built before the timer started."""
    lookup = case.prebuilt
    return [q in lookup for q in case.queries]
```

I report both but use only the end-to-end variant for the verdict. Mixing them up is how "100× faster" ends up in a slide deck when the real gain, after paying for the index, is much smaller or negative.

There is a second boundary, and my example sits on the inside of it. The contract talks about a new list for every batch, but the benchmark doesn't generate or read a new list for each timed call. Each case is generated once, and every timed call reuses the same lists, already in memory and, after the warmups, likely in CPU caches. The candidate really does rebuild its set on every call. What isn't timed is producing the input: reading or generating it, allocating new integer objects, and touching memory that hasn't been touched yet. So the benchmark is a **warm, in-memory proxy** for the decision, not a measurement of first contact with fresh data. If producing the input dominated the real pipeline, I'd need a different benchmark.

The same split appears everywhere: index build versus query time in vector search, model load versus inference, compile versus run. None of these boundaries is wrong. What's wrong is leaving the reader to guess which one you used.

## Warm, cold and repetitions

The first call of a function is often slower than the following ones. Memory gets allocated, caches fill up, the operating system pages things in. Whether that cold path matters depends on the contract. A command-line tool that runs once is a cold-start problem. A long-running service mostly lives in the warm state.

My example measures the warm state. Each case gets two untimed warmup passes over every variant before the timed repetitions start. That is a choice, and the report records it.

A single measurement says very little, so each case is repeated nine times. Very short operations add another problem: the work can be close to the timer's own resolution. The clock I use is `time.perf_counter_ns()`, a high-resolution counter for short durations. Only the difference between two readings is meaningful. Returning integer nanoseconds avoids float precision loss, but the unit is not the resolution: on the machine I used, `time.get_clock_info("perf_counter")` reported a resolution of about 42 ns. So the cheapest cases run the batch many times inside one sample (2,000 times for the smallest case) and the result is divided by the loop count. That is the same idea `timeit` uses with its `number` argument. It also means each per-batch figure is an average over the loops in one sample, not the time of one individual batch.

One more difference from `timeit`: by default it switches garbage collection off while timing, to make runs more comparable. Its documentation also warns that GC can be an important part of the performance being measured. I left GC on, because a real batch job would run with it on, and recorded that in the report.

## Pair the measurements

If I time all runs of the baseline first and all runs of the candidate afterwards, anything that changes over time (another process starting, the CPU changing frequency, the machine warming up) lands on one variant only. That's a bias that looks exactly like a performance difference.

So each repetition times **all** variants on the same case, one after another, and the order inside each repetition is shuffled with a seeded random generator. Then I compare the candidate with the baseline repetition by repetition: nine ratios of candidate time to baseline time. The order of every repetition is saved next to the raw samples, so I can check later whether running first or last mattered.

This is a shortened excerpt of `paired_verdict` from `harness.py`. The full function also returns the ratios and their summary, and needs the `statistics` import and the two threshold constants defined in that file:

```python
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
    ...  # shortened: the full function returns a dict with the verdict
```

The second condition requires every ratio to be strictly above 1.0. A tie in one pair therefore can't turn into a baseline win, and a unit test checks exactly that.

Pairing doesn't remove noise, and it doesn't fully stop drift either. A disturbance that hits the middle of one repetition still lands on whichever variant was running. What pairing does is reduce the risk that slow drift or a fixed execution order is charged consistently to one side, and it makes it obvious when noise is larger than the effect.

## Tuning set versus held-out cases

In model evaluation everyone knows that you don't tune hyperparameters on the test set. The same applies to performance work, even if it's rarely said out loud. If I keep changing the cases, the thresholds or the number of repetitions until the candidate wins, the final run isn't a measurement any more. It's a search for a favourable configuration.

Here, in plain terms, is what I changed and why. Originally the two "many queries" cases had 2,000 queries, and the 64-item/many-queries case ran 20 loops per sample. In a quick single-call check (an ad hoc measurement recorded only in my working session log, not part of the published results), scanning a 20,000-item list for 2,000 queries took about four seconds for one batch. That made the whole run far longer than the one-minute budget I wanted for an example readers can rerun. I lowered the query count to 200 in both "many queries" cases and, so that the cheaper small case stays well above the timer resolution, raised its loops from 20 to 200. Those were changes for runtime, made before the run reported below. Fewer queries make the comparison less favourable to the set, not more. The verdict thresholds and the warmup and repeat counts were not changed.

To be clear about the status of these cases: the four cases I report are the same cases I developed the code against. They are not a held-out set. For a real decision I'd go further: pick a few cases to develop on, then confirm on cases I didn't look at while tuning, such as a different seed, a different size or, best of all, real inputs.

## The example

The full code is standard-library only and split into small files:

- [`workload.py`]({{ '/assets/examples/benchmark/workload.py' | relative_url }}): deterministic cases and the three variants,
- [`harness.py`]({{ '/assets/examples/benchmark/harness.py' | relative_url }}): correctness gate, paired timing and the verdict rule,
- [`env_info.py`]({{ '/assets/examples/benchmark/env_info.py' | relative_url }}): Python, OS, CPU, uv and clock information,
- [`run_benchmark.py`]({{ '/assets/examples/benchmark/run_benchmark.py' | relative_url }}): command-line entry point that writes raw samples to JSON,
- [`test_benchmark.py`]({{ '/assets/examples/benchmark/test_benchmark.py' | relative_url }}): correctness and report-schema tests, with no assertions about speed,
- [`pyproject.toml`]({{ '/assets/examples/benchmark/pyproject.toml' | relative_url }}) and [`README.md`]({{ '/assets/examples/benchmark/README.md' | relative_url }}).

Download the files into one directory and run:

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m unittest -v test_benchmark
.venv/bin/python run_benchmark.py --out results/membership.json
```

Any Python 3.12+ interpreter works without uv too: `python -m unittest -v test_benchmark` and `python run_benchmark.py --out results/membership.json`. On Windows the interpreter is `.venv\Scripts\python`.

The snippets below are excerpts. They rely on imports and on the `CaseSpec`, `Case` and `Variant` types defined in the full files, so copy the files rather than the snippets if you want to run the code.

The data generator is the piece that makes the cases comparable (excerpt from `workload.py`, docstring omitted):

```python
def make_case(spec: CaseSpec, seed: int) -> Case:
    if not 0.0 <= spec.hit_ratio <= 1.0:
        raise ValueError("hit_ratio must be between 0 and 1")
    rng = random.Random(f"{seed}:{spec.name}")
    items = [rng.randrange(0, 4 * spec.n_items + 1) for _ in range(spec.n_items)]
    n_hits = round(spec.n_queries * spec.hit_ratio) if items else 0
    hits = [rng.choice(items) for _ in range(n_hits)]
    misses = [-1 - rng.randrange(0, 4 * spec.n_items + 1)
              for _ in range(spec.n_queries - n_hits)]
    queries = hits + misses
    rng.shuffle(queries)
    return Case(spec, items, queries, frozenset(items))
```

Items are drawn with replacement, so duplicates are expected, which is the situation a set quietly collapses. The baseline is a single line (excerpt, docstring omitted):

```python
def list_scan(case: Case) -> list[bool]:
    items = case.items
    return [q in items for q in case.queries]
```

And the timing itself is deliberately boring (excerpt from `harness.py`, docstring omitted):

```python
def time_once(fn: Variant, case: Case, loops: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(case)
    return time.perf_counter_ns() - start
```

Each raw sample, with the shuffled order of every repetition, goes into the JSON file. The summary is derived from those samples, never the other way round.

## What I measured

The tables below retain the original run, before the tie-handling correction: 13 tests passed, and the benchmark exited with code 0 and no correctness failures. Both were run on 12 October 2026 from the directory above the example, with commands equivalent to the ones shown earlier: `examples/.venv/bin/python -m unittest discover -s examples -v` and `examples/.venv/bin/python examples/run_benchmark.py --out evidence/membership.json`. Only the paths and the output file name differ. No pair in that stored run was an exact tie, so the correction leaves its verdicts unchanged. The environment: CPython 3.12.11 on macOS 26.5.2, Apple M4 Max (arm64, 14 logical CPUs), uv 0.7.8, GC enabled, `perf_counter` backed by `mach_absolute_time()`. The machine was not isolated. Other jobs were running on it at the same time, and the spread in the raw samples is consistent with that.

Per-batch elapsed time in milliseconds, minimum and median of nine samples each:

| Case (items / queries) | `list_scan` min | `list_scan` median | `set_build_and_probe` min | `set_build_and_probe` median | `set_probe_only` min (amortised view) |
|---|---:|---:|---:|---:|---:|
| 64 / 4 | 0.0007 | 0.0025 | 0.0004 | 0.0008 | 0.0001 |
| 64 / 200 | 0.0300 | 0.0349 | 0.0024 | 0.0024 | 0.0020 |
| 20,000 / 4 | 0.2392 | 18.9504 | 0.3284 | 13.3023 | 0.0002 |
| 20,000 / 200 | 10.4154 | 313.1589 | 0.2828 | 0.4503 | 0.0042 |

And the paired end-to-end comparison (candidate time divided by baseline time, per repetition):

| Case | Median of paired ratios | Range of the 9 ratios | Candidate faster in | Verdict |
|---|---:|---|---:|---|
| 64 / 4 | 0.561 | 0.003 to 33.4 | 7 of 9 | inconclusive |
| 64 / 200 | 0.079 | 0.001 to 2.01 | 8 of 9 | inconclusive |
| 20,000 / 4 | 0.751 | 0.012 to 137 | 5 of 9 | inconclusive |
| 20,000 / 200 | 0.003 | 0.0004 to 0.757 | 9 of 9 | candidate faster |

The median of the paired ratios is not the same thing as the ratio of the two medians in the first table. In the 20,000 / 4 case, dividing the candidate's median by the baseline's gives 13.3023 / 18.9504 ≈ 0.702, while the median of the nine per-repetition ratios is 0.751. The verdict rule uses the paired version because it compares measurements taken under the same conditions within one repetition.

These are batch timings from one session on one machine. The median is itself a percentile, the 50th, and nothing stops me from computing an empirical 95th percentile from nine values. But with nine samples that number sits at or next to the largest value and tells me very little about the tail of the underlying distribution. More importantly, each figure is a batch time averaged over the loops in one sample, not an observation of a single request, so none of these numbers are request-latency percentiles of any kind.

## Reading the result honestly

The first thing the raw vectors show is noise far larger than any algorithmic effect in three of the four cases. In the 20,000 / 4 case the baseline's raw samples, each covering 20 batches, range from about 4.8 ms to 575 ms. That is about 0.24 ms to 28.7 ms per batch, a factor of over 100. In the 20,000 / 200 case a single list-scan batch took 10.4 ms at best and 887 ms at worst. The code and the input don't change between those repetitions. The spread is consistent with interference from the other work on the machine, but these samples can't prove the cause.

This is what the `timeit` documentation warns about: higher values in the result vector are typically caused by other processes interfering, and the minimum is a lower bound on how fast the machine can run the code. It is not a typical time. That is why I show both, and why the medians in the table above shouldn't be read as "typical cost on a quiet machine".

So what can I say?

**20,000 items, 200 queries: the set wins, and the contract supports it.** The candidate was faster in all nine pairs. Even the worst pair had a ratio of 0.757. The minimums tell the same story: about 10.4 ms for the scan against 0.28 ms for building the set and probing it. With half the queries missing, each miss walks all 20,000 items. In this warm, in-memory setting, building one set is cheaper than that, even counting the build.

**The other three cases are inconclusive, and that is the honest answer.** In the small cases the set looks better at the minimum and median, but one or two pairs went the other way by a large factor, so the rule I fixed in advance doesn't let me call it.

**20,000 items, 4 queries is the interesting one.** Here the minimums point the *other* way: 0.239 ms for scanning against 0.328 ms for building the set and probing. That fits the intuition that building a 20,000-element set costs more than four scans when only half of them run to the end. The paired data can't confirm it. The candidate won five pairs out of nine, and the ratios range from 0.012 to 137. If this case mattered to me, the right next step would be to rerun it on a quiet machine with more repetitions, not to pick whichever statistic agrees with my expectation.

The amortised column is a reminder of why boundaries matter. With the set built outside the timer, probing four queries took about 0.0002 ms at best, three orders of magnitude below the end-to-end figure. Both numbers are real. They answer different questions.

## Count failures and limit the claim

In this run no variant failed the correctness gate. In a real comparison some will: a configuration runs out of memory, a model times out on long inputs, a retrieval setup returns nothing for a class of queries. Those outcomes belong in the report with a reason, not in a footnote, and certainly not filtered out before computing the average. The example writes them to a `failures` list and exits with code 1 if any variant fails, so a broken candidate can't produce a clean-looking table.

Then there's the limit on what the result says. The conclusion above holds for:

- exact integer membership, one batch at a time,
- these sizes and a 50% hit ratio,
- the same in-memory case timed repeatedly after warmups, with the set rebuilt on every call but no new input generated, read or allocated inside the timer,
- CPython 3.12.11 on this machine, under the load it had at the time.

It doesn't say that sets are always faster, how the result looks when every batch brings genuinely new data from a file or network, or what happens with strings, other hit ratios, or PyPy. If those matter, they're new cases to add, not conclusions to extrapolate.

## The same discipline beyond micro-benchmarks

None of this is specific to `list` and `set`. The same five steps carry over to the comparisons I care more about.

**Model evaluation.** The contract is the business decision and its metric. The competent baseline is a tuned simple model. Correctness becomes valid predictions on every record, with failures counted. Pairing means evaluating both models on the same examples and comparing per-example errors. The tuning set versus held-out split is the classic train/validation/test discipline.

**RAG and retrieval.** The boundary question is index build versus query time, and whether re-embedding a changed corpus counts. Correctness is relevance against a fixed set of questions with known evidence. Pairing means the same questions for every configuration. The bounded conclusion is "this chunking was better on this corpus and question set", not "this chunking is better".

**Packing and optimisation heuristics.** The baseline is a decent greedy algorithm. The correctness gate is a validator that checks every solution is feasible before scoring it. Pairing means the same instances and the same time budget for each method. The next planned article in this series applies those rules to a greedy packing baseline and a local improvement, again without declaring a universal winner.

In each case the cheapest mistake to avoid is the same: comparing numbers that were produced under different contracts.

## Where I start

When someone asks me which option is faster or better, I don't start with a tool. I write the decision down, choose a baseline I'd actually be willing to ship, and define what a correct output is. Then I decide what goes inside the timer, generate the cases, and fix the verdict rule before I look at a single number.

The result is often less exciting than a headline. Here it's one clear win, three "inconclusive" cases and a note that the machine was busy. But it's a result I can defend, rerun and extend. A benchmark that ends with a bounded conclusion is more useful than one that ends with a winner.

### Sources

- [Python 3.12: `timeit`](https://docs.python.org/3.12/library/timeit.html): setup time is excluded from the timed run; passing callables adds call overhead; `timeit()` temporarily turns off garbage collection by default; the minimum of a repeat vector is a lower bound, and higher values typically come from interference by other processes. Checked against the `Doc/library/timeit.rst` file at the CPython `v3.12.0` tag.
- [Python 3.12: `time`](https://docs.python.org/3.12/library/time.html): `perf_counter()` is a high-resolution clock for short durations whose reference point is undefined; `perf_counter_ns()` was added in Python 3.7; `get_clock_info()` reports a clock's resolution. Checked against `Doc/library/time.rst` at the `v3.12.0` tag.
- [Python 3.12.0 release](https://www.python.org/downloads/release/python-3120/): release date 2 October 2023.
- [Example code and the raw result of the run described above]({{ '/assets/examples/benchmark/README.md' | relative_url }}): the measured JSON report is [`results/membership-2026-10-12.json`]({{ '/assets/examples/benchmark/results/membership-2026-10-12.json' | relative_url }}).
