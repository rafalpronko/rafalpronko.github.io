---
layout: post
lang: en
title: "Greedy packing versus local improvement: measure, don't declare victory"
date: 2024-09-26 09:00:00 +0200
permalink: /blog/greedy-packing-local-improvement/
author: Rafał Prońko
categories: [optimization, benchmarking]
description: "First-fit decreasing and a bounded local repair for one-dimensional bin packing in Python 3.11: a paired benchmark, gaps to the lower bound and where a local search stops helping."
image: /assets/images/greedy-packing-local-improvement.svg
---

# Greedy packing versus local improvement: measure, don't declare victory

![Diagram: first-fit decreasing packs items 5, 4, 4, 3, 2, 2 into three bins of capacity 10; a local repair re-packs those three bins into two full ones. A note reports the corpus outcome: uniform 3 improve, 17 tie, 0 worse; triplet 20 improve, still 3 bins above optimum.]({{ '/assets/images/greedy-packing-local-improvement.svg' | relative_url }})

<time datetime="2024-09-26">26 September 2024</time>

A greedy heuristic produces a solution. You add a local search on top of it, run it on a small example and one bin disappears. The demo looks convincing, so the obvious conclusion is "the repair works".

That conclusion skips three questions. How often does the repair help on instances it has never seen? How far is the result from the best possible answer, not just from the greedy one? And what does the extra step cost?

In this post I build a small experiment to answer those questions for one-dimensional bin packing: a first-fit decreasing baseline, a bounded local repair, a test suite and a paired benchmark on a fixed synthetic corpus. Python 3.11 and the standard library are enough; pytest is the only test dependency.

A note on dates: the post is dated September 2024, but every number below comes from runs made on 12 October 2026 with Python 3.11.12 on the machine described in the method section. Nothing here is a historical measurement.

## The problem and the only bound I trust

Bin packing: there are bins of equal capacity and items with sizes. Every item has to be packed exactly once, no bin may exceed its capacity, and the goal is to use as few bins as possible. The [OR-Tools guide](https://developers.google.com/optimization/pack/bin_packing) uses the same formulation and distinguishes it from the multiple knapsack problem. I use the guide only for the definition; no solver appears in this post.

In my version capacities and sizes are positive integers. Every item has its own identifier, so two items of size 4 are still two different items. That matters in tests: a packing that loses one 4 and duplicates the other has the right sizes and the wrong content.

I need one more number: a **lower bound**. No packing can use fewer than `ceil(total size / capacity)` bins, because even perfectly full bins hold exactly `capacity` each.

It is a bound, not the optimum. Three items of size 6 in bins of capacity 10 give `ceil(18 / 10) = 2`, but no two of them fit together, so the optimum is 3. When I report "gap to the lower bound" below, a gap of 1 can mean "one bin too many" or "already optimal". I'll come back to that.

## Baseline: first-fit decreasing

First-fit decreasing (FFD) sorts items from largest to smallest and puts each one into the first open bin that still has room. If none has room, it opens a new bin.

Fragment of [`packing/ffd.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/ffd.py' | relative_url }}) (docstring omitted; runs as part of the downloadable package):

```python
def first_fit_decreasing(capacity, items):
    order = sorted(validate_instance(capacity, items), key=lambda it: -it.size)
    bins, free = [], []
    for item in order:
        for index, room in enumerate(free):
            if item.size <= room:
                bins[index].append(item)
                free[index] -= item.size
                break
        else:
            bins.append([item])
            free.append(capacity - item.size)
    return tuple(tuple(bin_) for bin_ in bins)
```

Two details make the result reproducible. `sorted()` is [guaranteed to be stable](https://docs.python.org/3.11/library/functions.html#sorted), so items of equal size keep their input order. And the function returns tuples built from the validated input, so it never modifies the caller's list.

`validate_instance` rejects anything that is not a positive `int`, including `True`. In Python `bool` is a subclass of `int`, so a check like `isinstance(size, int)` would quietly accept `True` as an item of size 1. The code checks `type(value) is int` instead. It also rejects items larger than the capacity and duplicate identifiers.

## Why "just move items out of a bin" does not work after FFD

The first repair idea is simple: take a lightly loaded bin and try to move its items into the free space of other bins. If all of them fit, the bin is gone.

After FFD this idea is weaker than it looks. When FFD placed an item in bin `j`, every earlier bin had less free space than the item's size. Free space only shrinks afterwards, so at the end that item still fits into no earlier bin. For the last bin there are no later bins at all: in the packing exactly as FFD returns it, none of its items can be moved to another bin on its own. So no single move can start emptying the last bin; other items would have to move first, and that is already a change to several bins. (This argument covers the last bin of an unmodified FFD packing only; items in earlier bins may still have single moves into later bins.)

So the repair has to change several bins at once.

## The repair: re-pack a small window of bins

The repair looks at a **window**: one target bin plus one or two partner bins. It pools their items and runs an exhaustive search that tries to fit the pool into as many bins as there are partners. Success means one bin less.

The rules are part of the result, so I keep them explicit in the module docstring of [`packing/repair.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/repair.py' | relative_url }}):

- **Order.** Targets from the least to the most loaded bin, ties broken by index. Partner sets come from `itertools.combinations`, which [emits them in lexicographic order](https://docs.python.org/3.11/library/itertools.html#itertools.combinations), smaller sets first.
- **Acceptance.** A move is accepted only if a bin disappears. There are no sideways or worsening moves, so the result can never be worse than the input.
- **Budget.** One unit of work is one window examined or one tentative placement during the search. The budget is shared by the whole call.
- **Termination.** The repair stops when it reaches the lower bound, when a full scan finds no improving move, or when the budget runs out.
- **Rollback.** The search works on copies. A failed or interrupted search leaves the last accepted packing untouched.

Fragment of `packing/repair.py`, the scan:

```python
def improve_once(capacity, bins, window, budget):
    """First improving move in the documented order, or None."""
    loads = [load(b) for b in bins]
    targets = sorted(range(len(bins)), key=lambda i: (loads[i], i))
    for t in targets:
        others = [i for i in range(len(bins)) if i != t]
        for k in range(1, window):
            for partners in combinations(others, k):
                budget.spend("windows")
                if loads[t] + sum(loads[p] for p in partners) > k * capacity:
                    continue
                pool = bins[t] + tuple(it for p in partners for it in bins[p])
                packed = pack_exact(capacity, pool, k, budget)
                if packed is not None:
                    touched = {t, *partners}
                    kept = tuple(b for i, b in enumerate(bins)
                                 if i not in touched)
                    return kept + packed
    return None
```

The size check before the search is cheap and rejects most windows: if the pooled items weigh more than `k` full bins, no search is needed.

The search itself (`pack_exact`) is a plain depth-first search. Items go in from largest to smallest. Bins with the same free space are interchangeable for the current item, so it tries only one of them. Every placement costs one unit of the budget.

Stopping with `no_improving_move` means: **no window of up to three bins can be re-packed into fewer bins**. It is a local optimum for this neighbourhood. It is not a proof that the packing is optimal.

## What the tests pin down

The suite has 54 tests. Four test-first cycles are recorded with a failing (RED) and a passing (GREEN) run: model, FFD and repair against stubs that raised `NotImplementedError`; corpus and evaluation; the work budget described below; and a later fix for one-shot iterables. That does not mean every assertion failed first: the plateau fixture was replaced after the code existed, and the four invalid-budget cases passed at RED for the wrong reason (the old keyword name). The two budget tests for a mid-search stop and for a stop after an accepted move were added after review as regression tests for behaviour the code already had; they passed on their first run. Besides validation and the invariant checker, the repair tests cover:

- an empty packing, and a packing of full bins that is already at the lower bound (zero work);
- a manufactured fixture that the repair fixes;
- `6, 6, 6` with capacity 10: the search runs, fails and returns the input unchanged — exactly 9 windows and 6 placements;
- budget exhaustion after 1 and 2 units of work, also with the input returned unchanged;
- budget exhaustion in the middle of a search: on the manufactured fixture with `max_work=5`, after 3 windows and 2 placements, the partial search is discarded and the input comes back unchanged;
- budget exhaustion after an accepted move: five single-item bins `6, 6, 6, 2, 2` with capacity 10 and `max_work=3`, where the first window merges a 2 into a 6; the result keeps that improvement (4 bins, 1 removed) and reports `budget`;
- a plateau: the repair stops, while a hand-written packing with one bin fewer passes the same invariant checker;
- determinism, no mutation of the input and "never more bins than the input";
- bins passed as a generator: `repair()` reads its input once, so `removed_bins` has to be counted from that snapshot. The first version iterated the input a second time and reported −2 for a generator; the test now pins 1.

`demo.py` runs the two most instructive fixtures. Output:

```text
$ uv run --python 3.11 python demo.py
manufactured: capacity=10 lower_bound=2
  ffd    3 bins  [5,4]=9 [4,3,2]=9 [2]=2
  repair 2 bins  [5,3,2]=10 [4,4,2]=10
  stop=lower_bound windows=3 placements=13
plateau: capacity=20 lower_bound=3
  ffd    4 bins  [13,5,1]=19 [12,4,3]=19 [9,9]=18 [3]=3
  repair 4 bins  [13,5,1]=19 [12,4,3]=19 [9,9]=18 [3]=3
  stop=no_improving_move windows=24 placements=27
```

The first fixture is the one from the illustration. I built it so that the repair succeeds; it shows the mechanism and proves nothing about frequency.

The second one was selected by [`find_plateau.py`]({{ '/assets/examples/greedy-packing-local-improvement/find_plateau.py' | relative_url }}), a deterministic random search (seed 1, trial 22725) for instances where the repair stops above the optimum. Three bins are enough: `{13, 4, 3}`, `{12, 5, 3}`, `{9, 9, 1}`. That packing takes items from all four FFD bins, so a window of three cannot reach it; with `window=4` the same code finds it. The search stopped after 26,078 trials, once it had found three such instances. Those are selected examples, not a frequency estimate.

## The first benchmark showed that my budget counted the wrong thing

My first version counted only placements in the search as work. On the first benchmark run, the ten uniform instances where the repair found nothing all reported **0 nodes** ("nodes" was that version's name for what the code now counts as placements). Their candidate medians were still between 5.611 and 428.362 ms, against well under a millisecond for FFD.

Part of that turned out to be machine load (more on that below). The structural problem was real, though. On a packing of `b` bins with nothing to improve, the final scan examines `b · ((b − 1) + C(b − 1, 2))` windows. For 25 bins that is 7,500 windows. In those ten runs every window failed the size check, so the search never started and the counter stayed at zero. The budget bounded something that hardly ever happened.

I changed the definition of work to count windows as well, wrote the failing tests for it first, and reran everything. Besides counting windows, the revision changed two details of the scan: bin loads are computed once per scan and cached, and a window's size check sums those cached loads, so a window that cannot fit is rejected before its item pool is built (the first version built the pool and then summed it). Neither change alters which moves are tried or accepted: the bin counts and outcomes of all 40 instances were identical in both runs. Both outputs are kept in the evidence for this post. I'm mentioning this because it is the kind of mistake a single success demo never reveals.

## Method

**Corpus.** [`data/corpus.json`]({{ '/assets/examples/greedy-packing-local-improvement/data/corpus.json' | relative_url }}) holds 40 instances, generated with `random.Random(20240926)`. The seed, the families and the repair parameters were written down before the corpus was generated, and nothing was tuned on it.

- **uniform**, 20 instances: capacity 150, 60 items, sizes uniform in [20, 100].
- **triplet**, 20 instances: capacity 1000, 20 groups of three items that each sum to exactly 1000 (first item 380–490, second 250 up to the remaining room minus 250, third the rest), shuffled. Here the optimum is known by construction: the original triplets are one packing into 20 full bins, which equals the lower bound. The generator does not prove that this is the only perfect packing; items from different triplets may also add up to 1000. What holds either way is that any optimal packing has to fill every bin exactly, and in this corpus FFD never reached the optimum on this family.

The benchmark reads the JSON file rather than calling the generator. The Python documentation [guarantees only that `random()`](https://docs.python.org/3.11/library/random.html#notes-on-reproducibility) keeps its sequence across versions; `randint()` and `shuffle()` may change. A persisted file with item identifiers makes the corpus reproducible regardless.

**Parameters.** `window=3`, `max_work=20000` per instance.

**Measurement.** Each instance is measured end to end: validation, FFD, invariant check, and for the candidate also the repair. One warmup pass, then 21 repeats; in every repeat both variants run on the same instance one after the other, and the order alternates between repeats. Time comes from [`time.perf_counter_ns()`](https://docs.python.org/3.11/library/time.html#time.perf_counter_ns), which is meant for short durations and is valid only as a difference between two calls.

**Machine.** Apple M4 Max (14 cores, 36 GiB), macOS 26.5.2, CPython 3.11.12, uv 0.7.8.

## Results

Output of `bench.py`, the summary lines (the full per-instance table is in the same output). This run was made before a bookkeeping fix in `repair()` (the `removed_bins` counter mentioned above). The benchmark does not use that counter; after the fix I recomputed the deterministic columns (bins, outcomes, stop reasons, windows, placements) and they matched all 40 instances and both summaries exactly, so the timings below are the original ones from that run:

```text
$ uv run --python 3.11 python bench.py
python 3.11.12 on macOS-26.5.2-arm64-arm-64bit
instances=40 max_work=20000 window=3 repeats=21 warmup=1
load average at start: 412.04 410.85 395.86
...
uniform: {'instances': 20, 'improve': 3, 'tie': 17, 'worse': 0, 'ffd_gap_total': 13, 'repaired_gap_total': 10, 'at_lower_bound_ffd': 7, 'at_lower_bound_repaired': 10, 'work_total': 81242, 'work_max': 9477, 'budget_stops': 0}
uniform: median_ms ffd=0.094 ffd+repair=1.342
uniform: min_ms    ffd=0.058 ffd+repair=0.715
triplet: {'instances': 20, 'improve': 20, 'tie': 0, 'worse': 0, 'ffd_gap_total': 80, 'repaired_gap_total': 60, 'at_lower_bound_ffd': 0, 'at_lower_bound_repaired': 0, 'work_total': 117616, 'work_max': 5903, 'budget_stops': 0}
triplet: median_ms ffd=0.103 ffd+repair=3.906
triplet: min_ms    ffd=0.058 ffd+repair=1.396
load average at end: 424.72 415.69 399.34
```

The same in a table:

| | uniform | triplet |
|---|---|---|
| improve / tie / worse | 3 / 17 / 0 | 20 / 0 / 0 |
| total gap to lower bound, FFD → repaired | 13 → 10 bins | 80 → 60 bins |
| instances at the lower bound, FFD → repaired | 7 → 10 | 0 → 0 |
| work units, total (max per instance) | 81,242 (9,477) | 117,616 (5,903) |
| median of per-instance medians, FFD / FFD + repair | 0.094 / 1.342 ms | 0.103 / 3.906 ms |
| median of per-instance minima, FFD / FFD + repair | 0.058 / 0.715 ms | 0.058 / 1.396 ms |

The budget was never exhausted, so it did not influence any outcome.

## What these numbers say

**Uniform.** FFD was already at the lower bound in 7 of 20 instances. Of the remaining 13, the repair brought 3 down to the bound. The other 10 ended one bin above it.

I don't know whether those 10 are improvable. The lower bound may simply be unreachable for them, as in `6, 6, 6`. What I do know: in all 10 the repair made **zero placements**. Not a single window of up to three bins had enough combined free space to absorb a bin. The free space is spread thinly across many bins, so a small window cannot even start a search.

**Triplet.** Twenty improvements out of twenty, never a worse result. Taken alone, that looks like a clear win.

The other number tells a different story. The optimum is 20 bins for every instance. FFD used 24, the repair 23 — every time. The remaining gap is 3 bins per instance, 15% above the optimum. The repair removed a quarter of the gap and then stopped at a plateau in all twenty cases.

"Improves 100% of instances" and "stays 15% above optimal" describe the same run. The first is the number people put in a slide.

**Cost.** Wall-clock times were measured with a load average above 400: other work kept every core busy, and I did not control it. The medians show it. For `uniform-05` the per-instance median of the candidate is 182.643 ms and its minimum 1.914 ms, for the same deterministic 8,450 units of work. The medians and minima in the table are what this run observed, and I report them only as noisy observations. Neither statistic is protected from scheduling under that load, so I don't derive a speed-up or slow-down factor from them. The candidate runs FFD plus the repair, so it cannot be cheaper; how much more it costs in time needs a rerun on a quiet, controlled machine before anyone makes a performance decision.

The work counters do not depend on load at all. They also show where the time goes: in triplet instances, about 5,800 of the roughly 5,900 units are the final scan that proves there is nothing left to do.

## Where this result ends

- The corpus is synthetic and small: two families, 40 instances, 60 items each. Real data with a different size distribution can behave differently.
- I measure against the lower bound. Only for the triplet family is it equal to the optimum. For uniform instances a remaining gap of one bin may already be optimal.
- One neighbourhood, one window size, one budget. A window of four solves the plateau fixture, but the number of windows then grows roughly as the fourth power of the number of bins.
- The timing comes from one heavily loaded, uncontrolled machine. It supports no latency ratio between the variants; only the work counters are reliable cost measurements here.
- The repair proves local optimality for its own neighbourhood only. `no_improving_move` is not "optimal".

## Alternatives

- **Exact methods for evaluation.** For small instances an exact search or a MIP model (the OR-Tools guide shows the formulation) would give the true optimum and turn "gap to the lower bound" into "gap to the optimum". That is a better reference for the uniform family than the bound I used.
- **Better lower bounds.** Items larger than half the capacity can never share a bin, so their count is also a lower bound. Combining bounds narrows the uncertainty without solving the problem.
- **Searches that accept sideways moves.** My repair accepts only moves that remove a bin, which is exactly why it gets stuck. Moves that keep the bin count but concentrate free space can open new windows; they need a different stopping rule and a different cost model.
- **Doing nothing.** The repair is never worse than FFD in bin count, by construction; FFD can be worse, as in 23 of the 40 instances here. FFD still wins on cost: if one extra bin costs less than the engineering and the repair's runtime, the baseline is the better choice.

## So, is the repair worth it?

The answer depends on the price of a bin, not on the demo.

From this experiment I'd take the following decision rules:

1. Keep the repair behind a fixed work budget. When it stops early, use the last packing it accepted, or the FFD packing if it accepted nothing. Every accepted move removed a bin and passed the invariant check, so throwing those moves away gains nothing.
2. Report three numbers together: improve/tie/worse counts, the remaining gap to a bound (or to the optimum, where known) and the cost in deterministic work units plus wall-clock time measured under controlled conditions.
3. Treat a high improvement rate with suspicion until you've seen the remaining gap. The triplet family improved every time and still left 15% on the table.
4. Before tuning the search, check what limits it. Here, in the uniform ties, the size filter rejected every window: a better search would not help, only a different neighbourhood would.

## Run it yourself

All files are available for download:

- [complete example as a ZIP archive]({{ '/assets/examples/greedy-packing-local-improvement/greedy-packing-local-improvement.zip' | relative_url }})
- [`README.md`]({{ '/assets/examples/greedy-packing-local-improvement/README.md' | relative_url }}), [`pyproject.toml`]({{ '/assets/examples/greedy-packing-local-improvement/pyproject.toml' | relative_url }}), [`uv.lock`]({{ '/assets/examples/greedy-packing-local-improvement/uv.lock' | relative_url }})
- code: [`packing/model.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/model.py' | relative_url }}), [`packing/ffd.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/ffd.py' | relative_url }}), [`packing/repair.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/repair.py' | relative_url }}), [`packing/corpus.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/corpus.py' | relative_url }}), [`packing/evaluate.py`]({{ '/assets/examples/greedy-packing-local-improvement/packing/evaluate.py' | relative_url }}) (the package marker `packing/__init__.py` is included in the ZIP archive)
- scripts: [`demo.py`]({{ '/assets/examples/greedy-packing-local-improvement/demo.py' | relative_url }}), [`bench.py`]({{ '/assets/examples/greedy-packing-local-improvement/bench.py' | relative_url }}), [`make_corpus.py`]({{ '/assets/examples/greedy-packing-local-improvement/make_corpus.py' | relative_url }}), [`find_plateau.py`]({{ '/assets/examples/greedy-packing-local-improvement/find_plateau.py' | relative_url }})
- data: [`data/corpus.json`]({{ '/assets/examples/greedy-packing-local-improvement/data/corpus.json' | relative_url }})
- tests: [`tests/helpers.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/helpers.py' | relative_url }}), [`tests/test_model.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/test_model.py' | relative_url }}), [`tests/test_ffd.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/test_ffd.py' | relative_url }}), [`tests/test_repair.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/test_repair.py' | relative_url }}), [`tests/test_corpus.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/test_corpus.py' | relative_url }}), [`tests/test_evaluate.py`]({{ '/assets/examples/greedy-packing-local-improvement/tests/test_evaluate.py' | relative_url }})

Keep the directory layout (`packing/`, `tests/`, `data/`). With [uv](https://docs.astral.sh/uv/), from the unpacked directory:

```bash
uv run --python 3.11 --extra test pytest -q
uv run --python 3.11 python demo.py
uv run --python 3.11 python bench.py
```

Without uv, inside a Python 3.11 virtual environment:

```bash
python -m pip install pytest==8.3.3
python -m pytest -q
python demo.py
python bench.py
```

The quality numbers (bins, outcomes, work units) should match the output above exactly. Times will differ on your machine, and a single run is not a timing result either: if time matters for your decision, measure on a quiet machine, repeat the run and use your own data.

### Sources

- [OR-Tools: The Bin Packing Problem](https://developers.google.com/optimization/pack/bin_packing) — problem formulation and the difference from multiple knapsack (current page, used for the definition only).
- [What's New In Python 3.11](https://docs.python.org/3.11/whatsnew/3.11.html) — Python 3.11 was released on 24 October 2022.
- [Python 3.11: time.perf_counter_ns](https://docs.python.org/3.11/library/time.html#time.perf_counter_ns) — clock for short durations; only differences are meaningful; available since 3.7.
- [Python 3.11: random, notes on reproducibility](https://docs.python.org/3.11/library/random.html#notes-on-reproducibility) — only `random()` is guaranteed to keep its sequence across versions.
- [Python 3.11: sorted](https://docs.python.org/3.11/library/functions.html#sorted) — guaranteed stable sort.
- [Python 3.11: itertools.combinations](https://docs.python.org/3.11/library/itertools.html#itertools.combinations) — lexicographic order of the emitted tuples.
