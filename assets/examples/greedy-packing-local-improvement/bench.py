"""Paired benchmark: FFD versus FFD + repair on data/corpus.json.

Run: uv run --python 3.11 python bench.py
Quality comes from packing.evaluate; timing is end to end per instance
(validation, FFD, optional repair, invariant check), 1 warmup pass,
then REPEATS passes that alternate which variant runs first.
Wall-clock time depends on machine load, so the load average is printed
and the minimum is reported next to the median. The work counters
(windows, placements) are deterministic and do not depend on load.
"""
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from packing.corpus import load_corpus
from packing.evaluate import evaluate, summarize
from packing.ffd import first_fit_decreasing
from packing.model import check_packing
from packing.repair import repair

MAX_WORK, WINDOW, REPEATS = 20_000, 3, 21


def baseline(inst):
    bins = first_fit_decreasing(inst.capacity, inst.items)
    check_packing(inst.capacity, inst.items, bins)


def candidate(inst):
    bins = first_fit_decreasing(inst.capacity, inst.items)
    result = repair(inst.capacity, bins, max_work=MAX_WORK, window=WINDOW)
    check_packing(inst.capacity, inst.items, result.bins)


def elapsed_ms(fn, inst):
    start = time.perf_counter_ns()
    fn(inst)
    return (time.perf_counter_ns() - start) / 1e6


def main():
    corpus = load_corpus(Path(__file__).resolve().parent / "data" / "corpus.json")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"instances={len(corpus)} max_work={MAX_WORK} window={WINDOW} "
          f"repeats={REPEATS} warmup=1")
    print("load average at start: %.2f %.2f %.2f" % os.getloadavg())

    rows = [evaluate(i, max_work=MAX_WORK, window=WINDOW) for i in corpus]
    for inst in corpus:  # warmup, not measured
        baseline(inst)
        candidate(inst)
    times = {i.id: {"ffd": [], "ffd+repair": []} for i in corpus}
    for repeat in range(REPEATS):
        for inst in corpus:
            pair = [("ffd", baseline), ("ffd+repair", candidate)]
            for name, fn in pair if repeat % 2 == 0 else pair[::-1]:
                times[inst.id][name].append(elapsed_ms(fn, inst))

    print("\nid          LB  FFD  +rep  outcome  stop               windows"
          "  places  ffd_med  rep_med  rep_min")
    for row in rows:
        t = times[row["id"]]
        print(f"{row['id']:<11} {row['lower_bound']:>3} {row['ffd_bins']:>4}"
              f" {row['repaired_bins']:>5}  {row['outcome']:<7}  "
              f"{row['stop_reason']:<17} {row['windows']:>8}"
              f" {row['placements']:>7}"
              f"  {statistics.median(t['ffd']):6.3f}"
              f"  {statistics.median(t['ffd+repair']):7.3f}"
              f"  {min(t['ffd+repair']):7.3f}")

    print("\nper family (timing = median over instances of the per-instance"
          " median or minimum, in ms)")
    for family, s in summarize(rows).items():
        ids = [r["id"] for r in rows if r["family"] == family]
        med = {name: statistics.median(statistics.median(times[i][name])
                                       for i in ids)
               for name in ("ffd", "ffd+repair")}
        print(f"{family}: {s}")
        low = {name: statistics.median(min(times[i][name]) for i in ids)
               for name in ("ffd", "ffd+repair")}
        print(f"{family}: median_ms ffd={med['ffd']:.3f} "
              f"ffd+repair={med['ffd+repair']:.3f}")
        print(f"{family}: min_ms    ffd={low['ffd']:.3f} "
              f"ffd+repair={low['ffd+repair']:.3f}")
    print("load average at end: %.2f %.2f %.2f" % os.getloadavg())
    return 0


if __name__ == "__main__":
    sys.exit(main())
