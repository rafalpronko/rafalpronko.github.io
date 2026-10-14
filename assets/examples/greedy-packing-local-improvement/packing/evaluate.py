"""Paired evaluation of FFD versus FFD + repair (no timing)."""
from collections import defaultdict

from packing.ffd import first_fit_decreasing
from packing.model import check_packing, lower_bound
from packing.repair import repair


def evaluate(instance, *, max_work, window=3):
    cap, items = instance.capacity, instance.items
    bound = lower_bound(cap, items)
    baseline = first_fit_decreasing(cap, items)
    result = repair(cap, baseline, max_work=max_work, window=window)
    check_packing(cap, items, baseline)
    check_packing(cap, items, result.bins)
    before, after = len(baseline), len(result.bins)
    outcome = ("improve" if after < before
               else "worse" if after > before else "tie")
    return {
        "id": instance.id, "family": instance.family,
        "lower_bound": bound, "ffd_bins": before, "repaired_bins": after,
        "ffd_gap": before - bound, "repaired_gap": after - bound,
        "outcome": outcome, "stop_reason": result.stop_reason,
        "windows": result.windows, "placements": result.placements,
        "work": result.work,
    }


def summarize(rows):
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    return {family: {
        "instances": len(group),
        "improve": sum(r["outcome"] == "improve" for r in group),
        "tie": sum(r["outcome"] == "tie" for r in group),
        "worse": sum(r["outcome"] == "worse" for r in group),
        "ffd_gap_total": sum(r["ffd_gap"] for r in group),
        "repaired_gap_total": sum(r["repaired_gap"] for r in group),
        "at_lower_bound_ffd": sum(r["ffd_gap"] == 0 for r in group),
        "at_lower_bound_repaired": sum(r["repaired_gap"] == 0 for r in group),
        "work_total": sum(r["work"] for r in group),
        "work_max": max(r["work"] for r in group),
        "budget_stops": sum(r["stop_reason"] == "budget" for r in group),
    } for family, group in families.items()}
