"""Text report; build all lines before the CLI emits any stdout."""
from __future__ import annotations

import itertools
import math

from replay_metrics import Summary, exact_mcnemar, nearest_rank, pareto_front, shown, summarize, violations, wilson_interval
from replay_validation import Attempt, Constraints, LogError, Price, check_paired


def _ms(values: list[float]) -> str:
    if not values:
        return "n=0 (no values)"
    return (f"n={len(values)} p50={nearest_rank(values, 50):.0f} "
            f"p95={nearest_rank(values, 95):.0f} max={max(values):.0f}")


def _candidate_lines(s: Summary, rows: dict[str, Attempt]) -> list[str]:
    lo, hi = wilson_interval(s.useful, s.attempts)
    answerable = [a for a in rows.values() if a.expected is not None]
    abstainable = [a for a in rows.values() if a.expected is None]
    out = [f"[{s.name}]",
           f"  useful completions: {s.useful}/{s.attempts} = {s.quality:.3f} "
           f"(Wilson 95% formula {lo:.3f}-{hi:.3f}; denominator = all attempts)",
           "  status counts: " + " ".join(f"{k}={v}" for k, v in s.counts.items()),
           f"  timeout rate: {s.rate('timeout'):.3f}",
           f"  wrong answers: {s.wrong_answers}  wrong abstentions: {s.wrong_abstentions}",
           f"  routing correct: {sum(a.correct for a in answerable)}/{len(answerable)}; "
           f"correct abstentions: {sum(a.correct for a in abstainable)}/{len(abstainable)}",
           "  by slice: " + " ".join(f"{k}={g}/{t}" for k, (g, t) in s.by_slice.items()),
           f"  response duration ms, schema-valid only: {_ms(s.valid_latencies)}",
           "  attempt duration to completion or cancellation ms, all attempts: "
           + _ms(s.all_latencies),
           f"  mean output tokens per attempt: {s.mean_output_tokens:.1f}"]
    if s.cost_units is None:
        out.append("  cost: no prices supplied (two-objective selection)")
    else:
        per = s.cost_per_useful
        per_txt = "infinity (no useful completions)" if per == math.inf else f"{per:.4f}"
        out.append(f"  cost units: total={s.cost_units:.4f} per useful completion={per_txt}")
    return out + [""]


def _paired_lines(logs: dict[str, dict[str, Attempt]], case_ids: list[str]) -> list[str]:
    out = []
    for x, y in itertools.combinations(logs, 2):
        both = only_x = only_y = neither = 0
        for cid in case_ids:
            cx, cy = logs[x][cid].correct, logs[y][cid].correct
            both += cx and cy
            only_x += cx and not cy
            only_y += cy and not cx
            neither += not cx and not cy
        out += [f"paired {x} vs {y}: both={both} only_{x}={only_x} "
                f"only_{y}={only_y} neither={neither}",
                f"  exact McNemar two-sided p={exact_mcnemar(only_x, only_y):.3f} "
                f"on {only_x + only_y} discordant cases", ""]
    return out


def _selection_lines(summaries: list[Summary], c: Constraints) -> list[str]:
    out = [f"empirical hard constraints: quality >= {shown(c.min_quality, 2)}, "
           f"malformed rate <= {shown(c.max_malformed_rate, 2)}, "
           f"p95 attempt duration <= {shown(c.max_p95_all_ms, 0)} ms"]
    passing = []
    for s in summaries:
        bad = violations(s, c)
        out.append(f"  {s.name}: " + ("PASS" if not bad else "REJECT: " + "; ".join(bad)))
        if not bad:
            passing.append(s)
    front = pareto_front(passing)
    out.append("pareto front among passing candidates: "
               + (", ".join(s.name for s in front) if front else "(none)"))
    return out


def render(logs: dict[str, dict[str, Attempt]], prices: dict[str, Price] | None,
           constraints: Constraints, *, fixture_provenance: bool = False) -> list[str]:
    case_ids = check_paired(logs)
    if prices is not None and (missing := logs.keys() - prices.keys()):
        raise LogError("prices must cover every candidate; missing=" + repr(sorted(missing)))
    summaries = [summarize(n, a, None if prices is None else prices[n])
                 for n, a in logs.items()]
    source = ("FIXTURE elapsed_ms from the logs (hand-designed, not measured inference)"
              if fixture_provenance else "elapsed_ms supplied by logs; provenance not verified")
    statistics = ("formula demonstration only; deterministic curated fixtures provide no sampling evidence"
                  if fixture_provenance else "inference requires representative independent sampling units")
    objectives = "quality + p95 attempt duration"
    objectives += (" + cost per useful completion (three objectives)" if prices is not None
                   else " (two objectives)")
    out = [f"paired cases: {len(case_ids)} (identical case_id, expected, slice)",
           f"duration source: {source}", f"statistics: {statistics}",
           "percentile method: nearest-rank, rank = ceil(p/100 * n); p50 is lower median for even n",
           f"Pareto objectives: {objectives}", ""]
    for s in summaries:
        out += _candidate_lines(s, logs[s.name])
    tokens = [s.mean_output_tokens for s in summaries]
    if max(tokens) > 1.25 * min(tokens):
        out += ["WARNING: mean output tokens differ by more than 25%; duration "
                "and cost partly reflect output length, not only the model.", ""]
    return out + _paired_lines(logs, case_ids) + _selection_lines(summaries, constraints)
