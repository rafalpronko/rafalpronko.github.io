def _slice_table(comparison):
    lines = [f"{'slice':<26}{'n':>3}{'base':>6}{'cand':>6}{'fixed':>7}{'regr':>6}"]
    for (dim, value), s in sorted(comparison.slices.items()):
        label = "overall" if dim == "all" else f"{dim}={value}"
        lines.append(f"{label:<26}{s.n:>3}{s.baseline_pass:>6}{s.candidate_pass:>6}"
                     f"{s.fixed:>7}{s.regressed:>6}")
    return lines


def _case_line(pair, label, score):
    case = pair.case
    tag = "[CRITICAL] " if case["critical"] else ""
    return f"  {tag}{case['case_id']} ({case['kind']}, {case['split']}) {label}: {'; '.join(score.reasons)}"


def _critical_first(pairs):
    return sorted(pairs, key=lambda p: not p.case["critical"])


def render(comparison, gate):
    """Plain-text report: per-case changes are listed before any aggregate can hide them."""
    unchanged = [p for p in comparison.pairs if p.change == "kept_fail"]
    lines = ["Regressions (baseline passed, candidate fails):"]
    lines += [_case_line(p, "candidate", p.candidate)
              for p in _critical_first(comparison.regressions)] or ["  none"]
    lines += ["Fixes (baseline failed, candidate passes):"]
    lines += [_case_line(p, "baseline", p.baseline) for p in comparison.fixes] or ["  none"]
    lines += ["Unchanged failures (both runs fail):"]
    lines += [_case_line(p, "candidate", p.candidate) for p in _critical_first(unchanged)] or ["  none"]
    lines += ["", "Pass counts; denominator = all cases in slice, abstentions included:"]
    lines += _slice_table(comparison)
    lines += ["", f"GATE: {'PASS' if gate.passed else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in gate.failures]
    return "\n".join(lines)
