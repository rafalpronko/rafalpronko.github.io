from dataclasses import dataclass

from rageval.validate import ValidationError

COUNT_KEYS = ("max_critical_failures", "max_critical_regressions", "max_regressions")
POLICY_KEYS = {*COUNT_KEYS, "max_slice_drop", "min_slice_size"}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list


def _is_int(value):
    # bool is a subclass of int: without this check `true` would mean 1.
    return isinstance(value, int) and not isinstance(value, bool)


def validate_policy(policy):
    """Check keys, types and ranges. A threshold that cannot be compared
    meaningfully (true, "0.5", NaN) must be invalid input, never a pass."""
    if not isinstance(policy, dict):
        raise ValidationError("policy: expected an object")
    unknown, missing = set(policy) - POLICY_KEYS, POLICY_KEYS - set(policy)
    if unknown:
        raise ValidationError(f"unknown policy keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"missing policy keys: {', '.join(sorted(missing))}")
    for key in COUNT_KEYS:
        if not (_is_int(policy[key]) and policy[key] >= 0):
            raise ValidationError(f"policy: {key} must be an integer >= 0")
    drop = policy["max_slice_drop"]
    # The range check alone rejects NaN and infinities. math.isfinite would first
    # convert to float, and a JSON integer like 10**309 raises OverflowError there.
    if not ((_is_int(drop) or isinstance(drop, float)) and 0 <= drop <= 1):
        raise ValidationError("policy: max_slice_drop must be a finite number in [0, 1]")
    if not (_is_int(policy["min_slice_size"]) and policy["min_slice_size"] >= 1):
        raise ValidationError("policy: min_slice_size must be an integer >= 1")
    return policy


def _slice_failures(comparison, policy):
    for (dim, value), s in sorted(comparison.slices.items()):
        if s.n < policy["min_slice_size"]:
            continue
        # Absolute drop in pass rate (percentage points / 100), not a relative change.
        drop = (s.baseline_pass - s.candidate_pass) / s.n
        if drop > policy["max_slice_drop"]:
            label = "overall" if dim == "all" else f"{dim}={value}"
            yield f"slice {label} drop {drop:.2f} > allowed {policy['max_slice_drop']:.2f}"


def evaluate_gate(comparison, policy):
    """Release gate: an absolute floor for critical cases, then per-case
    regression counts, then slice rates. Thresholds come from the policy file."""
    validate_policy(policy)
    if not comparison.pairs:
        raise ValidationError("no cases to evaluate")
    failures = []
    failing = sum(p.case["critical"] and not p.candidate.passed for p in comparison.pairs)
    if failing > policy["max_critical_failures"]:
        failures.append(f"critical failures in candidate: {failing} > allowed {policy['max_critical_failures']}")
    critical = sum(p.case["critical"] for p in comparison.regressions)
    if critical > policy["max_critical_regressions"]:
        failures.append(f"critical regressions: {critical} > allowed {policy['max_critical_regressions']}")
    total = len(comparison.regressions)
    if total > policy["max_regressions"]:
        failures.append(f"regressions: {total} > allowed {policy['max_regressions']}")
    failures.extend(_slice_failures(comparison, policy))
    return GateResult(not failures, failures)
