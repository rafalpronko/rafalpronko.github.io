"""Toy release gate: may this exact artifact be released under this policy?"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate JSON key in {keys}")
    return dict(pairs)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_reject_constant,
                      object_pairs_hook=_reject_duplicates)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _is_count(x):
    return type(x) is int and x >= 0  # type() excludes bool


def _is_rate(x):
    return type(x) is float and math.isfinite(x) and 0.0 <= x <= 1.0


def _is_identifier(x):
    return isinstance(x, str) and bool(x.strip())


def check_identity(report, label, require_policy_version=False):
    if not isinstance(report, dict):
        raise ValueError(f"{label} must be a JSON object")
    corpus = report.get("eval_corpus")
    if (not isinstance(corpus, dict)
            or not all(_is_identifier(corpus.get(key)) for key in ("id", "version"))):
        raise ValueError(f"{label}: eval_corpus needs nonempty string id and version")
    if require_policy_version and not _is_identifier(report.get("policy_version")):
        raise ValueError(f"{label}: policy_version must be a nonempty string")


def check_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")
    check_identity(policy, "policy", require_policy_version=True)
    for key in ("overall_floor", "critical_floor", "max_regression"):
        if not _is_rate(policy.get(key)):
            raise ValueError(f"policy {key} must be a finite float in [0, 1]")
    critical = policy.get("critical_slices")
    if (not _is_count(policy.get("min_slice_total"))
            or not isinstance(critical, list) or not critical
            or not all(isinstance(name, str) for name in critical)):
        raise ValueError("policy is incomplete or malformed")


def slice_counts(report, label):
    """Return {slice: (correct, total)}; raise ValueError on bad input."""
    slices = report.get("slices") if isinstance(report, dict) else None
    if not isinstance(slices, dict) or not slices:
        raise ValueError(f"{label}: report has no slices")
    counts = {}
    for name, c in slices.items():
        if not isinstance(c, dict) or set(c) != {"correct", "total"}:
            raise ValueError(f"{label}: slice {name!r} needs correct and total")
        correct, total = c["correct"], c["total"]
        if (not (_is_count(correct) and _is_count(total))
                or total == 0 or correct > total):
            raise ValueError(f"{label}: invalid counts in slice {name!r}: {c}")
        counts[name] = (correct, total)
    return counts


def evaluate(policy, candidate, baseline, artifact_sha256):
    """Return blocking reasons. An empty list means: eligible for release."""
    check_policy(policy)
    check_identity(candidate, "candidate", require_policy_version=True)
    check_identity(baseline, "baseline")
    cand = slice_counts(candidate, "candidate")
    base = slice_counts(baseline, "baseline")
    reasons = []
    if candidate.get("artifact_sha256") != artifact_sha256:
        reasons.append("report does not describe this artifact digest")
    if candidate.get("policy_version") != policy["policy_version"]:
        reasons.append("report was produced for another policy version")
    for label, report in (("candidate", candidate), ("baseline", baseline)):
        if report.get("eval_corpus") != policy["eval_corpus"]:
            reasons.append(f"{label} evaluated on {report.get('eval_corpus')}, "
                           f"policy requires {policy['eval_corpus']}")
    critical = set(policy["critical_slices"])
    for name in sorted({"overall", *critical, *base}):
        if name not in cand:
            reasons.append(f"{name}: missing from candidate report")
            continue
        correct, total = cand[name]
        rate = correct / total
        if total < policy["min_slice_total"]:
            reasons.append(f"{name}: {total} examples, policy needs "
                           f"{policy['min_slice_total']}")
        floor = (policy["overall_floor"] if name == "overall"
                 else policy["critical_floor"] if name in critical else None)
        if floor is not None and rate < floor:
            reasons.append(f"{name}: {rate:.4f} below floor {floor}")
        if name not in base:
            reasons.append(f"{name}: no baseline to compare against")
        elif rate < base[name][0] / base[name][1] - policy["max_regression"]:
            reasons.append(f"{name}: {rate:.4f} regressed from baseline "
                           f"{base[name][0] / base[name][1]:.4f}")
    return reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--policy", "--candidate", "--baseline", "--artifact"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    try:
        digest = sha256_file(args.artifact)
        reasons = evaluate(load_json(args.policy), load_json(args.candidate),
                           load_json(args.baseline), digest)
    except (OSError, ValueError) as exc:
        print(f"BLOCKED: cannot evaluate gate: {exc}")
        return 2
    for reason in reasons:
        print(f"BLOCKED: {reason}")
    if reasons:
        return 1
    print(f"ELIGIBLE: sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
