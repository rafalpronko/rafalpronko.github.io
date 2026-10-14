"""Print the diagnostic report for the hand-authored policy-assistant fixtures.

Run: python3 demo.py [--k 2] [--fixtures fixtures]   (Python 3.11 or newer)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import harness as h

HERE = Path(__file__).resolve().parent


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def cell(result: h.ConditionResult | None) -> str:
    if result is None:
        return "-"
    if result.ok:
        return "ok"
    if not result.readable:
        return "unparsed"
    return "format" if result.correct else "wrong"


def gold_cell(r: h.CaseResult) -> str:
    if not r.answerable:
        return "n/a"
    return "yes" if r.gold_in_context else "no"


def report(fixtures: Path, k: int) -> list[str]:
    corpus, data = h.load_fixtures(fixtures)
    cases = data["cases"]
    exemplars = data.get("fewshot_exemplars", [])
    if not isinstance(exemplars, list) or not all(isinstance(e, str) for e in exemplars):
        raise ValueError("fewshot_exemplars must be a list of case ids")
    dev, holdout = h.grouped_split(cases, data.get("holdout_groups"))
    leaks = h.leaking_groups(dev, holdout)
    if leaks:
        raise ValueError(f"groups in both splits: {sorted(leaks)}")
    h.check_exemplars(exemplars, dev, holdout)
    holdout_ids = {c["id"] for c in holdout}
    evaluated = [h.evaluate_case(case, corpus, k) for case in cases]
    # Wide enough for the longest retrieved list, so large k stays readable.
    top_w = max(22, 2 + max(len(",".join(r.retrieved_ids)) for r in evaluated))

    lines = [
        "HAND-AUTHORED FIXTURES: every answer string was written by hand to",
        "illustrate a failure pattern. None is a model output. Counts are not",
        "a quality measurement of prompting, RAG or fine-tuning.",
        "",
        f"k={k}  passages={len(corpus)}  cases={len(cases)}",
        f"dev groups:     {sorted({c['group'] for c in dev})}",
        f"holdout groups: {sorted({c['group'] for c in holdout})}",
        f"groups in both splits: {sorted(leaks) or 'none'}",
        f"few-shot exemplars {exemplars}: all from dev groups",
        "",
        f"{'case':<5}{'split':<8}{'group':<14}{'retrieved top-k':<{top_w}}{'gold@k':<8}"
        f"{'closed':<9}{'oracle':<9}{'+ex':<9}{'rag':<9}label",
    ]
    labels: dict[str, Counter] = {"dev": Counter(), "holdout": Counter()}
    rag_errors = []
    results = []
    for r in evaluated:
        label = h.diagnose(r)
        split = "holdout" if r.case_id in holdout_ids else "dev"
        labels[split][label] += 1
        results.append((r, label))
        c = r.conditions
        lines.append(
            f"{r.case_id:<5}{split:<8}{r.group:<14}{','.join(r.retrieved_ids):<{top_w}}"
            f"{gold_cell(r):<8}{cell(c['closed_book']):<9}"
            f"{cell(c['oracle']):<9}{cell(c.get('oracle_fewshot')):<9}"
            f"{cell(c['retrieved']):<9}{label}"
        )
        if c["retrieved"].errors:
            rag_errors.append(f"  {r.case_id}: {'; '.join(c['retrieved'].errors)}")

    lines += [
        "",
        "cell legend: ok | format = right content, broken contract | wrong = wrong content",
        "             unparsed = decision/limit could not be read, so content is not graded",
        "gold@k: n/a for cases with no gold passage (abstention cases)",
    ]
    answerable = [r for r, _ in results if r.answerable]
    found = sum(r.gold_in_context for r in answerable)
    lines += [f"gold recall@{k} over answerable cases: {found}/{len(answerable)}", ""]
    lines.append("needs external knowledge (closed-book wrong, oracle right): "
                 + ", ".join(r.case_id for r, _ in results if r.needs_external_knowledge))
    lines += ["", "labels and next step:"]
    for label in sorted({lab for _, lab in results}):
        ids = [r.case_id for r, lab in results if lab == label]
        lines.append(f"  {label:<21}{','.join(ids):<22}{h.NEXT_STEP[label]}")
    lines += ["", "label counts (illustrative cases, not a metric):"]
    for split in ("dev", "holdout"):
        lines.append(f"  {split:<8}" + ", ".join(f"{k_}={v}" for k_, v in sorted(labels[split].items())))
    lines += ["", "contract gate on the end-to-end (rag) answers would reject:"] + rag_errors

    current = sum(p["status"] == "current" for p in corpus.values())
    k_current = min(k, current)
    lines += ["", f"retrieval with a status=current filter, k={k_current} of {current} current passages"
                  " (computed, answers not re-observed):"]
    changed = False
    for case in cases:
        before = h.retrieve(case["question"], corpus, k)
        after = h.retrieve(case["question"], corpus, k_current, current_only=True)
        if before != after:
            changed = True
            lines.append(f"  {case['id']}: {','.join(before)} -> {','.join(after)}")
    if not changed:
        lines.append("  no change")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=positive_int, default=2, help="passages retrieved per question")
    parser.add_argument("--fixtures", type=Path, default=HERE / "fixtures")
    args = parser.parse_args(argv)
    try:
        lines = report(args.fixtures, args.k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
