"""Command line entry point: python -m retrieval_lab.report OUT_DIR

Writes report.json (every candidate list, top-k and per-query metric) and
report.md (summary table plus per-query diagnostics).
"""

import json
import platform
import sys
from pathlib import Path

from retrieval_lab.experiment import run_experiment

COLUMNS = ["evidence_recall", "full_support", "reciprocal_rank", "self_contained",
           "parent_hit", "ceiling_full_support", "context_sentences", "redundant_evidence_chunks"]


def summary_table(report: dict) -> list[str]:
    lines = ["| chunking | method | chunks | " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 3)]
    for run in report["runs"]:
        cells = [str(run["summary"][c]) for c in COLUMNS]
        lines.append(f"| {run['chunking']} | {run['method']} | {run['n_chunks']} | " + " | ".join(cells) + " |")
    return lines


def query_table(report: dict) -> list[str]:
    lines = ["| query | purpose | " + " | ".join(f"{r['chunking']}/{r['method']}" for r in report["runs"]) + " |",
             "|" + "---|" * (len(report["runs"]) + 2)]
    for i, qid in enumerate(report["purpose"]):
        cells = []
        for run in report["runs"]:
            q = run["queries"][i]
            m = q["metrics"]
            cells.append(f"R={m['evidence_recall']:.2f} RR={m['reciprocal_rank']:.2f}"
                         f"{' P' if m['parent_hit'] else ''}{' C' if q['ceiling_full_support'] else ''}")
        lines.append(f"| {qid} | {report['purpose'][qid]} | " + " | ".join(cells) + " |")
    return lines


def candidate_lists(report: dict) -> list[str]:
    lines = []
    for run in report["runs"]:
        lines.append(f"\n### {run['chunking']} / {run['method']}\n")
        for q in run["queries"]:
            top = ", ".join(c["chunk_id"] for c in q["top_k"])
            cands = ", ".join(f"{c['chunk_id']}({c['score']})" for c in q["candidates"])
            lines.append(f"- {q['qid']}: top-k [{top}] | candidates [{cands}]")
    return lines


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_experiment(depth=5, k=3)
    report["platform"] = {"python": sys.version.split()[0], "platform": platform.platform()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# Retrieval lab report", "", f"Settings: {report['settings']}", "",
          f"Executed with Python {report['platform']['python']} on {report['platform']['platform']}", "",
          "## Means over 10 queries", "", *summary_table(report), "",
          "## Per query (R = evidence recall@k, RR = reciprocal rank of complete evidence,"
          " P = parent document in top-k, C = complete evidence inside candidate depth)", "",
          *query_table(report), "", "## Candidate lists", *candidate_lists(report)]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(summary_table(report)))
    # Supplementary, not part of the controlled grid: a larger k for sentence chunks,
    # to see how much of the window advantage is simply a bigger context budget.
    budget = run_experiment(depth=10, k=6)
    rows = summary_table(budget)
    (out / "report-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    (out / "report-budget.md").write_text("\n".join(["# Supplementary run: depth=10, k=6", "", *rows]) + "\n")
    print("\nSupplementary depth=10, k=6:")
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
