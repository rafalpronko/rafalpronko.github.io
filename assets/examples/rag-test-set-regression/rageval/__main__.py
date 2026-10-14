"""Usage: python -m rageval [--corpus PATH] [--dataset PATH] [--baseline PATH]
[--candidate PATH] [--policy PATH]. Exit 0 = gate pass, 1 = gate fail, 2 = invalid input."""
import argparse
import sys
from pathlib import Path

from rageval.compare import compare
from rageval.gate import evaluate_gate, validate_policy
from rageval.load import load_json
from rageval.report import render
from rageval.score import score_case
from rageval.validate import ValidationError, validate_corpus, validate_dataset, validate_run

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULTS = {"corpus": "corpus.json", "dataset": "dataset.json", "baseline": "baseline.json",
            "candidate": "candidate.json", "policy": "gate_policy.json"}


def _parse(argv):
    parser = argparse.ArgumentParser(prog="python -m rageval")
    for name, filename in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=Path, default=DATA / filename)
    return parser.parse_args(argv)


def _score_run(path, dataset, corpus):
    run = load_json(path)
    responses = validate_run(run, dataset, validate_corpus(corpus))
    scores = {c["case_id"]: score_case(c, responses[c["case_id"]]) for c in dataset["cases"]}
    return run["run_id"], scores


def run_report(args):
    corpus, dataset = load_json(args.corpus), load_json(args.dataset)
    cases = validate_dataset(dataset, corpus)
    policy = validate_policy(load_json(args.policy))
    base_id, base = _score_run(args.baseline, dataset, corpus)
    cand_id, cand = _score_run(args.candidate, dataset, corpus)
    comparison = compare(cases, base, cand)
    gate = evaluate_gate(comparison, policy)
    header = (f"dataset {dataset['dataset_id']}@{dataset['dataset_version']} | "
              f"corpus {corpus['corpus_id']}@{corpus['corpus_version']}\n"
              f"baseline {base_id} -> candidate {cand_id}\n")
    return header + render(comparison, gate), gate.passed


def main(argv=None):
    args = _parse(argv)
    try:
        text, passed = run_report(args)
    except ValidationError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
