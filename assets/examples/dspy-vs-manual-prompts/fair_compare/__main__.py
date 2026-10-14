"""CLI: python -m fair_compare --out DIR (DIR must be new or empty)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_compare.experiment import run_experiment

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    report = run_experiment(args.data, args.out)
    text = json.dumps(report, indent=2)
    (args.out / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
