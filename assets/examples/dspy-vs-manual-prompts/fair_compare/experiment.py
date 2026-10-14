"""Wire the pieces together: split, search, freeze, one holdout run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fair_compare.budget import Ledger
from fair_compare.candidates import load_candidates
from fair_compare.cases import load_dataset
from fair_compare.freeze import HoldoutError, HoldoutGuard, freeze
from fair_compare.runner import Outcome, run_case
from fair_compare.search import ArmLog, search_arm, select_on_dev
from fair_compare.splits import check_label_coverage, split_cases
from fair_compare.stats import paired_compare


@dataclass(frozen=True)
class Config:
    salt: str
    train_pct: int
    dev_pct: int
    optimization_cap: int
    feedback_cap: int
    max_attempts: int
    final_eval_cap: int
    resamples: int
    seed: int


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Config(
        salt=raw["split"]["salt"],
        train_pct=raw["split"]["train_pct"],
        dev_pct=raw["split"]["dev_pct"],
        optimization_cap=raw["optimization_cap_per_arm"],
        feedback_cap=raw["feedback_cap_per_arm"],
        max_attempts=raw["max_attempts_per_case"],
        final_eval_cap=raw["final_eval_cap_per_arm"],
        resamples=raw["bootstrap"]["resamples"],
        seed=raw["bootstrap"]["seed"],
    )


def _arm_report(log: ArmLog) -> dict[str, object]:
    return {
        "dev": [
            {
                "candidate": r.candidate.candidate_id,
                "correct": sum(o.correct for o in r.outcomes),
                "n": len(r.outcomes),
                "invalid": r.count("invalid"),
                "errors": r.count("error"),
                "human_minutes": r.candidate.human_minutes,
            }
            for r in log.results
        ],
        "stop_reason": log.stop_reason,
        "optimization": log.ledger.summary(),
    }


def _final_report(
    outcomes: tuple[Outcome, ...], ledger: Ledger
) -> dict[str, object]:
    return {
        "correct": sum(o.correct for o in outcomes),
        "n": len(outcomes),
        "invalid": sum(o.status == "invalid" for o in outcomes),
        "errors": sum(o.status == "error" for o in outcomes),
        "wrong_ids": [o.case_id for o in outcomes if not o.correct],
        "final_eval": ledger.summary(),
    }


def run_experiment(data_dir: Path, out_dir: Path) -> dict[str, object]:
    # Refuse BEFORE reading inputs, searching or overwriting any artifact.
    # Empty directories are allowed; any partial or completed run is closed.
    if out_dir.is_symlink() or (out_dir.exists() and (
        not out_dir.is_dir() or any(out_dir.iterdir())
    )):
        raise HoldoutError(
            f"output directory already used or incomplete: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_started.txt").open("x", encoding="utf-8") as marker:
        marker.write("Run started; preserve this directory even on failure.\n")
    config = load_config(data_dir / "experiment.json")
    dataset = load_dataset(data_dir / "cases.json")
    fixtures_path = data_dir / "candidates.json"
    arms = load_candidates(fixtures_path)
    labels = dataset.labels

    splits = split_cases(
        dataset.cases, config.salt, config.train_pct, config.dev_pct
    )
    check_label_coverage(splits, labels)
    test = splits.pop("test")
    if config.final_eval_cap < len(test.cases) * config.max_attempts:
        raise ValueError("final eval cap could truncate the holdout run")
    guard = HoldoutGuard(test, out_dir / "holdout_receipt.txt")
    del test  # from here on, only the guard can hand out test cases

    logs = {
        arm: search_arm(
            arm, candidates, splits["train"], splits["dev"], labels,
            Ledger(f"{arm}/optimization", config.optimization_cap),
            config.feedback_cap, config.max_attempts,
        )
        for arm, candidates in arms.items()
    }
    chosen = {arm: select_on_dev(log).candidate for arm, log in logs.items()}
    fixtures_sha = hashlib.sha256(fixtures_path.read_bytes()).hexdigest()
    manifest = freeze(chosen, guard.test_ids_sha256, fixtures_sha)
    (out_dir / "freeze_manifest.json").write_text(
        manifest.to_json() + "\n", encoding="utf-8"
    )

    holdout = guard.open(manifest, chosen)
    final: dict[str, tuple[Outcome, ...]] = {}
    final_ledgers: dict[str, Ledger] = {}
    for arm, candidate in chosen.items():
        ledger = Ledger(f"{arm}/final", config.final_eval_cap)
        final[arm] = tuple(
            run_case(candidate, case, labels, ledger, config.max_attempts)
            for case in holdout.cases
        )
        final_ledgers[arm] = ledger

    paired = paired_compare(
        final["manual"], final["dspy"], config.resamples, config.seed
    )
    return {
        "splits": {
            name: {"cases": len(s.cases), "groups": sorted(s.groups)}
            for name, s in {**splits, "test": holdout}.items()
        },
        "arms": {arm: _arm_report(log) for arm, log in logs.items()},
        "frozen": json.loads(manifest.to_json()),
        "manifest_sha256": manifest.sha256,
        "holdout": {
            arm: _final_report(final[arm], final_ledgers[arm])
            for arm in final
        },
        "paired_dspy_minus_manual": {
            "n": paired.n,
            "only_manual_correct": paired.only_a_correct,
            "only_dspy_correct": paired.only_b_correct,
            "accuracy_diff": round(paired.accuracy_b - paired.accuracy_a, 4),
            "groups": paired.groups,
            "nonzero_groups": paired.nonzero_groups,
            "group_mean_accuracy_diff": round(paired.group_mean_diff, 4),
            "group_sign_flip_exact_p": round(paired.group_sign_flip_p, 4),
            "inference_endpoint": "equally weighted group accuracy difference",
            "null_assumptions": (
                "independent clusters; exchangeable arm labels within each "
                "cluster under the null (uniform independent delta signs)"
            ),
            "group_bootstrap_ci95": [round(x, 4) for x in paired.diff_ci95],
        },
    }
