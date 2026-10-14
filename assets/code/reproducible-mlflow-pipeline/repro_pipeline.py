"""Reproducible scaler + Ridge pipeline tracked with MLflow and checked by a gate."""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_SEED = 20240801
N_ROWS = 240
FEATURES = ("x0", "x1", "x2", "x3")
PINNED = ("mlflow", "scikit-learn", "numpy", "cloudpickle", "scipy")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- data ---------------------------------------------------------------------

def make_dataset(path, seed=DATA_SEED, n_rows=N_ROWS) -> Path:
    """Generate the synthetic table once; the written CSV is the source of truth."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 4))
    x[:, 2] = 50 + 10 * x[:, 2]  # one feature on a much larger scale
    y = 3.0 + x @ np.array([1.5, -2.0, 0.05, 0.0]) + rng.normal(scale=0.5, size=n_rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["row_id", "split", *FEATURES, "y"])
        for i in range(n_rows):
            row_id = f"r{i:04d}"
            # Split depends only on the row id, not on the RNG or row order.
            split = "test" if int(sha256_bytes(row_id.encode()), 16) % 5 == 0 else "train"
            writer.writerow([row_id, split, *(f"{v:.6f}" for v in x[i]), f"{y[i]:.6f}"])
    return path


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    data_sha256: str
    split_sha256: str


def load_dataset(path) -> Dataset:
    raw = Path(path).read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    expected = ["row_id", "split", *FEATURES, "y"]
    if not rows or list(rows[0]) != expected:
        raise ValueError(f"expected columns {expected}")
    ids = {"train": [], "test": []}
    values = {"train": [], "test": []}
    for row in rows:
        if row["split"] not in ids:
            raise ValueError(f"unknown split {row['split']!r} in row {row['row_id']}")
        numbers = [float(row[c]) for c in (*FEATURES, "y")]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError(f"non-finite value in row {row['row_id']}")
        ids[row["split"]].append(row["row_id"])
        values[row["split"]].append(numbers)
    if not values["train"] or not values["test"]:
        raise ValueError("both train and test split must be non-empty")
    train, test = np.array(values["train"]), np.array(values["test"])
    return Dataset(
        X_train=train[:, :-1], y_train=train[:, -1],
        X_test=test[:, :-1], y_test=test[:, -1],
        data_sha256=sha256_bytes(raw),
        split_sha256=sha256_bytes(json.dumps(ids, sort_keys=True).encode()),
    )


# --- training and evaluation ------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 1.0
    features: tuple = FEATURES

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))


def select_features(X: np.ndarray, config: TrainConfig) -> np.ndarray:
    # Fancy indexing returns a Fortran-ordered array. With the same values, BLAS may then
    # sum in another order and change the last bits, so always hand the model C order.
    return np.ascontiguousarray(X[:, [FEATURES.index(name) for name in config.features]])


def train(dataset: Dataset, config: TrainConfig) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=config.alpha))])
    # The scaler sees training rows only; Pipeline.fit never touches the test split.
    return model.fit(select_features(dataset.X_train, config), dataset.y_train)


def regression_metrics(y_true, y_pred) -> dict:
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error)))}


def lock_files() -> list:
    return [HERE / "uv.lock"] if (HERE / "uv.lock").exists() else []


def environment_snapshot() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PINNED},
        "lock_files_sha256": {p.name: sha256_bytes(p.read_bytes()) for p in lock_files()},
    }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    model_uri: str
    config: TrainConfig
    metrics: dict
    test_predictions: np.ndarray
    data_sha256: str
    split_sha256: str
    source_sha256: str


def run_experiment(data_path, tracking_dir, config: TrainConfig, run_name: str) -> RunResult:
    data_path = Path(data_path)
    dataset = load_dataset(data_path)
    source = Path(__file__).resolve()
    source_sha256 = sha256_bytes(source.read_bytes())
    X_train = select_features(dataset.X_train, config)
    X_test = select_features(dataset.X_test, config)

    mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
    mlflow.set_experiment("article-42-reproducible-pipeline")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"alpha": config.alpha, "features": ",".join(config.features),
                           "data_seed": DATA_SEED, "model": "StandardScaler+Ridge"})
        mlflow.set_tags({"data_sha256": dataset.data_sha256,
                         "split_sha256": dataset.split_sha256,
                         "source_sha256": source_sha256})
        mlflow.log_artifact(str(source), artifact_path="source")
        mlflow.log_artifact(str(data_path), artifact_path="data")
        mlflow.log_dict(environment_snapshot(), "environment/environment.json")
        for lock in lock_files():
            mlflow.log_artifact(str(lock), artifact_path="environment")

        model = train(dataset, config)
        test_predictions = model.predict(X_test)
        naive = np.full_like(dataset.y_test, dataset.y_train.mean())
        metrics = {
            "train_rmse": regression_metrics(dataset.y_train, model.predict(X_train))["rmse"],
            **{f"test_{k}": v for k, v in regression_metrics(dataset.y_test, test_predictions).items()},
            "naive_test_rmse": regression_metrics(dataset.y_test, naive)["rmse"],
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            pip_requirements=[f"{name}=={version(name)}" for name in PINNED])
        model_uri = f"runs:/{run.info.run_id}/model"

        # Reload what MLflow stored and require identical predictions in this environment.
        reloaded = mlflow.sklearn.load_model(model_uri)
        reload_max_abs_diff = float(np.max(np.abs(reloaded.predict(X_test) - test_predictions)))
        mlflow.log_metric("reload_max_abs_diff", reload_max_abs_diff)
        if reload_max_abs_diff != 0.0:
            raise RuntimeError(f"reloaded model differs: {reload_max_abs_diff}")

    return RunResult(run.info.run_id, model_uri, config, metrics, test_predictions,
                     dataset.data_sha256, dataset.split_sha256, source_sha256)


# --- approved baseline and gate -----------------------------------------------------

def load_baseline(path) -> dict:
    baseline = json.loads(Path(path).read_text())
    if baseline.get("status") != "APPROVED":
        raise ValueError("baseline fixture is not APPROVED")
    return baseline


@dataclass
class GateReport:
    reproducibility_failures: list
    quality_failures: list

    @property
    def passed(self) -> bool:
        return not self.reproducibility_failures and not self.quality_failures

    def summary(self) -> str:
        lines = [f"reproducibility: {'OK' if not self.reproducibility_failures else 'FAIL'}"]
        lines += [f"  - {f}" for f in self.reproducibility_failures]
        lines.append(f"quality: {'OK' if not self.quality_failures else 'FAIL'}")
        lines += [f"  - {f}" for f in self.quality_failures]
        return "\n".join(lines)


def same_evaluation_data(result: RunResult, baseline: dict) -> list:
    failures = []
    if result.data_sha256 != baseline["data_sha256"]:
        failures.append("data_sha256 differs from the approved data")
    if result.split_sha256 != baseline["split_sha256"]:
        failures.append("split_sha256 differs: not the approved test split")
    return failures


def check_reproducibility(result: RunResult, baseline: dict) -> list:
    """Same data, split and config must give the approved predictions."""
    failures = same_evaluation_data(result, baseline)
    approved = TrainConfig(**baseline["config"])
    if result.config != approved:
        failures.append(f"config {result.config} differs from approved {approved}")
    expected = np.array(baseline["test_predictions"])
    actual = np.asarray(result.test_predictions)
    atol = baseline["tolerances"]["prediction_atol"]
    if actual.shape != expected.shape:
        failures.append(f"predictions shape {actual.shape} != {expected.shape}")
    elif not np.allclose(actual, expected, rtol=0.0, atol=atol):
        worst = float(np.max(np.abs(actual - expected)))
        failures.append(f"test predictions differ by up to {worst:.3g} (atol {atol:g})")
    return failures


def check_quality(result: RunResult, baseline: dict) -> list:
    """Any candidate must not be meaningfully worse on the evaluation fixture.

    The "test" rows are a reusable regression fixture, not an untouched holdout: every
    gate run looks at them. Generalization needs a separate final test set.
    """
    failures = same_evaluation_data(result, baseline)
    rmse, naive = result.metrics["test_rmse"], result.metrics["naive_test_rmse"]
    limit = baseline["metrics"]["test_rmse"] * baseline["tolerances"]["max_test_rmse_ratio"]
    # NaN, infinity or a negative value is a broken run, not a good model.
    for name, value in (("test_rmse", rmse), ("naive_test_rmse", naive)):
        if not (math.isfinite(value) and value >= 0):
            failures.append(f"{name} {value} is not a finite non-negative RMSE")
    # Written as "not (x <= limit)" so that NaN fails instead of slipping through.
    if not rmse <= limit:
        failures.append(f"test_rmse {rmse:.4f} exceeds limit {limit:.4f}")
    if not rmse < naive:
        failures.append("test_rmse does not beat the train-mean predictor")
    return failures


def gate(result: RunResult, baseline: dict) -> GateReport:
    return GateReport(check_reproducibility(result, baseline), check_quality(result, baseline))


def accepted(report: GateReport, config: TrainConfig, baseline: dict) -> bool:
    """The acceptance policy, shared by the CLI exit code and the CI test."""
    if config == TrainConfig(**baseline["config"]):
        return report.passed  # a rerun must reproduce and keep quality
    # A new candidate is expected to change predictions; it only has to keep quality.
    return not report.quality_failures


# --- command line -------------------------------------------------------------------

TOLERANCES = {"prediction_atol": 1e-9, "max_test_rmse_ratio": 1.05}
TOLERANCE_REASONS = {
    "prediction_atol": (
        "Approved test predictions lie between about 1.5 and 10; one float64 ulp there is "
        "at most about 1.8e-15. Another BLAS, CPU or summation order may move the last "
        "few bits, so exact equality across machines is not promised. 1e-9 leaves about "
        "five orders of magnitude for that noise, while a real change such as alpha "
        "1.0 -> 1.01 moves predictions by far more (checked by the tests)."),
    "max_test_rmse_ratio": (
        "A deliberate model change may cost at most 5% test RMSE. This is a product "
        "decision, not a statistical test: with 43 test rows the gate only catches clear "
        "degradation, and the test split is evaluated once per proposed candidate."),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["make-data", "propose-baseline", "gate"])
    parser.add_argument("--data", default=HERE / "data" / "synthetic_regression.csv", type=Path)
    parser.add_argument("--tracking-dir", default=HERE / "mlruns", type=Path)
    parser.add_argument("--baseline", default=HERE / "baseline" / "approved_baseline.json", type=Path)
    parser.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    parser.add_argument("--output", type=Path, help="new file for propose-baseline")
    args = parser.parse_args(argv)

    if args.command == "make-data":
        print(f"wrote {make_dataset(args.data)}")
        return 0
    if args.command == "propose-baseline" and (args.output is None or args.output.exists()):
        # Never overwrite the approved fixture (or anything else) with an unreviewed proposal.
        print("propose-baseline needs --output pointing to a file that does not exist yet",
              file=sys.stderr)
        return 2

    config = TrainConfig(alpha=args.alpha)
    result = run_experiment(args.data, args.tracking_dir, config, run_name=args.command)
    print(json.dumps({"run_id": result.run_id, "metrics": result.metrics}, indent=2))

    if args.command == "propose-baseline":
        # Only a proposal: a person reviews the run and, if they accept it, replaces the
        # approved fixture in a separate, reviewed commit.
        baseline = {
            "status": "PENDING_REVIEW",
            "config": {"alpha": config.alpha, "features": list(config.features)},
            "data_sha256": result.data_sha256,
            "split_sha256": result.split_sha256,
            "n_train": len(load_dataset(args.data).y_train),
            "n_test": len(result.test_predictions),
            "metrics": result.metrics,
            "test_predictions": result.test_predictions.tolist(),
            "tolerances": TOLERANCES,
            "tolerance_reasons": TOLERANCE_REASONS,
            "source_run_id": result.run_id,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"wrote {args.output} (PENDING_REVIEW)")
        return 0

    baseline = load_baseline(args.baseline)
    report = gate(result, baseline)
    print(report.summary())
    return 0 if accepted(report, config, baseline) else 1


if __name__ == "__main__":
    sys.exit(main())
