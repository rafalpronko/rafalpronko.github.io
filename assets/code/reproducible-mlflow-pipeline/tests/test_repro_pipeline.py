import dataclasses
import json
import math
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

import repro_pipeline as rp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_regression.csv"
BASELINE = ROOT / "baseline" / "approved_baseline.json"


@pytest.fixture(scope="session")
def baseline():
    return rp.load_baseline(BASELINE)


@pytest.fixture(scope="session")
def tracking_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mlruns")


@pytest.fixture(scope="session")
def approved_run(tracking_dir, baseline):
    config = rp.TrainConfig(**baseline["config"])
    return rp.run_experiment(DATA, tracking_dir, config, run_name="rerun-approved")


@pytest.fixture(autouse=True)
def session_tracking_store(tracking_dir):
    # rp.main(...) in some tests points MLflow's global tracking URI at another store.
    # Reset it before every test so approved_run's ids resolve whatever the test order.
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())


# --- data and split -------------------------------------------------------

def test_materialized_data_matches_approved_fingerprints(baseline):
    dataset = rp.load_dataset(DATA)
    assert dataset.data_sha256 == baseline["data_sha256"]
    assert dataset.split_sha256 == baseline["split_sha256"]
    assert (len(dataset.y_train), len(dataset.y_test)) == (
        baseline["n_train"], baseline["n_test"])


def test_regenerating_data_in_pinned_environment_gives_identical_bytes(tmp_path):
    # Held for the pinned numpy on the article's machine; that is not a promise for other
    # builds or platforms. The committed CSV, not the seed, is the source of truth.
    regenerated = rp.make_dataset(tmp_path / "regenerated.csv")
    assert regenerated.read_bytes() == DATA.read_bytes()


def test_scaler_statistics_come_from_training_rows_only(baseline):
    dataset = rp.load_dataset(DATA)
    model = rp.train(dataset, rp.TrainConfig(**baseline["config"]))
    all_rows = np.vstack([dataset.X_train, dataset.X_test])
    np.testing.assert_allclose(model["scaler"].mean_, dataset.X_train.mean(axis=0))
    assert not np.allclose(model["scaler"].mean_, all_rows.mean(axis=0))


def test_select_features_returns_c_ordered_columns_in_config_order():
    dataset = rp.load_dataset(DATA)
    selected = rp.select_features(dataset.X_test, rp.TrainConfig(features=("x2", "x0")))
    assert selected.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(selected, dataset.X_test[:, [2, 0]])


def test_memory_order_noise_stays_within_prediction_tolerance(baseline):
    # Same values, same model, same process: only the memory order differs.
    # On some BLAS builds this changes the last bits, so the gate must not demand equality.
    dataset = rp.load_dataset(DATA)
    model = rp.train(dataset, rp.TrainConfig(**baseline["config"]))
    c_order = model.predict(np.ascontiguousarray(dataset.X_test))
    f_order = model.predict(np.asfortranarray(dataset.X_test))
    atol = baseline["tolerances"]["prediction_atol"]
    np.testing.assert_allclose(f_order, c_order, rtol=0, atol=atol)


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_non_finite_value_in_data_is_rejected(tmp_path, bad_value):
    lines = DATA.read_text().splitlines()
    cells = lines[5].split(",")
    cells[3] = bad_value
    lines[5] = ",".join(cells)
    broken = tmp_path / "broken.csv"
    broken.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        rp.load_dataset(broken)


def test_unknown_split_label_is_rejected(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text(DATA.read_text().replace(",test,", ",holdout,", 1))
    with pytest.raises(ValueError, match="split"):
        rp.load_dataset(broken)


def test_model_rejects_non_finite_input_at_prediction(approved_run):
    model = mlflow.sklearn.load_model(approved_run.model_uri)
    row = np.array([[0.1, float("nan"), 50.0, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        model.predict(row)


# --- MLflow metadata and reload --------------------------------------------

def test_run_records_params_metrics_fingerprints_and_artifacts(approved_run, baseline):
    run = mlflow.get_run(approved_run.run_id)
    assert run.data.params["alpha"] == str(baseline["config"]["alpha"])
    assert run.data.params["features"] == ",".join(baseline["config"]["features"])
    assert run.data.tags["data_sha256"] == baseline["data_sha256"]
    assert run.data.tags["split_sha256"] == baseline["split_sha256"]
    assert run.data.tags["source_sha256"] == approved_run.source_sha256
    assert run.data.metrics["test_rmse"] == approved_run.metrics["test_rmse"]

    client = MlflowClient()
    logged = {f.path for d in ["source", "data", "environment", "model"]
              for f in client.list_artifacts(approved_run.run_id, d)}
    assert {"source/repro_pipeline.py", "data/synthetic_regression.csv",
            "environment/environment.json", "environment/uv.lock", "model/MLmodel"} <= logged
    assert "environment/requirements.txt" not in logged  # uv.lock is the only lock file


@pytest.mark.parametrize("features", [None, ("x2", "x0", "x3", "x1"), ("x0", "x2")],
                         ids=["approved", "reordered", "reduced"])
def test_reloaded_model_reproduces_in_run_predictions(approved_run, tracking_dir, features):
    run = approved_run
    if features is not None:
        config = dataclasses.replace(approved_run.config, features=features)
        run = rp.run_experiment(DATA, tracking_dir, config, run_name="features")
    dataset = rp.load_dataset(DATA)
    reloaded = mlflow.sklearn.load_model(run.model_uri)
    # Same prepared input as in the run: config column order, C layout.
    np.testing.assert_array_equal(reloaded.predict(rp.select_features(dataset.X_test, run.config)),
                                  run.test_predictions)


# --- regression gate ---------------------------------------------------------

def test_baseline_fixture_is_approved_and_explains_tolerances(baseline):
    assert baseline["status"] == "APPROVED"
    for name in baseline["tolerances"]:
        assert baseline["tolerance_reasons"][name].strip()


def test_propose_baseline_never_overwrites_the_approved_fixture(tmp_path):
    approved = tmp_path / "approved_baseline.json"
    approved.write_bytes(BASELINE.read_bytes())
    code = rp.main(["propose-baseline", "--tracking-dir", str(tmp_path / "mlruns"),
                    "--baseline", str(approved)])
    assert code != 0
    assert approved.read_bytes() == BASELINE.read_bytes()


def test_propose_baseline_refuses_an_existing_output_file(tmp_path):
    existing = tmp_path / "approved_baseline.json"
    existing.write_bytes(BASELINE.read_bytes())
    code = rp.main(["propose-baseline", "--tracking-dir", str(tmp_path / "mlruns"),
                    "--output", str(existing)])
    assert code != 0
    assert existing.read_bytes() == BASELINE.read_bytes()


def test_propose_baseline_writes_a_pending_proposal_to_a_new_path(tmp_path):
    before = BASELINE.read_bytes()
    proposal = tmp_path / "proposal.json"
    code = rp.main(["propose-baseline", "--tracking-dir", str(tmp_path / "mlruns"),
                    "--output", str(proposal)])
    assert code == 0
    assert json.loads(proposal.read_text())["status"] == "PENDING_REVIEW"
    assert BASELINE.read_bytes() == before


def test_rerun_of_approved_config_passes_both_gates(approved_run, baseline):
    report = rp.gate(approved_run, baseline)
    assert report.reproducibility_failures == []
    assert report.quality_failures == []


def test_prediction_tolerance_separates_float_noise_from_real_change(approved_run, baseline):
    noise = dataclasses.replace(
        approved_run, test_predictions=approved_run.test_predictions + 1e-12)
    change = dataclasses.replace(
        approved_run, test_predictions=approved_run.test_predictions + 1e-6)
    assert rp.gate(noise, baseline).passed
    report = rp.gate(change, baseline)
    assert any("predictions" in f for f in report.reproducibility_failures)
    assert report.quality_failures == []


def test_small_config_drift_is_a_reproducibility_failure_not_a_quality_failure(
        tracking_dir, baseline):
    config = rp.TrainConfig(**{**baseline["config"], "alpha": 1.01})
    report = rp.gate(rp.run_experiment(DATA, tracking_dir, config, "drift"), baseline)
    assert report.reproducibility_failures
    assert report.quality_failures == []


def test_degraded_candidate_fails_quality_gate(tracking_dir, baseline):
    config = rp.TrainConfig(**{**baseline["config"], "alpha": 1000.0})
    report = rp.gate(rp.run_experiment(DATA, tracking_dir, config, "degraded"), baseline)
    assert any("test_rmse" in f for f in report.quality_failures)
    assert not report.passed


def test_nan_metric_never_passes_quality_gate(approved_run, baseline):
    broken = dataclasses.replace(
        approved_run, metrics={**approved_run.metrics, "test_rmse": math.nan})
    assert any("test_rmse" in f for f in rp.gate(broken, baseline).quality_failures)


@pytest.mark.parametrize("name, value", [
    ("test_rmse", -0.1), ("naive_test_rmse", math.inf), ("naive_test_rmse", math.nan)])
def test_metric_outside_rmse_domain_fails_quality_gate(approved_run, baseline, name, value):
    broken = dataclasses.replace(approved_run, metrics={**approved_run.metrics, name: value})
    assert any(f.startswith(name) for f in rp.gate(broken, baseline).quality_failures)


@pytest.mark.parametrize("alpha, shift, accepted", [
    (1.0, 1e-6, False),    # approved config, predictions drifted: not a reproduction
    (1.01, 0.0, True),     # new candidate that keeps quality
    (1000.0, 0.0, False),  # new candidate that loses quality
], ids=["approved-drift", "alpha-1.01", "alpha-1000"])
def test_cli_and_ci_test_share_the_acceptance_policy(
        approved_run, tracking_dir, baseline, monkeypatch, alpha, shift, accepted):
    result = approved_run
    if alpha != approved_run.config.alpha:
        config = dataclasses.replace(approved_run.config, alpha=alpha)
        result = rp.run_experiment(DATA, tracking_dir, config, run_name=f"alpha-{alpha}")
    if shift:
        # Recomputed metrics: the drift alone keeps quality, so only reproducibility can catch it.
        predictions = result.test_predictions + shift
        metrics = {**result.metrics, **{f"test_{k}": v for k, v in rp.regression_metrics(
            rp.load_dataset(DATA).y_test, predictions).items()}}
        result = dataclasses.replace(result, test_predictions=predictions, metrics=metrics)
    report = rp.gate(result, baseline)
    assert rp.accepted(report, result.config, baseline) is accepted  # used by test_candidate_gate
    monkeypatch.setattr(rp, "run_experiment", lambda *args, **kwargs: result)
    assert rp.main(["gate", "--alpha", str(alpha)]) == (0 if accepted else 1)


def test_different_test_split_is_not_comparable(approved_run, baseline):
    other = dataclasses.replace(approved_run, split_sha256="0" * 64)
    report = rp.gate(other, baseline)
    assert any("split" in f for f in report.reproducibility_failures)
    assert any("split" in f for f in report.quality_failures)


# --- the gate a CI job would run ----------------------------------------------

def test_candidate_gate(tracking_dir, baseline):
    """Candidate config comes from the environment; defaults to the approved one.

    CANDIDATE_ALPHA=1000 makes this test fail on purpose: that is the gate. It applies the
    same acceptance policy as the CLI, so a drifting rerun of the approved config fails too.
    """
    config = rp.TrainConfig(**baseline["config"])
    if "CANDIDATE_ALPHA" in os.environ:
        config = dataclasses.replace(config, alpha=float(os.environ["CANDIDATE_ALPHA"]))
    result = rp.run_experiment(DATA, tracking_dir, config, run_name="candidate")
    report = rp.gate(result, baseline)
    assert rp.accepted(report, config, baseline), report.summary()
