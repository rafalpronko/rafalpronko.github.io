import inspect
from types import SimpleNamespace

import numpy as np
import optuna
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate

import tuning


@pytest.fixture(scope="module")
def data():
    X, y = tuning.make_data()
    dev_idx, test_idx = tuning.split_holdout(y)
    return X, y, dev_idx, test_idx


@pytest.fixture(scope="module")
def tuned(data):
    X, y, dev_idx, _ = data
    return tuning.tune(X[dev_idx], y[dev_idx])


def test_holdout_split_is_disjoint_exhaustive_and_stratified(data):
    X, y, dev_idx, test_idx = data
    assert len(X) == 1000
    assert set(dev_idx).isdisjoint(test_idx)
    assert np.array_equal(np.sort(np.concatenate([dev_idx, test_idx])), np.arange(len(y)))
    assert abs(y[dev_idx].mean() - y[test_idx].mean()) < 0.01


def test_folds_are_fixed_and_partition_development_rows(data):
    _, y, dev_idx, _ = data
    folds = tuning.make_folds(y[dev_idx])
    assert len(folds) == tuning.N_FOLDS
    validation = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(validation), np.arange(len(dev_idx)))
    for train, val in folds:
        assert set(train).isdisjoint(val)
    again = tuning.make_folds(y[dev_idx])
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_scaler_is_fitted_inside_each_training_fold(data):
    X, y, dev_idx, _ = data
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    folds = tuning.make_folds(y_dev)
    fitted = cross_validate(tuning.build_model(tuning.BASELINE), X_dev, y_dev,
                            cv=folds, return_estimator=True)["estimator"]
    for model, (train, _) in zip(fitted, folds):
        mean = model.named_steps["scale"].mean_
        assert np.allclose(mean, X_dev[train].mean(axis=0))
        assert not np.allclose(mean, X_dev.mean(axis=0))


def test_suggested_parameters_reach_the_estimator():
    l2 = tuning.suggest_params(optuna.trial.FixedTrial({"penalty": "l2", "C": 0.1}))
    assert l2 == {"penalty": "l2", "C": 0.1}
    enet = tuning.suggest_params(optuna.trial.FixedTrial(
        {"penalty": "elasticnet", "C": 2.0, "l1_ratio": 0.3}))
    clf = tuning.build_model(enet).named_steps["clf"]
    assert (clf.penalty, clf.C, clf.l1_ratio) == ("elasticnet", 2.0, 0.3)


def test_cv_score_is_mean_roc_auc_over_the_fixed_folds(data):
    X, y, dev_idx, _ = data
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    folds = tuning.make_folds(y_dev)
    manual = []
    for train, val in folds:
        model = tuning.build_model(tuning.BASELINE).fit(X_dev[train], y_dev[train])
        manual.append(roc_auc_score(y_dev[val], model.predict_proba(X_dev[val])[:, 1]))
    assert tuning.cv_score(tuning.BASELINE, X_dev, y_dev, folds) == pytest.approx(np.mean(manual))


def test_search_interfaces_accept_development_data_only():
    assert list(inspect.signature(tuning.make_objective).parameters) == ["X_dev", "y_dev", "folds"]
    assert list(inspect.signature(tuning.tune).parameters) == ["X_dev", "y_dev"]


def test_search_sees_only_development_rows(data, monkeypatch):
    X, y, dev_idx, test_idx = data
    seen = []

    def spy(model, X_part, y_part, cv, scoring):
        seen.append(X_part)
        return np.array([0.5])

    monkeypatch.setattr(tuning, "cross_val_score", spy)
    tuning.tune(X[dev_idx], y[dev_idx])
    test_rows = {row.tobytes() for row in X[test_idx]}
    assert len(seen) == 1 + tuning.N_TRIALS
    assert all(len(X_part) == len(dev_idx) for X_part in seen)
    assert not any(row.tobytes() in test_rows for X_part in seen for row in X_part)


def test_study_maximises_with_the_declared_budget(tuned):
    study, _, _ = tuned
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert len(study.trials) == tuning.N_TRIALS
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    sampler = study.sampler
    assert isinstance(sampler, optuna.samplers.TPESampler)
    # Private attribute of Optuna 3.4.0's TPESampler; checks the constant is actually passed.
    assert sampler._n_startup_trials == tuning.N_STARTUP_TRIALS < tuning.N_TRIALS


def test_selection_rule_requires_predeclared_margin():
    tuned_params = {"penalty": "l1", "C": 0.5}
    gain = tuning.MIN_GAIN
    assert tuning.select_final(0.80, 0.80 + gain, tuned_params) == tuned_params
    assert tuning.select_final(0.80, 0.80 + gain / 2, tuned_params) == tuning.BASELINE


def test_selection_uses_cv_results(tuned):
    study, baseline, chosen = tuned
    assert chosen == tuning.select_final(baseline, study.best_value, study.best_params)


def test_run_routes_exact_rows_to_each_step(data, monkeypatch):
    X, y, dev_idx, test_idx = data
    chosen = {"penalty": "l1", "C": 0.5}
    calls = []

    def tune_spy(X_part, y_part):
        calls.append(("tune", X_part, y_part))
        study = SimpleNamespace(best_value=0.9, best_params=dict(chosen))
        return study, 0.8, dict(chosen)

    class Recorder:
        def __init__(self, params):
            calls.append(("build", params))
            self.model = real_build(params)

        def fit(self, X_part, y_part):
            calls.append(("fit", X_part, y_part))
            self.model.fit(X_part, y_part)
            return self

        def predict_proba(self, X_part):
            proba = self.model.predict_proba(X_part)
            calls.append(("predict", X_part, proba[:, 1]))
            return proba

    def metric_spy(y_true, scores):
        calls.append(("metric", y_true, scores))
        return real_metric(y_true, scores)

    real_build, real_metric = tuning.build_model, tuning.roc_auc_score
    monkeypatch.setattr(tuning, "tune", tune_spy)
    monkeypatch.setattr(tuning, "build_model", Recorder)
    monkeypatch.setattr(tuning, "roc_auc_score", metric_spy)
    result = tuning.run()

    assert [c[0] for c in calls] == ["tune", "build", "fit", "predict", "metric"]
    (_, tune_X, tune_y), (_, params), (_, fit_X, fit_y), (_, pred_X, proba), (_, met_y, met_s) = calls
    assert np.array_equal(tune_X, X[dev_idx]) and np.array_equal(tune_y, y[dev_idx])
    assert params == chosen
    assert np.array_equal(fit_X, X[dev_idx]) and np.array_equal(fit_y, y[dev_idx])
    assert np.array_equal(pred_X, X[test_idx])
    assert np.array_equal(met_y, y[test_idx]) and np.array_equal(met_s, proba)
    assert result["test_roc_auc"] == real_metric(y[test_idx], proba)


def test_run_computes_the_test_metric_directly_once(monkeypatch):
    calls = []
    real = tuning.roc_auc_score

    def spy(y_true, scores):
        calls.append(len(y_true))
        return real(y_true, scores)

    monkeypatch.setattr(tuning, "roc_auc_score", spy)
    result = tuning.run()
    assert calls == [250]
    assert 0.0 <= result["test_roc_auc"] <= 1.0
    assert set(result) >= {"baseline_cv", "best_cv", "chosen", "test_roc_auc"}
