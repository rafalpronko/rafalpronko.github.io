"""Tune a scaled logistic regression with Optuna without looking at the test set."""
import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Decisions fixed before any score is computed.
SEED = 7
N_TRIALS = 16
N_STARTUP_TRIALS = 6
N_FOLDS = 3
METRIC = "roc_auc"
MIN_GAIN = 0.005  # tuned params must beat the baseline mean CV score by this much
BASELINE = {"penalty": "l2", "C": 1.0}


def make_data():
    return make_classification(n_samples=1000, n_features=20, n_informative=5,
                               n_redundant=5, weights=[0.8], flip_y=0.03,
                               random_state=SEED)


def split_holdout(y):
    rows = np.arange(len(y))
    return train_test_split(rows, test_size=0.25, stratify=y, random_state=SEED)


def make_folds(y_dev):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y_dev)), y_dev))


def build_model(params):
    clf = LogisticRegression(solver="saga", max_iter=5000, random_state=SEED, **params)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def suggest_params(trial):
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    params = {"penalty": penalty, "C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    return params


def cv_score(params, X_dev, y_dev, folds):
    scores = cross_val_score(build_model(params), X_dev, y_dev, cv=folds, scoring=METRIC)
    return float(np.mean(scores))


def make_objective(X_dev, y_dev, folds):
    def objective(trial):
        return cv_score(suggest_params(trial), X_dev, y_dev, folds)
    return objective


def select_final(baseline_cv, best_cv, best_params):
    return dict(best_params) if best_cv >= baseline_cv + MIN_GAIN else dict(BASELINE)


def tune(X_dev, y_dev):
    folds = make_folds(y_dev)
    baseline_cv = cv_score(BASELINE, X_dev, y_dev, folds)
    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_dev, y_dev, folds), n_trials=N_TRIALS)
    return study, baseline_cv, select_final(baseline_cv, study.best_value, study.best_params)


def run():
    X, y = make_data()
    dev_idx, test_idx = split_holdout(y)
    study, baseline_cv, chosen = tune(X[dev_idx], y[dev_idx])
    # Selection is finished; the test rows are used once, for reporting only.
    final = build_model(chosen).fit(X[dev_idx], y[dev_idx])
    test_auc = roc_auc_score(y[test_idx], final.predict_proba(X[test_idx])[:, 1])
    return {"baseline_cv": baseline_cv, "best_cv": study.best_value,
            "best_params": study.best_params, "chosen": chosen,
            "test_roc_auc": float(test_auc), "study": study}


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    result = run()
    for t in result["study"].trials:
        print(f"trial {t.number:2d}  cv_auc={t.value:.4f}  {t.params}")
    print(f"baseline cv_auc={result['baseline_cv']:.4f}")
    print(f"best     cv_auc={result['best_cv']:.4f}  {result['best_params']}")
    print(f"chosen   {result['chosen']}")
    print(f"held-out test roc_auc={result['test_roc_auc']:.4f} (reported once, not used to choose)")
