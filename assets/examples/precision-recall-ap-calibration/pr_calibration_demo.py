"""Ranking, decisions and probabilities on one small synthetic dataset.

The scores are built so that every score group has an exact event rate.
That makes p a perfectly calibrated forecast for this data and p**2 a
strictly increasing distortion of it. Nothing here is fitted.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

GROUPS = [(0.1, 10), (0.3, 10), (0.6, 10), (0.9, 10)]  # (score, group size)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    y: list[int] = []
    p: list[float] = []
    for score, size in GROUPS:
        positives = round(score * size)
        y += [1] * positives + [0] * (size - positives)
        p += [score] * size
    return np.array(y), np.array(p)


def _check(
    y: ArrayLike, scores: ArrayLike, unit_interval: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or y.size == 0:
        raise ValueError(
            "y and scores must be non-empty 1-D arrays of equal length"
        )
    if y.dtype.kind not in "biu" or not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only integer 0/1 labels")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if unit_interval and ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return y.astype(int), scores


def confusion_at(
    y: ArrayLike, scores: ArrayLike, threshold: float
) -> dict[str, int | float | None]:
    """Counts for the rule `score >= threshold`.

    None marks an undefined ratio (zero denominator).
    """
    y, scores = _check(y, scores)
    if math.isnan(threshold):
        raise ValueError("threshold must not be NaN")
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def threshold_for_recall(
    y: ArrayLike, scores: ArrayLike, min_recall: float
) -> float:
    """Highest observed score whose `>=` rule reaches min_recall.

    Fit it on validation data, never on the test set.
    """
    if not 0 < min_recall <= 1:  # also rejects NaN
        raise ValueError("min_recall must be in (0, 1]")
    y, scores = _check(y, scores)
    if y.sum() == 0:
        raise ValueError("recall needs at least one positive")
    for t in np.unique(scores)[::-1]:  # whole tie groups enter together
        if confusion_at(y, scores, t)["recall"] >= min_recall:
            return float(t)
    raise AssertionError("unreachable: the lowest score flags every row")


def cost_threshold(cost_fp: float, cost_fn: float) -> float:
    """Bayes threshold for calibrated p when correct decisions cost nothing."""
    if not (math.isfinite(cost_fp) and math.isfinite(cost_fn)):
        raise ValueError("costs must be finite")
    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("costs must be positive")
    scale = max(cost_fp, cost_fn)  # keeps the sum finite for huge costs
    cost_fp, cost_fn = cost_fp / scale, cost_fn / scale
    return cost_fp / (cost_fp + cost_fn)


def expected_random_ap(n_pos: int, n: int) -> float:
    """Exact mean AP of a random ordering of n items with n_pos positives."""
    if not 0 < n_pos <= n:
        raise ValueError("need 0 < n_pos <= n")
    if n_pos == n:
        return 1.0
    harmonic = sum(1 / r for r in range(1, n + 1))
    return (harmonic + (n_pos - 1) / (n - 1) * (n - harmonic)) / n


def brier_decomposition(y: ArrayLike, p: ArrayLike) -> dict[str, float]:
    """Murphy decomposition over distinct forecast values.

    Exact when p takes a finite set of values, as in this example.
    """
    y, p = _check(y, p, unit_interval=True)
    base = y.mean()
    reliability = resolution = 0.0
    for value in np.unique(p):
        mask = p == value
        weight = mask.mean()
        reliability += weight * (value - y[mask].mean()) ** 2
        resolution += weight * (y[mask].mean() - base) ** 2
    return {"reliability": float(reliability), "resolution": float(resolution),
            "uncertainty": float(base * (1 - base))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n independent Bernoulli trials.

    Illustrative only on the deterministic synthetic labels used here.
    """
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def reliability_table(
    y: ArrayLike, p: ArrayLike, edges: Sequence[float]
) -> list[dict]:
    """Bins [a, b) with the last bin [a, 1] closed.

    Returns counts and 95% Wilson intervals per bin.
    """
    y, p = _check(y, p, unit_interval=True)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("edges must be 1-D with at least 2 finite values")
    if edges[0] != 0 or edges[-1] != 1 or (np.diff(edges) <= 0).any():
        raise ValueError("edges must increase strictly from 0 to 1")
    last = len(edges) - 2
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, last)
    rows = []
    for b in range(len(edges) - 1):
        mask = index == b
        n = int(mask.sum())
        row = {"bin": (float(edges[b]), float(edges[b + 1])),
               "closed": b == last, "n": n, "mean_score": None,
               "observed": None, "low": None, "high": None}
        if n:
            k = int(y[mask].sum())
            row.update(mean_score=float(p[mask].mean()), observed=k / n)
            row["low"], row["high"] = wilson(k, n)
        rows.append(row)
    return rows


def main() -> None:
    y, p = make_data()
    q = p**2  # strictly increasing on [0, 1]: same order and ties, new values
    ap_p, ap_q = average_precision_score(y, p), average_precision_score(y, q)
    random_ap = expected_random_ap(int(y.sum()), y.size)
    print(f"n={y.size} positives={y.sum()} prevalence={y.mean():.3f}")
    print(f"AP p={ap_p:.4f} p**2={ap_q:.4f}")
    print(f"AP unchanged by p**2: {ap_p == ap_q}")
    print(f"random-order AP: mean={random_ap:.4f}")
    for name, s in (("p", p), ("p**2", q)):
        c = confusion_at(y, s, 0.5)
        d = brier_decomposition(y, s)
        print(f"{name:>4} @0.5 tp={c['tp']} fp={c['fp']} fn={c['fn']} "
              f"tn={c['tn']} precision={c['precision']:.3f} "
              f"recall={c['recall']:.3f} | "
              f"brier={brier_score_loss(y, s):.4f} "
              f"log_loss={log_loss(y, s):.4f} "
              f"reliability={d['reliability']:.4f} "
              f"resolution={d['resolution']:.4f} "
              f"uncertainty={d['uncertainty']:.4f}")
    t_p, t_q = threshold_for_recall(y, p, 0.9), threshold_for_recall(y, q, 0.9)
    print(f"recall>=0.9 threshold: p={t_p:.2f} p**2={t_q:.2f} "
          f"same rows: {np.array_equal(p >= t_p, q >= t_q)}")
    print(f"cost threshold (FP=1, FN=4): {cost_threshold(1, 4):.2f}")
    print("reliability of p**2, bins [0,.25,.5,.75,1]:")
    for r in reliability_table(y, q, [0.0, 0.25, 0.5, 0.75, 1.0]):
        lo, hi = r["bin"]
        label = f"[{lo:.2f},{hi:.2f}{']' if r['closed'] else ')'}"
        if r["n"]:
            print(f"  {label} n={r['n']:>2} "
                  f"mean score={r['mean_score']:.3f} "
                  f"observed={r['observed']:.2f} "
                  f"95% CI=({r['low']:.2f}, {r['high']:.2f})")
        else:
            print(f"  {label} n= 0 empty")


if __name__ == "__main__":
    main()
