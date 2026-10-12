---
layout: post
lang: en
title: "Precision, recall, AP and calibration answer different questions"
date: 2023-11-30 09:00:00 +0100
permalink: /blog/precision-recall-ap-calibration/
author: Rafał Prońko
categories: [machine-learning]
description: "Ranking, decisions and probabilities are three different properties of a classifier. A small Python example shows AP staying fixed while thresholds, Brier score and log loss move."
image: /assets/images/precision-recall-ap-calibration.svg
---

# Precision, recall, AP and calibration answer different questions

![Diagram with three panels. Ranking: the same ordering of cases gives AP 0.783 for scores p and p squared. Decision: a threshold of 0.5 flags 20 cases for p and only 10 for p squared. Probability: on a reliability chart p lies on the diagonal and p squared lies above and to the left of it, so p squared under-states the risk.]({{ '/assets/images/precision-recall-ap-calibration.svg' | relative_url }})

<time datetime="2023-11-30">30 November 2023</time>

*A note on dates: 30 November 2023 is the editorial date this post is filed under, not a claim that it was published on that day. The code and outputs below were actually executed on 12 October 2026, with library versions that had already been released by the editorial date.*

A binary classifier rarely gets used for just one thing. Its scores sort a queue of cases for review. A threshold turns the same scores into yes/no decisions. Sometimes the score itself is read as a probability and multiplied by a cost or an amount of money.

Those are three different uses, and each one needs a different metric. Average precision (AP) says how well the model **ranks**. Precision and recall at a threshold say what a particular **decision rule** does. Calibration says whether a score of 0.3 can be read as **"about 30% of such cases are positive"**.

The metrics are easy to mix up because they are all computed from the same column of scores. In this post I take one small synthetic dataset, apply a transformation that keeps AP exactly the same, and watch the decisions and the probability metrics change anyway.

## Confusion counts first

Everything at a fixed threshold starts with four counts. For a rule "flag the case when `score >= t`":

- **TP**: flagged and positive,
- **FP**: flagged and negative,
- **FN**: not flagged but positive,
- **TN**: not flagged and negative.

Precision is `TP / (TP + FP)`: of the cases I flagged, how many were positive. Recall is `TP / (TP + FN)`: of all positives, how many I caught.

Both are ratios, and both can have a zero denominator. If the threshold is above every score, nothing is flagged and precision is `0 / 0`. If a validation slice happens to contain no positives, recall is `0 / 0`. Those are undefined values, not zeros. scikit-learn's `precision_score` returns 0 in that case and emits a warning, controlled by the `zero_division` argument. That is a reporting convention. If such a 0 gets averaged into a dashboard, it reads as "the model was wrong", when actually nothing was flagged or there was nothing to find. In my own helper I return `None` and make the caller decide.

Ties matter too. With `>=`, every case that has exactly the threshold score is flagged together. A model that outputs a few discrete values cannot be moved to an arbitrary recall: recall jumps by whole groups.

## Precision depends on prevalence

Recall (also called the true positive rate, TPR) is computed only among positives. The false positive rate, `FPR = FP / (FP + TN)`, is computed only among negatives. Precision mixes both groups, so it depends on how common positives are. With prevalence `π`:

```text
precision = π · TPR / (π · TPR + (1 − π) · FPR)
```

The same model, with the same TPR and FPR at the same threshold, gives lower precision on a population where positives are rarer. AP is built from precision values, so it inherits this dependence. An AP of 0.4 can be strong when positives are 1% of the data and weak when they are 35%. AP values are only comparable on populations with similar prevalence.

The usual reference point is "a random ranking has AP equal to prevalence". That holds for a constant score: with everything tied, there is a single threshold, recall jumps from 0 to 1 in one step, and the precision at that step equals prevalence. (`precision_recall_curve` also returns the conventional end point recall 0, precision 1, but it does not contribute to AP.) For a random *ordering* of a finite sample, the expected AP is a bit higher than prevalence. For `n` items with `m` positives it is exactly `(H_n + (m − 1)/(n − 1) · (n − H_n)) / n`, where `H_n` is the n-th harmonic number. On the 40-row example below (19 positives, prevalence 0.475) that gives 0.519. The gap goes away as `n` grows, but on small validation slices the baseline is not simply the prevalence.

## AP is not the area under a smoothed PR curve

scikit-learn's `average_precision_score` computes

```text
AP = Σ_k (R_k − R_{k−1}) · P_k
```

over thresholds sorted from the highest score down. Each step in recall is weighted by the precision reached at that step. Nothing is interpolated between points. The documentation calls out explicitly that this differs from computing the area under the precision–recall curve with the trapezoidal rule (`auc(recall, precision)`). Linear interpolation between PR points can be too optimistic, because precision does not change linearly between two thresholds.

In the example below AP is 0.7829 and the trapezoidal area over the same points is 0.8454. Neither number is wrong, but they are not the same metric, and they should not be compared across reports.

AP also never looks at the score values themselves. It only uses their **order** and their **ties**. That is the central point of this post.

## The example: same ranking, different probabilities

I wanted data where calibration can be seen without any estimation noise. So the synthetic dataset has four score groups of ten rows, and in each group the fraction of positives is exactly equal to the score:

| score `p` | rows | positives | event rate |
|---|---|---|---|
| 0.1 | 10 | 1 | 0.1 |
| 0.3 | 10 | 3 | 0.3 |
| 0.6 | 10 | 6 | 0.6 |
| 0.9 | 10 | 9 | 0.9 |

By construction, `p` is perfectly calibrated on these 40 rows. Next to it I put `q = p**2`. On `[0, 1]`, squaring is strictly increasing: it keeps the order and keeps ties as ties (0.1, 0.3, 0.6, 0.9 become 0.01, 0.09, 0.36, 0.81). It is **not** a fitted calibrator. It is a deliberate distortion, the kind of difference you get between a model that is calibrated and one that systematically under-states risk.

This is a demonstration of a mechanism, not a benchmark. Forty rows and four score values were chosen so that every number can be checked by hand.

The code uses only APIs available in numpy 1.26.2, scipy 1.11.4 and scikit-learn 1.3.2, all released by 30 November 2023. I ran it with those exact versions on Python 3.11.12, a later patch release of Python 3.11 (see [Reproducing the example](#reproducing-the-example)):

```python
import numpy as np
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve)

# Four score groups; inside each group the event rate equals the score.
groups = [(0.1, 10), (0.3, 10), (0.6, 10), (0.9, 10)]
y = np.concatenate([[1] * round(s * n) + [0] * (n - round(s * n)) for s, n in groups])
p = np.concatenate([[s] * n for s, n in groups])
q = p**2  # strictly increasing on [0, 1]; not a fitted calibrator

print(f"prevalence = {y.mean():.3f}")
for name, s in (("p", p), ("p**2", q)):
    precision, recall, _ = precision_recall_curve(y, s)
    flagged = s >= 0.5
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    print(f"{name:>4}: AP={average_precision_score(y, s):.4f} "
          f"trapezoid={auc(recall, precision):.4f} "
          f"flagged@0.5={tp + fp} precision={tp / (tp + fp):.3f} "
          f"recall={tp / y.sum():.3f} "
          f"brier={brier_score_loss(y, s):.4f} log_loss={log_loss(y, s):.4f}")

pooled = np.where((p == 0.3) | (p == 0.6), 0.45, p)  # merge two groups into a tie
print(f"pooled: AP={average_precision_score(y, pooled):.4f} "
      f"brier={brier_score_loss(y, pooled):.4f}")
print(f"constant 0.5: AP={average_precision_score(y, np.full(y.size, 0.5)):.4f}")

for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
    last = hi == 1.0  # the last bin is closed so that a score of 1.0 counts
    in_bin = (q >= lo) & ((q <= hi) if last else (q < hi))
    label = f"[{lo:.2f}, {hi:.2f}{']' if last else ')'}"
    if in_bin.any():
        print(f"p**2 bin {label}: n={in_bin.sum():>2} "
              f"mean score={q[in_bin].mean():.3f} observed={y[in_bin].mean():.2f}")
    else:
        print(f"p**2 bin {label}: n= 0")
```

Output:

```text
prevalence = 0.475
   p: AP=0.7829 trapezoid=0.8454 flagged@0.5=20 precision=0.750 recall=0.789 brier=0.1575 log_loss=0.4835
p**2: AP=0.7829 trapezoid=0.8454 flagged@0.5=10 precision=0.900 recall=0.474 brier=0.1870 log_loss=0.6013
pooled: AP=0.7355 brier=0.1688
constant 0.5: AP=0.4750
p**2 bin [0.00, 0.25): n=20 mean score=0.050 observed=0.20
p**2 bin [0.25, 0.50): n=10 mean score=0.360 observed=0.60
p**2 bin [0.50, 0.75): n= 0
p**2 bin [0.75, 1.00]: n=10 mean score=0.810 observed=0.90
```

Let me read it in three passes.

**Ranking.** AP is identical for `p` and `p**2`: 0.7829. So is the trapezoidal area. Both are computed from the same sequence of PR points, and that sequence depends only on the order of the rows. If the only question is "is this a good order for a review queue?", the two models are indistinguishable.

**Decision.** At a threshold of 0.5 they behave like different systems. `p` flags 20 rows with precision 0.75 and recall 0.789. `p**2` flags 10 rows with precision 0.90 and recall 0.474. "0.5" is a number on a scale, and the transformation moved the scale under the threshold. A threshold chosen for one score cannot be reused as the same number after the scores change, even when the ranking is the same. It has to be chosen again, or mapped through the same transformation: `p >= 0.5` and `p**2 >= 0.25` flag exactly the same rows.

**Probability.** Brier score goes from 0.1575 to 0.1870 and log loss from 0.4835 to 0.6013. The reliability bins show why: in the `[0.25, 0.5)` bin `p**2` says 0.36 on average while 60% of the rows are positive. The model under-states risk in every non-empty bin.

The `pooled` and `constant 0.5` lines are a check of the ties. Merging the 0.3 and 0.6 groups into a single score of 0.45 keeps calibration (that merged group has exactly 9 positives out of 20), but AP falls from 0.7829 to 0.7355. Ties throw away ordering information. A constant score is the extreme case: AP equals prevalence, 0.475.

## Strictly increasing transforms, ties and calibrators

The rule behind the example: **a strictly increasing transformation applied to all scores keeps AP, ROC AUC and any rank-based metric exactly the same**. It can change probability metrics such as Brier score and log loss, and it can change the decisions made at a fixed threshold. "Can" is deliberate: some transformations happen to leave a loss unchanged on a particular sample, and a threshold mapped through the same transformation (0.5 for `p` becomes 0.25 for `p**2`) keeps exactly the same decisions. What does not survive is reusing the old threshold *number* on the new scale.

The word "strictly" carries weight. A transformation that is only non-decreasing can merge different scores into one value. Isotonic regression does exactly that: it fits a step function, so whole ranges of raw scores receive the same calibrated value. The new ties can lower AP, as in the pooled example above. The scikit-learn documentation also warns that isotonic regression can overfit when the calibration set is small. Sigmoid (Platt) calibration is a single strictly increasing function, as long as the fitted slope is positive, so it keeps the order.

A second case is less obvious. `CalibratedClassifierCV` with the default `ensemble=True` fits a separate classifier and calibrator on each cross-validation fold and averages their predicted probabilities. That average is not one monotonic function of a single score, so the order of rows can change. AP can then go up or down. If ranking is what matters, I measure AP *after* calibration and do not assume it carried over.

## Calibration: reliability diagrams need counts

A model is calibrated when, among the cases that receive a score of about `s`, the fraction of positives is about `s`. A reliability diagram checks this: group predictions into bins, then plot the mean predicted score against the observed event rate in each bin.

`sklearn.calibration.calibration_curve` returns just those two arrays. It quietly drops empty bins and does not return how many rows went into each one. Without the counts, a dot computed from five rows looks as convincing as a dot computed from five thousand. So I always compute the bin counts next to it and add an uncertainty interval for each observed rate. The demo module computes a 95% Wilson interval (the last bin is closed, so a score of exactly 1.0 is counted):

```text
reliability of p**2, bins [0,.25,.5,.75,1]:
  [0.00,0.25) n=20 mean score=0.050 observed=0.20 95% CI=(0.08, 0.42)
  [0.25,0.50) n=10 mean score=0.360 observed=0.60 95% CI=(0.31, 0.83)
  [0.50,0.75) n= 0 empty
  [0.75,1.00] n=10 mean score=0.810 observed=0.90 95% CI=(0.60, 0.98)
```

These intervals are **illustrative** here. A Wilson interval describes the uncertainty of a rate estimated from independent Bernoulli trials with a common probability within the bin. My labels were not sampled at all: I constructed them deterministically. On real data the reading would be: *if* the rows in a bin were independent draws with one shared event probability, the interval shows which values of that probability are compatible with the observed count. Rows from the same customer, session or time window are not independent, and the interval can then be too narrow.

With ten rows per bin the intervals are wide. In the two upper bins the mean score (0.36 and 0.81) lies inside the interval, so under those assumptions those rows alone would not contradict calibration in that bin. Only the lowest bin, with twenty rows, excludes its mean score of 0.05. These are separate per-bin checks, not a test of calibration as a whole. A global assessment has to account for all bins at once (and for the choice of bins), for example with a dedicated calibration test or a resampling procedure. The miscalibration is visible in the example because I built it in, not because 40 rows would prove it. Two more practical points: the chosen bin edges change the picture (quantile bins with `strategy="quantile"` give more even counts), and an empty bin should show up as empty rather than disappear.

### Brier score and log loss are not calibration metrics

Both are proper scoring rules. They are minimised by reporting the true probability, so they reward calibration. But they also reward discrimination. For forecasts with a finite set of values, the Brier score splits exactly into three parts (Murphy's decomposition):

```text
Brier = reliability − resolution + uncertainty
```

- **reliability** is the weighted squared distance between each forecast value and the observed rate for that value. This is the calibration part, and lower is better.
- **resolution** says how far the per-group rates are from the overall rate. This is the discrimination part, and higher is better.
- **uncertainty** is `π(1 − π)` and depends only on the data.

On the example (values from the demo module's output, laid out side by side):

```text
   p   reliability=0.0000 resolution=0.0919 uncertainty=0.2494  → Brier 0.1575
p**2   reliability=0.0295 resolution=0.0919 uncertainty=0.2494  → Brier 0.1870
```

The square only damaged reliability. Resolution is the same, because it depends only on which rows share a group. The pooled scores go the other way. They stay perfectly calibrated (reliability 0) and lose resolution, and the Brier score still gets worse (0.1688). So a better Brier score does not have to mean better calibration, and a worse one does not have to mean worse calibration. The scikit-learn calibration guide says the same thing: Brier score and log loss assess calibration and discrimination together. If I want to talk about calibration, I show the reliability diagram with its counts. The score is a summary, not a diagnosis.

## Choosing a threshold

A threshold is a business decision expressed on the model's scale. There are three common ways to set it, and all of them are chosen on validation data.

**From costs.** If the scores are calibrated, flagging a case costs `(1 − p) · C_FP` in expectation and not flagging it costs `p · C_FN`. Flag when the first is smaller:

```text
p > C_FP / (C_FP + C_FN)
```

With a false negative four times as expensive as a false positive, the threshold is `1 / (1 + 4) = 0.2`. The derivation assumes:

- the scores are calibrated **on the population where the decision is made**. On `p**2` the same formula would act on under-stated risks,
- the costs are per case and the same for every case. When the cost depends on the amount, it belongs inside the expected value,
- a correct decision costs nothing. If correct decisions have costs too, the threshold becomes `(C_FP − C_TN) / ((C_FP − C_TN) + (C_FN − C_TP))`, but only under the usual assumption that each error costs more than the correct decision it replaces (`C_FP > C_TN` and `C_FN > C_TP`). Without that assumption the denominator can be zero or the direction of the rule can flip, and it is safer to compare the two expected costs directly for each case,
- each decision is independent and there is no capacity limit.

**From a recall target.** "We must catch at least 90% of positives" is a constraint on the ranking. I pick the highest threshold that reaches it on validation data. The block below is a **fragment**: a short version of the helper, without the input validation of the tested module:

```python
# FRAGMENT: simplified helper; the validated, tested version is in
# pr_calibration_demo.py (linked at the end of the post).
import numpy as np


def threshold_for_recall(y, scores, min_recall):
    """Highest observed score whose `>=` rule reaches min_recall."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not 0 < min_recall <= 1 or y.sum() == 0:
        raise ValueError("need 0 < min_recall <= 1 and at least one positive")
    for t in np.unique(scores)[::-1]:  # whole tie groups enter together
        if ((scores >= t) & (y == 1)).sum() / y.sum() >= min_recall:
            return float(t)
```

Called as `threshold_for_recall(y, p, 0.9)` on the example data, it needs 18 of 19 positives, so the whole 0.3 group must come in. The threshold is 0.3 for `p` and 0.09 for `p**2`, and it flags the same rows. A recall rule follows the ranking, so it survives a strictly increasing transformation, but the number itself has to be re-derived. The module version also validates the labels and score values, and its tests check these two thresholds.

**From workload.** "Reviewers can handle 200 cases per day" is a top-k rule. It also depends only on the ranking. Then precision at that k tells you how much of the reviewers' time goes to positives.

In all three cases the test set does not take part in the choice. If I try thresholds on the test set and report the best one, the reported precision and recall are optimistic: the threshold has fitted the noise of that particular sample. The test set is for measuring the complete rule once — model, calibrator and threshold together.

## Keep the stages on separate data

A calibrated model with a threshold is three fitted things. Each needs data that the previous stage did not see:

1. **train**: fit the model,
2. **calibration**: fit the calibrator on predictions for rows the model was not trained on. scikit-learn's documentation states this requirement directly. Calibrating on training predictions learns the model's overconfidence on data it has memorised,
3. **threshold selection**: choose the threshold on calibrated scores. This can be a separate slice, or held-out predictions from cross-fitting,
4. **test**: evaluate everything once.

Random row splits are only valid when rows are independent. When several rows belong to the same customer, device or patient, I split by group (`GroupKFold`, `GroupShuffleSplit`). Otherwise the calibration set shares people with the training set and both the calibration and the threshold look better than they are. When the model will run on future data, I split by time: train on the past, calibrate and pick the threshold on a later period, and test on the period after that. `TimeSeriesSplit` does this for cross-validation.

Even with clean splits, a calibrated model is calibrated **for the distribution it was calibrated on**. When prevalence changes and TPR and FPR at the threshold stay the same, precision changes (see the formula above). Whether calibration survives depends on *why* prevalence changed. Under prior (label) shift, where the class balance moves but the feature distribution within each class stays the same, calibrated probabilities do become wrong, even if the ranking is still fine. A change in the mix of risk groups is different: if a population moves from 50/50 to 90/10 between a group with true risk 0.1 and a group with true risk 0.9, prevalence falls from 0.50 to 0.18, yet scores of 0.1 and 0.9 are still calibrated. So a prevalence change *can* break calibration, but it does not have to. A changed relationship between features and labels can break all three properties. A fixed test set cannot detect any of this after deployment. It needs monitoring: the observed event rate per score bin, the fraction of rows flagged, and precision once labels arrive.

## Which number answers which question

- *Is this a good order for a review queue?* AP (with prevalence stated next to it), or precision at the k your team can handle.
- *What happens if we act on scores above t?* TP, FP, FN, TN at that t on validation data, then once on the test set. Report undefined values as undefined.
- *Can I multiply the score by money?* A reliability diagram with counts and intervals, plus the Brier score or log loss as a summary, evaluated on data the calibrator has not seen.

A model can be good at one of these and bad at another. In the example, `p` and `p**2` have the same AP, give different decisions at 0.5, and only one of them is calibrated.

## Reproducing the example

The full module with input validation, the reliability table, the Brier decomposition and the tests:

- [`pr_calibration_demo.py`]({{ '/assets/examples/precision-recall-ap-calibration/pr_calibration_demo.py' | relative_url }})
- [`tests/test_pr_calibration_demo.py`]({{ '/assets/examples/precision-recall-ap-calibration/tests/test_pr_calibration_demo.py' | relative_url }})
- [`pyproject.toml`]({{ '/assets/examples/precision-recall-ap-calibration/pyproject.toml' | relative_url }}) and [`uv.lock`]({{ '/assets/examples/precision-recall-ap-calibration/uv.lock' | relative_url }})

The direct dependencies are pinned to numpy 1.26.2, scipy 1.11.4, scikit-learn 1.3.2 and pytest 7.4.3. The lock file also fixes every transitive dependency (joblib, threadpoolctl, pluggy, packaging and the rest), and it was resolved with `exclude-newer = "2023-12-01T00:00:00Z"`, so every locked package was uploaded to PyPI before December 2023.

The tooling is modern, not historical. I ran everything on 12 October 2026 with [uv](https://docs.astral.sh/uv/) 0.7.8 and CPython 3.11.12; both are later releases used only to reproduce the environment today. The code itself relies only on APIs that already existed in the pinned library versions:

```bash
uv run --python 3.11 --locked --extra test pytest -q
uv run --python 3.11 --locked python pr_calibration_demo.py
```

The tests cover ties at the threshold, thresholds above and at the lowest score, undefined precision and recall, invalid labels, non-finite scores and costs, very large finite costs whose sum would overflow, invalid bin edges, the closed last bin, the exact random-ordering AP against brute-force enumeration, and the Brier decomposition against `brier_score_loss`.

Sources: scikit-learn 1.3 documentation for [`average_precision_score`](https://scikit-learn.org/1.3/modules/generated/sklearn.metrics.average_precision_score.html) and [probability calibration](https://scikit-learn.org/1.3/modules/calibration.html).
