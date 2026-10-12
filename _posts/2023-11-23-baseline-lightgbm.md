---
layout: post
lang: en
title: "A fair baseline for LightGBM: freeze the split before comparing models"
date: 2023-11-23 09:00:00 +0100
permalink: /blog/baseline-lightgbm/
author: Rafał Prońko
categories: [machine-learning]
description: "Dummy, logistic regression and LightGBM on one frozen stratified split: training-only preprocessing, selection by validation log loss and a single test evaluation."
image: /assets/images/baseline-lightgbm.svg
---

# A fair baseline for LightGBM: freeze the split before comparing models

![Diagram: the bundled dataset with a fixed row order and digest is split once into train, validation and test. Three candidates are fitted on train and compared by validation log loss. The choice is frozen in a file, and only then is the locked test set used once.]({{ '/assets/images/baseline-lightgbm.svg' | relative_url }})

<time datetime="2023-11-23">23 November 2023</time>

“LightGBM gets 0.98 ROC AUC” sounds like a result. On its own it isn't one.

Is 0.98 good? Compared to what? On which rows? Were those rows used to pick the parameters, the seed or the preprocessing? If the honest answer is “I tried a few things and kept what looked best on the test set”, then 0.98 describes the experiment more than the model.

So before I train anything clever, I want two things in place. A **frozen split**: the rows that will train, select and finally test the models, saved to a file before any model exists. And a **baseline**: something simpler that the clever model has to beat on exactly the same terms.

In this post I build both on a small public dataset. Three candidates — a model that ignores the features, a scaled logistic regression and LightGBM — are compared on validation rows. The choice is written down. Only then does the test set get used, once.

## Three kinds of baseline

“Baseline” gets used for different things. I find it useful to separate them, because they answer different questions.

1. **A no-skill baseline.** A model that ignores the input. Here it's `DummyClassifier(strategy="prior")`: it learns one number from the training labels — the share of malignant cases — and predicts that probability for every row. It sets the floor. A model that can't beat it on the chosen metric, on this split, has failed at the job I'm measuring. That doesn't prove the features carry no signal at all — a model can rank cases well and still give badly calibrated probabilities — but it does mean the model isn't usable for this purpose.
2. **A simple, competent baseline.** A model you would accept if nothing fancier existed. Here it's `StandardScaler` followed by `LogisticRegression`. This is the one LightGBM really has to beat to justify its extra complexity.
3. **The current solution.** A rule, a score or a model already in use. There isn't one in this example, but in real work it's often the most important comparison: the question is rarely “is this good?”, usually “is this better than what we have?”.

A baseline is only fair if it plays by the same rules as the model it's compared with: the same rows, the same features, the same metric and comparable effort. If I tune LightGBM for an afternoon and leave logistic regression at whatever settings I typed first, the comparison says more about my afternoon than about the models. In this example neither model is tuned. Both get one fixed configuration, written before any results existed.

## The data and the positive class

I use the Wisconsin Diagnostic Breast Cancer data that ships with scikit-learn: 569 rows, 30 numeric features computed from digitised images of fine-needle aspirates, 212 malignant and 357 benign cases. It loads from a file inside the package, so nothing is downloaded.

One detail matters before anything else. In `load_breast_cancer()` the target is encoded as `0 = malignant` and `1 = benign`. Most metric code treats `1` as the positive class, and “probability of malignancy” is the quantity I want to talk about. So I relabel explicitly:

```python
# fair_baseline/data.py (fragment)
original_malignant = list(bunch.target_names).index("malignant")
y = np.where(bunch.target == original_malignant, MALIGNANT, BENIGN).astype(np.int64)
```

`MALIGNANT = 1`, `BENIGN = 0`. A test checks the counts (212 ones, 357 zeros) and that the result is exactly the inverse of the original target. Without that step every probability in this post would quietly mean “probability of benign”.

## Freeze the split first

The split comes before any model, any scaler, any metric. Train / validation / test is 60 / 20 / 20 percent, stratified on the label, done with two calls to `train_test_split`:

```python
# fair_baseline/split.py (fragment)
def make_split(data: Dataset, seed: int = SPLIT_SEED) -> Split:
    rows = np.arange(len(data.y))
    rest, test = train_test_split(
        rows, test_size=TEST_FRACTION, stratify=data.y, random_state=seed
    )
    train, validation = train_test_split(
        rest, test_size=VALIDATION_FRACTION_OF_REST, stratify=data.y[rest], random_state=seed
    )
    return Split(seed, dataset_digest(data), np.sort(train), np.sort(validation), np.sort(test))
```

I split **row positions**, not the feature matrix. The result is three lists of integers: 341 training rows (127 malignant), 114 validation rows (43 malignant) and 114 test rows (42 malignant). A separate script writes them to `artifacts/split.json` once and refuses to overwrite an existing file.

Positions only mean something if the rows stay in the same order. So the file also stores:

- the seed (`2023`), chosen before any model was trained and never changed afterwards;
- an **ordering contract**: the indices refer to rows in the order returned by `load_breast_cancer()` in scikit-learn 1.3.2;
- a **SHA-256 digest** of the data: shape, feature names, feature bytes and labels, in that row order.

Every later run loads the split, recomputes the digest and stops if it differs. If someone sorts the data, drops a row or swaps the label encoding, the indices would point at different patients — and the check catches it before anything is trained. A test confirms that swapping just two rows changes the digest.

Why bother with a file when `random_state` already makes the split reproducible? Because the seed reproduces the split only if everything else stays equal: the data, its order and the splitting code. The file makes the split an artifact I can inspect, diff and check, not a side effect of a function call.

## Preprocessing belongs to training

For logistic regression, scaling helps the optimiser and puts the regularisation penalty on comparable terms for every feature. Tree models such as LightGBM generally don't need scaled features, because splits depend on the order of values within a feature rather than their units. The scaler is where leakage likes to hide. If I call `StandardScaler().fit(X)` on all 569 rows and split afterwards, the validation and test rows have already influenced the means and standard deviations the model trains with.

So the scaler lives inside the model:

```python
# fair_baseline/models.py (fragment)
def logistic_regression() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
```

Calling `fit` on the pipeline with the training rows fits the scaler on those rows only. Validation and test rows are transformed with training statistics. A test checks that the fitted scaler's mean equals the mean of the training rows and differs from the mean of the full dataset.

## What LightGBM actually learns

The LightGBM candidate has one fixed configuration:

```python
# fair_baseline/models.py (fragment)
def lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=MODEL_SEED,
        n_jobs=1,
        device_type="cpu",
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
```

These are **hyperparameters**: I chose them. Modest values for a dataset of 341 training rows, not the result of a search. `n_jobs=1`, `device_type="cpu"`, `deterministic=True` and `force_col_wise=True` ask for one CPU thread, stable results and a fixed way of building histograms (`verbose=-1` only silences log output); the LightGBM documentation asks for `force_col_wise` or `force_row_wise` together with `deterministic=True` to avoid numerical instability.

The **learned parameters** are something else. It helps to spell out the whole mechanism, because it explains several numbers later on.

- **Input**: one row of 30 numbers.
- **Target**: `y = 1` for malignant, `0` for benign.
- **Model output**: a raw score `F(x)`, turned into a probability with the sigmoid, `p = 1 / (1 + e^(−F(x)))`.
- **Loss** on one training row: binary log loss, `−[y·ln p + (1 − y)·ln(1 − p)]`.

Training starts from a constant. With the default `boost_from_average`, the initial score is the log-odds of the training prevalence: `ln(127/214) ≈ −0.52`, which corresponds to a probability of about 0.37 for every row.

Then come 200 rounds. In each round LightGBM computes, for every training row, how the loss would change if that row's score moved. For log loss the derivatives with respect to the score are simple:

- gradient: `p − y`,
- second derivative (the Hessian): `p·(1 − p)`.

A malignant row currently predicted at 0.2 has gradient `−0.8`: push its score up. A benign row at 0.9 has gradient `+0.9`: push it down. A row already predicted confidently and correctly has a gradient close to zero and barely affects the next tree.

LightGBM then grows one small decision tree. It picks the splits that best separate rows that need an upward push from rows that need a downward one, judged from the sums of gradients and Hessians on each side. Features are bucketed into histogram bins (at most 255 per feature by default), and the tree has at most 15 leaves here. Each leaf gets a value computed, in essence, as `−(sum of gradients) / (sum of Hessians + λ)`, where `λ` is `reg_lambda`. That value is scaled by the learning rate 0.05 and added to the score of every row that lands in the leaf.

After training, what the model has learned is:

- the initial score,
- for each of the 200 trees, which feature and threshold each split uses,
- the value in each leaf.

**Inference** is the same path without any learning: send the row through all 200 trees, add up the leaf values and the initial score, and apply the sigmoid. Nothing about the validation or test rows enters this model.

For comparison: the logistic regression learns 30 weights and an intercept on standardised features, plus the 30 means and 30 standard deviations stored in the scaler. The dummy learns a single number: 127 / 341 ≈ 0.372.

One consequence is worth seeing in numbers. Each boosting round seeks to reduce the training loss. That's its job. Non-zero gradients don't guarantee that it succeeds — tree size, the minimum number of rows per leaf, regularisation and the fixed number of rounds can all stop further progress — but in practice the training loss usually ends up far lower than on new rows. A low training loss therefore says almost nothing about rows the model hasn't seen.

## The metric: log loss, not ranking

I choose the model by **validation log loss**: the same formula as above, averaged over the 114 validation rows. Smaller is better.

I also report ROC AUC, but I don't use it for the decision. The two metrics answer different questions.

ROC AUC only looks at the **order** of predictions. It's the probability that a randomly chosen malignant row gets a higher score than a randomly chosen benign one, with ties counted as half: `P(score_malignant > score_benign) + ½·P(tie)`. The half-credit for ties is why a model that gives every row the same score gets exactly 0.5. Multiply every probability by 0.5, or square them all, and the AUC doesn't change. A model can rank very well and still claim 0.99 for rows that are malignant 60% of the time.

Log loss looks at the **probabilities themselves**. It rewards confident correct predictions a little and punishes confident wrong ones a lot: a malignant row predicted at 0.01 costs `−ln 0.01 ≈ 4.6` on its own, while a correct prediction at 0.99 costs 0.01. If the output is going to be read as a probability — and for this kind of data, it would be — log loss is the more honest yardstick.

There's no threshold anywhere in the selection. A threshold turns probabilities into decisions, and choosing one requires knowing what a false negative costs compared to a false positive. I only use a fixed threshold of 0.5 at the very end, to show what a confusion matrix looks like, and I don't change it.

## Results

All numbers below come from one run of the downloadable example. The second run on the same machine produced byte-identical output files. That shows the computation is reproducible: the same code, data and environment give the same numbers. It is not an independent replication of the result — both runs use the same rows and the same split — and since nothing in either run feeds back into the choice, repeating it is not tuning on the test set either.

**Training log loss** — the rows each model was fitted on. Context only, not used for anything:

| Model | Training log loss |
|---|---|
| Dummy (prior) | 0.6602 |
| Scaler + logistic regression | 0.0349 |
| LightGBM | 0.0052 |

**Validation** — the 114 rows used for the choice:

| Model | Validation log loss | Validation ROC AUC |
|---|---|---|
| Dummy (prior) | 0.6627 | 0.5000 |
| Scaler + logistic regression | 0.1188 | 0.9866 |
| LightGBM | **0.1082** | 0.9879 |

LightGBM has the smallest validation log loss, so it is the choice. The script writes it to `results/selection.json` together with all validation scores, the split seed, the data digest, a hash of the test indices and a hash of the record itself.

**Test** — 114 rows, one evaluation of the frozen choice:

| Model | Test log loss | Test ROC AUC | Accuracy at 0.5 |
|---|---|---|---|
| LightGBM | 0.1775 | 0.9821 | 0.9211 |

At the 0.5 threshold the test confusion matrix is 38 true positives, 4 false negatives, 5 false positives and 67 true negatives.

## Reading the numbers

**The dummy does exactly what it should.** It predicts 0.372 for everyone, so its training log loss of 0.6602 is just the entropy of a 127 / 214 label split; a test in the example checks this identity. On validation it scores 0.6627, and its ROC AUC is exactly 0.5 because it gives every row the same score. Both real models are far below it. The features carry plenty of signal.

**Training loss is not evidence.** LightGBM reaches 0.0052 on its own training rows and 0.1082 on validation — about twenty times higher. Logistic regression goes from 0.0349 to 0.1188. Had I chosen on training loss, LightGBM would have won by a huge margin, for the wrong reason.

**The validation win is small.** The gap between LightGBM and logistic regression is 0.0106 per row. Over 114 rows that's about 1.2 units of total log loss. A single confidently wrong prediction is worth more than that. And this isn't hypothetical: the example also reports how concentrated the loss is. For logistic regression, the worst validation row alone costs 6.33, and the three worst rows make up 82.5% of the total. For LightGBM the worst row costs 5.38 (a probability of about 0.005 for the true class) and the three worst rows make up 74.5%. With so few rows, one or two hard cases decide the ranking. I read the result as “on this split, LightGBM had a slightly lower observed validation loss”, not as “LightGBM is better” — and not as proof that it is at least as good, either. A gap this size doesn't establish a reliable difference in either direction.

**The test number is higher than the validation number.** 0.1775 versus 0.1082. Part of this is expected: I picked the candidate that looked best on validation, and the best-looking of several candidates tends to look a bit better than it is. Part of it is plain noise in 114 rows. On the test set five rows have a loss above 1.0, compared to three on validation, and the worst costs 5.87. I computed these per-row figures only for the frozen model, after the choice was saved; they don't feed back into anything.

**I don't know how logistic regression would score on the test set.** The script never evaluates it there, and that's deliberate. If I looked and it happened to be better, I'd be tempted to switch — and then the test set would have become a second validation set, and 0.1775 would no longer be an honest estimate for whatever I ended up choosing.

## Make the honest path the easy one

Writing “choose on validation, then test once” in a notebook is easy. Following it after an unexpected result is harder. So the example turns the rule into code and tests.

The test evaluation reads the frozen choice from disk and refuses to run if anything doesn't match:

```python
# fair_baseline/selection.py (fragment)
def evaluate_frozen_choice_on_test(selection_path, models, data, split):
    """Score only the frozen model on test rows. Nothing is refitted here."""
    chosen = _load_frozen(selection_path, split)["selected_model"]
    model = models[chosen]
    ...
```

`_load_frozen` recomputes the record's hash and compares three things with the current split: the data digest, the seed and a hash of the test indices. Edit the selected model name by hand and it raises; point it at a selection made with different data, a different seed or different test rows and it raises. It does not compare the training and validation indices — swapping those two parts would pass this function. That check lives in `run_experiment.py`, which before anything else regenerates the split from the saved seed and stops unless all three parts — train, validation and test — match the frozen file.

The record hash is a guard against accidents and casual edits, not a security boundary. It's a plain SHA-256 stored next to the data it covers, with no secret key, so anyone who edits the record can recompute it. It catches a hand-edited model name; it doesn't stop someone who sets out to forge a record.

Two tests check the isolation more directly:

- **Poisoned test rows.** I replace every test row's features with `NaN` and flip their labels, then rerun the whole validation comparison. The scores must be identical to the original ones. If any training or selection step touched a test row, they would change or the code would fail.
- **Untouchable models.** In the test-evaluation test, every non-selected model is replaced by an object that raises an error on any attribute access. The evaluation passes only if it uses nothing but the frozen choice.

The full suite has 22 tests: disjoint split coverage of all 569 rows, stratification, reproducibility against the saved file, the digest, the malignant mapping, the training-only scaler, metric ranges, the selection rule, the frozen-record checks, repeatability and the loss reporting. To check that the tests can actually fail, I also injected five deliberate bugs one at a time into a scratch copy — no relabelling, training on train plus test, selecting the largest loss, scoring a different model on test, overlapping splits. The relevant tests failed for each. The tests fit the models many times on the same rows. Like the repeated run, that checks reproducibility and the protocol's mechanics; it isn't a new experiment, and no test result is used to change a parameter or the choice.

None of this makes cheating impossible. Anyone can edit the code. It makes the honest sequence the default, and makes deviations visible.

## Scope and limits

This is a small, clean, well-studied dataset. 341 training rows, no missing values, measurements donated to the UCI repository in 1995. Many models score well on it, and it's a good place to practise evaluation discipline because the mechanics aren't drowned out by data problems. It's not a place to learn which algorithm wins in general. Neither is a single split: with a different seed the ranking between logistic regression and LightGBM could flip. I didn't try other seeds, because looking at several and reporting the friendliest would be exactly the problem this post is about. If I wanted to measure that instability, I'd decide beforehand to use repeated cross-validation on the training and validation rows, and still keep the test rows aside.

**This is not a diagnostic tool.** The numbers say nothing about clinical performance. The data are not a representative sample of any patient population, the 0.5 threshold was not chosen with the cost of a missed cancer in mind, and four false negatives out of 42 malignant test cases would be a serious matter in any real setting. Using a model like this in care would need clinical expertise, proper validation on relevant data and regulatory review — none of which a blog example provides.

About determinism: `deterministic=True` asks LightGBM for stable results with the same data, the same parameters and the same build. Its documentation also says that results are expected to differ across LightGBM versions, compilers and systems. My check is two identical runs on one machine. On a different machine the last decimals may differ.

## About the versions and this run

The date on this post, 23 November 2023, is an editorial label for the software setup, not a claim that the post was published then. It was written and the example executed on 12 October 2026. Every library it relies on had been released by the label date: LightGBM 4.1.0 (12 September 2023), scikit-learn 1.3.2 (23 October 2023), NumPy 1.26.2 (12 November 2023), SciPy 1.11.4 (18 November 2023) and pytest 7.4.3 (24 October 2023).

The numbers themselves are from a present-day run, made in October 2026 on macOS (arm64) with CPython 3.11.12. I used [uv](https://docs.astral.sh/uv/) to recreate the environment. uv is a modern tool, not something this post assumes was available in 2023; I use its `exclude-newer` setting so that even indirect dependencies resolve to files uploaded before the post date. PyPI has no LightGBM 4.1.0 wheel for Apple Silicon, so on that platform it is compiled from source, which needs the Xcode Command Line Tools and `libomp`.

## Run it yourself

Download [baseline-lightgbm-example.zip]({{ '/assets/downloads/baseline-lightgbm-example.zip' | relative_url }}). It contains the `fair_baseline` package, the two scripts, the tests, the frozen `artifacts/split.json`, `pyproject.toml`, `uv.lock` and a README with prerequisites. After unpacking:

```bash
cd baseline-lightgbm-example
uv sync --locked
uv run --locked pytest -v
uv run --locked python run_experiment.py
```

The first command needs internet access for the packages; the dataset itself is never downloaded. On a Mac, install `libomp` first (`brew install libomp`). If the run prints a digest error, the data no longer matches the frozen split, and the script stops instead of training on misaligned rows.

## What I'd keep from this

Freeze the split before any model exists, and save it as positions plus a digest of the data they point to. Fit every preprocessing step on training rows only. Compare against a no-skill model and against a simple model that's genuinely competent. Choose on validation with a metric that matches how the output will be used. Write the choice down. Then look at the test set once, and report what it says — even when it's less flattering than validation.

In this example the frozen choice was LightGBM, by a margin smaller than one bad prediction. That's a less exciting sentence than “LightGBM gets 0.98 AUC”. It's also one I can defend.

### Sources

- [LightGBM v4.1.0 release](https://github.com/microsoft/LightGBM/releases/tag/v4.1.0) — published 12 September 2023.
- [LightGBM 4.1.0: `LGBMClassifier`](https://lightgbm.readthedocs.io/en/v4.1.0/pythonapi/lightgbm.LGBMClassifier.html) — constructor parameters, `n_jobs`, `random_state`, extra parameters via `**kwargs`.
- [LightGBM 4.1.0: Parameters](https://lightgbm.readthedocs.io/en/v4.1.0/Parameters.html) — `deterministic`, `force_col_wise`, `boost_from_average`, `num_leaves`, `max_bin`, `lambda_l2`, binary log loss objective.
- [LightGBM 4.1.0: Installation Guide](https://lightgbm.readthedocs.io/en/v4.1.0/Installation-Guide.html) — OpenMP and `libomp` on macOS.
- [scikit-learn 1.3: `load_breast_cancer`](https://scikit-learn.org/1.3/modules/generated/sklearn.datasets.load_breast_cancer.html) — 569 samples, 30 features, 212 malignant / 357 benign.
- [scikit-learn 1.3: `train_test_split`](https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.train_test_split.html), [`DummyClassifier`](https://scikit-learn.org/1.3/modules/generated/sklearn.dummy.DummyClassifier.html), [`log_loss`](https://scikit-learn.org/1.3/modules/generated/sklearn.metrics.log_loss.html), [`roc_auc_score`](https://scikit-learn.org/1.3/modules/generated/sklearn.metrics.roc_auc_score.html), [`Pipeline`](https://scikit-learn.org/1.3/modules/generated/sklearn.pipeline.Pipeline.html), [`StandardScaler`](https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.StandardScaler.html).
- PyPI release files: [lightgbm 4.1.0](https://pypi.org/project/lightgbm/4.1.0/), [scikit-learn 1.3.2](https://pypi.org/project/scikit-learn/1.3.2/), [numpy 1.26.2](https://pypi.org/project/numpy/1.26.2/), [scipy 1.11.4](https://pypi.org/project/scipy/1.11.4/), [pytest 7.4.3](https://pypi.org/project/pytest/7.4.3/).
- W. Wolberg, O. Mangasarian, N. Street, W. Street. [Breast Cancer Wisconsin (Diagnostic)](https://doi.org/10.24432/C5DW2B). UCI Machine Learning Repository, 1995. CC BY 4.0.
- [uv settings: `exclude-newer`](https://docs.astral.sh/uv/reference/settings/) — modern reproduction tooling used for this run, not part of the 2023 setup.
