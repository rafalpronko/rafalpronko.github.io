---
layout: post
lang: en
title: "A small LoRA experiment with a baseline and a separate evaluation"
date: 2024-07-04 09:00:00 +0200
permalink: /blog/lora-baseline-separate-evaluation/
author: Rafał Prońko
categories: [machine-learning]
description: "LoRA from scratch in NumPy on a synthetic linear layer: frozen base, full adaptation, ranks 1/2/4, validation-only selection and a locked test split."
image: /assets/images/lora-baseline-separate-evaluation.svg
---

# A small LoRA experiment with a baseline and a separate evaluation

![Diagram: input x goes through a frozen weight W0 and, in parallel, through a trainable low-rank path A then B scaled by alpha over r; the outputs are added. Below, four separate data splits: source fits W0, target train updates weights, target validation picks learning rate, checkpoint and rank, target test is locked and read once.]({{ '/assets/images/lora-baseline-separate-evaluation.svg' | relative_url }})

<time datetime="2024-07-04">4 July 2024</time> — editorial label, not publication history.

**Execution date: 12 October 2026.** The code and all reported numbers were produced then, using macOS 26.5.2 arm64 and CPython 3.11.12. The date above identifies an editorial slot; it does not date the experiment.

“LoRA worked” needs two qualifications: compared with what, and measured on which data? An adapter can beat an untouched base and still lose to full fine-tuning. A rank chosen on the reported test set can look better than it generalises.

A single linear layer, synthetic data, NumPy and a CPU are enough to examine both problems. This is **not an LLM benchmark**: it demonstrates LoRA’s mechanics, checkpoint tests and a comparison under a shared training budget, without claiming how ranks behave in a Transformer.

## What LoRA changes in a layer

Take a dense layer with weight W₀ ∈ ℝ<sup>d×k</sup>: it maps an input x ∈ ℝ<sup>k</sup> to h = W₀ x ∈ ℝ<sup>d</sup>.

Full fine-tuning updates all d · k entries of W₀. LoRA (Hu et al., 2021) freezes W₀ and learns the *change* as a product of two thin matrices:

```text
h = W₀·x + (α/r)·B·A·x,    A ∈ ℝ^(r×k),  B ∈ ℝ^(d×r),  r ≪ min(d, k)
```

- W₀ is frozen: it receives no gradient updates.
- A and B are the only trainable parameters: r(d + k) numbers instead of d·k.
- Δ W = (α/r)·BA has rank at most r. That is the bet: the update needed for the new task is low-rank, even if W₀ is not.
- α is a constant. The paper sets it to the first r it tries and does not tune it; dividing by r helps reduce the need to retune hyperparameters when r changes. The paper relates α to the learning rate specifically for Adam; this example uses gradient descent.

In my example k = 16 and d = 8. Full adaptation trains 128 numbers; LoRA with r = 1, 2, 4 trains 24, 48 and 96.

### Why one factor is random and the other is zero

The paper initialises A randomly (Gaussian) and B with zeros. The official `loralib` code does the same thing for linear layers with Kaiming-uniform instead of a Gaussian for A; the essential part is identical: one factor random, B = 0.

B = 0 gives Δ W = 0: the adapted model starts exactly as the base.

Why not both zero? Look at the gradients. With the mean squared error over n samples and d outputs, let E = Ŷ - Y ∈ ℝ<sup>n×d</sup> and s = α / r. The gradient with respect to the effective weight is

```text
G = (2 / (n·d)) · Eᵀ·X  ∈ ℝ^(d×k)
```

and by the chain rule

```text
∂L/∂A = s · Bᵀ·G        ∂L/∂B = s · G·Aᵀ
```

With B = 0, A receives no gradient; random A lets B receive a learning signal. After B moves, A can learn too. With both factors zero, both gradients vanish and training cannot start; a test checks this.

## The experiment, fixed before the first run

I fixed the main protocol before its first run. A separate post hoc check, described below, examines the training budget using validation only.

**Task.** Multi-output linear regression y = W x + ε, x ~ 𝒩(0, I<sub>16</sub>), noise standard deviation 0.5 (noise variance 0.25).

**Domains.** A source weight W<sub>src</sub> has 𝒩(0, 1) entries. The target weight is

```text
W_tgt = W_src + U·Vᵀ/√2 + 0.05·N      # rank-2 shift + small dense shift
```

with Gaussian U ∈ ℝ<sup>8×2</sup>, V ∈ ℝ<sup>16×2</sup> and N ∈ ℝ<sup>8×16</sup>. The shift’s singular values are 9.56, 3.98, then 0.23 and below. **This favours low-rank adaptation by construction**; it cannot establish the rank of real fine-tuning updates.

**“Pretrained” base.** W₀ is a least-squares fit on 400 source samples. It plays the role of the base model and is never updated afterwards.

**Splits.** Each split is drawn from its own seed, so the samples are distinct:

| Split | Samples | Seed | Used for |
|---|---:|---:|---|
| source train | 400 | 2000 + replicate | fitting W₀ |
| source test | 2000 | 3000 + replicate | checking behaviour on the old domain |
| target train | 32 | 4000 + replicate | gradient updates |
| target validation | 200 | 5000 + replicate | learning rate, checkpoint, method and rank |
| target test | 2000 | 6000 + replicate | read once, after selection |

The world (W<sub>src</sub>, U, V, N) uses seed 1000; LoRA's A uses 7000 + replicate. Target training data is deliberately scarce: 32 samples × 8 outputs = 256 noisy targets for a layer with 128 weights.

**Methods.** Frozen base (no training), full adaptation (all of W, initialised from W₀), LoRA with r ∈ {1, 2, 4} and α = 2.

**Budget.** The same for every trainable method: 300 steps of full-batch gradient descent, learning rate from the grid {0.01, 0.03, 0.1, 0.3}, validation MSE recorded every 10 steps. The best checkpoint per run, the best learning rate per method, the best method overall and the best LoRA rank are all chosen by **target validation MSE**. A run that produces a non-finite training loss or parameter is stopped and keeps its best finite validation checkpoint, including step zero.

**Replicates.** Five replicates; each one changes the data seeds and LoRA's initialisation, not the world.

## The code

The experiment fits in one module. These code blocks are **excerpts**, not standalone programs: they use `numpy as np` and helpers from the downloadable module. The adapter and merge helpers are:

```python
class LoRA:
    def __init__(self, W0, r, alpha, seed):
        self.W0 = np.array(W0, dtype=float)
        self.W0.flags.writeable = False            # frozen base
        d, k = self.W0.shape
        rng = np.random.default_rng(seed)
        self.A = rng.normal(scale=1 / np.sqrt(k), size=(r, k))  # random
        self.B = np.zeros((d, r))                              # zero -> BA = 0 at start
        self.r, self.alpha, self.scaling = r, alpha, alpha / r

    n_trainable = property(lambda self: self.A.size + self.B.size)
    params = property(lambda self: [self.A, self.B])

    def delta(self):
        return self.scaling * self.B @ self.A

    def predict(self, X):
        return X @ self.W0.T + self.scaling * (X @ self.A.T) @ self.B.T

    def weight(self):
        return merge(self)

    def grads(self, X, Y):
        G = _grad_W(self.predict(X), X, Y)        # gradient w.r.t. W0 + s*BA
        return self.scaling * self.B.T @ G, self.scaling * G @ self.A.T  # (dA, dB)


def merge(m):
    return m.W0 + m.delta()


def unmerge(W, m):
    return W - m.delta()
```

Inputs are stored as rows, so a batch X ∈ ℝ<sup>n×k</sup> produces X Wᵀ ∈ ℝ<sup>n×d</sup>. `predict` never builds BA: it goes through the r-dimensional bottleneck, X Aᵀ first. `_grad_W` is the G from the formula above.

`W0` is read-only: accidental in-place updates raise an error.

Full adaptation has the same interface and trains W initialised from W₀. Both use this checkpoint helper and loop:

```python
def _checkpoint(model, step, val_mse):
    """Copy one coherent state, including the adapter factors at this step."""
    state = {"step": step, "val_mse": val_mse, "W": model.weight()}
    if isinstance(model, LoRA):
        state.update(A=model.A.copy(), B=model.B.copy(), scaling=model.scaling,
                     rank=model.r, alpha=model.alpha)
    return state


def train(model, train_xy, val_xy, steps, lr, eval_every):
    """Full-batch gradient descent; keep the checkpoint with the lowest validation MSE."""
    X, Y = train_xy
    best = _checkpoint(model, 0, mse(model.predict(val_xy[0]), val_xy[1]))
    history = [(0, mse(model.predict(X), Y), best["val_mse"])]
    for step in range(1, steps + 1):
        with np.errstate(over="ignore", invalid="ignore"):  # overflow is handled explicitly below
            for p, g in zip(model.params, model.grads(X, Y)):
                p -= lr * g
            tr = mse(model.predict(X), Y)
        if not (np.isfinite(tr) and all(np.isfinite(p).all() for p in model.params)):
            return {**best, "history": history, "diverged_at": step}  # keep last finite best
        if step % eval_every == 0:
            va = mse(model.predict(val_xy[0]), val_xy[1])
            history.append((step, tr, va))
            if va < best["val_mse"]:
                best = _checkpoint(model, step, va)
    return {**best, "history": history, "diverged_at": None}
```

Step zero is a valid checkpoint. `_checkpoint` copies A and B at that step and at every validation improvement, alongside W and the scale/rank metadata. `select` keeps that saved state rather than reading the factors at the final training step. If training never improves on the start, LoRA’s selected checkpoint is the base model.

The selection signature (fragment; its remaining body is in the module) removes a direct test-data argument:

```python
def select(cfg, source_train, target_train, target_val, *, replicate):
    """Every choice (lr, checkpoint, method, rank) uses target validation only."""
```

`select` accepts training and validation arrays, and rejects `test=`. This is an **API boundary, not an isolation guarantee**: code with `cfg` and the replicate seed could regenerate the test data, and module globals remain accessible. In this implementation, selection uses only the supplied training and validation arrays. The loop records its choices before evaluating the selected checkpoints on target and source test data. Stronger isolation would require a separate process or controlled access to held-out data.

Download the [complete ZIP]({{ '/assets/code/lora-baseline-separate-evaluation/lora-numpy.zip' | relative_url }}) for the module, tests, configuration, lock file, sensitivity script and measured results. The reproduction steps below preserve the required `tests/` directory. Individual files are also available: [`lora_lab.py`]({{ '/assets/code/lora-baseline-separate-evaluation/lora_lab.py' | relative_url }}), [`tests/test_lora_lab.py`]({{ '/assets/code/lora-baseline-separate-evaluation/tests/test_lora_lab.py' | relative_url }}), [`pyproject.toml`]({{ '/assets/code/lora-baseline-separate-evaluation/pyproject.toml' | relative_url }}), [`uv.lock`]({{ '/assets/code/lora-baseline-separate-evaluation/uv.lock' | relative_url }}) and [`sensitivity.py`]({{ '/assets/code/lora-baseline-separate-evaluation/sensitivity.py' | relative_url }}).

## What the tests check

The original suite had eleven tests in ten groups: the gradient group contained two tests. Checkpoint regressions added during correction now give **15 collected test cases in 14 test functions**, grouped below. The new regressions first failed on the old implementation (5 failed, 10 passed), then passed after the fix.

1. **Gradients.** A, B and full W match central finite differences (`rtol=1e-6`) with non-zero B.
2. **Initialisation and zero factors.** B = 0 gives exactly the base predictions; A = B = 0 gives zero gradients.
3. **Frozen base.** W₀ stays unchanged and read-only during training while B moves.
4. **Size and rank.** Counts are r(d + k) and d·k; rank(Δ W) ≤ r.
5. **Merge and unmerge.** Predictions agree; subtracting the adapter recovers W₀ within floating-point tolerance.
6. **Splits and API boundary.** Seeds and sample rows are distinct; `select` is repeatable and rejects the direct `test=` argument. This does not test isolation.
7. **Every saved checkpoint.** For all five replicates, saved A/B and scaling reconstruct W, target/source predictions, validation and test MSE, and unmerge within `atol=1e-12`, `rtol=0`.
8. **Earlier and step-zero checkpoints.** Selected factors are independent copies even when training continues. Step zero survives both ordinary training and divergence, including a saved divergent-run checkpoint.
9. **Divergence.** An absurd learning rate is detected at the offending step without NumPy warnings; the selected state stays finite.

The first green run logged overflow between validation points. Training now checks loss and parameters after every step and records `diverged_at`. In the final experiment, only r = 1 with lr = 0.3 diverged, at steps 16–29 in all five replicates; that learning rate was never selected.

## Results

Target columns show mean ± sample standard deviation over five replicates; the source column shows the mean only. Each method’s test result uses its checkpoint chosen on validation before test evaluation. These summaries describe this one synthetic world, not a general performance estimate.

| Method | Trainable params | Target val MSE | Target test MSE | Source test MSE |
|---|---:|---:|---:|---:|
| frozen base | 0 | 13.92 ± 0.89 | 13.63 ± 0.35 | 0.264 |
| full adaptation | 128 | 0.511 ± 0.068 | 0.512 ± 0.067 | 14.09 |
| LoRA r = 1 | 24 | 2.363 ± 0.159 | 2.381 ± 0.092 | 12.12 |
| LoRA r = 2 | 48 | 0.355 ± 0.028 | 0.356 ± 0.030 | 14.16 |
| LoRA r = 4 | 96 | 0.376 ± 0.022 | 0.380 ± 0.026 | 14.13 |

The oracle W<sub>tgt</sub> gives mean target test MSE **0.2497**, ranging from **0.2484 to 0.2514** across the five finite test sets. The noise variance 0.25 is the floor in expectation for independent test data. A finite test score can fall below 0.25 or the oracle’s score by chance. A real project does not have this generating-weight reference.

Validation chose LoRA with r = 2 in all five replicates, both as the best rank and as the best method overall. On the locked test split, full adaptation minus LoRA r = 2 was 0.148, 0.190, 0.085, 0.209 and 0.147 MSE in the five replicates.

How I read this:

- **The baseline matters.** r = 2 improves on the frozen base (13.6 versus 0.36) and full adaptation (0.51 versus 0.36). The second comparison measures its benefit beyond ordinary adaptation.
- **Rank 1 cannot express the rank-2 component.** No grid learning rate brought its validation MSE below 2.15.
- **Rank 4 loses to rank 2 here.** Fitting noise and the small dense shift with scarce data is a plausible explanation for the gap, not a separately measured cause.
- **Full adaptation has fewer constraints.** Its 128 free parameters can fit noise in directions outside the main shift. More data or a different shift could change the ranking; neither was tested.
- **The r = 2 means are close, with variation across replicates.** Its validation MSE ranges from 0.3295 to 0.3982 and target test MSE from 0.3345 to 0.4088. Within-replicate absolute gaps are 0.0051–0.0208. This statement applies to r = 2; other methods have larger gaps. Agreement does not prove absence of leakage, and choosing the minimum validation score makes that score optimistically biased. The test is a separate estimate after selection.

### “Forgetting” and what LoRA actually preserves

Source test MSE rises from 0.264 for the base to 12–14 after adaptation. The domains require different weights; this measures the tradeoff of using one adapted weight on both.

LoRA leaves W₀ untouched. Subtracting (α/r)·BA from the merged weight recovers it up to floating-point rounding; the regression checks use `atol=1e-12`. The immutable W₀ is also stored in each checkpoint, so loading that array recovers the base bit-for-bit. Full adaptation updates a separate W in this example, and its checkpoint also stores W₀: a full-adaptation workflow can retain the base too. LoRA’s distinction is a compact task-specific update alongside a shared base.

### Is the ranking an artefact of the budget?

Full adaptation selected the largest grid learning rate, often at step 300. Perhaps it needed more training.

I checked this **after** the main run: 3000 steps, a grid extended with lr = 0.6, and validation every **50** steps rather than every 10. The script generates all splits through `make_splits`, but passes only source train, target train and target validation to `select`; it never evaluates either test split. The best validation MSE for full adaptation barely changed (replicate 0: 0.4837 → 0.4837; replicate 2: 0.4305 → 0.4296), and LoRA r = 2 remained lower in every replicate. Under this extended budget, full adaptation’s selected validation error changes very little. The coarser checkpoint grid also makes this a sensitivity check, not a comparison changing only the step count. This check did not change any selection, and I didn't evaluate the longer runs on the test split.

## Merging and inference

After training, W = W₀ + (α/r)·BA can serve as a normal dense layer. This supports the paper’s inference-latency argument; the test checks prediction equivalence within floating-point tolerance.

Two caveats from the paper apply directly:

- A merged layer belongs to one task. Different adapters within one batch require another approach, such as unmerged paths with their extra matrix products.
- Subtractive recovery needs the matching A, B and scaling and is subject to floating-point rounding. Each LoRA checkpoint stores the selected factors, scaling, rank, α and step, alongside W₀ and merged W. Loading stored W₀ avoids subtraction entirely.

This example omits quantized weights and adapter dropout. Frameworks can also merge automatically: the inspected `loralib.Linear` merges on `eval()` and unmerges on `train()` when `merge_weights=True`.

## Fewer trainable parameters is not the same as less memory

LoRA reduces the number of **trainable** parameters: 48 instead of 128 here. For a large Transformer trained with Adam, the paper reports VRAM savings of up to two thirds, mainly because there is no optimizer state for frozen weights, and checkpoints that are about 10,000× smaller for GPT-3 175B.

The base is still needed at deployment. Here W₀ plus A/B holds 176 numbers, versus 128 for fully adapted W. Savings come from optimizer state and many compact adapters sharing a base. Activations still consume memory; backpropagation through intervening frozen layers is still needed in deep networks.

## Limitations

- One linear layer and full-batch gradient descent. MSE is convex in W for full adaptation but non-convex in the LoRA factors A and B. There's no Adam, no nonlinearity, no attention and no tokenisation; the paper's remark that tuning α is roughly like tuning the learning rate is about Adam and isn't tested here.
- The target shift is approximately rank 2 by construction. The experiment can't tell you which rank a real task needs.
- Five replicates of one synthetic world. The standard deviations describe variation across data samples and initialisations, not across tasks.
- The configuration (32 target samples, noise 0.5, the grid, 300 steps) was fixed before the first run and not tuned afterwards. A different configuration can produce a different ranking.
- NumPy’s `Generator` does not guarantee random-stream compatibility across versions; floating-point results can differ across platforms. Reproduction on this host matched the saved results; identical digits elsewhere are not promised.

## Reproducing it

The code was written and executed on **12 October 2026**, using CPython 3.11.12, NumPy 1.26.4, pytest 8.2.2 and uv 0.7.8. Python 3.11, NumPy 1.26.4 and pytest 8.2.2 were available before the editorial label; this is a modern execution with pinned dependencies, not a record of a 2024 run. [uv](https://docs.astral.sh/uv/) is present-day reproduction tooling. The lock file’s `exclude-newer = "2024-07-04T00:00:00Z"` also constrains transitive package uploads.

Install uv, then download `lora-numpy.zip` from the link above into your working directory. Unpack and run:

```bash
mkdir lora-reproduction
unzip lora-numpy.zip -d lora-reproduction
cd lora-reproduction/lora_numpy
uv sync --locked --python 3.11
uv run --locked pytest -v -W error::RuntimeWarning
uv run --locked python -W error::RuntimeWarning lora_lab.py reproduced-results
uv run --locked python -W error::RuntimeWarning sensitivity.py reproduced-results/sensitivity_val_only.json
```

The last command is optional and evaluates validation only. The ZIP contains `lora_numpy/lora_lab.py`, `sensitivity.py`, `pyproject.toml`, `uv.lock`, `README.md`, `tests/test_lora_lab.py` and `results/` with the measured JSON files and 25 checkpoints. No hidden file is required; `--python 3.11` selects the interpreter. New outputs go to `reproduced-results/`, preserving the supplied reference results. A fresh environment needs access to Python/package downloads or a populated package cache.

`metrics.json` records the configuration, seeds, environment, shift singular values and per-method learning-rate/checkpoint selections, validation scores, divergence steps and target/source test MSE. Each `.npz` contains the selected W, W₀ and step; LoRA files also contain the matching A, B, scaling, rank and α. The seeds and saved weights allow every reported score to be recomputed without retraining. The extracted ZIP was tested in a new environment and reproduced the results on this host.

## What I take from it

A full-adaptation baseline, validation for selection and test evaluation afterwards make this comparison useful. Coherent saved adapter states make it reproducible. Neither a clean API nor plausible scores alone establishes isolation.

In this constructed world, r = 2 beat full adaptation with scarce data, r = 1 lacked capacity, and the stored W₀ remained recoverable bit-for-bit. A real-model experiment needs its own evidence.

### Sources

All sources were accessed on **12 October 2026**. The paper establishes availability of the mechanism before the editorial label; the implementation link below is explicitly a modern source reading.

- [Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”, PDF v2](https://arxiv.org/pdf/2106.09685v2) — Section 4.1 (reparametrization, initialisation, α/r and merging), Section 4.2 (memory, checkpoint size and batching). The [submission history](https://arxiv.org/abs/2106.09685) dates v1 to 17 June 2021 and v2 to 16 October 2021; no historical availability of the HTML rendering is assumed.
- [microsoft/LoRA, `loralib/layers.py` at c4593f0](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py) — `Linear` shapes, frozen weight, Kaiming-uniform A, zero B, α/r and merge/unmerge. This commit is from **17 December 2024**, after the editorial label; it was inspected on 12 October 2026, not presented as a July 2024 snapshot.
- [NumPy 1.26: Random `Generator`](https://numpy.org/doc/1.26/reference/random/generator.html) — `default_rng` and no stream compatibility guarantee across versions.
- [NumPy 1.26: `numpy.savez`](https://numpy.org/doc/1.26/reference/generated/numpy.savez.html) — checkpoint format.
- [NumPy 1.26.4 release notes in the versioned 2.0 manual](https://numpy.org/doc/2.0/release/1.26.4-notes.html) — Python 3.9–3.12 support. The [1.26.4 release record](https://pypi.org/project/numpy/1.26.4/) dates it to 5 February 2024.
