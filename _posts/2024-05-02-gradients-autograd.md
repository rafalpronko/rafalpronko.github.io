---
layout: post
lang: en
title: "Gradients from scratch to autograd: what actually flows backward"
date: 2024-05-02 09:00:00 +0200
permalink: /blog/gradients-autograd/
author: Rafał Prońko
categories: [deep-learning]
description: "Derivative, partial derivatives and the gradient, derived by hand for linear regression and checked against PyTorch autograd and central finite differences."
image: /assets/images/gradients-autograd.svg
---

# Gradients from scratch to autograd: what actually flows backward

![Diagram of a computation graph for linear regression: the forward pass goes from the leaf tensors w and b through a matrix-vector product, addition, subtraction, squaring and a mean to a scalar loss; the backward pass runs in reverse, multiplying vector-Jacobian products until the gradients land in w.grad and b.grad.]({{ '/assets/images/gradients-autograd.svg' | relative_url }})

<time datetime="2024-05-02">2 May 2024</time>

In a PyTorch training loop, computing the gradients takes one line: `loss.backward()`. The forward pass, the optimizer step and zeroing the gradients are separate lines, but this is the one that hides the most. It's a convenient line, until something goes wrong.

The gradient doubles from one iteration to the next. `.grad` of a tensor I care about is `None`. The second `backward()` throws an error about a freed graph. Someone changes `reduction="sum"` to `"mean"` and the same learning rate suddenly stops working.

These aren't framework bugs. They follow directly from what a gradient is and how autograd computes it. So instead of memorising rules, I want a model in my head that explains them.

I'll start with a derivative of one variable, get to the gradient, derive it by hand for linear regression and then check the result in three independent ways: by the formula, by autograd and by finite differences. Small data, CPU, `float64`. First I need correct mechanics; scale comes later.

## A derivative is a local slope

I take a single sample `x = 2`, target `y = 1` and a model with one weight: the prediction is `w · x`. The loss is the squared error:

```
L(w) = (w·x − y)²
```

The derivative tells me how quickly `L` changes when I nudge `w`:

```
dL/dw = 2·(w·x − y)·x
```

At `w = 0` this is `2·(0 − 1)·2 = −4`. I read it like this: if I increase `w` by a small `δ`, the loss changes by approximately `−4·δ`. The sign says which direction makes the loss grow. The magnitude says how steep it is here.

The key word is *here*. A derivative is a linear approximation at one point. It says nothing about what happens a long way from it.

## Many parameters: partial derivatives

A real model has more than one parameter. For linear regression with three features I have a weight vector `w = (w₀, w₁, w₂)` and a bias `b`.

A partial derivative `∂L/∂w₀` answers the same question as before, but for one parameter at a time: how does the loss change when I move only `w₀` and freeze everything else?

I get four numbers like that: `∂L/∂w₀`, `∂L/∂w₁`, `∂L/∂w₂` and `∂L/∂b`.

## The gradient: a vector of partial derivatives

The gradient is simply those partial derivatives collected into a vector:

```
∇L = (∂L/∂w₀, ∂L/∂w₁, ∂L/∂w₂, ∂L/∂b)
```

It has two properties that matter in practice.

**Direction.** If I move the parameters by a small step `ε·u` in some direction `u` of unit Euclidean length, `‖u‖₂ = 1`, the loss changes by approximately `ε·(∇L · u)`. By the Cauchy–Schwarz inequality the dot product is largest when `u = ∇L / ‖∇L‖₂` (for a non-zero gradient). So the gradient points in the direction of the steepest *increase* of the loss. Gradient descent goes the opposite way.

"Steepest" here assumes the ordinary Euclidean length of a step. If I measured step length differently — per-parameter scaling, another norm — the steepest direction would change too. Plain gradient descent uses the Euclidean one.

**Magnitude.** The length `‖∇L‖₂` is the rate of that steepest increase. A long gradient means a steep slope, a short one means a flat area. A zero gradient means a flat point — which is not the same as a minimum.

One more thing I keep in mind: the gradient lives in parameter space. It has exactly the same shape as the parameters. For `w` with shape `(3,)`, `w.grad` also has shape `(3,)`.

## The chain rule, with shapes written out

Now the whole model. Data: `n` samples, `d` features.

```
X     (n, d)    features
y     (n,)      targets
w     (d,)      weights
b     ()        bias, a scalar

pred = X @ w + b              (n,)
r    = pred − y               (n,)    residuals
L    = (1/n) · Σᵢ rᵢ²         ()      scalar loss
```

I compute the gradient from the end, one step at a time.

The loss as a function of the residuals: `∂L/∂rᵢ = 2·rᵢ / n`. This is a vector `g_r` with shape `(n,)`.

The residuals as a function of the predictions: `r = pred − y`, so the gradient passes through unchanged: `g_pred = g_r`, shape `(n,)`.

The predictions as a function of the parameters: `predᵢ = Σⱼ Xᵢⱼ·wⱼ + b`. The derivative of `predᵢ` with respect to `wⱼ` is `Xᵢⱼ`, and with respect to `b` it is `1`. Summing over all samples:

```
∂L/∂w = Xᵀ @ g_pred = (2/n) · Xᵀ @ r        (d, n) @ (n,) → (d,)
∂L/∂b = Σᵢ g_pred,ᵢ = (2/n) · Σᵢ rᵢ          (n,) → ()
```

The shapes match the parameters, as they should.

Look at the order of computation. I started from the scalar loss, got a vector the size of the output of the last operation, and at each step multiplied that vector by the local derivative of the operation, moving towards the input. I never built a full matrix of derivatives of all residuals with respect to all parameters. This is exactly what autograd does.

## What autograd records

In normal grad mode (outside `torch.no_grad()`), when at least one input of a differentiable operation has `requires_grad=True`, PyTorch records the operation in a graph as it runs. Every result gets a `grad_fn` attribute — a `Function` node that knows how to compute the derivative of that operation and which nodes come before it.

The graph is built during the forward pass and, as the PyTorch documentation notes, it is recreated from scratch at every iteration. That's why ordinary Python `if` and `for` statements work in a model: the graph reflects the operations that were actually executed this time.

For the model above, the lab script lists the nodes reachable from `loss.grad_fn`, in breadth-first visiting order:

```
MeanBackward0, PowBackward0, SubBackward0, AddBackward0, MvBackward0, AccumulateGrad, AccumulateGrad
```

This is a list of visited nodes, not a path: the two `AccumulateGrad` nodes are separate endpoints, one per branch. `MvBackward0` is the matrix-vector product `X @ w`. The two `AccumulateGrad` nodes are the end of the road: they belong to the **leaf** tensors `b` and `w`, created by me with `requires_grad=True`. That's where the result lands, in the `.grad` attribute.

The graph is a directed acyclic graph, not a chain. The `AddBackward0` node has two inputs (the product `X @ w` and `b`), so the gradient branches there. If a tensor were used in two places, the gradients from both paths would be summed.

## Reverse mode: vector-Jacobian products

Each differentiable operation has a Jacobian at the current point — the matrix of derivatives of its outputs with respect to its inputs. (At points where an operation is not differentiable, such as `relu` at `0`, PyTorch uses a documented convention instead; my example has no such points.) For the elementwise steps the Jacobian would be an `n × n` matrix that is zero off the diagonal. For `X @ w` with respect to `w` the Jacobian is `X` itself, an `n × d` matrix that already exists as the data; autograd doesn't allocate a separate copy of it, and it doesn't build the `n × n` matrices at all.

Instead, every node receives a vector `v`: the gradient of the loss with respect to the node's output. It returns the gradient of the loss with respect to its inputs. Written with column vectors, for `J ∈ ℝⁿˣᵈ` and `v ∈ ℝⁿ` that is `Jᵀv ∈ ℝᵈ` — the same numbers as the row-vector form `vᵀJ ∈ ℝ¹ˣᵈ`, which gives the operation its name: the **vector-Jacobian product** (VJP). For the matrix-vector product it is simply `Xᵀ @ v`, computed straight from `X`.

The backward pass starts at the loss with `v = 1` and runs these products in reverse order through the graph. This is reverse-mode automatic differentiation.

Why reverse and not forward? Training has a specific shape: one scalar loss and many parameters. Reverse mode gives the derivative of one output with respect to *all* inputs in a single backward pass, whose cost is a small constant multiple of the forward pass. Finite differences need two evaluations of the loss per parameter. For four parameters that's eight forward passes — fine for a check. For a model with millions of parameters it's millions of forward passes per step.

Reverse mode has its price: to compute the VJP for `(residual**2)`, the node has to remember `residual` from the forward pass. The memory for those saved tensors is what grows with depth and batch size.

## The example: three ways to get the same gradient

The library pin is historical: `torch==2.2.2` (released on 27 March 2024) and `numpy<2`, on CPython 3.11. Synthetic data: 16 samples, 3 features, generated from a known linear rule with a little noise and a fixed seed. Everything on the CPU in `float64`.

**Dates.** This article carries the editorial date of 2 May 2024, but the code below was executed and tested on 12 October 2026 with a modern toolchain: uv 0.7.8 and CPython 3.11.12 on macOS arm64. The lockfile reproduces the 2024-era `torch` and `numpy`, but some verification tools in it are newer than the editorial date (for example `pytest` 9.1.1). Python versions other than 3.11 are not covered by the lockfile, and Linux and Windows were not run.

The script compares:

- the gradient from my hand-derived formula (`analytic`),
- the gradient from `loss.backward()` (`autograd`),
- the central difference `(L(θ + h) − L(θ − h)) / 2h` for each parameter separately (`finite diff`).

Then it accumulates gradients over two unequal microbatches, checks gradient accumulation and performs one gradient-descent step.

**Running it yourself.** Download [gradients-autograd.zip]({{ '/assets/downloads/gradients-autograd.zip' | relative_url }}). It contains `pyproject.toml`, `uv.lock`, the lab source, a README (at `gradients_autograd/README.md`) and the tests. With [uv](https://docs.astral.sh/uv/) installed:

```bash
unzip gradients-autograd.zip
cd gradients-autograd
uv sync --locked --python 3.11
uv run --frozen python -m gradients_autograd.lab
uv run --frozen python -m pytest -v
```

`uv sync --locked` refuses to re-resolve if the lockfile and `pyproject.toml` disagree, and `--frozen` runs from the lock as it is. The `torch` download is a few hundred MB. The script itself, `gradients_autograd/lab.py`:

```python
"""Gradients by hand, by autograd and by central finite differences.

Synthetic linear regression on CPU in float64:
    pred = X @ w + b, loss = mean((pred - y) ** 2)
"""

import math
import sys

import torch

DTYPE = torch.float64
N_SAMPLES = 16
N_FEATURES = 3
SEED = 0
FD_STEP = 1e-6
LEARNING_RATE = 0.1
MICROBATCH_SIZES = (5, 11)


def make_data(
    n: int = N_SAMPLES, d: int = N_FEATURES, seed: int = SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return X (n, d) and y (n,) from a known linear rule plus noise."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen, dtype=DTYPE)
    true_w = torch.tensor([2.0, -1.0, 0.5], dtype=DTYPE)[:d]
    noise = 0.1 * torch.randn(n, generator=gen, dtype=DTYPE)
    y = x @ true_w + 0.3 + noise
    return x, y


def mse_loss(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Mean squared error; returns a 0-dim tensor (a scalar)."""
    residual = x @ w + b - y  # (n,)
    return (residual**2).mean()


def analytic_grads(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-derived: dL/dw = 2/n X^T r, shape (d,); dL/db = 2/n sum(r)."""
    with torch.no_grad():
        residual = x @ w + b - y
        n = x.shape[0]
        return 2.0 / n * (x.T @ residual), 2.0 / n * residual.sum()


def autograd_grads(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients from one reverse-mode pass, on fresh leaf copies."""
    w_leaf = w.detach().clone().requires_grad_(True)
    b_leaf = b.detach().clone().requires_grad_(True)
    mse_loss(x, y, w_leaf, b_leaf).backward()
    return w_leaf.grad, b_leaf.grad


def finite_diff_grads(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    h: float = FD_STEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences (L(t + h) - L(t - h)) / 2h, per parameter."""
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"h must be a finite positive number, got {h!r}")
    theta = torch.cat([w.detach(), b.detach().reshape(1)])
    grad = torch.empty_like(theta)
    with torch.no_grad():
        for i in range(theta.numel()):
            plus, minus = theta.clone(), theta.clone()
            plus[i] += h
            minus[i] -= h
            loss_plus = mse_loss(x, y, plus[:-1], plus[-1])
            loss_minus = mse_loss(x, y, minus[:-1], minus[-1])
            grad[i] = (loss_plus - loss_minus) / (2.0 * h)
    return grad[:-1], grad[-1]


def microbatch_grads(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate .grad over consecutive microbatches of the given sizes.

    Each microbatch mean is weighted by its share of the reduced loss
    elements, numel_i / numel_total, so the accumulated gradient equals the
    full-batch mean gradient even for unequal sizes. w and b stay fixed.
    """
    if not sizes or min(sizes) <= 0 or sum(sizes) != x.shape[0]:
        raise ValueError(
            f"sizes must be positive and sum to {x.shape[0]}, got {sizes}"
        )
    w_leaf = w.detach().clone().requires_grad_(True)
    b_leaf = b.detach().clone().requires_grad_(True)
    start = 0
    for size in sizes:
        part = slice(start, start + size)
        weight = y[part].numel() / y.numel()
        (weight * mse_loss(x[part], y[part], w_leaf, b_leaf)).backward()
        start += size
    return w_leaf.grad, b_leaf.grad


def graph_nodes(loss: torch.Tensor) -> list[str]:
    """Names of backward nodes reachable from loss.grad_fn, in BFS order."""
    names: list[str] = []
    seen: list[object] = []  # keep references: id() of a freed node is reused
    queue = [loss.grad_fn]
    while queue:
        node = queue.pop(0)
        if node is None or any(node is other for other in seen):
            continue
        seen.append(node)
        names.append(type(node).__name__)
        queue.extend(child for child, _ in node.next_functions)
    return names


def gd_step(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    lr: float,
) -> tuple[float, float]:
    """One in-place gradient-descent update; returns (before, after)."""
    loss = mse_loss(x, y, w, b)
    w.grad, b.grad = None, None
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        loss_after = mse_loss(x, y, w, b)
    return loss.item(), loss_after.item()


def print_table(
    names: list[str], columns: dict[str, torch.Tensor], ref: torch.Tensor
) -> None:
    header = f"{'param':<6}" + "".join(f"{key:>22}" for key in columns)
    print(header + f"{'|auto-ana|':>12}{'|fd-ana|':>12}")
    for i, name in enumerate(names):
        row = f"{name:<6}" + "".join(
            f"{col[i].item():>22.15f}" for col in columns.values()
        )
        auto_err = abs(columns["autograd"][i] - ref[i]).item()
        fd_err = abs(columns["finite diff"][i] - ref[i]).item()
        print(row + f"{auto_err:>12.1e}{fd_err:>12.1e}")


def flat(grads: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def main() -> None:
    version = sys.version.split()[0]
    print(f"python {version}, torch {torch.__version__}, dtype {DTYPE}")
    x, y = make_data()
    w = torch.zeros(N_FEATURES, dtype=DTYPE, requires_grad=True)
    b = torch.zeros((), dtype=DTYPE, requires_grad=True)
    shapes = [tuple(t.shape) for t in (x, y, w, b)]
    print("shapes: X {}, y {}, w {}, b {}".format(*shapes))

    loss = mse_loss(x, y, w, b)
    print(f"loss at w=0, b=0: {loss.item():.6f}  (0-dim: {loss.dim() == 0})")
    print("backward nodes (BFS order):", ", ".join(graph_nodes(loss)))
    print(f"is_leaf: w={w.is_leaf}, loss={loss.is_leaf}")

    ana = flat(analytic_grads(x, y, w, b))
    auto = flat(autograd_grads(x, y, w, b))
    fd = flat(finite_diff_grads(x, y, w, b))
    print(f"\ngradient comparison (central differences, h={FD_STEP:g}):")
    columns = {"analytic": ana, "autograd": auto, "finite diff": fd}
    print_table(["w[0]", "w[1]", "w[2]", "b"], columns, ana)

    print("\nfinite-difference error on w[0] vs step h:")
    for h in (1e8, 1e4, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        fd_h = finite_diff_grads(x, y, w, b, h)[0][0]
        print(f"  h={h:<6g} |fd-ana| = {abs(fd_h - ana[0]).item():.1e}")

    sizes = MICROBATCH_SIZES
    weighted = flat(microbatch_grads(x, y, w, b, sizes))
    w_eq = w.detach().clone().requires_grad_(True)
    b_eq = b.detach().clone().requires_grad_(True)
    start = 0
    for size in sizes:
        part = slice(start, start + size)
        (mse_loss(x[part], y[part], w_eq, b_eq) / len(sizes)).backward()
        start += size
    equal = flat((w_eq.grad, b_eq.grad))
    print(f"\nmicrobatches {sizes}, max |accumulated - full batch|:")
    print(f"  weight 1/k:      {(equal - ana).abs().max().item():.1e}")
    print(f"  weight n_i/n:    {(weighted - ana).abs().max().item():.1e}")

    loss.backward()
    first = w.grad.clone()
    mse_loss(x, y, w, b).backward()
    ratio = (w.grad / first).tolist()
    print(f"\naccumulation: second backward without zeroing -> "
          f"w.grad / first = {ratio}")

    before, after = gd_step(x, y, w, b, LEARNING_RATE)
    print(f"\none GD step, lr={LEARNING_RATE}: "
          f"loss {before:.6f} -> {after:.6f}")
    print(f"w after step: {[round(v, 6) for v in w.tolist()]}, "
          f"b: {b.item():.6f}")


if __name__ == "__main__":
    main()
```

The output of the run on 12 October 2026 (macOS arm64, CPU):

```
python 3.11.12, torch 2.2.2, dtype torch.float64
shapes: X (16, 3), y (16,), w (3,), b ()
loss at w=0, b=0: 5.817236  (0-dim: True)
backward nodes (BFS order): MeanBackward0, PowBackward0, SubBackward0, AddBackward0, MvBackward0, AccumulateGrad, AccumulateGrad
is_leaf: w=True, loss=False

gradient comparison (central differences, h=1e-06):
param               analytic              autograd           finite diff  |auto-ana|    |fd-ana|
w[0]      -4.829593628378516    -4.829593628378516    -4.829593628130624     0.0e+00     2.5e-10
w[1]       1.286050634256510     1.286050634256510     1.286050634430325     0.0e+00     1.7e-10
w[2]      -0.814610034793499    -0.814610034793499    -0.814610035426711     0.0e+00     6.3e-10
b         -0.676279379114849    -0.676279379114849    -0.676279379163702     0.0e+00     4.9e-11

finite-difference error on w[0] vs step h:
  h=1e+08  |fd-ana| = 1.6e-09
  h=10000  |fd-ana| = 3.0e-13
  h=0.01   |fd-ana| = 4.3e-14
  h=0.0001 |fd-ana| = 8.0e-13
  h=1e-06  |fd-ana| = 2.5e-10
  h=1e-08  |fd-ana| = 1.5e-07
  h=1e-10  |fd-ana| = 3.6e-06

microbatches (5, 11), max |accumulated - full batch|:
  weight 1/k:      1.3e+00
  weight n_i/n:    1.1e-16

accumulation: second backward without zeroing -> w.grad / first = [2.0, 2.0, 2.0]

one GD step, lr=0.1: loss 5.817236 -> 3.508439
w after step: [0.482959, -0.128605, 0.081461], b: 0.067628
```

## Reading the comparison table

The autograd column is identical to my formula, digit for digit. In this run the difference is exactly zero, because both compute the same operations on the same data. I don't rely on that in tests. Reordering floating-point operations can change the last bits, so the test compares with a relative tolerance of `1e-9`. Over 16 samples I expect differences of a few units in the last place (around `1e-15` relative), and `1e-9` still catches real mistakes, such as a missing factor of 2 or `1/n`.

Finite differences differ by `5e-11` to `6e-10`. Where does that come from?

A central difference has two sources of error. For a function that is smooth near `θ` (three times differentiable), Taylor expansion gives

```
L(θ ± h) = L(θ) ± h·L′(θ) + h²/2·L″(θ) ± h³/6·L‴(ξ±)
```

The even terms cancel in the difference, so `(L(θ + h) − L(θ − h)) / 2h = L′(θ) + O(h²·L‴)`: the truncation error is proportional to `h²` and the third derivative. The rounding error is roughly `ε·|L(θ ± h)| / h`, where `ε ≈ 2.2e-16` for `float64`: I subtract two nearly equal numbers and divide by `h`. Balancing the two gives the usual heuristic `h ≈ ε^(1/3)`, about `6e-6` — but only when the parameters, the loss and its third derivative are all of order one. With a differently scaled problem the best step moves. I chose `h = 1e-6`.

The sweep shows that in this example the best value is `h = 0.01`, with an error of `4.3e-14`. The explanation is the model. The MSE loss of linear regression is a quadratic function of the parameters, so its third derivative is zero and the truncation error disappears. Only rounding is left. Across the small steps in the sweep, from `1e-10` up to `0.01`, it falls as `h` grows. It doesn't keep falling: the perturbed losses `L(θ ± h)` themselves grow like `h²`, and their rounding error grows with them. At `h = 1e4` the error is back to `3.0e-13` and at `h = 1e8` it reaches `1.6e-9`.

That's a property of this example, not a general rule. In a network with nonlinear activations a large `h` gives a noticeable truncation error. At a kink, such as `relu` at `0`, there is no classical derivative at all: the central difference averages the two one-sided slopes, while autograd returns the value chosen by PyTorch's convention for that operation, so the two can disagree without either being a bug. What does carry over is the right-hand side of the table: going down from `1e-6` to `1e-10`, the error increases from `2.5e-10` to `3.6e-6`. A smaller step is not more precise.

That's why the test compares finite differences with `rtol=1e-6` and `atol=1e-8`: well above the expected rounding level (around `1e-10·L` for `h = 1e-6`), but tight enough to catch a wrong formula. For `float32` I would set completely different values; `ε` is then about `1.2e-7`.

PyTorch has a built-in tool for this kind of check, `torch.autograd.gradcheck`. It's worth using when writing your own `autograd.Function`. Here I wanted to see each step myself.

## One step of gradient descent

The update is `θ ← θ − η·∇L`. With `η = 0.1`, one step lowered the loss from `5.817236` to `3.508439`.

Why does it work? For a small enough step:

```
L(θ − η·g) ≈ L(θ) − η·‖g‖²
```

The second term is negative whenever the gradient is non-zero. The negative gradient is a *descent direction*.

"Small enough" is a real condition. For this quadratic problem I checked the eigenvalues of the Hessian (`(2/n)·[X 1]ᵀ[X 1]`): they lie between `1.04` and `2.79`. Gradient descent on a quadratic function is stable when `η < 2 / 2.79 ≈ 0.72`. At `η = 0.1` I have a wide margin. At `η = 1.0` the loss would grow from step to step.

The test runs 2000 steps and compares the result with the least-squares solution from `torch.linalg.lstsq` (tolerance `1e-8`). It also checks that the loss never increases. A strictly monotonic version of that check failed when the test was first run: from step 159 onwards the loss was already at its minimum, around `0.0168`, and later steps changed it by up to `1e-16`, a few units in the last place. That is rounding, not divergence, so the check allows a relative slack of `1e-12`.

## Mechanics people get wrong

The tests in the lab check the specific behaviours demonstrated below on `torch 2.2.2`, including the exceptions. A few side remarks (`retain_grad()`, `optimizer.zero_grad()`, `detach()` sharing memory, leaving a parameter out of the optimizer) are explained but not tested.

### `.grad` accumulates

`backward()` doesn't overwrite `.grad`; it adds to it. The documentation of `Tensor.backward` says it directly: it "accumulates gradients in the leaves - you might need to zero .grad attributes or set them to None". The lab shows the effect: after a second `backward()` without zeroing, `w.grad` is exactly twice the first gradient.

In a training loop that's why I call `optimizer.zero_grad()` (or set `p.grad = None` myself) before each `backward()`. Accumulation is also useful: summing gradients from several microbatches before one optimizer step is a deliberate use of this behaviour, with a caveat about weights that I cover below.

### By default, only leaves get `.grad`

By default, during the backward pass only leaf tensors with `requires_grad=True` get gradients in `.grad`. Intermediate results like `residual` have `.grad` equal to `None` (and PyTorch warns when I read it). There are two ways around this. `residual.retain_grad()` before `backward()` keeps the gradient of an intermediate tensor in addition to the leaves. `loss.backward(inputs=[residual])` accumulates into `.grad` of exactly the listed tensors — the test shows `residual.grad` equal to `2·r/n` while `w.grad` stays `None`.

A subtler trap: what counts as a leaf. `torch.randn(3, requires_grad=True).double()` is *not* a leaf. `randn` creates `float32` by default, so the conversion really happens and is recorded in the graph (`ToCopyBackward0`); the gradient goes to the original `float32` tensor that I no longer have a reference to. (On a tensor that is already `float64`, `.double()` returns the same tensor, which stays a leaf.) The fix is to create the tensor with the right `dtype` and `device` straight away.

### Backward releases saved tensors

After the backward pass, autograd releases the tensors saved in the graph. My MSE graph needs saved tensors — `PowBackward0` keeps `residual` — so a second `loss.backward()` through this graph raises `RuntimeError` with the message "Trying to backward through the graph a second time". `retain_graph=True` keeps them, at the cost of memory.

The error comes from the freed saved tensors, not from a rule about calling `backward()` twice. A graph that saves no tensors can be traversed again: for `z = p + 1` the second `z.backward()` succeeds and `p.grad` becomes `2`. I don't build training code on that. In an ordinary loop I recompute the forward pass for every step, and if I have two losses on a shared graph, usually it's enough to add them up and call `backward()` once.

### A non-scalar output needs a `gradient` argument

`backward()` without arguments works only when the output has exactly one element. That includes a 0-dim scalar and also a tensor of shape `(1,)`. For `residual` with shape `(16,)` I get `RuntimeError: grad can be implicitly created only for scalar outputs`.

This follows from reverse mode: the backward pass needs a starting vector `v`. For a one-element output it is `1`. For anything larger I have to provide it myself, and I get the VJP `Jᵀv`. The test shows this: `residual.backward(gradient=2·r/n)` gives exactly the gradient of the mean loss, because `2·r/n` is `∂L/∂r`.

`torch.autograd.grad(outputs, inputs, grad_outputs=...)` does the same thing, but returns the gradients instead of adding them to `.grad`.

### `no_grad` and `detach` cut the graph in different ways

`torch.no_grad()` is a context: operations in the block aren't recorded and their results have `requires_grad=False`, even when the inputs require gradients. The documentation names one exception: factory functions that take a `requires_grad` argument are not affected. Inside `no_grad`, `torch.ones(1, requires_grad=True)` is still a leaf that requires gradients, and so is a newly created `nn.Parameter`. I use the context for inference, evaluation and the parameter update itself — in the lab, `w -= lr * w.grad` runs inside `no_grad`, because the update shouldn't become part of the graph.

`tensor.detach()` cuts one tensor out: it returns a tensor that shares memory with the original but no longer tracks history. I use it for values that should be treated as constants: a target computed by the model, a value I log, a "stop-gradient". In the test, after `pred.detach()` the gradient reaches neither `w` nor `b` through the prediction.

### `requires_grad=False` stops new gradients, not old ones

A leaf with `requires_grad=False` receives no new gradient: `backward()` doesn't add anything to its `.grad`. Its value can still be saved as a constant, and operations that use it are still recorded when other inputs require gradients, so those inputs are differentiated normally. That's how I freeze part of a model during fine-tuning.

But `requires_grad_(False)` doesn't clear a gradient that is already there. The built-in optimizers skip a parameter only when its `.grad` is `None`, so a stale gradient from an earlier `backward()` still moves the "frozen" parameter. The test shows it: `b` gets a gradient, I freeze it, and an `SGD` step still changes it by `lr` times the old gradient. Freezing therefore takes two lines:

```python
p.requires_grad_(False)
p.grad = None
```

Alternatively, leave the frozen parameters out of the optimizer altogether. Either approach prevents optimizer updates. Only clearing `p.grad` explicitly also removes the stale gradient; leaving `p` out of the optimizer keeps the old `.grad` in place, and that optimizer's `zero_grad()` doesn't touch it.

### Loss reduction rescales the gradient

`nn.MSELoss` has `reduction='mean'` by default, and the mean runs over *all elements* of the loss tensor, not over samples. In my example each sample has one output, so the number of elements equals `n = 16`. With `reduction='sum'` the loss — and therefore the gradient — is `16` times larger. The test confirms it: the gradient of the sum equals `16` times the gradient of the mean.

In practice the reduction and the batch size interact with the learning rate, but more narrowly than the usual rule of thumb suggests. With `sum`, a batch made of the same samples twice has exactly twice the gradient. A batch with *different* extra samples has a different gradient, larger or smaller. With `mean` the explicit multiplier is gone, and the duplicated batch has the same gradient as the original. That removes the scale factor; it doesn't make the gradient independent of which samples are in the batch.

### Microbatches must be weighted by their size

Accumulating gradients over `k` microbatches is a deliberate use of accumulation, but the weights matter. Dividing each microbatch mean by `k` reproduces the full-batch mean only when every microbatch has the same number of loss elements. In general each mean has to be weighted by its share of the elements:

```
weighted_loss_i = mean_loss_i · elements_i / total_elements
```

`microbatch_grads` in the lab does exactly that, with the parameters held fixed until all microbatches are processed. With the 16 samples split into 5 and 11, plain `1/k` weighting is off by up to `1.3` in a gradient component, while size weighting matches the full batch to `1.1e-16`.

## What gradient descent guarantees and what it doesn't

The gradient gives me local information: in which direction the loss falls fastest *at this point*. With a small enough step and a smooth function, a step against the gradient lowers the loss.

It doesn't guarantee that:

- a step of a specific size lowers the loss — too large a learning rate can make it grow,
- I'll reach the global minimum — that requires additional assumptions, such as convexity,
- a zero gradient is a minimum — it can also be a maximum or a saddle point.

My example is convex: the MSE of linear regression is a quadratic function of the parameters, so every minimum is global and gradient descent with a suitable step gets there. Neural networks don't have that property. The gradient mechanics are the same, but the guarantees are weaker.

## Limits of this example

This is a CPU example in `float64` with four parameters, executed in October 2026 with `torch 2.2.2` on macOS arm64 only. I didn't check CUDA, newer PyTorch releases, `float32`, mixed precision, `torch.compile` or a real training loop with an optimizer, a scheduler and a validation set, and I don't draw conclusions about them from these numbers.

The tolerances I gave are for `float64` and this particular loss. The `h = 0.01` result applies only to a quadratic loss. The stability bound `η < 0.72` comes from the Hessian of this particular data set.

What carries over directly is the model: the forward pass builds a graph, the backward pass computes vector-Jacobian products from a scalar loss to the leaves, and the result is *added* to `.grad`. Most surprises with `backward()` come down to one of these three facts.

### Sources

- [PyTorch 2.2: Autograd mechanics](https://pytorch.org/docs/2.2/notes/autograd.html) — graph recorded during the forward pass and recreated every iteration; by default only leaf tensors accumulate `.grad`; freezing with `requires_grad_(False)`; gradient conventions for non-differentiable functions; locally disabling gradient computation.
- [PyTorch 2.2: torch.autograd](https://pytorch.org/docs/2.2/autograd.html) — package overview, gradient accumulation, `gradcheck`.
- [PyTorch 2.2: torch.Tensor.backward](https://pytorch.org/docs/2.2/generated/torch.Tensor.backward.html) — `gradient` required for outputs with more than one element, `retain_graph`, `inputs`, accumulation in the leaves.
- [PyTorch 2.2: torch.autograd.grad](https://pytorch.org/docs/2.2/generated/torch.autograd.grad.html) — `grad_outputs` as the vector in the vector-Jacobian product.
- [PyTorch 2.2: torch.no_grad](https://pytorch.org/docs/2.2/generated/torch.no_grad.html) — results don't require gradients, with the factory-function exception.
- [PyTorch 2.2: torch.nn.MSELoss](https://pytorch.org/docs/2.2/generated/torch.nn.MSELoss.html) — the mean runs over all elements; `reduction='sum'`.
- [torch.Tensor.detach](https://pytorch.org/docs/2.2/generated/torch.Tensor.detach.html), [torch.Tensor.is_leaf](https://pytorch.org/docs/2.2/generated/torch.Tensor.is_leaf.html) (PyTorch 2.2).
- The finite-difference error analysis (Taylor expansion, `ε^(1/3)` heuristic) is standard numerical analysis rather than something these pages document; the numbers in the article are measured by the lab.
- [PyTorch 2.2.2 release](https://github.com/pytorch/pytorch/releases/tag/v2.2.2) — bug-fix release of 27 March 2024.
