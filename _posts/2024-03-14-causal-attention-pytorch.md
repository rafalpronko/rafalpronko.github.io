---
layout: post
lang: en
title: "Causal attention in PyTorch: a small implementation with shape and leakage tests"
date: 2024-03-14 09:00:00 +0100
permalink: /blog/causal-attention-pytorch/
author: Rafał Prońko
categories: [llm]
description: "A small square causal attention in PyTorch 2.2: shapes, scaling, masking before softmax, and tests that catch future-token leakage."
image: /assets/images/causal-attention-pytorch.svg
---

# Causal attention in PyTorch: a small implementation with shape and leakage tests

![A four by four attention grid. Rows are query positions, columns are key positions. Cells on and below the diagonal are allowed; cells above it are set to minus infinity before the softmax.]({{ '/assets/images/causal-attention-pytorch.svg' | relative_url }})

<time datetime="2024-03-14">14 March 2024</time>

> **Editorial note.** The date above is a proposed place for this post in the blog's series, not proof that it was published on that day. The code was written and run in October 2026, not in 2024. It pins package versions that were available on 14 March 2024: PyTorch 2.2.1 (on PyPI since 22 February 2024), NumPy 1.26.4 and pytest 8.1.1, and dependency resolution ignores anything uploaded from 00:00 UTC on that day onwards. Two things are newer than the date label. The interpreter was CPython 3.11.12, a later patch release of the Python 3.11 line. The environment was built with uv 0.7.8, a 2025 release, and its `uv.lock` file. This tooling only makes the old package versions easy to reinstall.

A decoder-only language model is trained to predict the next token. During training it sees the whole sequence at once, because that is what makes training on a GPU efficient. So the model needs a rule that stops position `i` from reading the tokens after it. One of those tokens is exactly what position `i` is supposed to predict.

That rule is the causal mask. It takes a few lines of code. It is also easy to get subtly wrong: the code still runs, the shapes still line up, and the loss looks better than it should.

In this post I build a small causal attention function in PyTorch. I write the tests before the implementation and compare the result with `torch.nn.functional.scaled_dot_product_attention`. The tests I care about most are not the shape checks. They are the ones that check whether a position can see the future.

## What this example covers, and what it doesn't

In a real transformer, `Q`, `K` and `V` are learned linear projections of the token representations: `Q = X W_Q`, `K = X W_K`, `V = X W_V`. They are then split into heads. I leave all of that out. My function takes `Q`, `K` and `V` that have **already been projected** and split into heads. It does only the attention step.

There are no embeddings, no output projection, no residual connections and no training loop here. This is not a model. It is one operation that I want to understand well enough to test.

The tensors use the layout `[B, H, T, D]`:

- `B` is the batch size, the number of sequences.
- `H` is the number of heads.
- `T` is the sequence length.
- `D` is the size of each query and key vector in one head.

`V` has the shape `[B, H, T, Dv]`. `Dv` may differ from `D`, and the function supports that. The output has the shape `[B, H, T, Dv]`. The attention weights have the shape `[B, H, T, T]`.

## Rows are queries, columns are keys

The core of attention is the score matrix `Q Kᵀ`. For one sequence and one head it is `T × T`. Entry `[i, j]` says how well query `i` matches key `j`. Row `i` holds everything position `i` could look at.

The softmax is computed **along each row**, over the keys. That is `dim=-1`. After the softmax, every row is a probability distribution over the key positions: non-negative weights that sum to 1. The output for position `i` is the weighted average of the value vectors with those weights.

The scores are divided by `√D`. The original transformer paper explains why. If the components of `q` and `k` are independent with mean 0 and variance 1, their dot product has variance `D`. With large `D`, the scores grow large and push the softmax into regions where its gradients are tiny. Dividing by `√D` counters that. PyTorch's `scaled_dot_product_attention` uses the same default: `1 / sqrt(E)`, where `E` is the last dimension of the query.

The matrix multiplication `q @ k.transpose(-2, -1)` works on the last two dimensions and treats `B` and `H` as batch dimensions. `[B, H, T, D] @ [B, H, D, T]` gives `[B, H, T, T]`, one score matrix per sequence and per head.

## The causal mask: what is allowed

For a causal model the rule is simple: query `i` may read key `j` only when `j ≤ i`.

- `j < i`: an earlier token. Allowed.
- `j = i`: the token itself. Allowed. The diagonal stays open.
- `j > i`: a future token. Forbidden.

Allowing the diagonal is not a leak. The training target for position `i` is not token `i`. It is token `i + 1`. The input sequence and the target sequence are shifted by one position. The transformer paper describes the same combination: the decoder masks attention to later positions, and the outputs are offset by one position, so the prediction for position `i` depends only on the known outputs before it.

For the toy sequence `<bos> the cat sat down`, the inputs are the first four tokens and the targets are the last four. The demo at the end of this post prints the alignment:

```text
position 0: reads ['<bos>'] -> predicts 'the'
position 1: reads ['<bos>', 'the'] -> predicts 'cat'
position 2: reads ['<bos>', 'the', 'cat'] -> predicts 'sat'
position 3: reads ['<bos>', 'the', 'cat', 'sat'] -> predicts 'down'
```

If I masked the diagonal as well, the first row would have no allowed key at all. The softmax of a row containing only `-inf` is `NaN`.

## Why the mask goes before the softmax

The forbidden scores have to be set to `-inf` **before** the softmax. `exp(-inf)` is exactly 0, so a future key gets weight 0. The remaining weights in the row are normalised among the allowed keys only.

A tempting mistake is to apply the softmax over the whole row and then zero out the future:

```python
weights = torch.softmax(scores, dim=-1) * allowed  # wrong
```

The future weights become 0, so it looks as if the future is hidden. It isn't. The softmax denominator has already summed `exp` over **all** keys, including the future ones. The rows no longer sum to 1, and how much they are scaled down depends on the future keys. Change a future token and the output of an earlier position changes. Information has leaked through the normalisation.

This is why I want a test that changes future inputs and checks whether past outputs move. The value vectors in that test have to be non-trivial. If the visible values were all zero, a wrong normalisation would still produce zeros, and the test would miss the error.

## Setting up the environment

I use PyTorch 2.2.1 on the CPU. The example works on tiny synthetic tensors and needs no GPU.

All five files can be downloaded:

- [`causal_attention.py`]({{ '/assets/code/causal-attention-pytorch/causal_attention.py' | relative_url }}): the implementation
- [`test_causal_attention.py`]({{ '/assets/code/causal-attention-pytorch/test_causal_attention.py' | relative_url }}): the tests
- [`demo.py`]({{ '/assets/code/causal-attention-pytorch/demo.py' | relative_url }}): the demo shown below
- [`pyproject.toml`]({{ '/assets/code/causal-attention-pytorch/pyproject.toml' | relative_url }}): the dependencies
- [`uv.lock`]({{ '/assets/code/causal-attention-pytorch/uv.lock' | relative_url }}): the exact resolved versions

The `pyproject.toml` pins the direct dependencies. `exclude-newer` makes uv ignore anything uploaded to PyPI on or after 14 March 2024, 00:00 UTC, transitive dependencies included:

```toml
[project]
name = "causal-attention-pytorch"
version = "0.1.0"
description = "Small square causal attention in PyTorch with shape and leakage tests."
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch==2.2.1",
    "numpy==1.26.4",
    "pytest==8.1.1",
]

[tool.uv]
# Resolve only distributions that were already published on PyPI
# before 14 March 2024, including transitive dependencies.
exclude-newer = "2024-03-14T00:00:00Z"
package = false
```

With all five files in one empty directory, these are the commands:

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python demo.py
```

`uv sync --locked` stops with an error if `uv.lock` does not match `pyproject.toml`, so it won't quietly resolve different versions.

I ran this on macOS on Apple silicon (arm64), with CPython 3.11.12 and the official `torch-2.2.1-cp311-none-macosx_11_0_arm64.whl` wheel. For Linux on x86_64, the same lock file also lists the CUDA packages that PyTorch 2.2.1 depends on there. I did not test Linux or Windows for this post.

## Tests first

I wrote the tests before the implementation. The first version of `causal_attention.py` only raised `NotImplementedError`. The run failed as expected:

```text
24 failed, 1 passed in 59.52s
```

The one test that passed doesn't call my implementation. It checks the leak detector itself against the deliberately wrong "softmax first, mask afterwards" variant from the previous section. I want to know the test can catch that mistake before I trust it to pass on correct code.

The first working implementation made all 25 tests pass. Then I refactored: I moved the validation into a separate function, added error messages and wrote comments about the axes. The tests passed again without any changes to them.

A review of that version found a hole in the leak detector itself, which I describe below. I added four regression tests for it and watched them fail against the old detector (`4 failed, 26 passed`) before I fixed it. I also added a `bfloat16` case to the dtype test. That one passed straight away, because the function already rejected `bfloat16`; it only closes a gap in coverage. The test file now has 30 tests, and all of them pass.

The tests cover the following. Each one guards against a specific mistake.

**Shapes.** With `B=2, H=3, T=5, D=4, Dv=6`, the output has the shape `(2, 3, 5, 6)` and the weights `(2, 3, 5, 5)`. Because `Dv ≠ D`, this test fails if I accidentally multiply by the wrong matrix or mix up `D` and `Dv`.

**Lower triangle and row sums.** Every weight above the diagonal is exactly 0. Every weight on the diagonal is positive. Every row sums to 1. The wrong variant fails the last check, because its rows sum to less than 1.

**First position and length one.** Position 0 can see only itself, so its weight is exactly 1 and its output equals `v[..., 0, :]`. For `T = 1` the whole output equals `V`. This is the edge case where an off-by-one error in the mask shows up as `NaN`.

**Changing the future.** This is the core test. The helper and the two tests below are excerpts from `test_causal_attention.py`, where they are not next to each other:

```python
def require_finite(x, what):
    # max() would silently skip a NaN, so a non-finite value must stop the check.
    if not torch.isfinite(x).all():
        raise AssertionError(f"leak check got a non-finite {what}")


def max_future_leak(attention, q, k, v):
    """Largest change in outputs 0..i after replacing positions i+1.. in Q, K and V."""
    out, _ = attention(q, k, v)
    require_finite(out, "baseline output")
    worst = 0.0
    g = torch.Generator().manual_seed(123)
    for i in range(q.shape[-2] - 1):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for x in (q2, k2, v2):
            noise = torch.randn(x[..., i + 1:, :].shape, generator=g, dtype=x.dtype)
            x[..., i + 1:, :] = 10.0 * noise
        out2, _ = attention(q2, k2, v2)
        require_finite(out2, f"output after changing positions {i + 1}..")
        diff = (out2[..., : i + 1, :] - out[..., : i + 1, :]).abs()
        require_finite(diff, f"difference at cut-off {i}")
        worst = max(worst, diff.max().item())
    return worst


def test_future_positions_do_not_change_past_outputs():
    q, k, v = make_qkv(t=6)
    assert v.abs().min() > 0  # visible values are not trivially zero
    assert max_future_leak(causal_attention, q, k, v) == 0.0


def test_leak_check_detects_mask_after_softmax():
    q, k, v = make_qkv(t=6)
    assert max_future_leak(mask_after_softmax, q, k, v) > 1e-3
```

For each cut-off `i`, the test replaces positions `i+1` onwards in `Q`, `K` and `V` with new random values. It then checks that outputs `0..i` did not change. It does not change `V` alone. With the wrong normalisation, the leak travels through the future **keys**, not the future values.

I compare with exact equality, not with a tolerance, and on this machine the test passes that way. In correct code a future key's weight is exactly 0, a finite value times 0 is exactly 0, and nothing from the future enters the result. If the test needed a tolerance here, I would want to know why. The second test makes sure the same detector reports a clear leak for the wrong variant.

The first version of this helper ended with `worst = max(worst, diff)` and had no finiteness checks. A review pointed out what that misses. Every ordered comparison with `NaN` is false, so `max(0.0, nan)` returns `0.0`. If changing the future turned an earlier output into `NaN`, the helper reported a leak of exactly zero, which is the "all clear" answer. I confirmed it with deliberately broken attention functions in the test file. The old helper returned `0.0` when an earlier output became `NaN`, when the baseline was already `NaN`, and when a future row became `inf`. When the difference itself overflowed, it returned `inf` as if that were a measurement. Now it stops with an error in all four cases, and a regression test checks each of them. This makes the detector harder to fool. It does not mean my attention function handles non-finite inputs; see the limits at the end.

**Gradients.** I backpropagate only from outputs `0..2` of a sequence of length 5. The gradients for `Q`, `K` and `V` at positions 3 and 4 have to be exactly 0, and all gradients have to be finite. This checks the same property as the previous test, but along the backward path. A `torch.autograd.gradcheck` in `float64` also compares the analytical gradients with numerical ones.

**Comparison with PyTorch.** The output has to match `F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)`:

```python
@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float64, dict(rtol=1e-12, atol=1e-12)), (torch.float32, dict(rtol=1e-5, atol=1e-6))],
)
def test_matches_sdpa_is_causal(dtype, tol):
    q, k, v = make_qkv(b=2, h=4, t=7, d=8, dv=5, dtype=dtype)
    out, _ = causal_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    torch.testing.assert_close(out, expected, **tol)
```

Here I do use a tolerance, on purpose. The PyTorch documentation says SDPA may pick different kernels and that the result can differ depending on which one runs. My code and the library don't have to perform floating-point operations in the same order, so bitwise equality would be the wrong contract. The tolerances are still many orders of magnitude smaller than the effect of a mask error. On the inputs of this test, I measured the largest difference from SDPA at about 1.8 (`float64`) and 2.2 (`float32`) for the "mask after softmax" variant, and about 2.6 for no mask at all. In the demo below the actual difference between my function and SDPA was `1.192e-07` for `float32` (about one unit in the last place for values near 1) and `1.110e-16` for `float64`.

**Boolean mask polarity.** A second comparison passes an explicit mask instead of `is_causal=True`: `torch.ones(T, T, dtype=torch.bool).tril()`. In SDPA, `True` in a boolean mask means "this position **takes part** in attention". This is the opposite of my internal `future` mask, where `True` means "hide this". Mixing up the two conventions inverts the mask, so I want a test that pins the polarity down.

**Input validation.** The function rejects inputs it was not designed for. It raises an error for a missing head axis, different `T` for queries and keys, different `D` for `Q` and `K`, `V` of a different length, an empty sequence (`T = 0`), an empty `D` or `Dv`, mismatched `B` or `H` (no broadcasting), mixed dtypes, integer tensors, `float16`, `bfloat16`, a non-tensor argument, and tensors on different devices. I test the last case with PyTorch's `meta` device, so it runs on a machine without a GPU.

## The implementation

This is the whole of `causal_attention.py` after the refactor:

```python
"""Square causal self-attention for already-projected Q, K and V.

Teaching implementation: no projections, padding mask, dropout or KV cache.
"""
import math

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, x in (("q", q), ("k", k), ("v", v)):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"dtype must be float32 or float64, got {q.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must share a device, got {q.device}, {k.device}, {v.device}")
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise ValueError("q, k, v must be 4-D: q, k [B, H, T, D] and v [B, H, T, Dv]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have equal shapes, got {tuple(q.shape)} and {tuple(k.shape)}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must match q in B, H, T, got {tuple(v.shape)} and {tuple(q.shape)}")
    if q.numel() == 0 or v.numel() == 0:
        raise ValueError("B, H, T, D and Dv must all be at least 1")


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output [B, H, T, Dv], weights [B, H, T, T]).

    weights[b, h, i, j] is how much query position i reads from key position j.
    Keys with j > i are masked out before the softmax; j == i is allowed.
    """
    _validate(q, k, v)
    t, d = q.shape[-2], q.shape[-1]

    # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]; rows are queries, columns keys.
    scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(d))

    # True above the diagonal = future key. The [T, T] mask broadcasts over B and H.
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # normalise over keys, per query row
    return weights @ v, weights
```

Most of the code is validation. The attention itself takes five lines.

`triu(diagonal=1)` marks the cells strictly above the diagonal, where `j > i`. With `diagonal=0` it would also mark the diagonal, and the first row would turn into `NaN`. The `[T, T]` mask is broadcast over `B` and `H`. Every sequence and every head gets the same causal structure.

The function also returns the weights. I can look at them, and the tests can check the triangle and the row sums directly. This has a cost. The full `[B, H, T, T]` matrix is kept in memory, so memory grows with the square of the sequence length. Fused kernels such as FlashAttention exist precisely to avoid storing that matrix. My implementation is for reading and testing, not for long sequences.

On purpose, the function does **not** broadcast its inputs. `Q`, `K` and `V` must have the same `B` and `H`. Variants that share keys and values across several query heads, such as multi-query and grouped-query attention, would need a deliberate change and new tests.

## The demo and its output

`demo.py` builds one sequence with one head, `T = 4`, `D = 2` and `Dv = 3`, and prints the weights:

```python
import torch
import torch.nn.functional as F

from causal_attention import causal_attention

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# One sequence, one head, 4 positions; D = 2 for Q/K and Dv = 3 for V.
q = torch.randn(1, 1, 4, 2)
k = torch.randn(1, 1, 4, 2)
v = torch.randn(1, 1, 4, 3)

out, weights = causal_attention(q, k, v)
print("weights[0, 0] (rows = query i, columns = key j):")
print(weights[0, 0])
print("row sums:", weights.sum(dim=-1)[0, 0])
print("output shape:", tuple(out.shape))

for dtype in (torch.float32, torch.float64):
    q2, k2, v2 = (x.to(dtype) for x in (q, k, v))
    mine, _ = causal_attention(q2, k2, v2)
    ref = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=True)
    print(f"max |mine - sdpa| {dtype}: {(mine - ref).abs().max().item():.3e}")

tokens = ["<bos>", "the", "cat", "sat", "down"]
inputs, targets = tokens[:-1], tokens[1:]
for i, (x, y) in enumerate(zip(inputs, targets)):
    print(f"position {i}: reads {inputs[: i + 1]} -> predicts {y!r}")

a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=True)
b = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=True)
print("dropout_p=0.5, two calls equal:", torch.equal(a, b))
```

This is the output I got:

```text
weights[0, 0] (rows = query i, columns = key j):
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4884, 0.5116, 0.0000, 0.0000],
        [0.5678, 0.2897, 0.1426, 0.0000],
        [0.1654, 0.2424, 0.3878, 0.2044]])
row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])
output shape: (1, 1, 4, 3)
max |mine - sdpa| torch.float32: 1.192e-07
max |mine - sdpa| torch.float64: 1.110e-16
position 0: reads ['<bos>'] -> predicts 'the'
position 1: reads ['<bos>', 'the'] -> predicts 'cat'
position 2: reads ['<bos>', 'the', 'cat'] -> predicts 'sat'
position 3: reads ['<bos>', 'the', 'cat', 'sat'] -> predicts 'down'
dropout_p=0.5, two calls equal: False
```

The matrix has the same shape as the illustration at the top. Everything above the diagonal is zero, the first row puts all its weight on itself, and each row sums to 1.

And the tests, run on a fresh copy of the five downloadable files after the detector fix:

```text
..............................                                           [100%]
30 passed in 10.20s
```

The same check on the files from before the fix, with the original 25 tests, took 99.85s. The run time varied between runs on the same machine. Most of it is starting PyTorch and the `gradcheck`, not the attention itself. I would not read it as any kind of performance measurement.

## Dropout is not switched off by `eval()`

The last line of the demo shows something that is easy to miss. In PyTorch 2.2, `F.scaled_dot_product_attention` applies dropout whenever `dropout_p > 0`. The reference code in the documentation calls `torch.dropout(attn_weight, dropout_p, train=True)`. A function knows nothing about `model.eval()`. Two calls with `dropout_p=0.5` gave different results.

If SDPA is used inside your own `nn.Module`, the probability has to depend on the mode explicitly, for example `dropout_p=self.p if self.training else 0.0`. In all my comparisons I pass `dropout_p=0.0`.

## Where this guarantee ends

The tests check one specific case: square self-attention, where queries and keys come from the same `T` positions, on the CPU, in `float32` and `float64`. Outside that case I don't claim anything.

**Padding.** A batch with sequences of different lengths needs a second mask that hides the padding tokens. My function has no such mask. There is also a trap here. A masked weight is exactly 0, but `0 × inf` is `NaN`. I checked this by putting `inf` into `V` only at the last position. The outputs of earlier positions became `NaN`, both in my function and in SDPA. The causal mask doesn't protect against non-finite values in the future positions. Padding should hold finite values, such as zeros.

**KV cache and non-square attention.** During generation with a KV cache, a new query is compared with keys that are already stored. Then `L` (the number of queries) differs from `S` (the number of keys). My function rejects that shape. SDPA accepts it, but in version 2.2 the reference code builds the `is_causal` mask as `ones(L, S).tril()`, which is aligned to the top-left corner. I checked this for `L = 1` and `S = 4`: the single query attended only to key 0, not to all four. For decoding with a cache you need an explicit mask or a different alignment. That is a topic for a separate post.

**Precision and hardware.** I accept only `float32` and `float64`. `float16` and `bfloat16` are rejected, because I haven't tested them. I didn't run the code on a GPU. The tolerances in the SDPA comparison are justified for the CPU and these small tensors. On another backend they may need to be checked again.

**Non-finite inputs.** I don't check `Q`, `K` and `V` for `NaN` or `inf`. A NaN or positive-infinity score in an allowed position spoils the row; an all-negative-infinity row also produces NaNs.

**The rest of the model.** Passing these tests doesn't mean a model built on this function doesn't leak the future. A leak can also come from data preparation, for example targets that aren't shifted, or from a layer that mixes positions without a mask. The tests cover only the attention operation.

## In short

Causal attention comes down to three decisions: which axis the softmax runs over (keys), what gets masked (`j > i`, with the diagonal kept) and when (before the softmax). Each of these can be gotten wrong without any error message.

So the tests don't stop at shapes. They change the future and check that the past didn't notice. They also check the checker, by showing that it catches the wrong variant.

The next post in this series is planned to cover prefill, decode and the KV cache. That is exactly where the square-mask assumption stops being enough.

### Sources

- [Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762](https://arxiv.org/abs/1706.03762): section 3.1 (decoder masking combined with outputs offset by one position), section 3.2.1 (scaled dot-product attention and the `1/√d_k` scaling, with the footnote on the variance of the dot product), section 3.2.3 (masking illegal connections by setting softmax inputs to `-∞`).
- [PyTorch 2.2 documentation: `torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html): reference code, `True` in a boolean mask means "takes part in attention", dropout applied when `dropout_p > 0`, default scale, `L`, `S`, `E` and `Ev` shapes, results that can differ between kernels.
- [PyTorch v2.2.1 source, `torch/nn/functional.py`](https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/torch/nn/functional.py): the same docstring at the release tag.
- [PyTorch 2.2 release blog post](https://pytorch.org/blog/pytorch2-2/): PyTorch 2.2 announced on 30 January 2024.
- [PyTorch v2.2.1 release on GitHub](https://github.com/pytorch/pytorch/releases/tag/v2.2.1) and [torch 2.2.1 on PyPI](https://pypi.org/project/torch/2.2.1/): bug-fix release; the GitHub release was published and the PyPI files were uploaded on 22 February 2024.
