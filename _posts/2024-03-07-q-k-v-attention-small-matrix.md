---
layout: post
lang: en
title: "Queries, keys and values: attention on a small matrix"
date: 2024-03-07 09:00:00 +0100
permalink: /blog/q-k-v-attention-small-matrix/
author: Rafał Prońko
categories: [llm]
description: "Single-head scaled dot-product attention computed by hand and in NumPy on three positions with two features: Q, K, V, scores, stable softmax and tests."
image: /assets/images/q-k-v-attention.svg
---

# Queries, keys and values: attention on a small matrix

![Diagram of single-head attention on a 3 by 2 input X. X is multiplied by three different 2 by 2 matrices W_Q, W_K and W_V, giving Q, K and V, each 3 by 2. Q times K transposed divided by the square root of 2 gives a 3 by 3 score matrix; a row-wise softmax turns it into 3 by 3 weights whose rows sum to 1; weights times V gives a 3 by 2 output. The first row is highlighted: scores 0, 0.7071, 0.7071, weights 0.1978, 0.4011, 0.4011 and output 0.5989, 2.2033.]({{ '/assets/images/q-k-v-attention.svg' | relative_url }})

<time datetime="2024-03-07">7 March 2024</time>

*Editorial note: the date above is an editorial label in a retrospective series, not a claim that this text or code existed on that day. The example was executed in October 2026 with Python 3.11.12 and NumPy 1.26.4. NumPy 1.26.4 was uploaded to PyPI on 5–6 February 2024 and Python 3.11.0 was released on 24 October 2022, so nothing in the code requires a release later than the date label.*

The attention formula fits on one line:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

It is easy to recite and easy to misread. The usual misreadings: that attention weights tell you which token the model will produce, that `Q`, `K` and `V` are three copies of the input, or that the `√d_k` is there to stop `exp` from overflowing.

I want to take this one line apart on a matrix small enough to check by hand: three positions, two features, one head. No framework, no training, no tokenizer. Just NumPy, the actual numbers, and a few tests that pin down the properties I rely on later in the series.

## Shapes first

Before the numbers, the axes. The mistakes worth guarding against are rarely in the formula itself; they are a transposed matrix or a softmax over the wrong axis.

In this example:

| Symbol | Shape | Axis meaning |
|---|---|---|
| `X` | `(3, 2)` | rows = positions, columns = features (`d_model = 2`) |
| `W_Q`, `W_K` | `(2, 2)` | `d_model → d_k`, here `d_k = 2` |
| `W_V` | `(2, 2)` | `d_model → d_v`, here `d_v = 2` |
| `Q`, `K`, `V` | `(3, 2)` | one row per position |
| `scores = Q Kᵀ / √d_k` | `(3, 3)` | row `i` = query position, column `j` = key position |
| `weights = softmax(scores)` | `(3, 3)` | softmax **along each row**, so every row sums to 1 |
| `output = weights V` | `(3, 2)` | one output row per query, `d_v` columns |

The indices matter. `scores[i, j]` answers: *how well does the query of position `i` match the key of position `j`?* Output row `i` is a weighted average of all value rows `V[j]`, with weights from row `i`.

The score and weight matrices are square here because I use the same three positions for queries and keys. The three projection matrices are also square because I pick `d_k = d_v = d_model = 2`. `Q`, `K`, `V` and the output are `(3, 2)`. Attention also supports rectangular scores: with 4 queries and 3 keys the score matrix is `(4, 3)`; with `d_v = 5` the output is `(4, 5)`. One of the tests checks exactly that, so the square case doesn't hide a transposition.

## The example

The file below is complete. Save it as `attention_demo.py` in an empty directory and run it in an isolated environment, for example:

```bash
uv run --no-project --python 3.11 --with numpy==1.26.4 python attention_demo.py
```

`--no-project` stops uv from picking up a `pyproject.toml` from the current or a parent directory, so only Python 3.11 and the pinned NumPy are used.

or with the standard library `venv`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install numpy==1.26.4
.venv/bin/python attention_demo.py
```

```python
"""Single-head scaled dot-product attention on a 3x2 synthetic matrix.

Run: python attention_demo.py  (needs NumPy; checked with NumPy 1.26.4)
"""
import numpy as np

# Three positions (rows) with two features each (columns). Synthetic numbers.
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])

# Fixed demo projections. In a trained model these are learned parameters.
W_Q = np.array([[1.0, 0.0],
                [1.0, 1.0]])
W_K = np.array([[0.0, 1.0],
                [1.0, 0.0]])
W_V = np.array([[1.0, 1.0],
                [0.0, 2.0]])


def check_matrix(name: str, a: np.ndarray) -> None:
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)  # row max becomes 0
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    for name, a in (("Q", q), ("K", k), ("V", v)):
        check_matrix(name, a)
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"Q and K need the same d_k: {q.shape} vs {k.shape}")
    if k.shape[0] != v.shape[0]:
        raise ValueError(f"K and V need the same rows: {k.shape} vs {v.shape}")
    scores = q @ k.T / np.sqrt(k.shape[1])  # (n_q, n_k): row i = query i
    weights = softmax_rows(scores)  # each row sums to 1
    return weights @ v, weights, scores  # output: (n_q, d_v)


def main() -> None:
    check_matrix("X", X)
    q, k, v = X @ W_Q, X @ W_K, X @ W_V
    out, weights, scores = attention(q, k, v)
    np.set_printoptions(precision=4, suppress=True)
    for name, a in (("Q", q), ("K", k), ("V", v), ("scores", scores),
                    ("weights", weights), ("output", out)):
        print(f"{name} {a.shape}:\n{a}\n")
    print("row sums:", weights.sum(axis=1))


if __name__ == "__main__":
    main()
```

The input `X` and all three projection matrices are synthetic. I chose them so that the arithmetic stays readable and the three projections are clearly different: for an input row `[x₀, x₁]`, `W_Q` gives `[x₀ + x₁, x₁]`, `W_K` swaps the features into `[x₁, x₀]`, and `W_V` gives `[x₀, x₀ + 2x₁]`.

That last point deserves a sentence of its own. **In a trained Transformer `W_Q`, `W_K` and `W_V` are learned parameters.** Here they are fixed by hand. The demo shows the mechanics of one forward pass, not what a model would learn.

This is the actual output:

```text
Q (3, 2):
[[1. 0.]
 [1. 1.]
 [2. 1.]]

K (3, 2):
[[0. 1.]
 [1. 0.]
 [1. 1.]]

V (3, 2):
[[1. 1.]
 [0. 2.]
 [1. 3.]]

scores (3, 3):
[[0.     0.7071 0.7071]
 [0.7071 0.7071 1.4142]
 [0.7071 1.4142 2.1213]]

weights (3, 3):
[[0.1978 0.4011 0.4011]
 [0.2483 0.2483 0.5035]
 [0.14   0.284  0.576 ]]

output (3, 2):
[[0.5989 2.2033]
 [0.7517 2.2552]
 [0.716  2.4359]]

row sums: [1. 1. 1.]
```

## Q and K decide where to look; V decides what comes back

The output shows the split of roles well.

`Q` and `K` only ever meet in `Q Kᵀ`. They produce the score matrix, and through softmax, the weights. They decide **routing**: how much each position pulls from every other position.

`V` doesn't take part in that decision at all. It is the **content** that gets mixed. If I multiply `V` by 10, the weights stay bit-for-bit identical and the output is 10 times larger. The tests check that.

This also explains why `Q`, `K` and `V` come from three separate projections of the same `X`. What makes a position a good *match* for a query doesn't have to be the same as what it should *contribute* once matched. With three projections the model can learn those as separate things.

A small surprise in this example: position 0 pays the *least* attention to itself. Its query is `[1, 0]` and its own key is `[0, 1]` — a dot product of zero. Self-attention means that keys come from the same sequence, not that a position prefers itself. Whether it does depends entirely on the projections.

## One row by hand

Take row `i = 0`.

The query is `Q[0] = [1, 0]`. The keys are `K[0] = [0, 1]`, `K[1] = [1, 0]`, `K[2] = [1, 1]`.

Dot products: `0`, `1`, `1`. Divided by `√2 ≈ 1.4142`: `0`, `0.7071`, `0.7071`. That is the first row of `scores`.

Softmax over that row:

- `exp(0) = 1`
- `exp(0.7071) ≈ 2.0281`, twice
- sum `≈ 5.0563`
- weights `≈ 1 / 5.0563 = 0.1978` and `2.0281 / 5.0563 = 0.4011`, `0.4011`

The output row is the weighted average of the value rows `[1, 1]`, `[0, 2]` and `[1, 3]`:

- first column: `0.1978·1 + 0.4011·0 + 0.4011·1 ≈ 0.5989`
- second column: `0.1978·1 + 0.4011·2 + 0.4011·3 ≈ 2.2033`

Exactly the first row of `output`. One of the tests repeats this calculation with `math.exp` in plain Python and compares it with the NumPy result.

Row `i = 2` shows the opposite pattern: the query `[2, 1]` matches the key `[1, 1]` of position 2 best (score `2.1213`), so it takes `0.576` of its output from `V[2]`. Note that the weights are still far from one-hot. With only two dimensions and small numbers the softmax stays soft.

## Why divide by √d_k

The scaling factor is easy to confuse with numerical stability. It is something else.

The paper motivates it with an assumption: if the components of `q` and `k` are independent random variables with mean 0 and variance 1, their dot product has mean 0 and variance `d_k`. As `d_k` grows, the scores spread out, the softmax saturates towards one-hot, and its gradients become very small. Dividing by `√d_k` brings the variance of the scores back to 1 under that assumption.

Three caveats follow from that wording:

- It is an argument about the *distribution* of scores, motivated by training: saturated softmax, small gradients. The division itself is part of every forward pass, in training and inference alike. It doesn't guarantee anything about a particular input.
- Real activations need not be independent with unit variance. The scaling is a sensible default, not an exact normalisation.
- With `d_k = 2`, as here, the effect is mild: scores are divided by about 1.41. In the base Transformer from the paper each head has `d_k = 64`, so the division is by 8.

## Stable softmax is a separate trick

`softmax_rows` subtracts the maximum of each row before calling `exp`. Mathematically this changes nothing: softmax is invariant to adding the same constant to every element of a row, because the constant cancels out between numerator and denominator.

Numerically it changes a lot. `exp(1000)` overflows a float64. After the shift the largest exponent is `exp(0) = 1`, and everything else is at most 1. A test feeds the row `[1000, 0, -1000]` with NumPy warnings turned into errors; the result is finite and equal to `[1, 0, 0]`.

The shift must be **per row**, with `keepdims=True`. Subtracting the global maximum of the whole matrix would still be correct mathematically, but a row whose values are all far below that maximum could underflow to zeros and then divide `0 / 0`.

`√d_k` scaling and max-subtraction solve different problems. The first is a modelling choice that shapes the score distribution. The second is an implementation detail that makes `exp` safe. A production kernel needs both.

## Self-attention and cross-attention

In this example `Q`, `K` and `V` all come from the same `X`. That is self-attention.

In encoder–decoder attention, as described in the original paper, queries come from the decoder and keys and values come from the encoder output. Nothing in the function `attention` changes. Only the shapes do: `n_q` decoder positions against `n_k` encoder positions give an `(n_q, n_k)` weight matrix. The validation in the code checks the two constraints that remain: `Q` and `K` must share `d_k`, and `K` and `V` must have the same number of rows, because each key has exactly one value.

## What attention weights are not

The weights matrix is tempting to read as an explanation. I'd be careful.

**They are not token probabilities.** Each row is a distribution over *positions in the input*, not over a vocabulary. The probability of the next token comes from a completely different softmax at the very end of the model, over logits for every token in the vocabulary.

**They are not causal explanations.** A weight says how much of `V[j]` was mixed into one output row of one head in one layer. It doesn't say that position `j` "caused" the model's answer. After this step come further projections, residual connections, more layers and more heads. A small weight on a value with a large norm can still matter more than a large weight on a small one. Treat attention maps as one diagnostic signal, not as a reason.

## What I deliberately left out

**The causal mask.** Every query here can see every key, including positions that come after it. For a decoder that generates text left to right this is wrong: position `i` must not look at `j > i`. The paper implements this by setting illegal scores to `−∞` before the softmax. I left it out to keep one idea per article. The causal mask, together with a small attention module in PyTorch and shape tests, is the subject of the next part.

**Multiple heads.** A real layer runs several heads in parallel, each with its own `W_Qᵢ`, `W_Kᵢ`, `W_Vᵢ` into a smaller `d_k`, concatenates the results and applies one more projection `W_O`. The base model in the paper uses `h = 8` heads with `d_k = d_v = 64`. One head here is a single slice of that, not a scaled-down copy of the whole layer.

**Position information.** The attention function has no idea of order. If I permute the rows of `X`, the output rows are permuted the same way and nothing else changes — one of the tests checks this. Order has to be injected from outside, which the paper does by adding positional encodings to the input embeddings.

**The rest of the Transformer.** This demo omits token embeddings and the residual connections, layer normalisation and feed-forward blocks used in the Transformer. It also omits learning the projection weights, the dropout used as a training regulariser in the paper, and batching for efficiency. These are outside the scope of this forward-pass example.

## What the tests check

The tests sit next to the file in my working copy and use pytest. They are not reproduced in this post; the list below describes each check precisely enough to rewrite it. They check:

1. Shapes `(3, 3)` for scores and weights and `(3, 2)` for the output, and that each weight row sums to 1.
2. The first row of scores, weights and output against a hand calculation in plain Python.
3. Cross-attention shapes: 4 queries, 3 keys and `d_v = 5` give weights `(4, 3)` and output `(4, 5)`.
4. Scaling `V` by 10 leaves the weights identical and scales the output by 10.
5. Adding a different constant to each score row doesn't change the softmax.
6. Extreme logits `[1000, 0, -1000]` produce finite weights without overflow warnings.
7. Equal scores (a zero query) give uniform weights and the plain mean of `V`.
8. Permuting the input positions only permutes the output rows.
9. `NaN`, `inf`, a 1-D query, a `d_k` mismatch and a `K`/`V` row mismatch raise `ValueError`.

The validation is deliberately minimal: two dimensions, finite values and the two shape constraints. For a three-row demo that is proportional. It isn't a substitute for the checks a real library does, such as dtype handling, batching or masking.

These tests prove the properties of this implementation on small inputs. They don't say anything about performance, about memory on long sequences — the score matrix grows with `n_q · n_k` — or about how a trained model distributes its attention.

## The takeaway

Attention on a small matrix comes down to three steps: compare every query with every key, turn each row of comparisons into weights, and average the values with those weights.

`Q` and `K` route. `V` carries content. `√d_k` is applied in every forward pass, motivated by a training-time assumption about score variance; max-subtraction is a numerical safety measure. And the weight matrix describes mixing inside one head, not the model's reasons and not its next token.

With that in place, the causal mask is a small change: some scores become `−∞` before the softmax. That is where the next part starts.

### Sources

- [Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762v7 (2 August 2023), sections 3.2.1–3.2.3 and 3.5](https://arxiv.org/html/1706.03762v7) — scaled dot-product attention, the `√d_k` variance argument, multi-head projections and shapes, encoder–decoder versus self-attention, masking with `−∞`, positional encodings. First version submitted 12 June 2017.
- [NumPy 1.26.4 on PyPI](https://pypi.org/project/numpy/1.26.4/) — release files uploaded 5–6 February 2024, with CPython 3.11 wheels.
- [PEP 664 — Python 3.11 Release Schedule](https://peps.python.org/pep-0664/) — Python 3.11.0 final released 24 October 2022.
