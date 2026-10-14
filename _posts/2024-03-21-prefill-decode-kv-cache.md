---
layout: post
lang: en
title: "Prefill, decode and the KV cache: where LLM latency comes from"
date: 2024-03-21 09:00:00 +0100
permalink: /blog/prefill-decode-kv-cache/
author: Rafał Prońko
categories: [llm]
description: "Why the first token and the following tokens cost different things: prefill, cached decode, KV cache memory and how to measure latency honestly."
image: /assets/images/prefill-decode-kv-cache.svg
---

# Prefill, decode and the KV cache: where LLM latency comes from

![Timeline of one request: queueing and tokenization, then a prefill block where layers run in sequence and prompt positions run in parallel, producing token 1; then decode steps that each consume the previous token, append one key/value entry per layer and emit the next token.]({{ '/assets/images/prefill-decode-kv-cache.svg' | relative_url }})

<time datetime="2024-03-21">21 March 2024</time>

A team reports that "the model is slow". One person means the blank screen before anything appears. Another means the text trickling out word by word. A third looks at the total time of a batch job.

These are three different numbers, and they are driven by two different phases of generation. Before I tune anything, I want to know which phase I am looking at.

I'll go through the mechanics with one attention head in plain Python, check the claims with tests, and only then talk about measurement.

## Two phases, one request

A decoder-only model generates one token at a time, each conditioned on everything before it. The PagedAttention paper (Kwon et al., §2.2) splits serving into two phases.

**Prefill** (the "prompt phase") takes the whole prompt of P tokens. All of them are known, so inside a layer every position can be processed at once with matrix-matrix products. The layers themselves still run one after another: layer 2 needs the output of layer 1.

The output at the **last prompt position** goes through the output head to logits, and that gives the distribution for the first new token. Selecting from that distribution produces token 1. There is no extra decode step before it.

**Decode** then runs once per further token. Step 1 takes token 1 as its input and produces logits for token 2. Step 2 consumes token 2 and produces token 3. If I count "steps" and "tokens" with different off-by-one conventions, my latency numbers will disagree with the server's, so I fix this convention early.

Decode cannot be parallelised across steps. The token emitted at step t is the input of step t+1 (Shazeer, §2.4).

## Why old keys and values can be kept

In causal self-attention, position i computes its query, key and value from its own hidden state and attends only to positions 1..i (Kwon et al., Eq. 2–3). At inference, assume fixed weights, unchanged prefix inputs, positional information and configuration fixed per position, and deterministic evaluation with dropout disabled. Under those assumptions, adding token i+1 leaves the earlier hidden states, and therefore their keys and values in every layer, unchanged mathematically. Different kernels or summation orders need not give bitwise identical results. This prefix-reuse argument does not apply to bidirectional attention, where earlier positions can attend to later ones.

That invariance is the whole justification for the KV cache. At each decode step, every layer keeps the K and V of all previous positions, computes K and V for the one new position, appends them, and attends over the lot.

Two things the cache is **not**:

- It is not memoization by token id. A cached entry is the K/V of a hidden state at a given position in a given layer, and that state depends on all preceding tokens. The paper says it directly: the same token at different positions has a different KV cache (Fig. 3 caption).
- It is not a shortcut around the model. Each decode step still computes the query for the new position, its K and V, the attention over all cached entries, the output projection and the feed-forward block, in every layer. The new token still uses the model weights; memory traffic depends on batching and on which weights remain on chip. The cache removes only the recomputation of the *old* positions' K and V (and everything that produced them).

## What it costs

For one dense layer with model width d, a fixed head layout, and attention over the whole context, rough operation counts look like this:

- **prefill** of P positions: O(P·d² + P²·d). The first term is the projections and feed-forward; the second is every position attending to its prefix.
- **one decode step** at context length T: O(d² + T·d). The d² term is the same per-token work as before; the T·d term is reading and scoring T cached keys and values.

So a cached decode step is not constant-time. It gets more expensive as the context grows, just much more slowly than recomputing the prefix (which would cost O(T·d² + T²·d) again on every step).

Arithmetic is only half of it. Shazeer analyses incremental multi-head attention and finds a memory-access to arithmetic ratio of Θ(n/d + 1/b), where n is sequence length and b batch size (§2.4.1). His simplified analysis assumes n ≤ d and key and value widths k = v = d/h, with h heads; the ratio should not be extrapolated unchanged to arbitrary context lengths. With b ≈ 1 or n ≈ d the ratio is of order 1, and memory bandwidth becomes the main limit on the hardware he considers. The 1/b term is what batching attacks: weight accesses are shared across requests in a batch (Kwon et al., §2.3). The n/d term comes from re-reading K and V and does not shrink with batching.

Kwon et al. state that decode "becomes memory-bound, being responsible for most portion of the latency of a single request" (§2.2). I read that as a statement about their setting and typical output lengths. With a very long prompt and a two-token answer, prefill can dominate instead.

## A one-head experiment

I want to check three claims with code, not with a picture:

1. prefill plus cached decode gives the same attention outputs as recomputing the whole prefix every step,
2. output 0 comes from the last prompt position and decode step t consumes continuation input t,
3. the cache saves K/V projections but every new position still needs its own Q, K and V.

The example below is **one causal attention head**, nothing more. No embeddings, positional encoding, residuals, normalisation, feed-forward, multiple layers, logits or sampling. The "continuation" vectors are fixed synthetic inputs that stand in for the hidden state of the token just emitted. In a real model they would come from sampling and embedding that token.

Projection matrices and vectors come from a seeded `random.Random`, so the numbers are deterministic. Only the standard library is used.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One causal attention head: full recomputation vs cached decode.

Deliberately NOT a decoder or an LLM: no embeddings, positions,
residuals, FFN, layers, logits or sampling. Continuation vectors are
synthetic inputs standing in for "the hidden state of the token that
was just emitted".
"""

import math
import random
from dataclasses import dataclass, field

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class ProjectionCounts:
    q: int = 0
    k: int = 0
    v: int = 0


def _shape(m: Matrix, name: str) -> tuple[int, int]:
    if not m or not m[0] or any(len(row) != len(m[0]) for row in m):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return len(m), len(m[0])


@dataclass(frozen=True)
class Head:
    wq: Matrix  # d_head x d_model
    wk: Matrix
    wv: Matrix

    def __post_init__(self) -> None:
        shapes = {
            _shape(self.wq, "wq"),
            _shape(self.wk, "wk"),
            _shape(self.wv, "wv"),
        }
        if len(shapes) != 1:
            raise ValueError(f"wq, wk, wv must share one shape, got {shapes}")

    @property
    def d_head(self) -> int:
        return len(self.wq)

    @property
    def d_model(self) -> int:
        return len(self.wq[0])


@dataclass
class KVCache:
    keys: list[Vector] = field(default_factory=list)
    values: list[Vector] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)


def _matvec(m: Matrix, x: Vector) -> Vector:
    return [sum(w * xi for w, xi in zip(row, x)) for row in m]


def _check_input(head: Head, x: Vector) -> None:
    if len(x) != head.d_model:
        raise ValueError(
            f"input has dim {len(x)}, expected d_model={head.d_model}"
        )


def _check_cache(head: Head, cache: KVCache) -> None:
    if len(cache.keys) != len(cache.values):
        raise ValueError("cache keys and values differ in length")
    if any(len(r) != head.d_head for r in cache.keys + cache.values):
        raise ValueError(f"cache entries must have d_head={head.d_head}")


def _attend(q: Vector, keys: list[Vector], values: list[Vector]) -> Vector:
    """Softmax(q.k_j / sqrt(d))-weighted sum of v_j.

    The caller passes only the causal prefix (positions 0..i).
    """
    scale = 1.0 / math.sqrt(len(q))
    scores = [scale * sum(a * b for a, b in zip(q, k)) for k in keys]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    width = len(values[0])
    return [
        sum(w * v[i] for w, v in zip(weights, values)) / total
        for i in range(width)
    ]


def _append_kv(
    head: Head, cache: KVCache, x: Vector, counts: ProjectionCounts
) -> None:
    cache.keys.append(_matvec(head.wk, x))
    cache.values.append(_matvec(head.wv, x))
    counts.k += 1
    counts.v += 1


def _query(head: Head, x: Vector, counts: ProjectionCounts) -> Vector:
    counts.q += 1
    return _matvec(head.wq, x)


def prefill(
    head: Head, prompt: list[Vector], counts: ProjectionCounts
) -> tuple[KVCache, list[Vector]]:
    """Project every prompt position once; i attends to 0..i."""
    if not prompt:
        raise ValueError("prompt must contain at least one position")
    for x in prompt:
        _check_input(head, x)
    cache = KVCache()
    for x in prompt:  # a real kernel does one matrix-matrix product
        _append_kv(head, cache, x, counts)
    queries = [_query(head, x, counts) for x in prompt]
    outputs = [
        _attend(q, cache.keys[: i + 1], cache.values[: i + 1])
        for i, q in enumerate(queries)
    ]
    return cache, outputs


def decode_step(
    head: Head, cache: KVCache, x: Vector, counts: ProjectionCounts
) -> Vector:
    """Project only the new position, append its K/V, attend to all."""
    _check_input(head, x)
    _check_cache(head, cache)
    if not cache.keys:
        raise ValueError("decode_step needs a cache produced by prefill")
    _append_kv(head, cache, x, counts)
    return _attend(_query(head, x, counts), cache.keys, cache.values)


def generate_cached(
    head: Head,
    prompt: list[Vector],
    continuation: list[Vector],
    counts: ProjectionCounts,
) -> list[Vector]:
    """Output 0: last prompt position; output t+1: continuation[t]."""
    cache, prompt_outputs = prefill(head, prompt, counts)
    decoded = [decode_step(head, cache, x, counts) for x in continuation]
    return [prompt_outputs[-1]] + decoded


def generate_recomputed(
    head: Head,
    prompt: list[Vector],
    continuation: list[Vector],
    counts: ProjectionCounts,
) -> list[Vector]:
    """Same outputs, but every step re-projects the whole prefix."""
    outputs = []
    for step in range(len(continuation) + 1):
        prefix = prompt + continuation[:step]
        _, prefix_outputs = prefill(head, prefix, counts)
        outputs.append(prefix_outputs[-1])
    return outputs


def kv_cache_bytes(
    *,
    layers: int,
    batch: int,
    tokens: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Raw K/V storage: 2 * L * B * T * H_kv * d_head * s bytes.

    Exact only if every sequence holds `tokens` positions. Excludes
    weights, activations, workspace and allocator overhead.
    """
    args = (layers, batch, tokens, kv_heads, head_dim, bytes_per_element)
    if any(isinstance(a, bool) or not isinstance(a, int) for a in args):
        raise TypeError("all parameters must be integers, not bool")
    if any(a <= 0 for a in args):
        raise ValueError("all parameters must be positive")
    return 2 * math.prod(args)


def synthetic_head(d_model: int, d_head: int) -> Head:
    rng = random.Random(1234)

    def matrix() -> Matrix:
        return [
            [rng.uniform(-1.0, 1.0) for _ in range(d_model)]
            for _ in range(d_head)
        ]

    return Head(wq=matrix(), wk=matrix(), wv=matrix())


def synthetic_vectors(count: int, dim: int, seed: int) -> list[Vector]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(count)]


def main() -> None:
    head = synthetic_head(d_model=4, d_head=3)
    prompt = synthetic_vectors(count=4, dim=4, seed=1)
    continuation = synthetic_vectors(count=3, dim=4, seed=2)

    cached_counts, full_counts = ProjectionCounts(), ProjectionCounts()
    cached = generate_cached(head, prompt, continuation, cached_counts)
    full = generate_recomputed(head, prompt, continuation, full_counts)
    diff = max(
        abs(a - b) for u, w in zip(cached, full) for a, b in zip(u, w)
    )
    c, f = cached_counts, full_counts
    print(f"outputs compared: {len(cached)}, max abs difference: {diff:.3e}")
    print(f"cached     projections q/k/v: {c.q}/{c.k}/{c.v}")
    print(f"recomputed projections q/k/v: {f.q}/{f.k}/{f.v}")

    config = dict(
        layers=32, batch=8, tokens=4096, head_dim=128, bytes_per_element=2
    )
    for name, kv_heads in (("MHA, 32 KV heads", 32), ("MQA,  1 KV head ", 1)):
        size = kv_cache_bytes(kv_heads=kv_heads, **config)
        gib = size / 2**30
        print(f"synthetic {name}: {size:>14,} bytes = {gib:6.3f} GiB")


if __name__ == "__main__":
    main()
```

`ProjectionCounts` is the important part. It counts how many times each projection runs, which is a statement about work done, not about time.

Save the complete block above as `kv_cache.py` in an empty directory. From that directory, run the standalone script with uv:

```bash
uv run --python 3.11 kv_cache.py
```

The block includes its own script metadata and needs no project file or third-party dependencies. I verified it with uv 0.7.8 and Python 3.11.12 in an isolated environment. Python 3.11 implementations should reproduce the output structure, projection counts and integer byte counts; floating-point details may vary. Output from my run:

```text
outputs compared: 4, max abs difference: 0.000e+00
cached     projections q/k/v: 7/7/7
recomputed projections q/k/v: 22/22/22
synthetic MHA, 32 KV heads: 17,179,869,184 bytes = 16.000 GiB
synthetic MQA,  1 KV head :    536,870,912 bytes =  0.500 GiB
```

Four prompt positions and three continuation inputs give four outputs: one from prefill, three from decode. The cached path projects each of the 7 positions exactly once. The recomputed path projects 4 + 5 + 6 + 7 = 22 positions.

The difference is exactly zero here because both paths call the same `_attend` on the same numbers in the same order. That equivalence alone cannot establish that the attention formula is correct. The companion tests use `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`, a tolerance suited to these bounded synthetic float64 vectors (width ≤ 4, projected values of order 1). It is not a general numerical proof. Independent analytic cases check nonuniform softmax weights, division by sqrt(d_head), and which positions are included.

## What the tests check

The private companion file `test_kv_cache.py` is author-verification material and is not reproduced or supplied for download here. I ran its original 13 `unittest` tests against a stub before implementation: 12 errors and 1 failure. Those 13 then passed. Later review led to analytic attention cases and stricter integer validation, bringing the final suite to 18 tests, all passing in the correction run. The new validation test first failed against the original implementation; this refinement happened after the original RED/GREEN cycle.

My internal verification command, run in the private companion directory containing both files, is `uv run --isolated --python 3.11 python -m unittest -v test_kv_cache`. Readers can run the standalone script above without that file. The companion tests cover:

- **equivalence** over the whole continuation, and that cached output 0 equals the last prefill output,
- **reuse**: 7 Q, 7 K and 7 V projections on the cached path, 22 K and 22 V on the recomputed one,
- **causality**: extending the prompt with extra vectors does not change outputs at earlier positions, and a decode step leaves existing cache entries untouched,
- **analytic attention outputs**: basis-vector cases have softmax weights 1 and exp(1/sqrt(d_head)); they check self inclusion and future exclusion in prefill, and inclusion of the new position in decode. A rectangular projection with d_model = 3 and d_head = 2 checks that scaling uses the head width,
- **growth**: the cache gains exactly one key and one value per decode step,
- **shapes**: wrong input width, empty prompt, ragged or mismatched matrices, and a cache with the wrong head dimension raise `ValueError`,
- **memory formula**: the byte count, integer-only inputs (excluding bool), positivity checks and the MHA/MQA ratio.

What they don't prove: anything about wall-clock time, GPU kernels, or a real model. Projection counts on a synthetic workload are evidence of reuse, not of latency.

## How big the cache gets

With equal K and V dimensions and every sequence in the batch at the same length, the cache holds:

**bytes = 2 · L · B · T · H_kv · d_head · s**

2 for K and V, L layers, B sequences, T positions, H_kv key/value heads per layer, d_head per head, and s bytes per element. This is the exact raw-element byte count for those equal lengths and dimensions. For unequal lengths T_i, replace B · T with sum(T_i). If T is instead the maximum length and all sequences are padded to it, B · T counts padded raw storage and bounds the unpadded element count. These counts exclude model weights, activations, workspace, allocator overhead and reserved capacity; they do not bound total allocated memory.

The script uses a synthetic configuration: 32 layers, batch 8, 4096 positions, d_head 128, 2-byte elements. With 32 KV heads (standard multi-head attention) it is 16 GiB. With a single shared KV head it is 0.5 GiB, exactly 32 times less.

That single-KV-head design is multi-query attention (MQA): every query head keeps its own projection, but all heads share one set of keys and values (Shazeer, §3). Under the same simplifying assumptions, it reduces the n/d term in the memory-access ratio by a factor of h (§3.1). In his WMT14 English-German experiments on a TPUv2, the reported amortized decoder cost fell from 46 to 3.8 µs per output token, with dev BLEU 26.7 versus 26.5 (§4.2–4.3, Tables 1–2). The speed comparison used greedy inference, a batch of 1024 sequences and source and target lengths of 128, with fixed padded attention shapes. Those are his numbers for his setup; I would not carry them over to another model.

Why the size matters for latency: Kwon et al. show a 13B model on a 40 GB A100 where about 65% of memory is weights and close to 30% is KV cache (§1, Fig. 1). Whatever is left for the cache caps the batch size, and the batch size caps how well the weight reads are amortized.

## Prefix reuse and paging are different tools

Two optimisations often get blurred together with the KV cache.

**Prefix reuse** keeps K/V for a shared prompt prefix, such as a long system prompt, and starts later requests from it. It requires causal attention and the deterministic inference assumptions above: identical prefix inputs and positions, fixed weights and positional configuration, and dropout disabled. One different token early on can change later hidden states, so the changed suffix must be recomputed. This is a server-side computation cache, not conversation memory, and it has nothing to do with what a chat UI or a browser remembers about a user.

**Paging** (PagedAttention) stores the cache in fixed-size blocks instead of one contiguous, pre-reserved region per request, which cuts fragmentation from pre-allocating each request's maximum length and lets sequences share blocks (§1). It changes where K and V live. Each decode step still attends over all of them.

## Three latency numbers

For a single request I measure three things:

- **TTFT (time to first token)**: from the moment the client sends the request until the first token arrives. It includes queueing, tokenization, host-device transfers, prefill, first-token selection, and client transport or streaming delays.
- **inter-token latency**: the gap between consecutive tokens during decode.
- **total response time**: TTFT plus roughly (output tokens − 1) inter-token gaps, plus any post-processing.

Batching moves these in opposite directions. A larger batch usually raises throughput by sharing weight reads, but each step serves more sequences, a new request may wait for a slot, and a long prefill landing in the batch can stall the decode steps of other requests. Whether a given server does this depends on its scheduler (Kwon et al., §2.3, describe iteration-level scheduling as one answer to the waiting problem).

## How I would diagnose "the model is slow"

I don't start with tuning. I start by splitting the number.

1. Log TTFT, inter-token latency and total time per request, together with prompt length, output length, and the batch or concurrency level at that moment.
2. Look at p50, p95 and p99, not the mean. Bucket by prompt length and output length.
3. Run controlled tests: fix the output length and vary the prompt; fix the prompt and vary the output; repeat each at several load levels.
4. If TTFT grows with prompt length at low load, prefill is the suspect. If it grows with load at a fixed prompt, queueing or scheduling contention is the suspect. Use server-side queue-wait, prefill and decode timings or tracing to distinguish those hypotheses from batching, execution and transport effects.
5. If inter-token latency grows with context length or batch size, look at KV cache reads and memory capacity: consider shorter context, fewer KV heads in the model choice, or a different batching policy. Check task quality before removing context; useful information may be lost.

I won't promise a speed-up from any of these steps. Each one only tells me which part of the request to look at next. A phase I haven't measured is not one I can tune.

### Sources

- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/html/2309.06180v1), arXiv:2309.06180v1, 12 September 2023 — §1 (memory breakdown, Fig. 1), §2.1 (attention equations), §2.2 (prompt and generation phases), Fig. 3 caption (position-dependent KV cache), §2.3 (batching); paging is described from the §1 overview only.
- Noam Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/html/1911.02150v1), arXiv:1911.02150v1, 6 November 2019 — §2.4 and §2.4.1 (incremental attention and its memory/arithmetic ratio), §3 and §3.1 (multi-query attention), §4.2–4.3 (quality and speed tables).

*Verification note: the example and tests were executed later with Python 3.11.12 via uv 0.7.8; the tooling commands reflect that run, not the article's editorial date.*
