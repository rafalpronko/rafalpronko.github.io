---
layout: post
lang: en
title: "Tokens, embeddings, and cosine similarity—without the magic"
date: 2023-12-14 09:00:00 +0100
permalink: /blog/tokens-embeddings-cosine-similarity/
author: Rafał Prońko
categories: [rag, machine-learning, python]
description: "Token IDs, lookup rows, contextual states and sentence vectors are different things. Cosine similarity in plain Python, with its edge cases and limits."
image: /assets/images/tokens-embeddings-cosine-similarity.svg
---

# Tokens, embeddings, and cosine similarity—without the magic

![Diagram: a tokenizer turns "refund policy" into token IDs, a lookup table returns one row per ID, encoder layers and pooling produce one sentence vector, and cosine similarity compares the directions of two vectors.]({{ '/assets/images/tokens-embeddings-cosine-similarity.svg' | relative_url }})

<time datetime="2023-12-14">14 December 2023</time>

A retrieval pipeline is often described in one sentence: embed the documents, embed the question, return the documents with the highest cosine similarity.

The sentence is correct, but it hides several decisions. The word "embedding" is used for at least four different objects. The cosine score is easy to read as a confidence level, and it is not one. And the question and documents need a compatible, versioned encoding contract, with the prescribed processing for each side.

In this post I separate those pieces. I want to know what each number is, where it comes from and what it can't tell me. The example is plain Python 3.11 with the standard library and a handful of hand-chosen vectors. There is no model, no download and no vector database. That's deliberate: first the semantics, then the tools.

A note on dates: the code was written and executed in 2026, on Python 3.11.12. Every standard-library function it uses was already available in Python 3.11.7 (released on 4 December 2023). I use [uv](https://docs.astral.sh/uv/) to reproduce the environment; it is modern tooling, not part of the historical setup.

## Four things people call "the embedding"

Take the text `refund policy`. On its way to a sentence vector it becomes four different kinds of data.

### 1. Token IDs

A tokenizer splits the text into pieces from a fixed vocabulary and replaces each piece with its position in that vocabulary. The output is a list of integers.

Many modern models use subword tokenizers. Byte Pair Encoding, as adapted for neural machine translation by Sennrich, Haddow and Birch, starts with a vocabulary of single characters plus an end-of-word symbol. It repeatedly counts adjacent symbol pairs in the training text and merges the most frequent pair into a new symbol. The number of merges is the hyperparameter that sets the vocabulary size. An unseen word can be composed from covered characters and subwords. This character-based scheme still needs a fallback for characters absent from its training alphabet; it does not guarantee arbitrary-character coverage. The paper also notes that symbols absorbed into larger merges may need those merges reversed.

What matters for retrieval: **token IDs are indices, not measurements.** ID 7 is not "close to" ID 8. The same text gets different IDs and even a different number of tokens under a different tokenizer. In my toy example two vocabularies produce:

```text
tokenizer A: [7, 3]
tokenizer B: [2, 11]
```

Both lists are valid. Neither is comparable with the other, and computing a similarity on raw IDs makes no sense at all.

### 2. Token embeddings: a lookup table

The first layer of the model is a learned matrix with one row per vocabulary entry. Embedding a token ID means selecting row number ID. That lookup itself adds no context; transformer encoders supply position information separately.

```text
lookup row for 'refund' (ID 7): (0.9, 0.2, 0.0, 0.1)
```

The values come from training. The row is the same wherever the token appears, so the word "bank" gets one row, whether the sentence is about a river or about money.

This also shows why a tokenizer and a model are a pair. Feed IDs from tokenizer B into a table trained with tokenizer A and an in-range ID can silently select the wrong token row. An out-of-range ID can fail explicitly. Passing a shape check does not establish tokenizer compatibility.

### 3. Contextual hidden states

The encoder layers then mix information between positions. After them each token has a new vector that depends on its neighbours, so "bank" near "river" and "bank" near "loan" end up different.

You still have one vector per token, and the number of tokens depends on the input. Two texts of different length can't be compared position by position.

### 4. A sentence or document embedding

To get one fixed-size vector per text, the model pools the token states. Reimers and Gurevych tried three strategies in Sentence-BERT: the output of the `[CLS]` token, the mean of all output vectors, and a max over them. Mean was their default.

Pooling alone does not make the vectors good for similarity. The same paper reports that averaged BERT outputs and the BERT `[CLS]` vector, used without fine-tuning, did worse on their semantic textual similarity benchmarks than averaged GloVe word vectors. Their fix was training: the same encoder processes both sentences of a pair (a siamese network) and the weights are tuned with one of three objectives:

- classification over `(u, v, |u − v|)` with cross-entropy, trained on natural language inference pairs,
- regression of `cos(u, v)` towards a gold similarity score with mean squared error,
- a triplet loss that pushes an anchor closer to a positive example than to a negative one.

That is the part that makes cosine meaningful. **The training objective decides what "close" means in this vector space.** A model trained to bring paraphrases together puts paraphrases together. It doesn't promise anything about questions and their answers unless it was trained for that too.

The paper also gives the main reason to encode texts separately. Finding the most similar pair among 10,000 sentences by feeding every pair through BERT takes about 50 million inference computations, which the authors estimate at roughly 65 hours. Encoding each sentence once and comparing vectors took about 5 seconds for the embeddings plus about 0.01 seconds for the cosine computation, on their hardware.

One practical detail: when mean pooling is applied to a padded batch, padding positions have to be excluded through the attention mask. Otherwise the vector of a short text depends on how long the other texts in the batch were.

In my demo, steps 3 and 4 are collapsed into the simplest possible toy encoder: look up the rows and take their mean. There are no contextual layers. It is enough to show the arithmetic, and it is also enough to show one of its blind spots later on.

## Cosine similarity, piece by piece

For two vectors `u` and `v` of the same length:

```text
cos(u, v) = (u · v) / (‖u‖ ‖v‖)
```

**The numerator** is the dot product: multiply coordinate by coordinate and add. It grows with the length of the vectors, so on its own it mixes direction with magnitude.

**The denominator** is the product of the two Euclidean lengths. Dividing by it removes magnitude. What is left depends only on the angle between the vectors.

A few consequences follow directly from that.

- **Dimensions must match.** Coordinate *i* of one vector is multiplied by coordinate *i* of the other. That only means something if the encoders produce compatible coordinates under the query/document contract; their weights need not be identical.
- **The range is −1 to 1.** By the Cauchy–Schwarz inequality the dot product can't exceed the product of the lengths. Same direction gives 1, orthogonal gives 0, opposite gives −1.
- **Scaling doesn't matter.** Multiplying a vector by 1000 leaves the cosine unchanged. Multiplying it by −3 flips the sign.
- **A zero vector has no cosine.** The denominator is 0. Returning 0.0 here would be a lie, because 0.0 is a real answer meaning "orthogonal". A zero vector usually means something went wrong upstream, and I want to see that error.
- **After normalisation, cosine is a plain inner product.** If both vectors are scaled to length 1 when they are stored, `u · v` *is* the cosine. For unit vectors the squared Euclidean distance is `2 − 2·cos(u, v)`, so ranking by distance and ranking by cosine give the same order. So if an index offers inner product or Euclidean distance rather than cosine, normalising both sides gives you cosine ranking anyway.
- **It is not a probability.** Scores can be negative, they don't sum to anything, and their spread depends on the model.

## The code

The whole module is below. It can be saved as `cosine_lab.py`.

```python
"""Cosine similarity with explicit edge cases; Python 3.11 standard library."""
import math
from collections.abc import Sequence
from dataclasses import dataclass

Vector = Sequence[float]


def _check(v: Vector) -> None:
    if len(v) == 0:
        raise ValueError("vector is empty")
    for x in v:
        if not math.isfinite(x):
            raise ValueError(f"vector contains a non-finite value: {x!r}")


def norm(v: Vector) -> float:
    """Validated length; may round at subnormal scale or overflow to inf."""
    _check(v)
    return math.hypot(*v)


def normalize(v: Vector) -> tuple[float, ...]:
    """Scale before taking a norm to preserve finite extreme directions."""
    _check(v)
    scale = max(abs(x) for x in v)
    if scale == 0.0:
        raise ValueError("zero vector has no direction")
    scaled = tuple(x / scale for x in v)
    length = math.hypot(*scaled)
    return tuple(x / length for x in scaled)


def cosine_similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    # Normalise first, then take the inner product of two unit vectors.
    ua, ub = normalize(a), normalize(b)
    c = math.fsum(x * y for x, y in zip(ua, ub, strict=True))
    # Rounding can land a hair outside [-1, 1]; the true value cannot.
    return max(-1.0, min(1.0, c))


def mean_pool(vectors: Sequence[Vector]) -> tuple[float, ...]:
    """Validated column means; scale each column to avoid raw-sum overflow.

    Finite precision still loses tiny terms beside much larger terms,
    especially when cancellation leaves a small residual.
    """
    if len(vectors) == 0:
        raise ValueError("cannot pool an empty sequence of vectors")
    dim = len(vectors[0])
    for v in vectors:
        _check(v)
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must have the same dimension")
    means = []
    for column in zip(*vectors, strict=True):
        scale = max(abs(x) for x in column)
        if scale == 0.0:
            means.append(0.0)
        else:
            total = math.fsum(x / scale for x in column)
            means.append((total / len(vectors)) * scale)
    return tuple(means)


@dataclass(frozen=True)
class Recipe:
    """Encoding metadata for this symmetric toy, not a general contract."""

    encoder: str
    version: str
    pooling: str
    normalized: bool


def check_same_recipe(index: Recipe, query: Recipe) -> None:
    """Guard this toy only; paired encoders/role prefixes need a contract."""
    if index != query:
        raise ValueError(
            f"query recipe {query} does not match index recipe {index}: "
            "embed the query with the index recipe or rebuild the index"
        )
```

Some choices worth a sentence each:

- `math.hypot(*v)` computes the Euclidean length of a vector of any length (n-dimensional support arrived in Python 3.8, and 3.10 improved its accuracy to under 1 ulp). Unlike `sqrt(sum(x * x))`, it doesn't overflow for values around `1e200` or underflow to zero for values around `1e-200`.
- I divide by the largest absolute component **before** taking the norm. For `[5e-324, 5e-324]`, directly dividing by its rounded subnormal norm loses the direction. Scaling first gives `(1, 1)`, then a unit vector. It also handles finite components whose raw norm would overflow. The dot product uses these unit vectors.
- `math.fsum` adds with extra precision. For four dimensions that hardly matters; it costs nothing here.
- `zip(..., strict=True)` (Python 3.10+) raises if the lengths differ. The explicit length check comes first only to give a clearer message.
- The result is clamped to `[-1, 1]`. Floating-point rounding can return something like `1.0000000000000002` for identical vectors, and downstream code that calls `math.acos` would then fail.
- `cosine_similarity` rejects empty, mismatched, non-finite and zero vectors. `norm` validates empty/non-finite inputs but may return `inf` for an unrepresentable length. `mean_pool` validates every row and dimensions, and scales each column before summing to avoid raw-sum overflow. A zero pooled vector is valid until a cosine comparison needs its direction.

This is a reference implementation for reading, not for production. A Python loop over vectors is slow; real systems use a numerical library or a vector index. The semantics, including what to do with a zero vector, are the part I want to carry over.

## The tests came first

The original implementation started with a stub: 26 failing tests, then 27 after adding a norm-overflow rejection test, followed by 27 passing tests. Independent review later found a subnormal-direction bug those tests missed.

For this correction I added regression tests before changing the implementation and replaced the overflow-rejection expectation with support for finite vectors whose raw norm overflows. The real RED run had **17 failures and 28 passes**. After the fix, all **45 tests passed**; the final suite and clean reproduction use the same 45 cases.

The suite covers direction and scaling, invalid vectors, subnormal magnitudes, overflowing raw norms, pooling validation and large means, unit-vector identities, and the symmetric toy's recipe guard. A fixed-seed loop also checks the cosine range on 2,000 random pairs. These are arithmetic and input-contract checks; they say nothing about any real model's retrieval quality.

## Ranking on synthetic vectors

The demo builds sentence vectors with the toy encoder: look up each token's hand-chosen row, then take the mean. For the query `refund policy`:

```text
3. ranking for query 'refund policy'
  +0.892  refund approved
  +0.513  delay policy
  +0.078  shipping delay
```

The order follows from numbers I picked, so it demonstrates the mechanics, not quality.

What I take from it is how to read the scores. 0.892 doesn't mean "89% relevant". It means the angle between two vectors is about 27 degrees in a space this toy encoder defines. Change the model and the same pair can land at a very different value. A threshold such as "keep everything above 0.8" belongs to one specific model and data set, and it has to be checked against labelled examples for that model.

Cosine can also be negative:

```text
5. not a probability
  cos((1, 0), (-1, 0.2)) = -0.981
```

Whether a real model's vectors ever point in opposite directions depends on the model. The formula allows it.

## What cosine doesn't know: negation and relevance

Here is the toy encoder on two sentences with opposite meaning:

```text
4. negation
  cos('refund approved', 'refund not approved') = +0.995
```

The mechanism is simple. The mean of three rows (`refund`, `not`, `approved`) points almost the same way as the mean of two (`refund`, `approved`). The word that flips the meaning moves the vector only a little.

Contextual encoders don't simply average fixed word rows, and Sentence-BERT was fine-tuned on inference data in which contradiction is one of the labels. So I don't claim that a particular model fails this case. I do claim that nothing in cosine similarity itself detects negation. Whether the encoder separates such pairs is a property of its training, and I would check it on my own pairs before relying on it.

The same applies to relevance. Cosine measures how similar two vectors are. A document can be very similar to a question, for example by repeating the question, and still not answer it. A short answer can be relevant while sharing few words with the question. Whether "similar" lines up with "relevant" depends on what the encoder was trained to bring together. That is one reason retrieval systems are evaluated with labelled queries rather than by looking at scores.

## One index, one encoding contract

The last part of the demo simulates an encoder update. Version 2 produces vectors of the same length, but its coordinates are a permutation of version 1's. That is a stand-in for retraining, after which coordinate *i* simply means something else.

```text
6. encoder version mismatch
  v1 query vs v1 doc: +0.892
  v2 query vs v2 doc: +0.892
  v2 query vs v1 doc: +0.125
```

Within one version the geometry is identical. Across versions the score changes, even though every shape check passes. Real retraining doesn't give you a clean permutation. You get a different space, and mixed scores are noise.

So a vector index has to remember a versioned query/document contract:

- the tokenizer and model weights for each role, including the exact versions,
- pooling (mean, `[CLS]`, …) and whether padding was masked,
- normalisation,
- truncation length and any text preprocessing,
- any instruction or prefix the model documentation prescribes for queries and documents.

Encode a query with the query recipe prescribed by that contract. It may use a different prefix or a paired encoder: [E5](https://arxiv.org/abs/2212.03533v1) uses `query:` and `passage:` prefixes with shared weights; [DPR](https://arxiv.org/abs/2004.04906v3) uses separate question and passage encoders. Compatibility requires the intended pair, rather than identical processing.

Re-embed the corpus and rebuild the index when the document representation changes incompatibly. A compatible query-side change need not re-embed unchanged documents, but it still needs retrieval evaluation and a recorded contract update. Mixing incompatible spaces can produce plausible-looking rankings.

The `Recipe` dataclass and `check_same_recipe` equality guard model **only this symmetric toy**. They deliberately reject every difference; a general contract must allow its prescribed role-specific prefixes or paired encoders:

```text
refused: query recipe Recipe(encoder='toy-static', version='2', pooling='mean', normalized=False) does not match index recipe Recipe(encoder='toy-static', version='1', pooling='mean', normalized=False): embed the query with the index recipe or rebuild the index
```

In a real system the complete compatible contract is stored next to the index and checked when a query arrives. A refusal is cheap. A silently wrong ranking is not.

## Running the example

The files are available for download:

- [`cosine_lab.py`]({{ '/assets/examples/tokens-embeddings-cosine-similarity/cosine_lab.py' | relative_url }}) — the module above,
- [`demo.py`]({{ '/assets/examples/tokens-embeddings-cosine-similarity/demo.py' | relative_url }}) — the walk-through on synthetic vectors,
- [`tests/test_cosine_lab.py`]({{ '/assets/examples/tokens-embeddings-cosine-similarity/tests/test_cosine_lab.py' | relative_url }}) — the 45 tests,
- [`pyproject.toml`]({{ '/assets/examples/tokens-embeddings-cosine-similarity/pyproject.toml' | relative_url }}) and [`uv.lock`]({{ '/assets/examples/tokens-embeddings-cosine-similarity/uv.lock' | relative_url }}) — the environment.

Put them in one directory, with the test file in a `tests/` subdirectory, then:

```bash
uv sync --locked                      # Python 3.11 + pytest 7.4.3 (test tooling only)
uv run pytest -v                      # 45 tests
uv run python demo.py                 # the output quoted in this post
```

The module and the demo need only the standard library, so `python3.11 demo.py` works without uv as well.

The run quoted here used Python 3.11.12, pytest 7.4.3 and uv 0.7.8 on macOS (arm64). All commands exited with code 0.

## Limits of this example

- The vectors are four-dimensional and hand-chosen. They are not output of any encoder, and the rankings are not a retrieval benchmark.
- The toy encoder has no contextual layers. It shows the mean-pooling step, not what a trained transformer does.
- There is no approximate nearest-neighbour index, batching or performance measurement.
- Scaling fixes the tested subnormal-direction and overflow cases, but does not make floating-point arithmetic exact. Components tiny relative to the largest can round away during scaling. Pooling can lose a small residual under extreme cancellation, and a mean below the smallest representable float rounds to zero. The cosine clamp only handles rounding at the range boundaries.

Next in this series: a first RAG system over local public documents, where these vectors have to find passages that a model then cites.

### Sources

- Rico Sennrich, Barry Haddow, Alexandra Birch. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). ACL 2016, pages 1715–1725. Section 3.2 and footnote 3: alphabet coverage, unseen-symbol limits, BPE initialisation with characters and an end-of-word symbol, merging the most frequent pair, the number of merges as the only hyperparameter.
- Nils Reimers, Iryna Gurevych. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). EMNLP-IJCNLP 2019, pages 3982–3992. Pooling strategies and MEAN default, the three training objectives, Table 1 comparison with averaged GloVe, the 65 hours versus about 5 seconds estimate.
- Vladimir Karpukhin et al. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906v3). 2020, section 3.1: separate passage and question encoders; this is a compatibility example, not a claim that DPR uses cosine as its default score.
- Liang Wang et al. [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533v1). Version 1, 7 December 2022, section 4.1: shared encoder with different `query:` and `passage:` prefixes.
- [Python 3.11: `math`](https://docs.python.org/3.11/library/math.html) — `hypot`, `fsum`, `isfinite`; [Python 3.11: built-in `zip`](https://docs.python.org/3.11/library/functions.html#zip) — `strict` argument since 3.10.
