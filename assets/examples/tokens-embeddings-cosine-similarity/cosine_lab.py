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
