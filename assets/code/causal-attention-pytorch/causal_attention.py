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
