"""Blockwise uniform symmetric (absmax) quantization on synthetic weights.

Educational CPU demo. This is NOT NF4, NOT bitsandbytes and NOT QLoRA training.
Memory figures are theoretical packed sizes, not measured NumPy or GPU allocations.
Validation is limited to these cases, which raise ValueError:
- quantize_blockwise: empty or non-finite x, bits not an integer in [2, 8],
  block_size not an integer >= 1, or a nonzero block whose float32 scale
  absmax / qmax underflows to 0 (only blocks of tiny subnormals, e.g. the
  smallest positive float32; such blocks are not representable here). This
  assumes NumPy's default floating-point error settings (underflow ignored):
  under a caller's np.errstate(under='raise') the scale calculation may raise
  FloatingPointError before the check, and underflow warnings enabled and
  treated as errors can also interrupt it before the check;
- dequantize_blockwise: codes not 2-D, not one finite positive scale per block,
  or a shape whose size does not fit the block count. The original length is
  not stored, so any size within the last block passes: a smaller one
  truncates data, a larger one returns padding zeros as values;
- outlier_experiment: index not an integer in [0, n).
Other malformed inputs (complex values, a non-integer shape, ...) are not
checked and may fail differently or be converted silently. This is a demo,
not a library.

Run from this directory (uses the shipped uv.lock, temporary environment):
    uv run --isolated --locked --python 3.12 python quant_demo.py
    uv run --isolated --locked --python 3.12 pytest -v
"""
from __future__ import annotations

import math
import numbers

import numpy as np


def _is_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def quantize_blockwise(x, bits=4, block_size=64):
    """Return int8 codes in [-qmax, qmax] and one float32 scale per block."""
    if not (_is_int(bits) and 2 <= bits <= 8 and _is_int(block_size) and block_size >= 1):
        raise ValueError("bits must be an integer in [2, 8] and block_size an integer >= 1")
    flat = np.asarray(x, dtype=np.float32).ravel()
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("x must be a non-empty array of finite values")
    qmax = 2 ** (bits - 1) - 1  # 4 bits -> 7: codes -7..7, exact zero, 15 of 16 used
    n_blocks = math.ceil(flat.size / block_size)
    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[: flat.size] = flat
    blocks = padded.reshape(n_blocks, block_size)
    absmax = np.abs(blocks).max(axis=1)
    scales = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    if np.any(scales == 0):  # checked before dividing by the scales
        raise ValueError("block scale underflows to 0 in float32; values too small to quantize")
    codes = np.clip(np.rint(blocks / scales[:, None]), -qmax, qmax)
    return codes.astype(np.int8), scales


def dequantize_blockwise(codes, scales, shape):
    """Map codes back to float32 values: code * scale of its block."""
    codes, scales = np.asarray(codes), np.asarray(scales)
    if codes.ndim != 2 or scales.shape != (codes.shape[0],):
        raise ValueError("codes must be (n_blocks, block_size) with one scale per block")
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("scales must be finite and positive")
    n = math.prod(shape)
    if not codes.size - codes.shape[1] < n <= codes.size:
        raise ValueError("shape does not match the number of quantized values")
    values = (codes.astype(np.float32) * scales[:, None]).ravel()
    return values[:n].reshape(shape)


def outlier_experiment(n=4096, sigma=0.02, outlier=1.0, index=100, seed=0,
                       block_sizes=(4096, 256, 64, 16), bits=4):
    """RMSE of the ordinary weights with and without one large outlier."""
    if not (_is_int(index) and 0 <= index < n):
        raise ValueError("index must be an integer in [0, n)")
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, sigma, n).astype(np.float32)
    spiked = clean.copy()
    spiked[index] = outlier
    ordinary = np.arange(n) != index  # the outlier itself is excluded from RMSE
    rows = []
    for block in block_sizes:
        start = (index // block) * block
        stop = min(start + block, n)  # the last block may be partial; skip padding
        neighbors = ordinary[start:stop]
        row = {"block": block, "others": int(neighbors.sum())}
        for name, data in (("clean", clean), ("outlier", spiked)):
            codes, scales = quantize_blockwise(data, bits, block)
            error = dequantize_blockwise(codes, scales, data.shape) - data
            row[f"rmse_{name}"] = float(np.sqrt(np.mean(error[ordinary] ** 2)))
            block_codes = codes.ravel()[start:stop][neighbors]
            row[f"zero_{name}"] = int(np.sum(block_codes == 0))
        rows.append(row)
    return rows


def packed_bytes(n_params, bits=4, block_size=64, scale_bits=32):
    """Theoretical bytes: tightly packed codes plus one scale per block."""
    codes = math.ceil(n_params * bits / 8)
    scales = math.ceil(n_params / block_size) * scale_bits // 8
    return codes, scales


def double_quant_bytes(n_params, block_size=64, block2=256):
    """Paper's DQ layout: 8-bit first-level scales, one FP32 scale per 256 of them.

    Mean-offset metadata omitted.
    """
    n_scales = math.ceil(n_params / block_size)
    return n_scales * 1 + math.ceil(n_scales / block2) * 4


def lora_params(h, o, r):
    """Trainable parameters of L1 (h x r) and L2 (r x o)."""
    return r * (h + o)


def main():
    print("1) One outlier, 4-bit uniform symmetric absmax; RMSE over the other weights")
    print("   'zero' = other weights in the outlier's block mapped to code 0")
    print(f"{'block':>6} {'rmse clean':>11} {'rmse +outlier':>14} "
          f"{'zero clean':>11} {'zero +outlier':>14}")
    for r in outlier_experiment():
        zc = f"{r['zero_clean']}/{r['others']}"
        zo = f"{r['zero_outlier']}/{r['others']}"
        print(f"{r['block']:>6} {r['rmse_clean']:>11.5f} {r['rmse_outlier']:>14.5f} "
              f"{zc:>11} {zo:>14}")

    print("\n2) Scale overhead per parameter (bits)")
    for block in (16, 64, 256, 4096):
        print(f"   block {block:>4}, FP32 scale: {32 / block:.4f}")
    dq = 8 / 64 + 32 / (64 * 256)
    print(f"   block 64, double quantization: {dq:.4f} (saves {0.5 - dq:.4f})")

    n = 7_000_000_000  # nominal weight count, every weight quantized
    gib = 2 ** 30
    codes, scales = packed_bytes(n)
    print(f"\n3) Theoretical weight-only storage for {n:,} parameters (GiB)")
    print(f"   16-bit weights:                      {2 * n / gib:6.2f}")
    print(f"   4-bit codes + FP32 scales (block 64): {(codes + scales) / gib:6.2f}")
    print(f"   4-bit codes + double-quant scales:   "
          f"{(codes + double_quant_bytes(n)) / gib:6.2f}")
    print("   Excludes activations, gradients, optimizer state, buffers, workspaces.")

    demo = np.zeros(4096, dtype=np.float32)
    c, _ = quantize_blockwise(demo)
    print(f"\n   This demo stores codes as int8: {c.nbytes} B for 4096 codes "
          f"vs {packed_bytes(4096)[0]} B if packed two per byte.")

    h = o = 4096
    print(f"\n4) LoRA on one {h}x{o} projection")
    for r in (8, 16, 64):
        p = lora_params(h, o, r)
        print(f"   r={r:>2}: L1 {h}x{r}, L2 {r}x{o}, {p:,} trainable "
              f"= {p / (h * o):.2%} of W")


if __name__ == "__main__":
    main()
