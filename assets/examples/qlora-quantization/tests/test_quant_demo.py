import math
import warnings

import numpy as np
import pytest

from quant_demo import (
    double_quant_bytes,
    dequantize_blockwise,
    lora_params,
    main,
    outlier_experiment,
    packed_bytes,
    quantize_blockwise,
)


def roundtrip(x, bits=4, block=64):
    codes, scales = quantize_blockwise(x, bits, block)
    return codes, scales, dequantize_blockwise(codes, scales, np.shape(x))


def test_codes_are_symmetric_and_in_range():
    x = np.random.default_rng(1).normal(0, 1, (37, 11)).astype(np.float32)
    codes, scales, rec = roundtrip(x)
    assert codes.dtype == np.int8 and codes.min() >= -7 and codes.max() <= 7
    assert scales.shape == (math.ceil(x.size / 64),)
    assert rec.shape == x.shape and rec.dtype == np.float32


def test_zero_is_exact_and_all_zero_block_is_safe():
    x = np.zeros(130, dtype=np.float32)
    x[:5] = [0.3, -0.1, 0.0, 0.2, 0.0]
    _, _, rec = roundtrip(x)
    assert np.all(rec[x == 0] == 0)


def test_error_bounded_by_half_step():
    x = np.random.default_rng(2).normal(0, 0.02, 1000).astype(np.float32)
    codes, scales, rec = roundtrip(x, block=64)
    step = np.repeat(scales, 64)[: x.size]
    assert np.all(np.abs(rec - x) <= step / 2 + 1e-7)


def test_absmax_element_reconstructs_exactly():
    x = np.array([0.01, -0.5, 0.02, 0.03], dtype=np.float32)
    _, _, rec = roundtrip(x, block=4)
    assert rec[1] == pytest.approx(-0.5, abs=1e-7)


def test_outlier_damage_is_confined_to_its_block():
    rows = {r["block"]: r for r in outlier_experiment()}
    # a 50-sigma outlier pushes (almost) every other weight in its block to code 0
    for r in rows.values():
        assert r["others"] == r["block"] - 1
        assert r["zero_outlier"] >= 0.99 * r["others"]
        assert r["zero_outlier"] > r["zero_clean"]
    # damage shrinks as blocks get smaller
    damage = [rows[b]["rmse_outlier"] for b in (4096, 256, 64, 16)]
    assert damage == sorted(damage, reverse=True)
    for r in rows.values():
        assert r["rmse_outlier"] >= r["rmse_clean"]


@pytest.mark.parametrize("n,index,block,others", [
    (65, 64, 64, 0),    # outlier alone in a partial last block
    (70, 69, 64, 5),    # partial last block with neighbors
    (65, 10, 128, 64),  # block larger than the whole array
])
def test_outlier_experiment_ignores_padding(n, index, block, others):
    (row,) = outlier_experiment(n=n, index=index, block_sizes=(block,))
    assert row["others"] == others
    assert 0 <= row["zero_clean"] <= others
    assert row["zero_outlier"] == others  # padding zeros are not counted


@pytest.mark.parametrize("index", [-1, 65, 10.0, True])
def test_outlier_experiment_rejects_index_outside_array(index):
    with pytest.raises(ValueError):
        outlier_experiment(n=65, index=index, block_sizes=(64,))


def test_blocks_without_outlier_are_unchanged():
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 0.02, 512).astype(np.float32)
    spiked = clean.copy()
    spiked[10] = 1.0
    a = roundtrip(clean)[2]
    b = roundtrip(spiked)[2]
    assert np.array_equal(a[64:], b[64:])
    assert not np.array_equal(a[:64], b[:64])


def test_packed_bytes_and_padding():
    assert packed_bytes(64) == (32, 4)  # 4.5 bits per parameter
    assert packed_bytes(65) == (33, 8)  # partial block still needs a scale
    assert packed_bytes(1000, bits=8, block_size=1000, scale_bits=16) == (1000, 2)


def test_double_quant_matches_paper_formula():
    n = 64 * 256 * 1000
    bits_per_param = double_quant_bytes(n) * 8 / n
    assert bits_per_param == pytest.approx(8 / 64 + 32 / (64 * 256))
    assert bits_per_param == pytest.approx(0.127, abs=5e-4)


def test_lora_param_count():
    assert lora_params(4096, 4096, 16) == 131_072
    assert lora_params(4096, 11008, 8) == 8 * (4096 + 11008)


@pytest.mark.parametrize("bits,block", [(1, 64), (9, 64), (4, 0), (4.0, 64), (4, 64.5),
                                        (True, 64), (4, True)])
def test_rejects_invalid_arguments(bits, block):
    with pytest.raises(ValueError):
        quantize_blockwise(np.ones(4), bits, block)


@pytest.mark.parametrize("x", [[], [1.0, np.nan], [np.inf, 0.0], [-np.inf]])
def test_rejects_empty_or_non_finite_input(x):
    with pytest.raises(ValueError):
        quantize_blockwise(np.array(x, dtype=np.float32))


def test_rejects_scale_that_underflows_to_zero():
    # smallest positive float32 (a subnormal) alone in a padded 64-wide block:
    # absmax / 7 rounds to 0 in float32, so the block has no representable scale.
    # The check must come before blocks / scales: no divide-by-zero, no nan cast.
    tiny = np.nextafter(np.float32(0), np.float32(1))
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise"):
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="scale underflows"):
            quantize_blockwise(np.array([tiny], dtype=np.float32))


@pytest.mark.parametrize("change", ["one_d_codes", "short_scales", "nan_scale", "inf_scale",
                                    "zero_scale", "negative_scale", "bad_shape", "empty_shape"])
def test_dequantize_rejects_inconsistent_inputs(change):
    codes, scales = quantize_blockwise(np.ones(130, dtype=np.float32))
    shape = (130,)
    bad_scale = {"nan_scale": np.nan, "inf_scale": np.inf,
                 "zero_scale": 0.0, "negative_scale": -1.0}
    if change == "one_d_codes":
        codes = codes.ravel()
    elif change == "short_scales":
        scales = scales[:-1]
    elif change in bad_scale:
        scales = scales.copy()
        scales[0] = bad_scale[change]
    elif change == "bad_shape":
        shape = (200,)
    else:
        shape = (0,)
    with pytest.raises(ValueError):
        dequantize_blockwise(codes, scales, shape)


def test_main_prints_report(capsys):
    main()
    out = capsys.readouterr().out
    assert "One outlier" in out
    assert "Theoretical weight-only storage" in out
    assert "Excludes activations" in out
