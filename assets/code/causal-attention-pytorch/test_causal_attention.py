import pytest
import torch
import torch.nn.functional as F

from causal_attention import causal_attention


def make_qkv(b=2, h=3, t=5, d=4, dv=6, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, t, d, generator=g, dtype=dtype)
    k = torch.randn(b, h, t, d, generator=g, dtype=dtype)
    v = torch.randn(b, h, t, dv, generator=g, dtype=dtype)
    return q, k, v


def mask_after_softmax(q, k, v):
    """Deliberately wrong: hides the future only after normalisation."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    allowed = torch.ones(t, t, dtype=torch.bool).tril()
    weights = torch.softmax(scores, dim=-1) * allowed
    return weights @ v, weights


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


def test_shapes_with_different_value_dim():
    q, k, v = make_qkv(b=2, h=3, t=5, d=4, dv=6)
    out, weights = causal_attention(q, k, v)
    assert out.shape == (2, 3, 5, 6)
    assert weights.shape == (2, 3, 5, 5)
    assert out.dtype == q.dtype and weights.dtype == q.dtype


def test_weights_are_lower_triangular_and_rows_sum_to_one():
    q, k, v = make_qkv()
    _, weights = causal_attention(q, k, v)
    t = q.shape[-2]
    future = torch.ones(t, t, dtype=torch.bool).triu(diagonal=1)
    assert torch.all(weights[..., future] == 0)
    assert torch.all(weights.diagonal(dim1=-2, dim2=-1) > 0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3, t, dtype=q.dtype))


def test_first_position_sees_only_itself():
    q, k, v = make_qkv()
    out, weights = causal_attention(q, k, v)
    assert torch.all(weights[..., 0, 0] == 1)
    torch.testing.assert_close(out[..., 0, :], v[..., 0, :])


def test_length_one_returns_value():
    q, k, v = make_qkv(t=1)
    out, weights = causal_attention(q, k, v)
    assert weights.shape == (2, 3, 1, 1)
    assert torch.all(weights == 1)
    torch.testing.assert_close(out, v)


def test_future_positions_do_not_change_past_outputs():
    q, k, v = make_qkv(t=6)
    assert v.abs().min() > 0  # visible values are not trivially zero
    assert max_future_leak(causal_attention, q, k, v) == 0.0


def test_leak_check_detects_mask_after_softmax():
    q, k, v = make_qkv(t=6)
    assert max_future_leak(mask_after_softmax, q, k, v) > 1e-3


def poison_after_future_change(rows, fill, baseline_fill=None):
    """Deliberately broken: correct attention until the helper's large future keys arrive."""
    def attention(q, k, v):
        out, weights = causal_attention(q, k, v)
        out = out.clone()
        if k.abs().max() > 5:  # make_qkv keys stay below 5; the 10x noise does not
            out[..., rows, :] = fill
        elif baseline_fill is not None:
            out[..., rows, :] = baseline_fill
        return out, weights
    return attention


@pytest.mark.parametrize(
    "attention",
    [
        pytest.param(poison_after_future_change(slice(0, 1), float("nan")), id="past-turns-nan"),
        pytest.param(poison_after_future_change(slice(5, 6), float("inf")), id="future-turns-inf"),
        pytest.param(poison_after_future_change(slice(0, 1), float("nan"), float("nan")), id="baseline-nan"),
        pytest.param(poison_after_future_change(slice(0, 1), -1e308, 1e308), id="difference-overflows"),
    ],
)
def test_leak_check_fails_closed_on_non_finite(attention):
    q, k, v = make_qkv(t=6)
    assert k.abs().max() <= 5
    with pytest.raises(AssertionError, match="non-finite"):
        max_future_leak(attention, q, k, v)


def test_future_inputs_get_zero_gradient_from_past_outputs():
    q, k, v = make_qkv(t=5)
    q, k, v = (x.requires_grad_() for x in (q, k, v))
    out, _ = causal_attention(q, k, v)
    out[..., :3, :].sum().backward()
    for x in (q, k, v):
        assert torch.isfinite(x.grad).all()
        assert torch.all(x.grad[..., 3:, :] == 0)
        assert x.grad[..., :3, :].abs().sum() > 0


def test_gradcheck_float64():
    q, k, v = make_qkv(b=1, h=2, t=4, d=3, dv=2)
    inputs = tuple(x.requires_grad_() for x in (q, k, v))
    assert torch.autograd.gradcheck(lambda *a: causal_attention(*a)[0], inputs)


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float64, dict(rtol=1e-12, atol=1e-12)), (torch.float32, dict(rtol=1e-5, atol=1e-6))],
)
def test_matches_sdpa_is_causal(dtype, tol):
    q, k, v = make_qkv(b=2, h=4, t=7, d=8, dv=5, dtype=dtype)
    out, _ = causal_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    torch.testing.assert_close(out, expected, **tol)


def test_boolean_mask_true_means_allowed():
    q, k, v = make_qkv()
    out, _ = causal_attention(q, k, v)
    t = q.shape[-2]
    allowed = torch.ones(t, t, dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3, 5), (2, 3, 5), (2, 3, 5)),              # missing head axis
        ((2, 3, 5, 4), (2, 3, 6, 4), (2, 3, 6, 4)),     # T_q != T_k
        ((2, 3, 5, 4), (2, 3, 5, 8), (2, 3, 5, 4)),     # D_q != D_k
        ((2, 3, 5, 4), (2, 3, 5, 4), (2, 3, 4, 4)),     # V length != K length
        ((2, 3, 5, 4), (1, 3, 5, 4), (1, 3, 5, 4)),     # no batch broadcasting
        ((2, 3, 5, 4), (2, 1, 5, 4), (2, 1, 5, 4)),     # no head broadcasting
        ((2, 3, 0, 4), (2, 3, 0, 4), (2, 3, 0, 4)),     # empty sequence
        ((2, 3, 5, 0), (2, 3, 5, 0), (2, 3, 5, 4)),     # empty head dimension
        ((2, 3, 5, 4), (2, 3, 5, 4), (2, 3, 5, 0)),     # empty value dimension
    ],
)
def test_rejects_invalid_shapes(shapes):
    q, k, v = (torch.zeros(s, dtype=torch.float32) for s in shapes)
    with pytest.raises(ValueError):
        causal_attention(q, k, v)


@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, torch.float64, torch.float32),  # mixed precision
        (torch.int64, torch.int64, torch.int64),        # not floating point
        (torch.float16, torch.float16, torch.float16),  # outside the tested set
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),  # outside the tested set
    ],
)
def test_rejects_unsupported_dtypes(dtypes):
    q, k, v = (torch.zeros(1, 1, 2, 2, dtype=dt) for dt in dtypes)
    with pytest.raises(TypeError):
        causal_attention(q, k, v)


def test_rejects_non_tensor_input():
    q, k, v = make_qkv()
    with pytest.raises(TypeError):
        causal_attention(q, k, v.tolist())


def test_rejects_mixed_devices():
    q, k, v = make_qkv()
    meta_v = torch.empty(v.shape, dtype=v.dtype, device="meta")
    with pytest.raises(ValueError):
        causal_attention(q, k, meta_v)
