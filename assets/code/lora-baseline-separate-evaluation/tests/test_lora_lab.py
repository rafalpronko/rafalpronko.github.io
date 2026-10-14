import json
import warnings

import numpy as np
import pytest

import lora_lab as L

CFG = L.Config()


def small_problem(seed=0, n=12, d=3, k=5):
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(d, k))
    X = rng.normal(size=(n, k))
    Y = rng.normal(size=(n, d))
    return W0, X, Y


def numeric_grad(f, P, eps=1e-6):
    G = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        up = f()
        P[idx] = old - eps
        down = f()
        P[idx] = old
        G[idx] = (up - down) / (2 * eps)
    return G


def test_lora_gradients_match_central_differences():
    W0, X, Y = small_problem()
    m = L.LoRA(W0, r=2, alpha=3.0, seed=1)
    m.B[:] = np.random.default_rng(2).normal(size=m.B.shape)  # non-zero B so dA is non-trivial
    gA, gB = m.grads(X, Y)
    loss = lambda: L.mse(m.predict(X), Y)
    np.testing.assert_allclose(gA, numeric_grad(loss, m.A), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(gB, numeric_grad(loss, m.B), rtol=1e-6, atol=1e-8)


def test_full_gradient_matches_central_differences():
    W0, X, Y = small_problem()
    m = L.FullAdapter(W0)
    (gW,) = m.grads(X, Y)
    loss = lambda: L.mse(m.predict(X), Y)
    np.testing.assert_allclose(gW, numeric_grad(loss, m.W), rtol=1e-6, atol=1e-8)


def test_initial_lora_model_equals_frozen_base():
    W0, X, _ = small_problem()
    m = L.LoRA(W0, r=2, alpha=2.0, seed=3)
    assert np.all(m.B == 0) and np.any(m.A != 0)
    np.testing.assert_array_equal(m.predict(X), X @ W0.T)


def test_both_factors_zero_never_moves():
    W0, X, Y = small_problem()
    m = L.LoRA(W0, r=2, alpha=2.0, seed=3)
    m.A[:] = 0.0
    gA, gB = m.grads(X, Y)
    assert not gA.any() and not gB.any()  # saddle point: no signal reaches A or B


def test_base_weight_is_frozen_during_training():
    W0, X, Y = small_problem()
    before = W0.copy()
    m = L.LoRA(W0, r=1, alpha=1.0, seed=0)
    L.train(m, (X, Y), (X, Y), steps=20, lr=0.05, eval_every=5)
    np.testing.assert_array_equal(m.W0, before)
    assert not m.W0.flags.writeable
    assert m.B.any()  # the adapter did learn something


def test_trainable_parameter_count_and_rank():
    W0 = np.zeros((CFG.d, CFG.k))
    for r in (1, 2, 4):
        m = L.LoRA(W0, r=r, alpha=2.0, seed=0)
        m.B[:] = 1.0
        assert m.n_trainable == r * (CFG.d + CFG.k)
        assert np.linalg.matrix_rank(m.delta()) <= r
    assert L.FullAdapter(W0).n_trainable == CFG.d * CFG.k


def test_merge_matches_unmerged_and_unmerge_recovers_base():
    W0, X, _ = small_problem()
    m = L.LoRA(W0, r=2, alpha=2.0, seed=0)
    m.B[:] = np.random.default_rng(5).normal(size=m.B.shape)
    W = L.merge(m)
    np.testing.assert_allclose(X @ W.T, m.predict(X), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(L.unmerge(W, m), W0, rtol=0, atol=1e-12)


def test_splits_are_disjoint_and_seeds_distinct():
    seeds = CFG.seeds(replicate=0)
    assert len(set(seeds.values())) == len(seeds)
    splits = L.make_splits(CFG, replicate=0)
    rows = [tuple(np.round(x, 12)) for name in splits for x in splits[name][0]]
    assert len(rows) == len(set(rows))
    assert splits["target_train"][0].shape == (CFG.n_train, CFG.k)
    assert splits["target_test"][1].shape == (CFG.n_test, CFG.d)


def test_selection_api_rejects_direct_test_argument():
    splits = L.make_splits(CFG, replicate=0)
    a = L.select(CFG, splits["source_train"], splits["target_train"], splits["target_val"], replicate=0)
    b = L.select(CFG, splits["source_train"], splits["target_train"], splits["target_val"], replicate=0)
    assert a["chosen"] == b["chosen"]
    with pytest.raises(TypeError):
        L.select(CFG, splits["source_train"], splits["target_train"], splits["target_val"],
                 replicate=0, test=splits["target_test"])


def assert_checkpoint(ck, row, splits):
    W = ck["W"]
    if "A" in ck:
        # Assert factor consistency first: this exposes v1 even before metadata checks.
        rank = ck["A"].shape[0]
        scaling = float(ck["scaling"])
        restored = L.LoRA(ck["W0"], rank, scaling * rank, seed=0)
        restored.A[:] = ck["A"]
        restored.B[:] = ck["B"]
        np.testing.assert_allclose(L.merge(restored), W, rtol=0, atol=1e-12)
        np.testing.assert_allclose(L.unmerge(W, restored), ck["W0"], rtol=0, atol=1e-12)
        for split in ("target_val", "target_test", "source_test"):
            X, _ = splits[split]
            np.testing.assert_allclose(restored.predict(X), X @ W.T, rtol=0, atol=1e-12)
        assert {"scaling", "rank", "alpha", "step"} <= set(ck.files)
        assert int(ck["rank"]) == rank == ck["B"].shape[1]
        assert float(ck["alpha"]) / rank == scaling
        assert int(ck["step"]) == row["step"]
    for split, metric in (("target_val", "val_mse"), ("target_test", "target_test_mse"),
                          ("source_test", "source_test_mse")):
        X, Y = splits[split]
        assert L.mse(X @ W.T, Y) == pytest.approx(row[metric], rel=0, abs=1e-12)
        if "A" in ck:
            assert L.mse(restored.predict(X), Y) == pytest.approx(row[metric], rel=0, abs=1e-12)


def test_run_writes_reconstructable_checkpoints(tmp_path):
    cfg = L.Config()  # all five replicates, every saved adapter, including pre-final choices
    report = L.run(cfg, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == json.loads(json.dumps(report))
    early = []
    for rep in saved["replicates"]:
        splits = L.make_splits(cfg, replicate=rep["replicate"])
        for name, row in rep["methods"].items():
            with np.load(tmp_path / row["checkpoint"]) as ck:
                assert_checkpoint(ck, row, splits)
            if name.startswith("lora") and 0 < row["step"] < cfg.steps:
                early.append((rep["replicate"], name))
        assert rep["chosen"] in rep["methods"]
    assert early and (3, "lora_r2") in early


def test_best_checkpoint_copies_factors_before_final_step():
    m = L.LoRA(np.zeros((1, 1)), r=1, alpha=3.0, seed=0)
    m.A[:] = 1.0
    X = np.ones((1, 1))
    # First GD step: B=0.3, A=1, W=0.9. Later steps move toward training Y=1.
    val = (X, np.full((1, 1), 0.9))
    res = L.train(m, (X, X), val, steps=30, lr=0.05, eval_every=1)
    assert res["step"] == 1 < 30
    assert res["scaling"] == 3.0 and res["rank"] == 1 and res["alpha"] == 3.0
    np.testing.assert_allclose(res["W"], 3.0 * res["B"] @ res["A"], rtol=0, atol=1e-12)
    assert L.mse(X @ res["W"].T, val[1]) == pytest.approx(res["val_mse"], rel=0, abs=1e-12)
    assert not np.shares_memory(res["A"], m.A) and not np.shares_memory(res["B"], m.B)
    A, B = res["A"].copy(), res["B"].copy()
    m.A[:] = 123.0
    m.B[:] = 456.0
    np.testing.assert_array_equal(res["A"], A)
    np.testing.assert_array_equal(res["B"], B)


@pytest.mark.parametrize("lr,steps,eval_every,diverges", [(0.05, 30, 1, False), (50.0, 200, 50, True)])
def test_step_zero_checkpoint_survives_training_and_divergence(lr, steps, eval_every, diverges):
    W0, X, Y = small_problem()
    m = L.LoRA(W0, r=1, alpha=2.0, seed=0)
    initial_A = m.A.copy()
    res = L.train(m, (X, Y), (X, X @ W0.T), steps, lr, eval_every)
    assert res["step"] == 0
    assert (res["diverged_at"] is not None) == diverges
    np.testing.assert_array_equal(res["A"], initial_A)
    np.testing.assert_array_equal(res["B"], np.zeros_like(res["B"]))
    np.testing.assert_array_equal(res["W"], W0)
    np.testing.assert_allclose(W0 + res["scaling"] * res["B"] @ res["A"], res["W"], rtol=0, atol=1e-12)
    assert not np.shares_memory(res["A"], m.A) and not np.shares_memory(res["B"], m.B)


def test_saved_step_zero_after_divergence_reconstructs(tmp_path):
    cfg = L.Config(lrs=(50.0,), steps=200, eval_every=50, replicates=1)
    report = L.run(cfg, tmp_path)
    rep = report["replicates"][0]
    splits = L.make_splits(cfg, 0)
    for name, row in rep["methods"].items():
        with np.load(tmp_path / row["checkpoint"]) as ck:
            assert_checkpoint(ck, row, splits)
        if name.startswith("lora"):
            assert row["step"] == 0 and row["diverged_at"] < cfg.eval_every


def test_divergence_is_detected_at_the_step_it_happens_without_warnings():
    W0, X, Y = small_problem()
    m = L.LoRA(W0, r=1, alpha=2.0, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        res = L.train(m, (X, Y), (X, Y), steps=200, lr=50.0, eval_every=50)
    assert res["diverged_at"] is not None and res["diverged_at"] < 50
    assert np.isfinite(res["val_mse"]) and np.all(np.isfinite(res["W"]))
