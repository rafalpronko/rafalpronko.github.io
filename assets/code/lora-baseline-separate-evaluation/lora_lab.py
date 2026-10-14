"""A small synthetic LoRA experiment in NumPy (CPU, seconds).

Task: multi-output linear regression y = W x + noise, x in R^k, y in R^d.
A frozen "base" W0 is fitted on a source domain; the target domain differs by a
constructed low-rank shift plus a small dense shift. We compare: frozen base,
full adaptation of W, and LoRA W0 + (alpha/r) B A for several ranks r.
Learning rate, checkpoint and model are chosen on target validation only;
the target test split is evaluated once, after selection.
This is not an LLM benchmark and uses no pretrained model.
"""
import dataclasses
import json
import platform
import sys
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Config:
    k: int = 16                 # input dimension
    d: int = 8                  # output dimension
    shift_rank: int = 2         # rank of the constructed target shift
    shift_scale: float = 1.0
    dense_shift_scale: float = 0.05
    noise: float = 0.5          # std of label noise in both domains
    n_source: int = 400
    n_train: int = 32           # scarce target training data
    n_val: int = 200
    n_test: int = 2000
    ranks: tuple = (1, 2, 4)
    alpha: float = 2.0          # fixed, not tuned (as in the paper's recipe)
    lrs: tuple = (0.01, 0.03, 0.1, 0.3)
    steps: int = 300            # identical full-batch GD budget for every method
    eval_every: int = 10
    replicates: int = 5

    def seeds(self, replicate):
        r = replicate
        return {"world": 1000, "source_train": 2000 + r, "source_test": 3000 + r,
                "target_train": 4000 + r, "target_val": 5000 + r,
                "target_test": 6000 + r, "lora_init": 7000 + r}


def mse(pred, Y):
    return float(np.mean((pred - Y) ** 2))


def make_world(cfg):
    rng = np.random.default_rng(cfg.seeds(0)["world"])
    W_src = rng.normal(size=(cfg.d, cfg.k))
    U = rng.normal(size=(cfg.d, cfg.shift_rank))
    V = rng.normal(size=(cfg.k, cfg.shift_rank))
    low_rank = cfg.shift_scale * U @ V.T / np.sqrt(cfg.shift_rank)
    dense = cfg.dense_shift_scale * rng.normal(size=(cfg.d, cfg.k))
    return W_src, W_src + low_rank + dense


def sample(W, n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, W.shape[1]))
    return X, X @ W.T + noise * rng.normal(size=(n, W.shape[0]))


def make_splits(cfg, replicate):
    W_src, W_tgt = make_world(cfg)
    s = cfg.seeds(replicate)
    return {
        "source_train": sample(W_src, cfg.n_source, cfg.noise, s["source_train"]),
        "source_test": sample(W_src, cfg.n_test, cfg.noise, s["source_test"]),
        "target_train": sample(W_tgt, cfg.n_train, cfg.noise, s["target_train"]),
        "target_val": sample(W_tgt, cfg.n_val, cfg.noise, s["target_val"]),
        "target_test": sample(W_tgt, cfg.n_test, cfg.noise, s["target_test"]),
    }


def fit_base(X, Y):
    """Least squares on the source domain: our stand-in for a 'pretrained' weight."""
    return np.linalg.lstsq(X, Y, rcond=None)[0].T


def _grad_W(W_pred, X, Y):
    """d mse / d W_effective for predictions X @ W.T, shape (d, k)."""
    n, d = Y.shape
    return 2.0 / (n * d) * (W_pred - Y).T @ X


class FullAdapter:
    def __init__(self, W0):
        self.W = np.array(W0, dtype=float)  # all d*k entries are trainable

    n_trainable = property(lambda self: self.W.size)
    params = property(lambda self: [self.W])

    def predict(self, X):
        return X @ self.W.T

    def weight(self):
        return self.W.copy()

    def grads(self, X, Y):
        return (_grad_W(self.predict(X), X, Y),)


class LoRA:
    def __init__(self, W0, r, alpha, seed):
        self.W0 = np.array(W0, dtype=float)
        self.W0.flags.writeable = False            # frozen base
        d, k = self.W0.shape
        rng = np.random.default_rng(seed)
        self.A = rng.normal(scale=1 / np.sqrt(k), size=(r, k))  # random
        self.B = np.zeros((d, r))                              # zero -> BA = 0 at start
        self.r, self.alpha, self.scaling = r, alpha, alpha / r

    n_trainable = property(lambda self: self.A.size + self.B.size)
    params = property(lambda self: [self.A, self.B])

    def delta(self):
        return self.scaling * self.B @ self.A

    def predict(self, X):
        return X @ self.W0.T + self.scaling * (X @ self.A.T) @ self.B.T

    def weight(self):
        return merge(self)

    def grads(self, X, Y):
        G = _grad_W(self.predict(X), X, Y)        # gradient w.r.t. W0 + s*BA
        return self.scaling * self.B.T @ G, self.scaling * G @ self.A.T  # (dA, dB)


def merge(m):
    return m.W0 + m.delta()


def unmerge(W, m):
    return W - m.delta()


def _checkpoint(model, step, val_mse):
    """Copy one coherent state, including the adapter factors at this step."""
    state = {"step": step, "val_mse": val_mse, "W": model.weight()}
    if isinstance(model, LoRA):
        state.update(A=model.A.copy(), B=model.B.copy(), scaling=model.scaling,
                     rank=model.r, alpha=model.alpha)
    return state


def train(model, train_xy, val_xy, steps, lr, eval_every):
    """Full-batch gradient descent; keep the checkpoint with the lowest validation MSE."""
    X, Y = train_xy
    best = _checkpoint(model, 0, mse(model.predict(val_xy[0]), val_xy[1]))
    history = [(0, mse(model.predict(X), Y), best["val_mse"])]
    for step in range(1, steps + 1):
        with np.errstate(over="ignore", invalid="ignore"):  # overflow is handled explicitly below
            for p, g in zip(model.params, model.grads(X, Y)):
                p -= lr * g
            tr = mse(model.predict(X), Y)
        if not (np.isfinite(tr) and all(np.isfinite(p).all() for p in model.params)):
            return {**best, "history": history, "diverged_at": step}  # keep last finite best
        if step % eval_every == 0:
            va = mse(model.predict(val_xy[0]), val_xy[1])
            history.append((step, tr, va))
            if va < best["val_mse"]:
                best = _checkpoint(model, step, va)
    return {**best, "history": history, "diverged_at": None}


def select(cfg, source_train, target_train, target_val, *, replicate):
    """Every choice (lr, checkpoint, method, rank) uses target validation only."""
    W0 = fit_base(*source_train)
    methods = {"frozen_base": {"val_mse": mse(target_val[0] @ W0.T, target_val[1]),
                               "W": W0, "n_trainable": 0, "lr": None, "step": 0}}
    makers = {"full": lambda: FullAdapter(W0)}
    for r in cfg.ranks:
        makers[f"lora_r{r}"] = lambda r=r: LoRA(W0, r, cfg.alpha, cfg.seeds(replicate)["lora_init"])
    for name, make in makers.items():
        runs = []
        for lr in cfg.lrs:
            model = make()
            res = train(model, target_train, target_val, cfg.steps, lr, cfg.eval_every)
            runs.append({**res, "lr": lr, "n_trainable": model.n_trainable})
        methods[name] = min(runs, key=lambda run: run["val_mse"])
        methods[name]["lr_grid"] = {str(r["lr"]): {"best_val_mse": r["val_mse"], "best_step": r["step"],
                                                    "diverged_at": r["diverged_at"]} for r in runs}
    lora_names = [f"lora_r{r}" for r in cfg.ranks]
    return {"W0": W0, "methods": methods,
            "chosen": min(methods, key=lambda n: methods[n]["val_mse"]),
            "chosen_lora": min(lora_names, key=lambda n: methods[n]["val_mse"])}


def run(cfg, outdir):
    outdir = Path(outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    W_src, W_tgt = make_world(cfg)
    report = {"config": dataclasses.asdict(cfg),
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "platform": platform.platform(), "machine": platform.machine()},
              "true_shift_singular_values": [float(s) for s in np.linalg.svd(W_tgt - W_src, compute_uv=False)],
              "replicates": []}
    for rep in range(cfg.replicates):
        splits = make_splits(cfg, rep)
        sel = select(cfg, splits["source_train"], splits["target_train"], splits["target_val"],
                     replicate=rep)
        # Selection is final here. The locked test split is read only below.
        Xt, Yt = splits["target_test"]
        Xs, Ys = splits["source_test"]
        rows = {}
        for name, m in sel["methods"].items():
            ck = f"checkpoints/rep{rep}_{name}.npz"
            extra = {k: m[k] for k in ("A", "B", "scaling", "rank", "alpha") if k in m}
            np.savez(outdir / ck, W=m["W"], W0=sel["W0"], step=m["step"], **extra)
            rows[name] = {"n_trainable": int(m["n_trainable"]), "lr": m["lr"], "step": int(m["step"]),
                          "val_mse": m["val_mse"], "target_test_mse": mse(Xt @ m["W"].T, Yt),
                          "source_test_mse": mse(Xs @ m["W"].T, Ys), "checkpoint": ck,
                          "diverged_at": m.get("diverged_at"),
                          "lr_grid": m.get("lr_grid")}
        report["replicates"].append({
            "replicate": rep, "seeds": cfg.seeds(rep), "chosen": sel["chosen"],
            "chosen_lora": sel["chosen_lora"], "methods": rows,
            # Reference only: uses the generating weights, which a real project never has.
            "reference_true_W_target_test_mse": mse(Xt @ W_tgt.T, Yt)})
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rep = run(Config(), out)
    for r in rep["replicates"]:
        print(f"replicate {r['replicate']}: chosen={r['chosen']} chosen_lora={r['chosen_lora']}")
        for name, m in r["methods"].items():
            print(f"  {name:12s} params={m['n_trainable']:4d} lr={m['lr']!s:5s} step={m['step']:3d} "
                  f"val={m['val_mse']:.4f} test={m['target_test_mse']:.4f} source_test={m['source_test_mse']:.4f}")
        print(f"  reference(true W_tgt) test={r['reference_true_W_target_test_mse']:.4f}")
