"""Four experiment traps on CPU: train/eval, padding, checkpoints, seeds.

Python 3.11, torch==2.2.2, numpy<2. Run: uv run python lab.py
"""
import copy
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PAD, VOCAB, MAX_LEN, BATCH = 0, 12, 8, 32
Batch = tuple[torch.Tensor, torch.Tensor]


def make_batch(gen: torch.Generator, size: int = BATCH) -> Batch:
    lengths = torch.randint(2, MAX_LEN + 1, (size,), generator=gen)
    tokens = torch.randint(1, VOCAB, (size, MAX_LEN), generator=gen)
    tokens[torch.arange(MAX_LEN) >= lengths[:, None]] = PAD
    low = ((tokens >= 1) & (tokens <= 5)).sum(dim=1)
    high = (tokens >= 6).sum(dim=1)
    # class 1: more "low" tokens (1-5) than "high" ones (6-11)
    return tokens, (low > high).long()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=1, keepdim=True)  # (batch, 1) real tokens
    if (counts == 0).any():
        raise ValueError("a sequence contains only padding")
    return (x * mask.unsqueeze(-1)).sum(dim=1) / counts


class TinyClassifier(nn.Module):
    def __init__(self, dim: int = 8, hidden: int = 16,
                 p: float = 0.3) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.hidden = nn.Linear(dim, hidden)
        self.norm = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean(self.embed(tokens), tokens != PAD)
        h = self.drop(torch.relu(self.norm(self.hidden(pooled))))
        return self.out(h)  # logits: unnormalised score per class


def build(seed: int) -> tuple[TinyClassifier, torch.optim.Optimizer]:
    torch.manual_seed(seed)  # before construction: decides init weights
    model = TinyClassifier()
    return model, torch.optim.Adam(model.parameters(), lr=0.01)


def train_step(model: nn.Module, opt: torch.optim.Optimizer,
               tokens: torch.Tensor, targets: torch.Tensor) -> float:
    model.train()
    # cross_entropy expects raw logits and class indices
    loss = F.cross_entropy(model(tokens), targets)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train(model: nn.Module, opt: torch.optim.Optimizer,
          gen: torch.Generator, steps: int) -> list[float]:
    return [train_step(model, opt, *make_batch(gen))
            for _ in range(steps)]


def save_checkpoint(path: Path, model: nn.Module,
                    opt: torch.optim.Optimizer, gen: torch.Generator,
                    step: int) -> None:
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),  # dropout masks
        "data_rng": gen.get_state(),  # which batches come next
    }, path)


def load_checkpoint(path: Path, model: nn.Module,
                    opt: torch.optim.Optimizer, gen: torch.Generator,
                    restore_rng: bool = True) -> int:
    # weights_only=True restricts unpickling; it does not make an
    # untrusted file safe to load. Load only files you trust.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    gen.set_state(ckpt["data_rng"])
    if restore_rng:
        torch.set_rng_state(ckpt["torch_rng"])
    return ckpt["step"]


def same_state(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(x, y) for x, y in
               zip(a.state_dict().values(), b.state_dict().values()))


def resume_matches(restore_rng: bool, steps: int = 6,
                   split: int = 3) -> bool:
    straight, opt = build(seed=0)
    train(straight, opt, torch.Generator().manual_seed(100), steps)

    first, first_opt = build(seed=0)
    gen = torch.Generator().manual_seed(100)
    train(first, first_opt, gen, split)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, first, first_opt, gen, split)
        # a "new process": different init and global RNG position
        resumed, resumed_opt = build(seed=123)
        new_gen = torch.Generator()
        done = load_checkpoint(path, resumed, resumed_opt, new_gen,
                               restore_rng)
    train(resumed, resumed_opt, new_gen, steps - done)
    return same_state(straight, resumed)


def check(ok: bool, name: str) -> None:
    if not ok:
        raise AssertionError(f"FAIL: {name}")
    print(f"PASS: {name}")


def demo_train_eval(tokens: torch.Tensor, targets: torch.Tensor) -> None:
    model, opt = build(seed=0)
    stats = model.norm.running_mean.clone()
    with torch.no_grad():
        check(model.training, "no_grad leaves the model in training mode")
        check(not torch.equal(model(tokens), model(tokens)),
              "dropout stays random under no_grad")
    check(not torch.equal(stats, model.norm.running_mean),
          "train-mode forward leaks batch into BatchNorm stats")
    model.eval()
    stats = model.norm.running_mean.clone()
    with torch.no_grad():
        check(torch.equal(model(tokens), model(tokens)),
              "eval forward is repeatable")
    check(torch.equal(stats, model.norm.running_mean),
          "eval forward leaves BatchNorm stats alone")
    weight = model.out.weight.detach().clone()
    loss = F.cross_entropy(model(tokens), targets)  # eval, grads on
    opt.zero_grad()
    loss.backward()
    opt.step()
    check(not torch.equal(weight, model.out.weight),
          "eval mode does not freeze weights")


def demo_padding() -> None:
    model, opt = build(seed=0)
    emb = model.embed
    short = torch.tensor([[3, 7, PAD, PAD]])
    long = torch.tensor([[3, 7] + [PAD] * 6])
    with torch.no_grad():
        check(not emb.weight[PAD].any(), "padding_idx row starts at zero")
        plain_short = emb(short).mean(dim=1)
        plain_long = emb(long).mean(dim=1)
        check(torch.allclose(plain_short, 2 * plain_long),
              "plain mean divides by padded length")
        check(torch.allclose(masked_mean(emb(short), short != PAD),
                             masked_mean(emb(long), long != PAD)),
              "masked mean ignores padding length")
    all_pad = torch.zeros(1, 4, dtype=torch.long)
    try:
        masked_mean(emb(all_pad), all_pad != PAD)
        check(False, "all-padding sequence is rejected")
    except ValueError:
        check(True, "all-padding sequence is rejected")
    train(model, opt, torch.Generator().manual_seed(1), 5)
    check(not emb.weight[PAD].any(),
          "padding row stays zero (Adam, no weight decay)")

    logits = torch.randn(8, 5)  # flattened (batch * time, classes)
    labels = torch.tensor([1, 2, -100, -100, 3, -100, -100, -100])
    keep = labels != -100
    check(torch.allclose(
        F.cross_entropy(logits, labels, ignore_index=-100),
        F.cross_entropy(logits[keep], labels[keep])),
        "ignore_index averages loss over non-ignored targets only")


def demo_checkpoints(tokens: torch.Tensor) -> None:
    model, opt = build(seed=0)
    train(model, opt, torch.Generator().manual_seed(1), 5)
    alias = model.state_dict()
    snapshot = copy.deepcopy(model.state_dict())
    train(model, opt, torch.Generator().manual_seed(2), 1)
    check(torch.equal(alias["out.weight"], model.out.weight),
          "state_dict() alias follows later updates")
    check(not torch.equal(snapshot["out.weight"], model.out.weight),
          "deepcopy snapshot keeps old values")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, opt, torch.Generator(), step=6)
        restored, restored_opt = build(seed=1)
        load_checkpoint(path, restored, restored_opt, torch.Generator())
    model.eval()
    restored.eval()
    with torch.no_grad():
        check(torch.equal(model(tokens), restored(tokens)),
              "reloaded model gives identical CPU logits")
    check(resume_matches(restore_rng=True),
          "resume with optimizer + RNG state matches straight run")
    check(not resume_matches(restore_rng=False),
          "resume without torch RNG state diverges (dropout)")


def demo_seeds() -> None:
    check(same_state(build(0)[0], build(0)[0]), "same seed, same init")
    check(not same_state(build(0)[0], build(1)[0]),
          "different seed, different init")
    for seed in (0, 1, 2):
        model, opt = build(seed)
        losses = train(model, opt, torch.Generator().manual_seed(100), 30)
        print(f"seed={seed} train loss: "
              f"first={losses[0]:.3f} last={losses[-1]:.3f}")


def main() -> None:
    torch.set_num_threads(1)  # tiny tensors; avoid CPU oversubscription
    print(f"torch {torch.__version__}, CPU only, "
          f"threads={torch.get_num_threads()}")
    tokens, targets = make_batch(torch.Generator().manual_seed(7))
    demo_train_eval(tokens, targets)
    demo_padding()
    demo_checkpoints(tokens)
    demo_seeds()


if __name__ == "__main__":
    main()
