---
layout: post
lang: en
title: "Train/eval, padding, checkpoints and seeds: four experiment traps"
date: 2024-05-16 09:00:00 +0200
permalink: /blog/train-eval-padding-checkpoints-seeds/
author: Rafał Prońko
categories: [machine-learning, pytorch]
description: "Four silent PyTorch experiment bugs on CPU: eval vs no_grad, padding vs masks, state_dict aliasing and resumable checkpoints, and seeds — with a runnable lab and tests."
image: /assets/images/train-eval-padding-checkpoints-seeds.svg
---

# Train/eval, padding, checkpoints and seeds: four experiment traps

![Four panels: model.eval() and torch.no_grad() as separate switches; a padded sequence where a plain mean divides by the padded length and a masked mean by the real tokens; a checkpoint holding model, optimizer, torch RNG and data RNG state; seeds set before the model is built and reported across several runs.]({{ '/assets/images/train-eval-padding-checkpoints-seeds.svg' | relative_url }})

<time datetime="2024-05-16">16 May 2024</time>

A crashing experiment is honest: it tells you something is wrong. The harder bugs run to the end, write a number to the log and leave you comparing two runs that were never comparable.

This article covers four of them in PyTorch:

1. The model is in the wrong mode — or in the right mode for the wrong reason.
2. Padding leaks into a computation that was supposed to ignore it.
3. A "checkpoint" is either a reference to live weights or a file that cannot actually resume training.
4. A seed is set, so the run is assumed to be reproducible.

None of them raises an exception by default. Each can be checked with a small test on a tiny CPU model. That's the plan: what each switch really does, and a small lab that checks it.

## The setup

I use Python 3.11, `torch==2.2.2` and `numpy<2`, CPU only. PyTorch 2.2.2 is a bug-fix release from 27 March 2024; everything below uses APIs and defaults documented for the 2.2 line. That matters for one of the traps — `torch.load` behaves differently in later versions.

A note on dates. The 16 May 2024 date on this post is an editorial label, not proof that it was published then. I actually ran the lab in October 2026: the PyTorch pin is historical, but the tooling is not. The `uv` project file with a `[dependency-groups]` table, the `uv run` commands and the transitive dependencies in `uv.lock` (sympy, networkx, filelock and others) are what uv resolved and ran in October 2026. Don't read them as a setup that was available in May 2024. The same goes for the documentation links: they point to the archived 2.2 docs on the current docs.pytorch.org host.

The data is synthetic. Each example is a sequence of up to eight token IDs from a vocabulary of twelve; ID `0` is padding. The label says whether the sequence has more "low" tokens (1–5) than "high" ones (6–11). The task is trivial on purpose. I'm not measuring a model here; I'm checking the mechanics around it.

The model is just as small: an embedding, a mean over tokens, a linear layer, `BatchNorm1d`, ReLU, `Dropout` and an output layer. BatchNorm and Dropout are there because they are exactly the layers whose behaviour depends on the mode.

The complete lab is at the end of this article. You can also download it with tests: [`pyproject.toml`]({{ '/assets/examples/article-31/pyproject.toml' | relative_url }}), [`lab.py`]({{ '/assets/examples/article-31/lab.py' | relative_url }}), [`tests/test_lab.py`]({{ '/assets/examples/article-31/tests/test_lab.py' | relative_url }}) and a short [`README.md`]({{ '/assets/examples/article-31/README.md' | relative_url }}).

## What one training step actually does

Before the traps, the vocabulary. One step in the lab looks like this:

```python
def train_step(model, opt, tokens, targets):
    model.train()
    # cross_entropy expects raw logits and class indices
    loss = F.cross_entropy(model(tokens), targets)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()
```

- **Inputs** (`tokens`) are a `(batch, length)` tensor of integer token IDs, padded to a common length.
- **Targets** are a `(batch,)` tensor of class indices, `0` or `1`. Not one-hot vectors, not probabilities.
- **Parameters** are the tensors the optimizer owns: embedding rows, linear weights and biases, BatchNorm's scale and shift. BatchNorm's running mean and variance are *buffers*, not parameters. The optimizer never touches them, yet they change during training. Remember that; it comes back twice.
- **Logits** are what the model returns: one unnormalised score per class. `F.cross_entropy` applies log-softmax internally and expects exactly these raw scores. If I apply softmax in the model and then call `cross_entropy`, nothing crashes; the loss is simply computed on the wrong quantity.
- `loss.backward()` fills `.grad` on every parameter that took part in the computation. `opt.step()` updates the parameters **in place**. `opt.zero_grad()` clears the gradients, because PyTorch accumulates them by default.

At inference I use the same logits. `logits.argmax(dim=1)` gives the predicted class. If I need probabilities, I call `softmax` explicitly at that point — not inside the model that is trained with `cross_entropy`.

## Trap 1: `model.eval()` and `torch.no_grad()` are different switches

The two are easy to conflate, and the PyTorch 2.2 autograd notes address it directly: evaluation mode "is not a mechanism to locally disable gradient computation". The switches are orthogonal.

- `model.train()` / `model.eval()` flip the `training` flag on the module and all its children. Only modules that read that flag change behaviour. In this model those are Dropout and BatchNorm.
- `torch.no_grad()` stops autograd from recording operations. It does not know about modules at all.

So each one does half of what people expect.

**`no_grad` does not switch the mode.** Inside `with torch.no_grad():` the model is still in training mode if it was before. Dropout still drops a random subset of activations — two forward passes on the same batch give different logits. BatchNorm still normalises with statistics of the current batch and, more importantly, **updates its running mean and variance**.

That last part is a leak. If I run a validation batch through a model in training mode, even under `no_grad`, the validation data's statistics flow into the buffers with the default momentum of 0.1. Nothing is learned through gradients, but the model I save afterwards is no longer the model I trained. The lab checks exactly that: a forward pass in train mode under `no_grad` changes `running_mean`; the same pass in eval mode leaves it untouched.

**`eval()` does not freeze weights.** In eval mode, autograd still records the graph. If I call `backward()` and `opt.step()`, the weights change. The lab does that on purpose and checks that `out.weight` moved. "I set eval, so the model is frozen" is not true.

Freezing is its own decision. The most robust form is not to give those parameters to the optimizer at all. `requires_grad_(False)` stops new gradients from being computed, but on its own it is not a guarantee: if a parameter is still in the optimizer and carries a stale `.grad` from an earlier backward pass, the next `opt.step()` will apply it. If I freeze that way, I also set the old gradients to `None` (`opt.zero_grad(set_to_none=True)`, the default in 2.2) before the next step.

The assumptions behind the two mode-dependent layers are worth spelling out, because they explain why the mode matters:

- **Dropout** in training zeroes each element with probability `p` and scales the survivors by `1/(1-p)`, so the expected activation stays the same. In eval mode it is the identity. The model is trained under noise and evaluated without it. Evaluating in train mode gives a noisy, randomly perturbed model — and a different one every call.
- **BatchNorm** in training normalises with the current batch's mean and variance and updates running estimates. In eval it uses the running estimates. That assumes training batches are large and representative enough for those estimates to make sense. For `BatchNorm1d` on 2D input `(batch, features)`, as in this model, a training-mode batch of one has a single value per channel, and PyTorch raises an error instead of normalising. Small or very homogeneous batches don't fail; they just give noisy statistics and poor running estimates.

For inference I want both switches, and I want them explicitly:

```python
model.eval()
with torch.no_grad():
    logits = model(tokens)
```

And because `eval()` is a state that persists on the object, every training step sets `model.train()` again. In the lab that call is the first line of `train_step`. Otherwise an evaluation in the middle of training silently turns off Dropout for the rest of the run.

## Trap 2: `padding_idx` is not a mask

Batches of sequences are padded to a common length. The padding token then has to be ignored by every computation that should not depend on how much padding there is. PyTorch gives me one tool that looks like it handles this and doesn't.

The 2.2 documentation for `nn.Embedding` says two things about `padding_idx`: the embedding at that index does not contribute to the gradient and stays fixed, and for a newly constructed embedding it defaults to all zeros. That's all. It says nothing about pooling, attention or loss.

A zero vector seems harmless. Summing zeros adds nothing. But averaging is a sum **divided by a count**, and `x.mean(dim=1)` divides by the padded length.

Take the same two real tokens, `[3, 7]`, padded once to length 4 and once to length 8. With a zero padding row, the plain mean of the first is exactly twice the plain mean of the second. The numerator is right; the denominator is wrong. The model's representation of a sentence now depends on the longest sentence in the batch — which depends on how batches were formed. Shuffle differently, get different features.

A masked mean divides by the number of real tokens:

```python
def masked_mean(x, mask):
    counts = mask.sum(dim=1, keepdim=True)  # (batch, 1) real tokens
    if (counts == 0).any():
        raise ValueError("a sequence contains only padding")
    return (x * mask.unsqueeze(-1)).sum(dim=1) / counts
```

Two details here are decisions, not boilerplate.

First, I multiply by the mask instead of relying on the padding row being zero. `padding_idx` only guarantees a zero row at construction. If someone loads pretrained embeddings, re-initialises the weight or writes to it directly, the row can be something else — and since it receives no gradient, training will not pull it back to zero. The mask makes the pooling correct regardless.

Second, a row with no real tokens is rejected. Dividing by zero would produce `NaN`, which can contaminate the current batch's loss and BatchNorm buffers. Clamping the count to 1 would silently return a zero vector and let the classifier produce a prediction for an empty input. I'd rather find out where the empty sequences come from. That's a choice; if your data legitimately contains empty inputs, handle them explicitly and visibly, not with a hidden `clamp`.

The lab checks the full chain: the padding row starts at zero, the plain mean depends on padding length, the masked mean doesn't, an all-padding row raises `ValueError`, and after a few Adam steps the padding row is still zero because it never received a gradient. That last check holds for Adam without weight decay; I don't generalise it to every optimizer configuration.

### `ignore_index` and attention masks are different contracts

The same confusion shows up in token-level tasks such as tagging or language modelling, in a different form.

`F.cross_entropy(..., ignore_index=-100)` is a contract about the **loss**. Targets equal to `-100` contribute nothing to the loss or the gradient, and with `reduction="mean"` the average is taken over the remaining targets only. The lab checks that, for a batch with some real targets, this equals a plain cross-entropy over the non-ignored positions. A batch in which every target is ignored leaves nothing to average; I don't test that case here, so if it can happen in your data, guard against it explicitly before the loss.

It says nothing about the **forward pass**. The model has already processed the padded positions by the time the loss sees them. If those positions were visible to other tokens — through attention, pooling or a recurrent state — they have already influenced the predictions for real tokens. Preventing that is the job of an attention or pooling mask inside the model.

So I keep two separate things, built from the same lengths:

- an input-side mask (`tokens != PAD`) used by pooling or attention,
- a target-side label of `-100` at every position that should not be scored.

They often coincide, but not always. In a language model the last real token has no next-token target, yet it is not padding. Deriving one from the other "because they are the same" is how off-by-one bugs get in.

## Trap 3: a checkpoint that is not a checkpoint

### `state_dict()` returns references

The 2.2 documentation for `Module.state_dict()` is explicit: the returned object is a shallow copy that contains references to the module's parameters and buffers. And optimizers update parameters in place.

So this "best model so far" snapshot is not a snapshot:

```python
best = model.state_dict()   # references, not values
train(model, opt, gen, 1)
# best["out.weight"] is now the *current* weight
```

The keys belong to a new dictionary, but every value points at the same storage as the live tensor. After the next optimizer step, `best` contains the new weights. Early stopping built this way restores the last epoch and reports it as the best one. The lab checks both sides: the alias follows the update, and `copy.deepcopy(model.state_dict())` keeps the old values.

Writing the state to disk with `torch.save` also breaks the alias, because the file holds values. The trap is specifically the in-memory "keep the best one" pattern.

### Inference needs weights; resume needs everything

There are two reasons to save a checkpoint, and they need different contents.

For **inference** I need the model's `state_dict` — parameters and buffers. BatchNorm's running statistics are buffers, so they are included; that's why the earlier leak ends up in saved files.

For **resuming training** that's not enough. To continue a run as if it had not stopped, I need:

- the model state,
- the **optimizer** state — for Adam, the running first and second moments and the step count per parameter. Without them, the first steps after a restart behave like the first steps of training,
- the **scheduler** state, if there is one, otherwise subsequent learning-rate updates may differ because the scheduler's position and history are lost (the lab has no scheduler; with one, add `scheduler.state_dict()`),
- the **RNG state** that drives Dropout masks and any random augmentation,
- the **data order state** — which examples remain in the current epoch and in what order. In the lab this is simple: batches are drawn from a dedicated `torch.Generator`, so its state is enough. With a `DataLoader` there is no single universal recipe. Saving the loader's generator and a step count is not sufficient in general: I need the sampler's permutation for the current epoch and a cursor into it, and with worker processes and prefetching, some batches have already been produced but not consumed. Either the loader has to be rebuilt to skip exactly the consumed indices, or I accept that a resumed epoch is not identical and say so,
- the step or epoch counter.

```python
torch.save({
    "step": step,
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "torch_rng": torch.get_rng_state(),  # dropout masks
    "data_rng": gen.get_state(),  # which batches come next
}, path)
```

The lab then tests the property that matters: a run of six steps straight through ends with exactly the same parameters as a run of three steps, a checkpoint, a "new process" with a different seed and fresh objects, a load, and three more steps. On CPU, in one environment, `torch.equal` holds for every tensor in the state.

The negative test is just as important. If I restore model, optimizer and data order but **not** the torch RNG state, the resumed run diverges, because Dropout draws different masks. Nothing errors. The loss curve looks plausible. It is simply a different run.

### Loading: `weights_only=True`, explicitly

In PyTorch 2.2 the signature is `torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=False, ...)`. The default is `False`, which means full pickle, which can execute arbitrary code from the file. The documentation's own advice is to load only data you trust.

So I load with:

```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
```

`weights_only=True` restricts the unpickler to tensors, primitive types and dictionaries (plus lists and tuples, which the optimizer state uses). Everything I store above fits: RNG states are `ByteTensor`s, the step is an `int`, optimizer state is nested dicts of tensors and numbers. The test suite also checks the other direction: a checkpoint containing a dataclass is refused with `pickle.UnpicklingError`. That's useful as a diagnostic — it tells me when someone has started storing objects in checkpoints that will break on refactors.

**Trust warning.** `weights_only=True` is not a licence to load checkpoints from unknown sources. It reduces what a file I already trust is allowed to contain; it is not a security review of the file. Where a checkpoint came from, and whether it could have been tampered with, is a separate question that I answer before calling `torch.load` at all.

`map_location="cpu"` makes the load independent of the device the checkpoint was saved on.

The final check in this section is prediction parity: the original model and a model built with a different seed and then loaded from the checkpoint produce identical logits on the same batch, both in eval mode, on CPU. If that check fails in a real project, the usual suspects are a missing buffer (`strict=False` hiding it), a model left in train mode, or preprocessing that differs between the two paths.

## Trap 4: a seed is not a reproducibility guarantee

### Seed before construction

Initial weights are drawn from the global RNG when the module is **constructed**. A seed set after `Model()` affects the next random operation, not the weights that already exist. In the lab, `build(seed)` calls `torch.manual_seed(seed)` and only then constructs the model and optimizer. Two `build(0)` calls give identical weights; `build(1)` gives different ones; and a test checks that calling `manual_seed` after construction leaves the weights unchanged.

### Seed at start is not RNG state at step N

Setting a seed at the start of a run says where the RNG started, not where it is now. After three steps, Dropout has consumed random numbers; after a restart with the same seed, the RNG would be back at the beginning. That's why the resumable checkpoint stores `torch.get_rng_state()` and the data generator's state, not just the original seed.

### What the documentation actually promises

The 2.2 reproducibility notes are careful, and I quote them rather than paraphrase: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."

The same page describes the further steps for one environment: `torch.use_deterministic_algorithms(True)`, which makes PyTorch select deterministic implementations of operations where they exist and raise an error for affected operations that have no deterministic implementation, and a `worker_init_fn` plus a seeded `generator` for multi-worker `DataLoader`s. My lab runs single-process on CPU, with one intra-op thread, and doesn't use them. A GPU training job very likely needs them.

So even "same seed, same result" is not unconditional. What I can say is narrower: in my environment — this code, these versions, this CPU, one thread — repeated runs with the same seed produced identical tensors in the checks above. That is an observation about one setup, not a guarantee I can hand to another machine.

### Several seeds, reported together

A seed fixes one sample from the distribution of possible runs. If I try ten seeds and report the best, I'm reporting the tail of that distribution. The lab ends by training three seeds on the same data order and printing the first and last training loss for each, all of them. In a real comparison I'd evaluate each seed on a validation set and report the mean and spread — and choose seeds before looking at results.

The printed losses are a sanity trace on a toy task, not a benchmark. They differ between seeds, and that's the point: seed is part of the experiment's variance, not a constant to tune.

## The complete lab

Save as `lab.py` next to the `pyproject.toml` below. Every function the checks use is in this file.

```python
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
```

```toml
[project]
name = "train-eval-traps-lab"
version = "0.1.0"
description = "CPU lab: train/eval, padding, checkpoints and seeds in PyTorch 2.2.2"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch==2.2.2",
    "numpy<2",
]

[dependency-groups]
dev = ["pytest>=8.0,<8.2"]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`main()` only calls one demonstration per trap: `demo_train_eval`, `demo_padding`, `demo_checkpoints` and `demo_seeds`. It also sets `torch.set_num_threads(1)`. I use one intra-op thread to limit CPU oversubscription. The test module does the same.

Run it with:

```bash
uv run python lab.py
uv run pytest -q
```

This is the output of `lab.py` from my run (CPU, Python 3.11.12, torch 2.2.2, numpy 1.26.4):

```text
torch 2.2.2, CPU only, threads=1
PASS: no_grad leaves the model in training mode
PASS: dropout stays random under no_grad
PASS: train-mode forward leaks batch into BatchNorm stats
PASS: eval forward is repeatable
PASS: eval forward leaves BatchNorm stats alone
PASS: eval mode does not freeze weights
PASS: padding_idx row starts at zero
PASS: plain mean divides by padded length
PASS: masked mean ignores padding length
PASS: all-padding sequence is rejected
PASS: padding row stays zero (Adam, no weight decay)
PASS: ignore_index averages loss over non-ignored targets only
PASS: state_dict() alias follows later updates
PASS: deepcopy snapshot keeps old values
PASS: reloaded model gives identical CPU logits
PASS: resume with optimizer + RNG state matches straight run
PASS: resume without torch RNG state diverges (dropout)
PASS: same seed, same init
PASS: different seed, different init
seed=0 train loss: first=0.869 last=0.551
seed=1 train loss: first=1.010 last=0.697
seed=2 train loss: first=0.987 last=0.600
```

The per-seed numbers are a training-loss trace on synthetic data with Dropout active, not an evaluation. What they show is that three seeds give three different runs on identical data — enough to make the point about reporting several seeds, and nothing more. I ran this in one environment only; on another machine or PyTorch version I would rerun the checks rather than assume them, and I would not expect the exact loss values to match.

The separate test file covers the same properties one by one, plus a few that don't fit a linear script: `eval()` does not disable autograd (`logits.requires_grad` is still `True`), the full model's logits agree within `torch.allclose` tolerances when the same sequence is padded to two different lengths, `weights_only=True` refuses a checkpoint containing a dataclass, and a seed set after construction leaves the weights unchanged.

## Diagnostic tests worth copying

The lab's checks are generic enough to port to a real project. The ones I'd add first:

1. **Mode assertions.** A forward pass in eval mode under `no_grad` twice gives identical logits; BatchNorm buffers and `num_batches_tracked` are unchanged after the evaluation loop.
2. **Padding invariance.** The same sequence padded to two lengths gives the same logits in eval mode. An all-padding sequence raises an error instead of producing `NaN` or a silent zero vector.
3. **Loss masking.** `cross_entropy` with `ignore_index` equals the loss over valid targets only — and a separate test that the forward pass uses an input mask.
4. **Snapshot independence.** After one more optimizer step, the "best" checkpoint in memory still has its old values.
5. **Round trip.** Save, load with `weights_only=True` into a model built with a different seed, compare eval logits with `torch.equal` on CPU.
6. **Resume equivalence.** N steps straight equals k steps + checkpoint + N−k steps in a fresh object, on a tiny model. And the negative: the same without RNG state differs. The negative test shows that the positive one isn't passing by accident.
7. **Seed placement.** Two builds with the same seed give identical weights; a seed set after construction changes nothing.

They use tiny models and synthetic batches, so they are cheap compared with a real training run. How cheap depends on the machine, so measure it before you put them in CI.

## Limits

This is a CPU-only lab with one tiny model. The equality checks hold in the environment described above; I don't claim them for CUDA, MPS, other PyTorch versions or other machines, and the documentation doesn't either.

The checkpoint here is written with a plain `torch.save`. I don't address crash consistency — a process killed while writing, partial files, atomic rename, keeping the previous good checkpoint. Those matter for long runs and are a separate topic.

The resume test covers model, Adam state, the global torch RNG and a dedicated data generator. It doesn't cover LR schedulers, `DataLoader` samplers, workers and prefetching, gradient scalers for mixed precision or Python/NumPy RNGs used by data augmentation. Each of them adds state that has to be saved and has a test of the same shape.

Nothing here is an accuracy result. The task is synthetic and the lab measures behaviour of the training machinery, not the quality of a model.

### Sources

- [PyTorch 2.2.2 release](https://github.com/pytorch/pytorch/releases/tag/v2.2.2) — bug-fix release of 27 March 2024.
- [PyTorch 2.2: Reproducibility](https://docs.pytorch.org/docs/2.2/notes/randomness.html) — limits of seeding, deterministic algorithms, DataLoader seeding.
- [PyTorch 2.2: Autograd mechanics](https://docs.pytorch.org/docs/2.2/notes/autograd.html) — evaluation mode versus no-grad and inference mode.
- [PyTorch 2.2: `torch.nn.Module`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.Module.html) — `train`/`eval`, `state_dict` returning references, `load_state_dict`.
- [PyTorch 2.2: `torch.nn.Embedding`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.Embedding.html) — `padding_idx`.
- [PyTorch 2.2: `torch.load`](https://docs.pytorch.org/docs/2.2/generated/torch.load.html) — `weights_only` (default `False` in 2.2), `map_location`, pickle warning.
- [PyTorch 2.2: `BatchNorm1d`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.BatchNorm1d.html), [`Dropout`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.Dropout.html), [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.CrossEntropyLoss.html) — mode-dependent behaviour, logits input and `ignore_index`.
- [PyTorch 2.2: Serialization semantics](https://docs.pytorch.org/docs/2.2/notes/serialization.html) — saving `state_dict` instead of whole modules.
