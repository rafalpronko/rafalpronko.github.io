---
layout: post
lang: en
title: "QLoRA and quantization: the memory–quality trade-off"
date: 2024-07-11 09:00:00 +0200
permalink: /blog/qlora-quantization-memory-quality/
author: Rafał Prońko
categories: [machine-learning]
description: "How QLoRA stores a frozen base in 4 bits, computes in 16 bits and trains small adapters — what it saves, what it doesn't, with a NumPy demo."
image: /assets/images/qlora-quantization.svg
---

# QLoRA and quantization: the memory–quality trade-off

![Diagram of one QLoRA linear layer: the frozen base weight is stored as 4-bit NF4 codes with constants, dequantized to BF16 for the matrix multiplication, and added to the output of two small trainable LoRA matrices. Gradients update only the adapters, but still flow through the dequantized base weight to earlier layers.]({{ '/assets/images/qlora-quantization.svg' | relative_url }})

<time datetime="2024-07-11">11 July 2024</time>

*Editorial note: 11 July 2024 is a proposed editorial date for this post, not a record of historical publication. The example was last run on 13 October 2026 (details below).*

"Fine-tune a big model on one GPU" is an attractive promise. QLoRA is often summarized exactly like that, and the summary hides the interesting part: which memory actually goes away, which stays, and what I have to measure before I believe the model is as good as one trained without quantization.

In this article I go through the mechanism, not the marketing. I'll look at what QLoRA keeps frozen, what it trains, in which precision it computes, and where its memory savings come from. Then I'll run a small CPU experiment with blockwise quantization to see why block size and outliers matter, and finish with how I would design a fair quality comparison.

The source for the method is the QLoRA paper by Dettmers et al., submitted to arXiv on 23 May 2023. Every number I quote from it is the authors' result in their setup, not my benchmark.

## Two ideas stacked on top of each other

QLoRA combines two things that are worth separating.

**LoRA** freezes the pretrained weight matrix and learns a low-rank correction next to it. For one linear layer with input `X` of shape `b × h` and weight `W` of shape `h × o`, the paper writes the output as:

```text
Y = X·W + s·X·L1·L2
```

Here `b` is the number of rows fed through the layer (batch items or tokens), `h` is the input width, `o` is the output width and `r` is the adapter rank. `L1` has shape `h × r`, `L2` has shape `r × o`, `s` is a scalar scaling factor, and the output `Y` has shape `b × o`. The rank `r` is small compared with `h` and `o`, so the two adapters together have `r·(h + o)` parameters instead of `h·o`. Only `L1` and `L2` are trained.

**Quantization** stores `W` with fewer bits per value. In QLoRA the frozen `W` is stored in 4 bits.

The combination is what matters: because `W` is never updated, it doesn't need to be stored in a precision suitable for training. It only needs to be good enough to compute with.

## Storage precision is not compute precision

The paper is explicit that QLoRA has one *storage* data type and one *computation* data type. Storage is usually 4-bit NormalFloat (NF4); computation is usually BFloat16. Whenever a weight tensor is used, it is dequantized to BF16 and the matrix multiplication runs in 16 bits.

So the 4-bit weights are not multiplied as 4-bit numbers. They are a compressed representation that is unpacked on demand. That has two consequences that I keep in mind:

- the dequantized copy is temporary working memory, not free;
- the dequantization is extra work on every forward and backward pass.

The adapters `L1` and `L2` are kept in BF16 in the paper's setup. They are small, and they are the only weights that receive updates.

## A frozen base still carries gradients

A common misunderstanding is that a frozen base means "no backpropagation through the base". That's not the case.

The paper puts it this way: parameter updates need only the gradient of the error with respect to the adapters, `∂E/∂Lᵢ`, not with respect to the 4-bit weights. But computing the adapter gradients still requires going backwards through the base layers, using the BF16 dequantized `W`.

Think of a stack of layers, each with its own adapters. To compute gradients for the adapters in layer 3, the error signal from layers 4, 5, 6 and beyond must pass back through those layers. Each pass goes through `X·W` as well as through the adapter branch. The gradient with respect to the *input* of a layer always involves `W`.

What QLoRA saves is the gradient with respect to `W` itself and the optimizer state for `W`. It does not save the backward pass through the base.

## NF4: a codebook shaped like the weights

Most quantizers I see in tutorials are uniform: they divide the range into equally spaced steps. NF4 is not uniform.

The paper builds it from quantile quantization. The idea is that each quantization bin should receive the same expected number of values. Pretrained weights are, in the authors' framing, approximately zero-centered and normally distributed, so the bins are placed at quantiles of a standard normal distribution and normalized into `[-1, 1]`. The levels are dense near zero, where most weights are, and sparse in the tails.

Two details from the paper are worth knowing:

- **Exact zero.** A symmetric quantile construction has no exact representation of zero. NF4 is made deliberately asymmetric — `2^(k-1)` quantiles for the negative side and `2^(k-1) + 1` for the positive side, with one duplicate zero removed — so zero is exactly representable and all 16 codes are used.
- **Blockwise absmax scaling.** Before mapping to the codebook, each block of weights is divided by its absolute maximum. The codebook is fixed; the per-block scale (the "quantization constant") carries the magnitude.

That last point is where an outlier hurts. If one weight in a block is much larger than the others, the absmax of that block grows, and everything else in the block is squeezed into a small part of the codebook.

The authors report that NF4 gave better results than 4-bit floats and 4-bit integers in their experiments. I take that as their empirical finding; it's one reason I don't treat "4-bit" as a single thing.

## A CPU experiment: uniform blockwise quantization

To get a feel for block size and outliers, I wrote a small NumPy script. It is important to be clear about what it is:

- it is **uniform symmetric absmax** quantization, not NF4;
- it does not use bitsandbytes, PEFT or a GPU;
- it does not train anything;
- the memory figures it prints are **theoretical packed sizes**, computed by arithmetic, not measured allocations.

With 4 bits and symmetric codes, the script uses the codes `-7 … 7`. That gives an exact zero but uses only 15 of the 16 possible values — exactly the waste NF4's asymmetric construction avoids.

The core is two functions:

```python
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
```

The validation is deliberately narrow, because this is a demo, not a library. `quantize_blockwise` raises `ValueError` for an empty array, `nan` or `inf` values, and `bits` or `block_size` that are not integers in range. It also raises `ValueError` for a nonzero block whose float32 scale `absmax / qmax` underflows to 0. That only happens for blocks containing nothing but the tiniest float32 subnormals (for example the smallest positive float32, about 1.4e-45), and the script simply doesn't represent them; the check runs before the division by the scales, so there is no divide-by-zero or `nan`. This guarantee assumes NumPy's default floating-point error settings, in which underflow is ignored. If the caller sets `np.errstate(under='raise')`, computing the scale can raise `FloatingPointError` before the check runs; enabling underflow warnings and treating them as errors can likewise interrupt it before the check. `dequantize_blockwise` raises `ValueError` unless `codes` is 2-D with one finite positive scale per block and the size of `shape` is consistent with the number of blocks. That check is loose in both directions, because the original length is not stored: any size that falls within the last block passes. For 130 values in blocks of 64, any size from 129 to 192 is accepted, so a smaller `shape` silently truncates data and a larger one returns padding zeros as if they were values. Other malformed inputs, such as complex values or a non-integer `shape`, are not checked and may fail with a different error or be converted silently. The tests cover each `ValueError` case listed here, including non-2-D `codes` and a non-integer or `bool` outlier index. (`_is_int` is a small helper in the script that accepts integers but not `bool`.)

The experiment draws 4096 synthetic weights from a normal distribution with standard deviation 0.02, then sets one weight to 1.0 — about 50 standard deviations. It quantizes both versions with different block sizes and reports the RMSE over the *other* 4095 weights, plus how many of the outlier's block neighbors end up at code 0.

### Download and run

The complete example is four files. Keep this layout:

```text
qlora-quantization/
├── pyproject.toml
├── uv.lock
├── quant_demo.py
└── tests/
    └── test_quant_demo.py
```

- [quant_demo.py]({{ '/assets/examples/qlora-quantization/quant_demo.py' | relative_url }})
- [tests/test_quant_demo.py]({{ '/assets/examples/qlora-quantization/tests/test_quant_demo.py' | relative_url }})
- [pyproject.toml]({{ '/assets/examples/qlora-quantization/pyproject.toml' | relative_url }})
- [uv.lock]({{ '/assets/examples/qlora-quantization/uv.lock' | relative_url }})

The commands need [uv](https://docs.astral.sh/uv/getting-started/installation/), a modern Python package and project manager; the link is the official installation guide. From the `qlora-quantization` directory:

```bash
uv run --isolated --locked --python 3.12 python quant_demo.py
uv run --isolated --locked --python 3.12 pytest -v
```

`--locked` makes uv fail instead of silently re-resolving if the lock file doesn't match `pyproject.toml`. `--isolated` runs in a temporary environment, so no virtual environment is created in the project directory. Caches may still be written even in isolated mode: uv's own package cache, and Python's `__pycache__` and pytest's `.pytest_cache` next to the files.

A note on dates. The concepts here — LoRA, NF4, double quantization, paged optimizers — were all public before this article's proposed editorial date: the QLoRA paper appeared in May 2023, and the PEFT v0.11.0 quantization guide I cite was released in May 2024. The *execution* of the example is different: I first ran it on 12 October 2026 and re-ran the corrected versions on 13 October 2026, all dates in Warsaw local time (CEST, UTC+2; the evidence logs record UTC, so the 13 October runs carry late 12 October UTC timestamps), every time with uv 0.7.8, CPython 3.12.11, NumPy 2.5.4 and pytest 9.1.1 on macOS (arm64). Those versions did not exist in July 2024. I only tested Python 3.12. The script only uses basic NumPy operations, but the run I recorded is the modern one above, and I'm not claiming results for other Python or NumPy versions.

### What the run printed

This is the exact output of `quant_demo.py` from the final 13 October re-run (every earlier run printed the same):

```text
1) One outlier, 4-bit uniform symmetric absmax; RMSE over the other weights
   'zero' = other weights in the outlier's block mapped to code 0
 block  rmse clean  rmse +outlier  zero clean  zero +outlier
  4096     0.00324        0.01994    851/4095      4093/4095
   256     0.00245        0.00558      44/255        255/255
    64     0.00211        0.00325        6/63          63/63
    16     0.00168        0.00207        1/15          15/15

2) Scale overhead per parameter (bits)
   block   16, FP32 scale: 2.0000
   block   64, FP32 scale: 0.5000
   block  256, FP32 scale: 0.1250
   block 4096, FP32 scale: 0.0078
   block 64, double quantization: 0.1270 (saves 0.3730)

3) Theoretical weight-only storage for 7,000,000,000 parameters (GiB)
   16-bit weights:                       13.04
   4-bit codes + FP32 scales (block 64):   3.67
   4-bit codes + double-quant scales:     3.36
   Excludes activations, gradients, optimizer state, buffers, workspaces.

   This demo stores codes as int8: 4096 B for 4096 codes vs 2048 B if packed two per byte.

4) LoRA on one 4096x4096 projection
   r= 8: L1 4096x8, L2 8x4096, 65,536 trainable = 0.39% of W
   r=16: L1 4096x16, L2 16x4096, 131,072 trainable = 0.78% of W
   r=64: L1 4096x64, L2 64x4096, 524,288 trainable = 3.12% of W
```

The `3.12%` is rounded output; the exact value is 524,288 / 16,777,216 = 3.125%.

And the tests: 37 collected, 37 passed.

### Reading the outlier table

With one block of 4096, the outlier sets the scale for the whole tensor. The scale becomes `1.0 / 7`, so a weight needs a magnitude above half of that step — roughly 0.07, or about 3.5 standard deviations here — to avoid code 0. Almost none do: 4093 of the 4095 other weights collapse to zero, and RMSE over the ordinary weights grows from 0.00324 to 0.01994.

With blocks of 16, the outlier still wipes out its block — 15 of 15 neighbors go to zero — but that block contains only 15 other weights. The rest of the tensor is quantized with its own, unaffected scales, and the RMSE moves from 0.00168 to 0.00207. One test checks the confinement on a separate small sample (512 weights, seed 3, blocks of 64), not on the table's data: it compares reconstructions, not codes, and the blocks that don't contain the outlier reconstruct to identical values with and without it.

Smaller blocks also reduce error on the clean data (0.00324 → 0.00168 from 4096 down to 16), because each scale fits a smaller neighborhood.

So blockwise quantization doesn't make outliers harmless. It confines the damage to one block.

What this table does *not* tell me is how NF4 would behave on the same data, or how a real model's loss would change. It is a tensor-level reconstruction error on synthetic weights, nothing more.

## Double quantization: paying for the scales

Small blocks have a price: one scale per block. Section 2 of the output shows it. With FP32 scales and blocks of 64, the scales alone add 0.5 bits per parameter — on top of 4 bits for the codes. With blocks of 16 the overhead would be 2 bits per parameter.

The paper's **double quantization** quantizes those scales themselves. The first-level constants are stored as 8-bit floats in blocks of 256, with one FP32 constant per block of 256 (the paper also subtracts the mean of the constants before quantizing them). For blocks of 64 the overhead becomes:

```text
8/64 + 32/(64·256) = 0.127 bits per parameter
```

That's the 0.1270 in the output and the 0.373 bits saved per parameter that the paper reports. In my calculation the mean-offset metadata is omitted, as the code comment says.

The authors describe the quality effect of double quantization as minor; its value is the extra memory headroom that lets certain model sizes fit into certain GPUs.

## Paged optimizers: a safety valve, not more memory

The third QLoRA component is easy to misread. Paged optimizers use NVIDIA unified memory so that optimizer state can be evicted to CPU RAM when the GPU runs out of memory, and paged back when the optimizer update needs it. The authors introduced it to survive memory spikes, which they associate with long sequences during gradient checkpointing.

Three things follow from that description:

- It moves optimizer state between GPU and CPU. It doesn't make GPU memory bigger, and CPU RAM is finite too.
- Paging costs transfer time. The authors report that for a 65B model on 48 GB GPUs with batch size 16 they saw the same training speed as with regular optimizers, but they also say they don't provide hard measurements and that future work should characterize when slow-downs occur.
- In QLoRA, optimizer state exists only for the adapters, so it is already much smaller than in full fine-tuning.

I treat paging as protection against occasional spikes. If a run pages constantly, I have a memory budget problem to fix, not a feature to rely on.

## Memory accounting: the weights are only one line

Section 3 of the output is a deliberately narrow calculation: 7 billion weights, every one of them quantized, nothing else counted.

| Weight storage only | GiB |
| --- | ---: |
| 16-bit weights | 13.04 |
| 4-bit codes + FP32 scales, block 64 | 3.67 |
| 4-bit codes + double-quantized scales | 3.36 |

That is roughly 4.5 bits per parameter with FP32 scales and about 4.13 with double quantization. It's a useful lower bound for the frozen base. It is not the memory a training run needs.

A real fine-tuning run also holds:

- **Adapter weights** — small, but in 16-bit (or higher) precision.
- **Adapter gradients** and **optimizer state** — for Adam, typically two extra values per trainable parameter.
- **Activations** saved for the backward pass. These scale with batch size, sequence length, hidden size and number of layers, and they don't shrink because the weights are 4-bit. Gradient checkpointing trades them for recomputation.
- **Temporary dequantized weights** and other workspaces used by the kernels.
- **Mixed-precision overhead**, such as copies of values in different dtypes, plus framework and runtime overhead.

Section 4 shows why the adapter side is cheap. On one `4096 × 4096` projection, `r = 64` gives `L1` of `4096 × 64` and `L2` of `64 × 4096`: 524,288 trainable parameters, exactly 3.125% of `W` (printed as 3.12%). At `r = 8` it's 0.39%. Multiply by the number of adapted layers to get the trainable total.

Two warnings about my own example. First, the demo stores codes as `int8`, one byte per 4-bit code: 4096 bytes for 4096 codes, against 2048 bytes if packed two per byte. The 3.67 and 3.36 GiB figures are packed-size arithmetic, not what NumPy allocates. Second, the calculation assumes every weight is quantized. Real models contain layers that a given library may keep in higher precision, so I check what my setup actually quantizes before trusting a number.

The paper's headline figure — the memory needed to fine-tune a 65B model going from more than 780 GB of GPU memory to less than 48 GB — is the authors' reported figure. I wouldn't try to reproduce it from the weight-only arithmetic above.

## Quality: reconstruction error is not the evaluation

The RMSE table is tempting to over-interpret. A lower reconstruction error on a tensor says the quantizer preserved the weights better. It says nothing directly about whether the fine-tuned model solves my task as well.

The question I actually care about is: *given the same budget, does QLoRA produce a model that is as good as 16-bit LoRA on my task?* The paper reports that in their experiments NF4 with double quantization matched 16-bit LoRA on average on 5-shot MMLU for LLaMA 7B–65B fine-tuned on Alpaca and FLAN v2, while 4-bit floats were about one percentage point behind. They also state the limitation directly: they did not establish that QLoRA matches full 16-bit fine-tuning at 33B and 65B.

Those results come from the authors' models, data and benchmarks. To make a decision for my own case, I would run a controlled comparison:

1. **Same base checkpoint** for both runs. One is LoRA on the 16-bit base, the other is QLoRA on the 4-bit base.
2. **Same data splits** — train, validation and test — fixed before any run.
3. **Same number of training tokens**, not just the same number of steps, so that differences in batching don't silently change the budget.
4. **Same adapter configuration**: rank, scaling, dropout and target modules. The paper's own setup put adapters on all linear layers with `r = 64` and `α = 16`, and the authors found rank mattered little when all layers were adapted. I'd keep that as their observation and keep the configuration identical across my two arms.
5. **Hyperparameters tuned on validation only**, with the same tuning budget for each arm. If I tune one arm harder, I'm measuring my effort, not the method.
6. **Test set used once**, at the end, for the selected configuration of each arm.
7. **More than one seed** where I can afford it, so I can see whether a difference is larger than run-to-run noise.
8. **A task metric I believe.** The paper itself has a long discussion of benchmark validity and of moderate agreement between human annotators. An aggregate score I don't trust is not a basis for a decision.

## Deployment is a separate decision

A fine-tuned QLoRA model is a 4-bit base plus 16-bit adapters. At inference I have a few options, and each changes what I'm evaluating.

- **Keep the adapter separate.** The model is exactly what I trained, but every adapted layer computes the extra branch.
- **Merge the adapter into the base.** `W + s·L1·L2` is a new matrix that generally does not lie on the 4-bit grid. Merging into a 16-bit copy of the dequantized base gives a 16-bit model; quantizing it again introduces new quantization error on top of the trained correction. Library support also varies by quantization method — the PEFT v0.11.0 quantization guide, for example, notes that LoRA adapters trained on AQLM-quantized weights can't be merged with them.
- **Requantize with a different method** for serving. That is yet another model.

The rule I follow: whatever artifact I deploy is the artifact I evaluate. A test score measured on base-plus-adapter doesn't automatically carry over to a merged and requantized model.

The same applies to speed. Four-bit weights reduce memory and memory traffic, but dequantization is extra work. Whether a given setup is faster or slower depends on the hardware, the kernels, the batch size and the sequence length. I don't assume; I measure latency and throughput on the configuration I'll actually serve.

## When I'd reach for QLoRA

QLoRA makes sense to me when the frozen base weights are the thing that doesn't fit. If a 16-bit LoRA run fits comfortably in the hardware I have, I'd start there: it avoids quantization error and dequantization overhead, and it gives me the baseline I need anyway.

If I do use QLoRA, the knobs map onto the concepts above. In the PEFT v0.11.0 quantization guide they appear as the quantization type (`bnb_4bit_quant_type`, e.g. `"nf4"`), double quantization (`bnb_4bit_use_double_quant`), the compute dtype (`bnb_4bit_compute_dtype`), a preparation step for k-bit training (`prepare_model_for_kbit_training`) and adapter targets (including `"all-linear"`). I check the documentation for the exact versions I install rather than copying settings across versions.

## Troubleshooting checklist

- **Out of memory despite 4-bit weights.** Look at activations first: micro-batch size, sequence length, gradient checkpointing. The weight-only number is a lower bound.
- **Training much slower than expected.** Check whether the optimizer is paging continuously, and measure the dequantization overhead rather than guessing.
- **Quality worse than the 16-bit baseline.** Confirm both runs used the same data, token budget, adapter targets and tuning effort. Check which compute dtype is actually used, and which modules are actually quantized.
- **Good validation, worse production results.** Check whether the deployed model was merged or requantized after evaluation.
- **Comparing quantizers by reconstruction error.** Fine as a diagnostic, not as the final decision. Evaluate the task.

## Limitations of this article

The demo uses uniform symmetric quantization on synthetic normally distributed data with a single artificial outlier. It does not implement NF4, does not run on a GPU and does not train a model. The memory numbers are arithmetic, not measurements, and they cover weights only. The quality results I quote are the QLoRA authors', from their models and benchmarks; I haven't reproduced them here.

What the example does show, with tests behind it, is the mechanism: per-block scales confine an outlier's damage to its block, smaller blocks cost more scale storage, and double quantization buys most of that storage back.

### Sources

- [Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1](https://arxiv.org/html/2305.14314v1) — submitted 23 May 2023. Storage vs. compute dtype, NF4 construction and exact zero, double quantization arithmetic, paged optimizers, adapter gradients through the frozen base, and all quoted experimental results.
- [Hugging Face PEFT v0.11.0: Quantization developer guide](https://huggingface.co/docs/peft/v0.11.0/developer_guides/quantization) — PEFT 0.11.0 was uploaded to PyPI on 16 May 2024. Configuration names for 4-bit QLoRA-style training and the note on merging adapters with AQLM-quantized weights.
- Example code, tests and lock file: [quant_demo.py]({{ '/assets/examples/qlora-quantization/quant_demo.py' | relative_url }}), [tests/test_quant_demo.py]({{ '/assets/examples/qlora-quantization/tests/test_quant_demo.py' | relative_url }}), [pyproject.toml]({{ '/assets/examples/qlora-quantization/pyproject.toml' | relative_url }}), [uv.lock]({{ '/assets/examples/qlora-quantization/uv.lock' | relative_url }}).
