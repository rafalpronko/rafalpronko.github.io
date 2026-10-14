# LoRA NumPy lab

Synthetic CPU-only linear-regression experiment. The article date 4 July 2024 is
an editorial label; the code and reference results were produced on 12 October
2026 with CPython 3.11.12, NumPy 1.26.4, pytest 8.2.2 and uv 0.7.8.

From the directory containing `lora-numpy.zip`, with uv installed:

```bash
mkdir lora-reproduction
unzip lora-numpy.zip -d lora-reproduction
cd lora-reproduction/lora_numpy
uv sync --locked --python 3.11
uv run --locked pytest -v -W error::RuntimeWarning
uv run --locked python -W error::RuntimeWarning lora_lab.py reproduced-results
uv run --locked python -W error::RuntimeWarning sensitivity.py reproduced-results/sensitivity_val_only.json
```

The final command is optional. Fresh setup needs package/Python download access
or a populated uv cache. No hidden project file is required. The ZIP contains:

```text
lora_numpy/
  README.md
  lora_lab.py
  sensitivity.py
  pyproject.toml
  uv.lock
  tests/test_lora_lab.py
  results/metrics.json
  results/sensitivity_val_only.json
  results/checkpoints/rep{0..4}_{frozen_base,full,lora_r1,lora_r2,lora_r4}.npz
```

The reference `results/` stays intact; the commands write new outputs under
`reproduced-results/`. Five replicates share one constructed world. Each method's
learning rate and checkpoint, and the overall method/rank choices, use target
validation MSE. Source/target test MSE is computed after selection. The selection
API rejects a direct test argument, but does not provide process/data isolation.

There are 15 pytest cases in 14 test functions. Adapter checkpoints store copies
of the selected A/B, scaling, rank, alpha and step, as well as W and W0. Regression
tests check every saved adapter's reconstructed W, predictions, validation/test
MSE and subtractive recovery at `atol=1e-12`, including selections before the last
training step and step-zero/divergence paths. Stored W0 can be loaded bit-for-bit.

The post hoc sensitivity script evaluates validation only. Its 300-step setting
checks every 10 steps; its 3000-step setting checks every 50 steps and adds
learning rate 0.6. It generates test arrays through `make_splits`, but never
evaluates them or changes the main experiment's choices.

This is not an LLM benchmark. The approximately rank-2 target shift favours low
rank by construction. Numerical results may vary with Python/NumPy/platform;
NumPy gives no random-stream compatibility guarantee across versions.
