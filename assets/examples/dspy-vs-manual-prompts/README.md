# Fair prompt-arm comparison harness (offline)

Companion code for *DSPy vs hand-written prompts: a fair comparison with a
holdout and a fixed budget*. The complete download includes source, synthetic
data, configuration, tests and `uv.lock`. It contains no model outputs.

## Download and run

Download `dspy-vs-manual-prompts.zip` using the article's download link, save it
in a fresh working directory, then run:

```bash
unzip dspy-vs-manual-prompts.zip
cd dspy-vs-manual-prompts
uv run --frozen --isolated --python 3.11 --group dev pytest -q
uv run --frozen --isolated --python 3.11 python -m fair_compare --out run-1
# Expected exit 1; the existing artifacts remain byte-identical.
uv run --frozen --isolated --python 3.11 python -m fair_compare --out run-1
```

Requires uv and Python 3.11+. The first setup can download Python and the
locked pytest dependencies. Optional lint/type checks use uvx to fetch ruff
and mypy too. The harness and tests make no provider calls. Once dependencies
are cached, add `--offline` to uv/uvx. Validation used uv 0.7.8, Python 3.11.12
and pytest 9.1.1 on 12 October 2026; the dependency-group workflow and tool
versions are reproduction tooling, separate from the June 2024 editorial date.

All 67 tests passed in both the author directory and the extracted ZIP,
including six tests of our provider adapter with stubs. The fixture experiment
selects `manual-v2` and `dspy-bootstrap-k4`, charges 37 and 60 optimization
attempts, and scores the holdout 7/10 and 8/10. These are scripted outcomes,
not evidence about model quality.

## Scope

- `fair_compare/` enforces group-disjoint splits, train-only demos, separate
  per-arm attempt ledgers, dev feedback caps, selection on dev and a frozen
  manifest before a one-shot holdout. Invalid and failed cases count as wrong.
- `data/candidates.json` contains **hand-authored lookup-table fixtures**, not
  DSPy programs or LM outputs. Declared teacher calls simulate compiler costs.
- Output directories must be new or empty. `run_started.txt` is written before
  input loading/search; any non-empty, completed or partial run is refused
  before any writes/search. Keep failed runs. Changed-input reruns preserve all
  existing files. This cooperative sequential protocol is not a tamper-proof
  seal or a concurrent security boundary; deleting files defeats the rule.
- The report gives descriptive discordant rows and row accuracies separately
  from its inferential endpoint: the equally weighted mean of each group's
  accuracy difference. The exact paired sign-flip test enumerates group signs,
  requiring independent clusters and exchangeable arm labels per cluster under
  the null (independent uniform signs conditional on magnitudes). Equal means
  alone do not imply this null. Zero deltas are omitted without changing the
  tail ratio; more than 20 non-zero groups is explicitly refused. A group
  bootstrap interval uses that same endpoint. Five tiny clusters cannot
  establish adequacy, equivalence or noninferiority.
- `dspy_recipe/real_dspy_arm.py` is **SOURCE_CHECKED / STUB_ONLY**. The primary
  pins are `dspy-ai==2.4.9` and `openai==1.30.5`; neither real dependency nor the
  compiler was installed or executed. Tests import our adapter with fake
  modules and a socket guard, not real SDK integration. `requirements.txt` is
  not a tested full provider dependency lock. The recipe has no benchmark CLI.
- DSPy 2.4.9 calls the OpenAI module client, so the recipe sets
  `openai.max_retries = 0` at that boundary, never in completion kwargs. It
  retains DSPy's outer rate-limit backoff and counts its retries. Dedicated
  single-threaded process only: module-client settings are global. A real run
  still needs transport-level verification, including lower-layer retries.
- Both real arms use `Router`/`Predict`, the same signature fields, template
  extraction, missing-field recovery and final label parser. The manual arm
  changes only instructions and train demos. Copying the LM retains shared
  ledger/meter identity and honors caller decoding overrides.
- The token meter observes successful-response usage; **it does not enforce a
  token cap**. Before a real run, reserve a conservative input-plus-output
  upper bound before each transport attempt, including retries/follow-ups and
  failures with unknown usage, and test that reservation with the pinned SDK.

## Checks

```bash
uvx --python 3.11 ruff@0.6.9 check --select E,F,W,I,UP,B \
  --line-length 79 --target-version py311 .
uvx --python 3.12 mypy@1.11.2 --strict --python-version 3.11 fair_compare
```

Strict mypy checks the 11 `fair_compare/` modules; it does not check the
unexecuted provider package integration. The mypy engine ran on Python 3.12
with Python 3.11 as the target.

## Layout

| Path | Role |
|---|---|
| `data/cases.json` | 48 synthetic tickets in 24 groups |
| `data/experiment.json` | Attempt/feedback caps, split salt, group endpoint and null assumptions |
| `data/candidates.json` | Hand-authored manual/DSPy stand-in outcomes |
| `fair_compare/cases.py` | Data loading and shared `parse_label` contract |
| `fair_compare/splits.py` | Group split, disjointness and label coverage |
| `fair_compare/budget.py` | Attempt ledger and cache-hit counts per arm |
| `fair_compare/runner.py` | Charge attempts, retry transient, score invalid/error |
| `fair_compare/search.py` | Dev-only search and train-only demos |
| `fair_compare/freeze.py` | Manifest and sequential holdout receipt |
| `fair_compare/stats.py` | Exact paired group sign-flip and group bootstrap |
| `fair_compare/experiment.py` | Early rejection and fixture orchestration |
| `dspy_recipe/real_dspy_arm.py` | Incomplete pinned provider recipe, source-checked/stub-tested only |
| `tests/test_correction_regressions.py` | Cluster dependence, input validation and artifact preservation |
| `tests/test_provider_recipe.py` | No-network adapter stub checks |
| `tests/provider_stubs.py` | Behavioral fakes, not real SDK/DSPy implementations |
