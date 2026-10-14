---
layout: post
lang: en
title: "DSPy vs hand-written prompts: a fair comparison with a holdout and a fixed budget"
date: 2024-06-20 09:00:00 +0200
permalink: /blog/dspy-vs-manual-prompts/
author: Rafał Prońko
categories: [llm-evaluation]
description: "How to compare a DSPy program with a hand-written prompt honestly: shared model and output contract, train/dev/test boundaries, equal per-arm budgets, a frozen choice and one paired holdout run."
image: /assets/images/dspy-vs-manual-prompts.svg
---

# DSPy vs hand-written prompts: a fair comparison with a holdout and a fixed budget

![Diagram of a fair comparison: cases split by group into train, dev and a sealed test set; a manual arm and a DSPy arm with separate, equal ledgers; selection on dev; a frozen manifest; one holdout run; a paired comparison.]({{ '/assets/images/dspy-vs-manual-prompts.svg' | relative_url }})

<time datetime="2024-06-20">20 June 2024</time>

"We tried DSPy and it beat our prompt" usually describes an unequal contest. On one side there is an afternoon of prompt tweaking. On the other there is an optimizer that made as many model calls as it needed. Both were then scored on the examples they were tuned on. Sometimes the DSPy arm also had a different model, a longer prompt, or a cache that was warm from yesterday's run.

The opposite claim, "a good prompt engineer beats DSPy", usually has the same problem the other way round.

The question I want answered is narrower:

> With the same model, the same labelled examples and the same budget, which process produces a better **frozen** artifact on data that neither process has seen?

This post is a recipe for that experiment. It has two parts. First, I look at what `BootstrapFewShot` in `dspy-ai` 2.4.9 actually does, because the fairness rules follow from it. Then I build a small offline harness that enforces those rules: group-level splits, an independent budget per arm, selection on dev only, a frozen manifest and exactly one holdout run.

One thing up front. The harness runs on **hand-authored fixtures**, not on model outputs. It tests the rules of the experiment, not DSPy. Nothing below tells you which arm wins on your task.

## What BootstrapFewShot does in 2.4.9

The [DSPy paper's abstract](https://arxiv.org/abs/2310.03714) describes LM pipelines as text transformation graphs in which LMs are invoked through declarative modules. A compiler optimizes a pipeline for a given metric, and the modules learn by creating and collecting demonstrations. That is the idea. The fairness questions depend on the details, so I read [`dspy/teleprompt/bootstrap.py` at the v2.4.9 tag](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dspy/teleprompt/bootstrap.py). Version 2.4.9 was [uploaded to PyPI on 29 April 2024](https://pypi.org/project/dspy-ai/2.4.9/).

The signature is:

```python
BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings={},
                 max_bootstrapped_demos=4, max_labeled_demos=16,
                 max_rounds=1, max_errors=5)
optimizer.compile(student, *, teacher=None, trainset, valset=None)
```

`compile()` does four things.

1. **It builds a teacher.** Without an explicit `teacher`, the teacher is a fresh copy of the student. If `max_labeled_demos` is non-zero and the teacher is not already compiled, `LabeledFewShot` gives it up to that many randomly sampled, labelled **train** examples as demos.
2. **It bootstraps traces.** It walks through `trainset`, removes the current example from the teacher's demos and runs the teacher on that example's inputs. It then calls `metric(example, prediction, trace)`. When the metric accepts the result, every predictor step in the trace becomes an "augmented" demo. It stops after `max_bootstrapped_demos` accepted examples. With `max_rounds > 1`, later rounds copy the LM with temperature 0.7 (plus 0.001 per round) to get different samples.
3. **It fills up with raw labelled demos.** Each student predictor receives the accepted traces and, if `max_labeled_demos` allows more, labelled examples sampled from a pool. The pool is `valset` if you passed one. Otherwise it is the train examples that were not bootstrapped.
4. **It returns the student** with those demos attached.

That is the whole algorithm. It does not update model weights. It does not generate or rewrite instructions. It also does not evaluate candidate programs on `valset` or select between them. The search over candidates lives in `BootstrapFewShotWithRandomSearch`, which scores each candidate on `valset` and, in 2.4.9, uses `trainset` when `valset` is not given.

Three consequences matter for a fair comparison.

**`valset` is a leak in 2.4.9.** Point 3 says it all: if you pass your dev set as `valset=` to `BootstrapFewShot`, the labelled demos in the compiled prompt are sampled from dev. Your dev score now measures a prompt that contains dev examples. If you pass the test set, the comparison is over. The fix is to leave `valset` out, and then check where the demos came from.

**Prompt length is part of the treatment.** With the defaults, the compiled predictor can carry up to sixteen demos total, including at most four bootstrapped demos. The remaining slots can hold raw labelled examples; twelve is the limit only when all four bootstrap slots are filled. A hand-written zero-shot prompt is much shorter. If you cap only the number of calls, the arm with long prompts gets more tokens for the same budget. Cap tokens too.

**One `Predict` call is not one request.** In `dsp/primitives/predict.py`, a completion that is missing an output field triggers a follow-up request, up to two levels deep. So one `Predict` call can make up to three requests. The OpenAI client in `dsp/modules/gpt3.py` also retries rate-limit errors with exponential backoff for up to 1000 seconds. Neither the follow-ups nor the retries are visible when you count program calls.

## The rules of the experiment

The rules have to be written down before the first candidate runs.

### What is shared

Both arms get the same model snapshot, decoding settings (temperature, `max_tokens`), message format and tools. In this post that means no tools. They also share the output contract, the parser, the retry policy and the metric.

The output contract matters more than it seems. In the harness, a valid answer is exactly one of four queue names, after stripping whitespace and lowercasing. `"Billing."` is invalid. If one arm's evaluation is lenient about punctuation and the other's is strict, you are comparing parsers, not prompts.

The only thing allowed to differ is how the instructions and demos are produced: a person choosing them, or DSPy compiling them. In the provider recipe, both arms use the same `Router`, signature fields, `Predict`, template extraction and missing-field follow-ups. A raw manual completion followed by the same label parser would still be a different recovery policy.

### Who sees what

| Split | Manual arm | DSPy arm | Purpose |
|---|---|---|---|
| train | may read it, pick examples | `trainset` for bootstrapping and labelled demos | building the prompt |
| dev | may score candidates and read errors | may score candidates (random search uses it) | choosing one candidate |
| test | nothing | nothing | one final measurement |

Two rules make this table hold in practice.

- **Split by group, not by row.** A ticket and its follow-up from the same customer are near-duplicates. If one sits in train and the other in test, the test measures memory. The harness hashes the group id with a fixed salt, assigns whole groups to a split, and checks that no case id, group or ticket text is shared between splits. It also refuses a split where any label is missing.
- **Demos come from train only.** For the manual arm this is a declaration ("I used c01, c13, c25 and c37"), which the harness checks. For a real DSPy run, inspect `predictor.demos` after `compile()` and assert that every demo's input appears in train.

### What each arm may spend

Each arm gets its own budget. One arm running out never stops or subsidises the other.

- **Optimization LM cap: attempts and tokens.** Every HTTP attempt is charged: teacher, student, retries, follow-up requests and judge calls (if your metric uses a model). Compile calls on train and evaluation calls on dev share this cap.
- **Developer feedback cap.** This is the number of dev evaluations an arm may run. A person who reruns dev after each prompt edit uses one per edit. With random search, each candidate scored on dev uses one.
- **Final evaluation cap.** This is separate and equal for both arms, and large enough to finish the holdout. A holdout run that hits a budget is broken, not "lower".
- **Cache.** Turn it off, or give each arm its own empty cache. Record cache hits either way. In 2.4.9 the on-disk cache is on by default, keyed on the full request. Cache hits still go through `basic_request()` and land in `lm.history`, together with the `usage` block of the original response. Token totals read from history therefore overstate spending whenever the cache is on.
- **Human time** goes in its own column. Writing the signature and the metric counts for the DSPy arm, just as editing the prompt counts for the manual arm. Do not convert minutes into calls. They are different resources, and the reader should see both.

### When to stop

The search ends when the feedback cap is reached, when the next step would exceed the LM cap, or when the candidates run out. A candidate whose compile or dev run hits the cap is **discarded**. A partial dev score cannot be compared with a full one.

## The harness

The [complete example ZIP]({{ '/assets/examples/dspy-vs-manual-prompts.zip' | relative_url }}) contains the source, fixtures, tests, configuration, `uv.lock` and [README]({{ '/assets/examples/dspy-vs-manual-prompts/README.md' | relative_url }}). Individual files are available too, including [the statistics module]({{ '/assets/examples/dspy-vs-manual-prompts/fair_compare/stats.py' | relative_url }}). The offline harness uses only the standard library, Python 3.11+, and `pytest` for the tests. The modules are small: `splits.py`, `budget.py`, `runner.py`, `search.py`, `freeze.py` and `stats.py`.

The data is 48 synthetic support tickets in 24 groups. Each ticket is labelled `billing`, `technical`, `account` or `shipping`. I took the first salt in a numbered series that put every label into every split. That choice looked only at labels, before any candidate ran. It gives 26 train, 12 dev and 10 test cases.

The candidates are fixtures. Each one is a lookup table that returns the true label unless I wrote an override for that case. For example, `manual-v1` answers `"Billing."` for case `c08`. `!transient` simulates a retryable failure and `!fatal` a non-retryable one. A list scripts successive attempts. The `dspy-bootstrap-k*` entries are **stand-ins**: they carry a declared number of teacher calls, so compile cost is charged where it would be. They are not DSPy programs, and their answers are mine.

### Every attempt is charged before it runs

```python
def charge(self, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if self.billable >= self.max_attempts:
        raise BudgetExceeded(
            f"{self.name}: cap of {self.max_attempts} attempts reached"
        )
    self.by_role[role] += 1
```

`ROLES` is `teacher`, `student`, `retry` and `judge`. Each arm has one `Ledger` for optimization and another for the final evaluation.

One case runs under the shared policy. Transient failures are retried and every retry is charged. A fatal failure is not retried. An invalid output is scored as wrong, reported as invalid and not retried. Only completed responses are cached.

```python
for attempt in range(max_attempts):
    ledger.charge("student" if attempt == 0 else "retry")
    raw = candidate.output(case, attempt)
    if raw == TRANSIENT:
        continue
    if raw == FATAL:
        return Outcome(case.case_id, case.group, "error", None, False,
                       attempt + 1)
    label = parse_label(raw, labels)
    status: Status = "ok" if label is not None else "invalid"
    ...
```

When retries run out, the result is `error`, which counts as wrong. It is never skipped. Dropping failed cases from the denominator rewards the arm that fails more often.

### Dev only, with a feedback cap

```python
for candidate in candidates:
    if len(log.results) >= feedback_cap:
        log.stop_reason = (...)  # "feedback cap ... reached"
        break
    check_demos_from_train(candidate, train)
    try:
        for role, count in candidate.compile_attempts.items():
            for _ in range(count):
                ledger.charge(role)
        outcomes = tuple(
            run_case(candidate, case, labels, ledger, max_attempts, cache)
            for case in dev.cases
        )
    except BudgetExceeded as exc:
        log.stop_reason = f"{exc}; {candidate.candidate_id} discarded"
        break
    log.results.append(DevResult(candidate, outcomes))
```

`search_arm()` refuses to run unless it gets the splits named `train` and `dev`. Selection takes the highest dev accuracy. Ties go to the earlier candidate, a rule written down in advance.

### Freeze, then open the holdout once

Before reading inputs or searching, the harness rejects any non-empty output directory, including one containing only a partial manifest or an in-progress marker. A new or empty directory receives `run_started.txt`; a failed run keeps that marker. Before any test case is scored, the harness writes a manifest. It holds the selected candidate id per arm, the SHA-256 of each selected artifact, a digest of the test ids and the SHA-256 of the fixture file. The manifest is a frozen dataclass whose mappings are read-only.

The test split lives inside a `HoldoutGuard`. Once the experiment has wrapped it, the orchestrator deletes its own reference. The guard hands the split out once, and only for a manifest that matches:

```python
def open(self, manifest, candidates):
    if self._receipt.exists():
        raise HoldoutError(f"holdout already used: {self._receipt}")
    if manifest.test_ids_sha256 != self.test_ids_sha256:
        raise HoldoutError("test split changed after freeze")
    for arm, candidate in candidates.items():
        if manifest.selected.get(arm) != candidate.candidate_id:
            raise HoldoutError(f"{arm}: candidate differs from manifest")
        if manifest.artifact_sha256.get(arm) != candidate.artifact_sha256:
            raise HoldoutError(f"{arm}: artifact changed after freeze")
    self._receipt.write_text(manifest.sha256 + "\n", encoding="utf-8")
    return self._test
```

The receipt file makes the rule survive a restart. If you run the experiment again into the same output directory, it refuses before any search or write, even if the inputs changed. A regression test checks that every existing artifact remains byte-identical. Preserve completed and failed directories. This is a cooperative rule for sequential runs, not a tamper-proof seal or a concurrent security boundary; someone can delete the artifacts or choose a fresh directory. "Let me just check the test score once more" is how a holdout turns into a second dev set.

## Running it

Download the ZIP above to a fresh working directory, then extract and run it:

```bash
unzip dspy-vs-manual-prompts.zip
cd dspy-vs-manual-prompts
uv run --frozen --isolated --python 3.11 --group dev pytest -q
uv run --frozen --isolated --python 3.11 python -m fair_compare --out run-1
# Expected refusal: this command exits 1 and preserves the original files.
uv run --frozen --isolated --python 3.11 python -m fair_compare --out run-1
```

All 67 tests pass, including the six provider-adapter tests that use stubs. The same suite and fixture experiment also ran from the extracted ZIP. The run writes `run_started.txt`, `freeze_manifest.json`, `holdout_receipt.txt` and `report.json`. Condensed from that report:

| | Manual arm | DSPy arm (stand-ins) |
|---|---|---|
| dev evaluations | v1 8/12, v2 11/12, v3 10/12, v4 11/12 | zero-shot 9/12, k2 11/12, k4 12/12 |
| why the search stopped | feedback cap of 4 reached, v5 not evaluated | LM cap of 60 reached, k8 discarded |
| optimization attempts | 37 (student 36, retry 1), 12 cache hits | 60 (teacher 21, student 38, retry 1) |
| invalid / error on dev | v1: 2 invalid; v3: 1 fatal error | zero-shot: 1 invalid |
| frozen choice | `manual-v2` | `dspy-bootstrap-k4` |
| holdout | 7/10, 1 invalid | 8/10, 1 retry |
| final eval attempts (cap 20) | 10 | 11 |

Each line shows one of the mechanisms at work.

- `manual-v4` has exactly the same prompt text as `manual-v2`, as if the developer reverted an edit. All 12 dev answers come from the cache. That is 0 billable attempts, but it still uses one of the 4 feedback slots, because someone looked at a dev score. The harness rejects a fixture file in which two candidates share a prompt text but script different outputs. A cache keyed on the prompt is only valid if the same prompt behaves the same way.
- `dspy-bootstrap-k8` charges 12 teacher attempts, starts on dev and hits the cap after two cases. Everything spent up to the cap stays in the ledger. The partial score is thrown away.
- The `!fatal` failure on `manual-v3` costs one attempt and shows up as an error, not as a missing row.

### What these numbers do and do not say

They say nothing about DSPy. I wrote every override. That includes the pattern that both selected candidates score lower on test than on dev, which is what selecting the maximum of several noisy dev scores leads you to expect. Whether it happens in your data is for your run to show.

What they do show is how little a small holdout can tell you. The manual arm alone is right on one case, the DSPy stand-in alone on two. Those are descriptive row counts. Ordinary case-wise McNemar inference would assume independent matched rows, which these related tickets do not provide.

The executable statistic instead computes each group's accuracy difference, DSPy minus manual, and gives every group equal weight. It enumerates the positive and negative signs of those group differences and counts sums at least as large in absolute value as the observed sum. Fractions avoid rounding at the tail boundary. Zero differences can be omitted because their duplicated sign assignments cancel; the code refuses more than twenty non-zero groups rather than silently switching methods.

This is an exact paired sign-flip test **under its null assumptions**: clusters are independent, and swapping the arm labels for an entire cluster is exchangeable under the null, so the difference signs are independent and uniform conditional on their magnitudes. Equal mean accuracy alone does not imply those assumptions. For a real study, choose the group definition and endpoint in advance and justify that null model; a deterministic prompt comparison does not make exchangeability automatic.

Here the ten rows form five groups. Two discordant rows cancel within one group, leaving just one non-zero group difference. The mean group difference is +0.1 and the two-sided sign-flip p-value is 1.0. Six discordant rows in **one** group also give p = 1.0. Six independent groups with equal differences all pointing one way give 2/64 = 0.03125, assuming the null above. The number of related rows cannot substitute for the number of independent clusters.

The group bootstrap interval for that same equally weighted group endpoint is [0.0, 0.3]. Here every group has two rows, so its point estimate also equals the row accuracy difference. Unequal group sizes can make the two endpoints differ; the report labels both. Do not read the interval as "DSPy is never worse". With only five groups, all observed group differences are zero except one positive value, so resampling them cannot produce a negative value. That is a limitation of this tiny fixture, not evidence of adequate quality.

These are arithmetic outputs from fixtures I wrote. They establish no model-quality difference, equivalence or noninferiority.

## The recipe for a real run

The harness fixes the rules. A real experiment needs model calls under those rules. The downloadable [provider recipe]({{ '/assets/examples/dspy-vs-manual-prompts/dspy_recipe/real_dspy_arm.py' | relative_url }}) and its [historical pins]({{ '/assets/examples/dspy-vs-manual-prompts/dspy_recipe/requirements.txt' | relative_url }}) target `dspy-ai==2.4.9` with `openai==1.30.5`. They are an incomplete integration sketch, not a runnable benchmark command. The offline `uv.lock` locks the test harness, not a provider environment.

**Validation is SOURCE_CHECKED / STUB_ONLY.** The tests execute our adapter with fake DSPy and SDK modules and a socket guard. Neither real package was installed or executed, and no provider calls were made. The stubs exercise retry accounting, budget refusal, shared prediction routing and copy overrides; they do not prove SDK integration or real DSPy extraction behavior.

The retry setting belongs on the SDK client DSPy actually uses. In the pinned [DSPy source](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dsp/modules/gpt3.py), chat requests call `openai.chat.completions.create(...)`. The [OpenAI 1.30.5 module client](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/src/openai/__init__.py) reads `openai.max_retries`; its [README](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/README.md) documents two automatic retries by default. Constructing an unrelated `OpenAI(max_retries=0)` would not configure DSPy's call path. Passing `max_retries` as a completion parameter would put a client option into the wrong API. The core pieces are:

```python
os.environ["DSP_CACHEBOOL"] = "false"  # read once, when dsp is imported
import openai

if openai.__version__ != "1.30.5":
    raise RuntimeError("this recipe requires openai==1.30.5")
openai.max_retries = 0  # module client used by DSPy 2.4.9

import dspy
from dspy.teleprompt import BootstrapFewShot


class BudgetedOpenAI(dspy.OpenAI):  # dspy.OpenAI is dsp.GPT3 in 2.4.9
    def request(self, prompt, **kwargs):
        self._tries = 0  # super().request is the backoff-wrapped loop
        return super().request(prompt, **kwargs)

    def basic_request(self, prompt, **kwargs):
        if openai.max_retries != 0:
            raise RuntimeError("SDK retries must remain disabled")
        self.ledger.charge(self.role if self._tries == 0 else "retry")
        self._tries += 1
        response = super().basic_request(prompt, **kwargs)
        ...  # meter usage on successful responses; this is NOT a token cap
        return response
```

With SDK retries disabled and caches off, the adapter charges every `basic_request()` entry, including DSPy's outer rate-limit backoff and `generate()` follow-ups. The no-network stub shows one simulated dispatch per charge and stops a retry before dispatch when the cap is reached. A real integration still needs a transport-level check, including any lower-layer retries or redirects, before claiming that every HTTP attempt is accounted for. SDK connection and server errors now propagate; DSPy's outer retry covers rate-limit errors. Both arms must use that same policy.

The recipe's token meter records successful-response usage; **it does not enforce a token cap**, and missing usage on failed attempts is not zero cost. Before a real budgeted run, reserve a conservative upper bound for the complete serialized input plus allowed output before each transport attempt, including follow-ups and retries. Do not refund an unknown failed attempt on the assumption that it used no tokens. Test the reservation and refusal at that boundary with the pinned SDK. The fixture harness has no tokens to count.

The class also overrides `copy()`. The base `LM.copy()` merges saved kwargs with caller overrides, removes `model`, then reconstructs `self.__class__(model=model, **merged_kwargs)`. A subclass with a required ledger must preserve it explicitly. The recipe keeps the same ledger and token meter while honoring overrides such as the later bootstrap temperature; teacher and student share those counters within one arm. Use a dedicated single-threaded process. The module client's global settings make this unsuitable as a concurrent accounting boundary.

Compile with separate teacher and student clients that charge the same ledger. Leave `valset` out:

```python
optimizer = BootstrapFewShot(
    metric=exact_queue,          # parse_label(...) == example.queue
    max_bootstrapped_demos=k,
    max_labeled_demos=k,
    max_rounds=1,
    max_errors=1,                # first exception (BudgetExceeded too) aborts
    teacher_settings={"lm": teacher_lm},
)
program = optimizer.compile(Router(), trainset=to_examples(train))
```

Then follow these steps.

1. **Pin everything.** The source-checked primary pins are `dspy-ai==2.4.9` and `openai==1.30.5`; the latter's [tagged package metadata](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/pyproject.toml) confirms the version. A full provider dependency lock and executed SDK integration remain prerequisites for a real run. In 2.4.9, `dspy.LM` is the abstract base class from `dsp`; configure the concrete `dspy.OpenAI` client.
2. **Write `experiment.json` first.** It holds the split salt and percentages, the model id, decoding settings, `max_attempts_per_case`, the per-arm optimization cap in attempts **and** tokens, the feedback cap, the final evaluation cap, the metric and the decision rule.
3. **Put both arms through the same program.** `manual_candidate(instruction, demos)` creates a `Router`, changes the signature's instructions with `with_instructions()` and sets train-only demos. The compiled arm uses that same `Router` and `Predict`. Both run through `predict_ticket(program, lm, case)` with an LM context, so template extraction and missing-field recovery are shared before `parse_label()`. In 2.4.9, binding `predictor.lm` instead would set `query_only=True` and omit instructions and demos. Keep settings and error handling identical, charge separate ledgers and log human minutes.
4. **Search each arm on dev.** For DSPy, each `compile()` plus its dev evaluation is one feedback slot. `BootstrapFewShotWithRandomSearch` uses one slot per candidate it scores, including its three baselines, so set `num_candidate_programs` to fit the cap. Pass `valset=dev` explicitly. After each compile, check that every demo comes from train.
5. **Freeze.** For both arms, hash the JSON written by `program.save()`. It includes demos and signature instructions. Record the common signature, settings and recovery policy too, and write the manifest together with the test id digest.
6. **Score the holdout with your own loop, once.** Do not use `dspy.Evaluate` here. In 2.4.9 it turns an exception into a score of 0.0 until `max_errors` is reached, and then re-raises. A budget error or provider failure then looks just like a wrong answer. Record `ok`, `invalid` or `error` per case, as `runner.py` does, under each arm's separate final evaluation ledger.
7. **Report the pairs.** Give accuracy, invalid rate and error rate per arm. Add descriptive discordant row counts, the group endpoint, the exact paired group sign-flip p-value with its null assumptions, and a bootstrap interval for that same endpoint that resamples whole groups. Then add attempts, tokens and cache hits per role, and human minutes. If the optimizer's result depends on the order of `trainset`, a single compile is a single sample. Repeat it with different seeds if the budget allows, and say that you did.

## How to choose

There is no general winner, and this setup is not built to find one. It answers a local question: for this task, model, data and budget, is the extra machinery worth it?

DSPy is a natural candidate when:

- you have a few dozen or more labelled examples and a metric you trust;
- the pipeline has several LM steps, so demos for intermediate steps are tedious to write by hand;
- you expect to change the model or the pipeline and want to recompile rather than rewrite prompts.

A hand-written prompt is a strong baseline when:

- you have few labels, or the metric is mostly human judgement;
- a short prompt matters for cost or latency;
- domain rules can be written down directly, like the "a locked-out user is `account` even if they mention invoices" rule in the fixtures.

In both cases, run the comparison as above before you adopt either. If the holdout is small, the most likely honest outcome is "inconclusive". You can choose the cheaper option provisionally as a cost decision under uncertainty, but an inconclusive result does not show that its quality is adequate. If adequacy is required, set an acceptable noninferiority margin in advance and collect enough independent groups to support it; failure to establish a difference is not evidence of noninferiority.

## Verification scope

- **Date.** The post date, 20 June 2024, places it in the series. It is an editorial label, not a record of when anything ran. The API facts are taken from the `v2.4.9` tag. That release was uploaded to PyPI on 29 April 2024 (PyPI JSON metadata, checked on 12 October 2026).
- **Executed.** The corrected offline harness and all 67 tests ran on 12 October 2026 with CPython 3.11.12 and pytest 9.1.1, including from the extracted reader ZIP. Fixture experiments completed and repeat runs were refused with the artifacts unchanged. Ruff 0.6.9 and strict mypy 1.11.2 checked the code (mypy checks `fair_compare/`, targeting Python 3.11). The original run used uv/uvx to fetch pytest, ruff and mypy; correction validation reused cached tools offline. The harness makes no network requests or model calls. Six tests import our provider adapter with fake dependencies only.
- **Tooling date.** The uv 0.7.8 workflow, dependency groups and test dependency versions describe the October 2026 reproduction environment. They are separate from the June 2024 editorial label and historical provider API pins.
- **Not executed.** Neither the real DSPy compiler nor the real OpenAI SDK was installed or run. Their call boundary was checked in pinned source; stubs are not proof of their integration. The provider recipe does not yet enforce a token cap or supply a complete real-run orchestration. I read the DSPy paper's abstract, not the full paper, so I do not quote its results.
- **Not covered.** The harness counts attempts but not tokens, latency or money. It handles one task type with one exact-match metric. It does not model judge calls, although the ledger has a role for them.

### Sources

Retrieved and archived on 12 October 2026.

- [dspy-ai 2.4.9 on PyPI](https://pypi.org/project/dspy-ai/2.4.9/) and its [JSON metadata](https://pypi.org/pypi/dspy-ai/2.4.9/json): upload time 2024-04-29T20:56 UTC, `requires_python >=3.9`, `openai>=0.28.1,<2.0.0`.
- [`dspy/teleprompt/bootstrap.py` at v2.4.9](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dspy/teleprompt/bootstrap.py): the `BootstrapFewShot` algorithm, the use of `valset` as the raw-demo pool, and `max_errors`.
- [`dspy/teleprompt/random_search.py` at v2.4.9](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dspy/teleprompt/random_search.py): candidate scoring on `valset`, which defaults to `trainset`.
- [`dsp/modules/gpt3.py`](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dsp/modules/gpt3.py), [`dsp/modules/lm.py`](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dsp/modules/lm.py) and [`dsp/modules/cache_utils.py`](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dsp/modules/cache_utils.py) at v2.4.9: `basic_request`, backoff on rate limits, `copy()`, history and the `DSP_CACHEBOOL` / `DSP_CACHEDIR` cache.
- [`dsp/primitives/predict.py` at v2.4.9](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dsp/primitives/predict.py): follow-up requests when an output field is missing.
- [`dspy/evaluate/evaluate.py` at v2.4.9](https://raw.githubusercontent.com/stanfordnlp/dspy/v2.4.9/dspy/evaluate/evaluate.py): exceptions scored as 0.0 until `max_errors`, and `return_all_scores`.
- [OpenAI Python SDK v1.30.5 README](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/README.md), [module client](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/src/openai/__init__.py) and [package metadata](https://raw.githubusercontent.com/openai/openai-python/v1.30.5/pyproject.toml): default automatic retries, the module client's `max_retries` setting and the exact version pin. Read through the web tool; the shell download failed, and no SDK was executed.
- O. Khattab et al., [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714), arXiv:2310.03714 (abstract only).
