---
layout: post
lang: en
title: "Choosing a model for the task: benchmark quality and latency, not marketing labels"
date: 2024-03-28 09:00:00 +0100
permalink: /blog/model-selection-quality-latency/
author: Rafał Prońko
categories: [llm, evaluation]
description: "Compare paired replay logs: quality over all attempts, duration to completion or cancellation, failure rates, cost per useful completion, hard constraints and Pareto trade-offs."
image: /assets/images/model-selection-quality-latency.svg
---

# Choosing a model for the task: benchmark quality and latency, not marketing labels

![Paired replay logs are validated, scored on quality, attempt duration to completion or cancellation and cost, filtered by empirical hard constraints, then compared on a consistent Pareto front.]({{ '/assets/images/model-selection-quality-latency.svg' | relative_url }})

<time datetime="2024-03-28">28 March 2024</time>

Model names carry promises: "fast", "mini", "pro", first place on a leaderboard. None of those labels describes the task a particular application needs to perform.

A public benchmark measures someone else's prompts, output format and definition of a correct answer. A product page's latency numbers depend on hardware, region, prompt length and load. Both can suggest candidates worth testing. A deployment decision needs evidence from the intended task and operating conditions.

The useful question is narrower: **on these cases, under these constraints, which candidate produces useful answers at an acceptable duration and cost?** This article develops the evaluation step of that decision. The accompanying Python program compares replay logs; it sends no requests to a model.

The example was prepared with **hand-designed synthetic fixtures**. Its candidates are fictional, its durations invented and its prices hypothetical. It demonstrates evaluation mechanics. It makes no claim about a real model, an internal support system or the author's execution history.

## Quality means quality on this task

Consider a support router that assigns a ticket to `billing`, `shipping` or `account`, returning JSON such as `{"queue": "billing"}`. An out-of-scope ticket should trigger an abstention and human review.

That task requires several measurements:

1. **Routing correctness:** whether an answerable ticket gets the right queue.
2. **Schema validity:** whether the downstream system can parse the response. A correct label inside broken JSON is unusable.
3. **Abstention:** whether the system abstains on out-of-scope tickets and avoids abstaining on answerable ones.
4. **Slices:** performance by queue and on out-of-scope cases, because an average can hide a serious local failure.

The abstract of [HELM](https://arxiv.org/abs/2211.09110) describes multi-metric measurement across standardized scenarios, exposing trade-offs across accuracy, calibration, robustness, fairness, bias, toxicity and efficiency. That is a useful principle for this router: several measurements under the same conditions for every candidate. Only the abstract is used here; no HELM ranking or result is claimed.

## Evaluation data: representative holdout and separate challenges

A decision set should reflect the target traffic: ticket types, lengths, languages, typos and whatever input transformations production applies. Keep an untouched **representative final holdout**, separate from the development set used to tune prompts, schemas and parsing. Freeze those choices before evaluating candidates on the holdout. Repeatedly choosing configurations against that set gradually turns it into development data.

Rare, expensive cases deserve a separate **challenge suite**. Deliberately oversample ambiguous tickets or out-of-scope requests there, then report each slice and any agreed slice-specific gates. Pooling that suite unweighted with the holdout changes the mixture being evaluated. It can change the overall quality and cost per useful completion merely by adding more abstention cases. To estimate production performance, preserve the representative holdout or specify production weights and an appropriate weighted analysis. The simple evaluator here implements neither weighting nor weighted uncertainty estimates.

Split related material together: a conversation thread, a customer's related requests, duplicates and near-duplicates should not straddle development and holdout sets. For statistical inference, those groups may also be the sampling units; treating every correlated message as an independent observation overstates the amount of evidence.

Protecting gold labels limits deliberate tuning against the answer key. It does **not** guarantee freedom from contamination. Inputs can already be public, repeated, shared with a provider or present in training data even when the labels are private. Data provenance and overlap checks remain separate work.

## Failures stay in the denominator

Suppose a harness submits 16 cases. Two time out, one returns broken JSON, and eleven of the thirteen parseable responses are correct. `11/13` describes correctness conditional on receiving a parseable response. The system delivered something useful for `11/16` requests.

The example therefore divides useful completions by **all attempts**. A timeout or malformed response contributes zero usefulness. That binary convention does not say that all failures have the same operational harm; retries, misroutes and human review can have different costs.

A completion is useful when it returns the correct label or correctly abstains:

```python
@property
def correct(self) -> bool:
    if self.status == "ok":
        return self.expected is not None and self.output == self.expected
    if self.status == "abstain":
        return self.expected is None
    return False
```

Here `expected=None` means abstention is the gold behaviour. A queue label on such a case is wrong; abstaining on an answerable ticket is also wrong. Validation requires `output=null` for every abstention, so an object masquerading as an abstention cannot score as useful.

The aggregate gives a correct route and a correct abstention **equal usefulness, one each**. That is an explicit scoring convention, not an economic claim. The report also separates correct routing on answerable cases from correct abstention on out-of-scope cases. If the application values them differently, define separate gates or a justified utility function before selecting a model. The bundled unweighted aggregate describes its constructed mixture of 13 answerable and three out-of-scope cases, rather than production traffic.

## Latency: which measurement?

A streamed response has several useful timing boundaries:

- **Client TTFT:** time from starting a request to observing the first streamed content or content chunk. Network transit, queueing, prefill and buffering affect it. A chunk can contain several tokens, so this is not necessarily the instant the model generates its first token. Metadata-only events should not count as content.
- **Decode duration:** generation of subsequent output tokens. Under comparable serving conditions it often increases with output length, but batching, hardware, scheduling and pauses matter.
- **Complete response duration:** time until the full response is received. In an ideal unbuffered stream it roughly consists of time to first content plus subsequent generation and delivery. Client observations do not isolate server decode time exactly.

A router waiting for complete JSON needs the full response. When a request times out, however, its eventual completion time is unknown. The log records the observed **attempt duration to completion or cancellation**. For a completed request this ends at completion; for a timeout it ends at cancellation.

For the question "how long would completion have taken?", a timeout is a **right-censored observation**: completion has not occurred by the observed cutoff. Substituting that cutoff as if it were the eventual completion time cannot recover completion-latency p95. The all-attempt percentile in this evaluator is a percentile of observed attempt durations, not an estimate of that unobserved completion distribution.

Compare timeout policies as well as workload. Reducing one candidate's timeout from four seconds to half a second can improve its reported attempt-duration p95 while leaving every timed-out request unsuccessful. The report therefore keeps **timeout rate** visible. Comparable deadlines and cancellation behaviour are necessary for a meaningful duration comparison. A study of eventual completion times would need a suitable censoring analysis and its assumptions; this program performs none.

A harness can measure observed duration with Python's high-resolution [`time.perf_counter()`](https://docs.python.org/3.11/library/time.html#time.perf_counter):

```python
start = time.perf_counter()
response = call_candidate(case)  # harness illustration, not executable example code
elapsed_ms = (time.perf_counter() - start) * 1000
```

The documentation says only differences between calls are meaningful. The harness must also record the elapsed duration on cancellation, not drop the request. `time.time()` can go backwards when the system clock is changed and is unsuitable for this purpose.

### Compare the same work under the same load

- **Output length:** constrain both candidates to the same minimal JSON or report token counts alongside durations. An explanation adds work before model speed even enters the comparison.
- **Concurrency:** reproduce the expected number of requests in flight and load profile. A single-request test does not describe a busy service.
- **Time and place:** use comparable region, network path and time windows; interleaving candidates helps avoid a morning-versus-afternoon load difference.
- **Cache policy:** record cold or warm behaviour consistently. Historically generic examples include warmed connections, a self-hosted model already loaded in memory, self-hosted prefix reuse, and an application's own response cache. This is not a claim that a particular hosted provider offered a later prompt-caching feature in March 2024.
- **Timeout policy:** record the deadline, retries and cancellation handling. These affect both observed durations and the failure rate.

### Percentiles and tiny samples

The example uses **nearest-rank percentiles**: sort the `n` observations and select 1-based rank `ceil(p/100 × n)`.

```python
def nearest_rank(values: list[float], percentile: int) -> float:
    n = len(values)
    rank = (percentile * n + 99) // 100
    return sorted(values)[rank - 1]
```

The complete function also rejects empty input and percentiles outside `1..100`. There is no interpolation, so the result is an observed value. Nearest-rank **p50 is the lower median at even `n`**. It differs from `statistics.median()`, which averages the two middle values: the 16 all-attempt durations for A give p50 1310 ms versus an arithmetic median of 1320 ms. B gives 630 ms versus 635 ms.

Nearest-rank p95 describes the upper part of this sample. It provides **no exact one-in-twenty guarantee** about future requests, or even exactly that share above the value in a finite sample with ties. Python's `statistics.quantiles()` interpolates by default, so the method must be stated when comparing reports.

With 16 values, p95 has rank 16 and is the maximum. With 20 it selects rank 19, the second-largest observation. A tail calculation based on so few observations cannot establish a production SLO. Here the values were invented; even a larger synthetic fixture would provide no empirical latency evidence.

## Quality intervals and paired comparisons

The report evaluates the **Wilson 95% score interval formula** using the conventional normal critical value near 1.96. For independently sampled binary outcomes from a defined target population, that interval expresses sampling uncertainty under its binomial model. It is not a guarantee about future traffic or protection against a biased evaluation set.

Both candidates must run on the **same cases**. Count cases where both are useful, only A is useful, only B is useful, or neither is useful. The **exact two-sided McNemar calculation** conditions on the discordant count: under the equal-marginal-probability null, the direction of each discordant pair has probability one-half. This implementation doubles the smaller binomial tail and caps the result at one. If there are no discordant pairs it returns one. Fagerland, Lydersen and Laake found this exact conditional test highly conservative with poor power: its type I error rate often stays well below the nominal 5% level, so it can miss real differences.

Pairing removes case mismatch between candidates; it does not remove dependence between different cases. Real inference needs representative, independently sampled units. Threads, customers or duplicated requests may require grouping, cluster-aware methods or another study design rather than these simple formulas.

For the deterministic curated fixtures, **Wilson limits and McNemar p-values are formula demonstrations only**. The designed outcomes contain no random-sampling evidence. On real data with the required assumptions, a nonsignificant paired result means insufficient evidence to reject the stated null; it does not demonstrate equivalence. Overlap or non-overlap of two marginal Wilson intervals is not the paired significance test. Equivalence needs a predefined acceptable difference and an analysis designed for that question.

## Cost per useful completion

Price per token does not describe the cost of finishing a task. Include costs from failed attempts wherever they are actually billed. Whether tokens generated before a timeout are billable depends on the provider contract and usage record. Broken JSON may also incur charges. Retries, fallback models and human review add downstream costs.

```text
cost per useful completion = cost of all attempts / useful completions
```

The example computes a simplified direct token cost from logged usage and hypothetical prices. It does not model downstream handling, tiered billing or provider-specific cancellation rules. A real harness must supply usage appropriate to the billing policy being evaluated.

When prices are supplied and useful completions are zero, the report and selection use **infinity**, including when the recorded total cost is zero. This is an explicit decision convention: a candidate delivering no useful result does not receive a favourable per-result cost. Without any supplied price file, cost is unknown and the entire comparison uses two objectives. A supplied price file must cover **every candidate**, even one that later fails a gate; partial or empty coverage is an input error.

## Hard constraints first, then trade-offs

The implemented empirical gates are:

- useful-completion rate at least a chosen threshold;
- malformed-output rate at most a chosen limit, with all attempts in its denominator;
- p95 **attempt duration to completion or cancellation** within a chosen budget.

A malformed limit is not a minimum schema-valid-rate guarantee. Timeouts are another status: 12 correct responses and four short timeouts can meet all the default gates despite only 75% schema-valid responses. A production requirement on schema-valid rate or timeout frequency needs an explicit additional gate. This example reports timeout rate without imposing such a gate.

The quality gate compares the empirical rate, not the lower Wilson confidence bound. Gate lines keep the usual rounding unless it would misstate a value: a 0.4 ms budget prints as `0.4 ms`, not `0 ms`, and a malformed rate of 0.0625 against a 0.0624 limit prints both exactly. Other requirements — data region, retention, licence or context capacity — must be checked separately; the evaluator does not verify them.

Among candidates passing all implemented gates, use the **Pareto front**. A candidate is dominated when another is at least as good on every selected objective and strictly better on at least one. The objective set stays global:

- Without `--prices`: higher quality and lower p95 attempt duration, two objectives.
- With complete `--prices`: those two plus lower cost per useful completion, three objectives.

Cost is never silently dropped from individual pairs. The front exposes remaining trade-offs; product requirements determine which survivor to choose. A `PASS` in this report means that a candidate met empirical thresholds on the supplied cases. It does not certify population quality, a latency SLO or an independent review.

## A complete replay-log example

One JSON Lines file represents each candidate. The candidate name is its **file stem**, so `candidate_a.jsonl` maps to `candidate_a` in the price file. Stems must not start with `_`: price-file keys with that prefix, such as `_note`, are comments, so the evaluator rejects a file like `_a.jsonl` before reading any log. A harness would write records such as:

```json
{"case_id": "c04", "slice": "billing", "expected": "billing", "status": "timeout",
 "output": null, "elapsed_ms": 4000, "input_tokens": 305, "output_tokens": 160}
```

The harness assigns `ok`, `malformed`, `timeout` or `abstain` after checking the response; the evaluator trusts that classification rather than parsing raw model responses itself. The preserved fixtures cover correct routes, wrong labels, malformed output, timeouts and correct abstentions. They contain **zero incorrect abstentions**. A dedicated regression test checks that abstaining on an answerable case scores zero usefulness and increments the wrong-abstention count.

Before reporting, validation checks required fields, their types, unique case IDs, identical gold labels and slices across logs, and identical case sets. `output` must be a string or null for every status; `ok` requires a string and `abstain` requires null. Durations must be finite and positive; token counts must be integers between zero and `2**63-1`. That explicit numeric bound is an input-safety limit, not a provider's context limit. Text must be valid Unicode, so an escaped unpaired surrogate such as `"\ud800"` is rejected. Case IDs, slices and candidate names appear in the report and therefore may not contain control characters or line separators; a slice cannot add a fake line to the report. Accented letters and emoji remain valid, and `expected` and `output` may span several lines.

Rates supplied through CLI flags must be finite and in `0..1`; the duration budget must be finite and positive. Prices must be finite and nonnegative. Invalid UTF-8, oversized JSON integers, numeric conversion or cost overflow, bad types, invalid text and incomplete price coverage produce **exit 2**, a reason on stderr, and no stdout report or traceback. All report lines are built before printing, so later validation cannot leave a partial report. If the console encoding cannot represent the report, the run also exits 2 rather than printing part of it.

The implementation uses Python 3.11 and the standard library. It is split into small modules for validation, metrics and report generation, with a short CLI. Download **all** these files and retain the layout:

- [`replay_eval.py`]({{ '/assets/examples/model-selection-quality-latency/replay_eval.py' | relative_url }}) — CLI and example API;
- [`replay_validation.py`]({{ '/assets/examples/model-selection-quality-latency/replay_validation.py' | relative_url }}), [`replay_metrics.py`]({{ '/assets/examples/model-selection-quality-latency/replay_metrics.py' | relative_url }}), [`replay_report.py`]({{ '/assets/examples/model-selection-quality-latency/replay_report.py' | relative_url }}) — companion modules;
- [`test_replay_eval.py`]({{ '/assets/examples/model-selection-quality-latency/test_replay_eval.py' | relative_url }}), [`test_validation_cli.py`]({{ '/assets/examples/model-selection-quality-latency/test_validation_cli.py' | relative_url }}), [`test_selection.py`]({{ '/assets/examples/model-selection-quality-latency/test_selection.py' | relative_url }}) — full test suite;
- [`fixtures/candidate_a.jsonl`]({{ '/assets/examples/model-selection-quality-latency/fixtures/candidate_a.jsonl' | relative_url }}), [`fixtures/candidate_b.jsonl`]({{ '/assets/examples/model-selection-quality-latency/fixtures/candidate_b.jsonl' | relative_url }}), [`fixtures/prices.json`]({{ '/assets/examples/model-selection-quality-latency/fixtures/prices.json' | relative_url }}) — unchanged synthetic fixtures;
- [`pyproject.toml`]({{ '/assets/examples/model-selection-quality-latency/pyproject.toml' | relative_url }}), [`uv.lock`]({{ '/assets/examples/model-selection-quality-latency/uv.lock' | relative_url }}), [`README.md`]({{ '/assets/examples/model-selection-quality-latency/README.md' | relative_url }}) — project and reproduction instructions.

Run with a plain interpreter:

```bash
python3.11 -m unittest -v
python3.11 replay_eval.py fixtures/candidate_a.jsonl fixtures/candidate_b.jsonl \
  --prices fixtures/prices.json --fixture-provenance
```

Or create an isolated environment with modern uv tooling:

```bash
uv run --python 3.11 python -m unittest -v
uv run --python 3.11 python replay_eval.py fixtures/candidate_a.jsonl \
  fixtures/candidate_b.jsonl --prices fixtures/prices.json --fixture-provenance
```

`--fixture-provenance` explicitly labels this demonstration as hand-designed. Without it, arbitrary external logs receive neutral wording: supplied `elapsed_ms`, provenance not verified. The flag records a declaration, not a verification of the input's origin.

Defaults are `--min-quality 0.70 --max-malformed-rate 0.10 --max-p95-ms 3000`. Exit zero means a report was produced, including when every candidate was rejected.

## Reading the executed report

The correction run used the command above with uv's `--offline` option and an isolated Python 3.11 environment. Its output was:

```text
paired cases: 16 (identical case_id, expected, slice)
duration source: FIXTURE elapsed_ms from the logs (hand-designed, not measured inference)
statistics: formula demonstration only; deterministic curated fixtures provide no sampling evidence
percentile method: nearest-rank, rank = ceil(p/100 * n); p50 is lower median for even n
Pareto objectives: quality + p95 attempt duration + cost per useful completion (three objectives)

[candidate_a]
  useful completions: 13/16 = 0.812 (Wilson 95% formula 0.570-0.934; denominator = all attempts)
  status counts: abstain=2 malformed=1 ok=13 timeout=0
  timeout rate: 0.000
  wrong answers: 2  wrong abstentions: 0
  routing correct: 11/13; correct abstentions: 2/3
  by slice: account=3/3 billing=4/5 out_of_scope=2/3 shipping=4/5
  response duration ms, schema-valid only: n=15 p50=1310 p95=1880 max=1880
  attempt duration to completion or cancellation ms, all attempts: n=16 p50=1310 p95=1880 max=1880
  mean output tokens per attempt: 60.9
  cost units: total=3.9865 per useful completion=0.3067

[candidate_b]
  useful completions: 10/16 = 0.625 (Wilson 95% formula 0.386-0.815; denominator = all attempts)
  status counts: abstain=1 malformed=1 ok=12 timeout=2
  timeout rate: 0.125
  wrong answers: 3  wrong abstentions: 0
  routing correct: 9/13; correct abstentions: 1/3
  by slice: account=2/3 billing=4/5 out_of_scope=1/3 shipping=3/5
  response duration ms, schema-valid only: n=13 p50=630 p95=720 max=720
  attempt duration to completion or cancellation ms, all attempts: n=16 p50=630 p95=4000 max=4000
  mean output tokens per attempt: 37.6
  cost units: total=0.7452 per useful completion=0.0745

WARNING: mean output tokens differ by more than 25%; duration and cost partly reflect output length, not only the model.

paired candidate_a vs candidate_b: both=8 only_candidate_a=5 only_candidate_b=2 neither=1
  exact McNemar two-sided p=0.453 on 7 discordant cases

empirical hard constraints: quality >= 0.70, malformed rate <= 0.10, p95 attempt duration <= 3000 ms
  candidate_a: PASS
  candidate_b: REJECT: quality 0.625 < 0.700; p95 attempt duration 4000 ms > 3000 ms
pareto front among passing candidates: candidate_a

measured: evaluator wall time 18.1 ms via time.perf_counter() (this machine's parsing/report time, not model latency)
```

Every candidate duration comes from the synthetic input. The final line measures only parsing and report generation on the execution machine; it varies between runs and says nothing about model speed. The 18.1 ms shown comes from the original correction run; a rerun after the final code changes reproduced every other line exactly and timed 0.4 ms.

**Completion-only reporting can hide timeouts.** B's schema-valid responses have p95 720 ms. Its all-attempt duration to completion or cancellation has p95 4000 ms, while timeout rate is 2/16, or 0.125. The latter duration includes the wait until cancellation; it does not reveal how long those requests would have needed to finish. A shorter deadline could reduce that percentile without increasing usefulness.

**The statistical numbers demonstrate arithmetic.** A's Wilson formula yields 0.570–0.934 and B's 0.386–0.815. Seven constructed cases are discordant, five favouring A and two favouring B. Doubling the smaller binomial tail gives `58/128 = 0.453125`. None of these designed numbers establishes uncertainty, a chance explanation or equivalence for real traffic. Interval overlap supplies no paired significance conclusion.

**Routing and abstention can disagree with the average.** A routes 11/13 answerable cases correctly and abstains correctly on 2/3 out-of-scope cases. B achieves 9/13 and 1/3. Those separate denominators make the equal-usefulness aggregate easier to assess.

**Failure costs remain included.** B's hypothetical cost per useful completion is about a quarter of A's. A also produces around 61 output tokens per attempt versus B's 38. A real comparison should constrain the output format and investigate how much of the observed duration and cost difference comes from output length.

**The gates precede the front.** A passes the default empirical gates even though its Wilson lower bound is only about 0.570; the gate is not a lower-bound requirement. B fails quality and the attempt-duration budget. Relaxing those two thresholds lets both enter selection:

```bash
python3.11 replay_eval.py fixtures/candidate_a.jsonl fixtures/candidate_b.jsonl \
  --prices fixtures/prices.json --fixture-provenance \
  --min-quality 0.50 --max-p95-ms 5000
```

The executed selection excerpt is:

```text
empirical hard constraints: quality >= 0.50, malformed rate <= 0.10, p95 attempt duration <= 5000 ms
  candidate_a: PASS
  candidate_b: PASS
pareto front among passing candidates: candidate_a, candidate_b
```

A has higher quality and lower p95 attempt duration; B has lower cost per useful completion. With prices, both are nondominated. Omit `--prices` and the same relaxed comparison explicitly has two objectives, with A alone on the front. That change follows a declared global objective set, not missing cost in selected pairs.

## What the tests establish

The full standard-library `unittest` suite has **55 tests**. New regression cases first failed against the implementation they targeted, then the full suite passed in both source and public download copies under isolated CPython 3.11.12. The captured runs retain exact commands, stdout, stderr and exit codes.

Coverage includes CLI rate ranges and finite budgets; status types before membership checks; output types for every status; null abstentions; huge numbers, cost overflow and invalid UTF-8; unpaired surrogates and control characters in text; reserved underscore candidate names; complete prices; neutral and explicit fixture provenance; incorrect-abstention scoring; nearest-rank p50/p95; unrounded display of fractional or near-threshold gate values; numerical Wilson limits and exact McNemar values; all-attempt denominators; timeout reporting; and constraints before globally consistent Pareto selection, including zero-useful cost infinity.

These tests verify evaluator behaviour. They perform no inference, load generation, streaming or TTFT measurement. The harness that produces real logs must control load, record timing boundaries, cancellations, usage and cache policy.

## Dates and reproduction environment

28 March 2024 is the series' **historical editorial label**, not a publication or execution date. Python 3.11 was released on 24 October 2022. The code uses standard-library APIs available by the label date; Wilson, exact McNemar and right-censoring methods also predate it. No later provider feature is required.

The correction was executed on **12 October 2026**, using CPython 3.11.12, macOS 26.5.2 (Darwin 25.5.0, arm64) and uv 0.7.8. uv is modern reproduction tooling, expressly separate from the historical label. The code also runs directly with Python 3.11. These are execution details for the prepared example, not an account of the author's personal work or a published deployment.

A model-selection study starts by defining the task, the meaning of usefulness, separate challenge slices, a representative holdout, sampling units and operating conditions. Freeze the empirical gates and timeout policies, run candidates on paired cases, retain failures, and select among the survivors using the same objectives throughout. The result is a defensible trade-off tied to a particular workload.

### Sources

- [Liang et al., HELM abstract page](https://arxiv.org/abs/2211.09110) — v1 16 November 2022, v2 1 October 2023; abstract-only attribution.
- [Python 3.11 time documentation](https://docs.python.org/3.11/library/time.html#time.perf_counter) — duration measurement and clock differences.
- [Python 3.11 statistics documentation](https://docs.python.org/3.11/library/statistics.html) — arithmetic median and interpolated quantiles.
- [Python 3.11.0 release page](https://www.python.org/downloads/release/python-3110/) — release date 24 October 2022.
- [NIST/SEMATECH handbook: confidence limits for a proportion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) — Wilson score interval reference.
- [Fagerland, Lydersen and Laake (2013), McNemar methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/) — exact conditional binomial calculation and its limitations.
- [NIST/SEMATECH handbook: censoring](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm) — observations ending before an event occurs.
