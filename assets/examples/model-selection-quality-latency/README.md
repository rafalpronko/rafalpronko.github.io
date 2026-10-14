# Replay-log evaluator: quality, duration, failures and cost

Companion to *Choosing a model for the task: benchmark quality and latency,
not marketing labels*. Python 3.11+, standard library only; no model calls.

The preserved fixtures are hand-designed synthetic logs. Candidates are fictional,
`elapsed_ms` values are invented, and prices are hypothetical cost units. They cover
correct routes, wrong labels, malformed output, timeouts and correct abstentions;
a separate test covers incorrect abstention. Their unweighted aggregate evaluates
a constructed mixture, not production traffic. A correct route and a correct
abstention each count as one useful result; the report also separates them.

## Run

Download all listed files together; retain the `fixtures/` directory.
Modern uv tooling can create an isolated environment:

```bash
uv run --python 3.11 python -m unittest -v
uv run --python 3.11 python replay_eval.py \
  fixtures/candidate_a.jsonl fixtures/candidate_b.jsonl \
  --prices fixtures/prices.json --fixture-provenance
```

Or run directly:

```bash
python3.11 -m unittest -v
python3.11 replay_eval.py fixtures/candidate_a.jsonl fixtures/candidate_b.jsonl \
  --prices fixtures/prices.json --fixture-provenance
```

For external logs omit `--fixture-provenance`. The default report says
`elapsed_ms supplied by logs; provenance not verified`. The flag explicitly
declares hand-designed fixtures; it does not independently verify provenance.

Defaults: `--min-quality 0.70 --max-malformed-rate 0.10 --max-p95-ms 3000`.
Rates must be finite in `0..1`; the duration limit must be finite and positive.
Exit `0` means a report was produced, even if all candidates were rejected.
Invalid inputs produce exit `2`, a reason on stderr, no report on stdout and no
traceback. The report is built completely before printing; if stdout's encoding
cannot represent it (for example non-ASCII slice names under an ASCII console),
the run also exits `2` instead of printing part of it. UTF-8 output always works.

## Log format

One UTF-8 JSON object per line. Candidate name = file stem; stems must be unique
and must not start with `_`, because price-file keys starting with `_` (such as
`_note`) are comments. Such a stem is rejected before any log is read.
Every candidate must have the same case IDs, gold labels and slices. Duplicate
IDs, empty logs, missing fields and incorrect types are errors.

| Field | Type | Meaning |
|---|---|---|
| `case_id` | nonempty string, unique per log, no control characters | paired case identifier |
| `slice` | nonempty string, no control characters | segment; must match across candidates |
| `expected` | string or `null` | gold label; null means abstain |
| `status` | `ok`, `malformed`, `timeout`, `abstain` | classification assigned by the harness |
| `output` | string or `null` for every status | `ok` requires a string; `abstain` requires null; timeout may retain partial text |
| `elapsed_ms` | finite number > 0 | attempt duration to completion or cancellation |
| `input_tokens`, `output_tokens` | integers in `0..2**63-1` | usage per attempt, including billed failures |

Boolean numeric fields, nonfinite numbers, oversized JSON integers, conversion
or cost overflow and invalid UTF-8 are rejected. Every text field must be valid
Unicode: escaped unpaired surrogates such as `"\ud800"` are rejected. Report
identifiers (`case_id`, `slice`, candidate names) also reject control characters
and line or paragraph separators, so one value cannot create extra report lines;
other Unicode text, including emoji and accented letters, is accepted. `expected`
and `output` may contain newlines because they are compared, not printed. The token bound is input safety,
not a model context-window limit. Malformed/timeout outputs may be text or null;
they never score as useful. The evaluator trusts harness status classification,
rather than parsing raw model JSON itself.

## Metrics and selection

- **Useful rate:** correct routes plus correct abstentions / all attempts.
  Malformed output, wrong answers, wrong abstentions and timeouts score zero.
- **Duration:** schema-valid completed responses (`ok`, `abstain`) are shown
  separately. The all-attempt metric is **attempt duration to completion or
  cancellation**, with timeout rate beside it. A timed-out eventual completion
  time is right-censored; this program performs no censoring analysis and does
  not estimate latent completion-time p95. Use comparable timeout policies.
- **Percentiles:** nearest rank `ceil(p/100 * n)` without interpolation. p50 is
  the lower median for even n, unlike the arithmetic mean of the middle values.
  p95 of 16 values is the maximum; it gives no exact one-in-twenty guarantee.
- **Cost:** direct token cost of all logged usage / useful completions. Timeout
  billing depends on the provider contract; the harness must supply appropriate
  usage. Downstream handling costs are outside this simplified model. A priced
  candidate with zero useful completions has cost per useful completion
  **infinity**, even when total cost is zero.
- **Global objectives:** without `--prices`, quality and p95 attempt duration
  form a two-objective comparison. With `--prices`, every candidate must have
  finite nonnegative input/output prices, adding cost as the third objective.
  Empty or partial prices are exit-2 errors before reporting, including prices
  missing for candidates that would fail gates. Cost is never omitted pairwise.
- **Empirical gates first:** useful rate, malformed rate and p95 attempt duration.
  The quality gate uses the observed rate, not the lower Wilson bound.
  Thresholds and rejection reasons use the usual rounding (`0.70`, `3000 ms`)
  unless it would misstate a value; then the shortest exact form is printed,
  such as `<= 0.4 ms` or `malformed rate 0.0625 > 0.0624`.
  A malformed-rate limit is not a schema-valid-rate minimum: timeouts can reduce
  schema-valid rate without increasing malformed rate. Timeout rate is reported,
  without a separate timeout gate. Then compute the Pareto front of survivors.
- **Evaluator wall time:** the last line measures parsing/report generation using
  `perf_counter()`. It is the only measured duration in the fixture report.

## Statistical scope

Wilson 95% score limits and exact two-sided McNemar p-values on these deterministic
curated fixtures are **formula demonstrations**, not uncertainty or evidence
about real models. McNemar doubles the smaller binomial tail conditional on the
discordant count. No discordant pairs gives p=1. Fagerland et al. found this
exact conditional test highly conservative with poor power; its p-values here
only illustrate the formula.

Real inference needs a target population and representative independent sampling
units. Keep an untouched representative holdout separate from oversampled
challenge slices; the evaluator does not implement production weighting. Group
threads, customers and duplicates in splits, and account for dependence in the
analysis. Pairing candidates does not make correlated tickets independent.
Keeping gold labels private does not guarantee absence of training contamination.

Overlapping marginal intervals are not a paired significance test. A nonsignificant
result is insufficient evidence against the null, not proof of equivalence.

References: [NIST Wilson limits](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[Fagerland et al. (2013), McNemar](https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/),
[NIST censoring](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm).

## Files and tests

- `replay_eval.py` — CLI and re-exported example API.
- `replay_validation.py` — schema, numeric validation, prices and pairing.
- `replay_metrics.py` — scoring, percentiles, statistical formulas and selection.
- `replay_report.py` — report construction and empirical gates.
- `test_replay_eval.py`, `test_validation_cli.py`, `test_selection.py` — full suite,
  55 tests, including CLI subprocess regressions, text and candidate-name
  validation, threshold display and incorrect abstention.
- `fixtures/candidate_a.jsonl`, `fixtures/candidate_b.jsonl`, `fixtures/prices.json`
  — unchanged synthetic logs and hypothetical prices.
- `pyproject.toml`, `uv.lock` — project metadata and lock; no dependencies.
- `README.md` — this documentation.

New regressions were run RED before implementation, followed by GREEN runs in
source and public copies. No inference, streaming, TTFT or load test is included.
The 2024-03-28 date is an editorial label; the recorded reproduction is from
2026-10-12 with modern uv 0.7.8 and CPython 3.11.12. All runtime APIs are from
Python 3.11 or earlier. No private-ticket history or real-model result is claimed.
