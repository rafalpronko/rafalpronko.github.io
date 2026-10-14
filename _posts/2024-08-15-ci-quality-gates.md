---
layout: post
lang: en
title: "A green build is not a release decision: quality gates for ML in CI"
date: 2024-08-15 09:00:00 +0200
permalink: /blog/ci-quality-gates/
author: Rafał Prońko
categories: [mlops]
description: "A small fail-closed release gate in Python: quality floors, per-slice regression checks and evaluation evidence bound to one artifact digest, plus where CI actually enforces it."
image: /assets/images/ci-quality-gates.svg
---

# A green build is not a release decision: quality gates for ML in CI

![Diagram: a successful build produces one artifact digest; an evaluation report is bound to that digest, a corpus version and a policy version; a fail-closed release gate either blocks deployment or lets the same digest go to a staged rollout.]({{ '/assets/images/ci-quality-gates.svg' | relative_url }})

<time datetime="2024-08-15">15 August 2024</time>

A CI pipeline for a model service can be entirely green while the model inside it is worse than the one in production. The container builds, the unit tests pass, the inference code loads the weights and returns a response in the right shape. None of that says whether the answers are good enough to replace what users get today.

For ordinary application code, a passing test suite is a reasonable proxy for "safe to ship", because the tests exercise the behaviour that changed. For an ML, data or LLM application the behaviour that changed often lives in weights, prompts, retrieval indexes or training data. Code tests touch it barely or not at all.

So I separate two questions. Did the build work? And may this particular artifact be released? The second one needs its own job, its own evidence and its own failure mode. In this post I build a small release gate in Python and then look at the part that is easy to get wrong: where CI actually enforces the decision, and where it only looks like it does.

## What a successful build proves

A build job proves that the source could be packaged and that whatever tests ran in that job passed. For a model service that typically means: dependencies resolved, the image or wheel was produced, the serving code imports, perhaps a smoke test returned a valid response.

It does not prove that:

- the model's quality on a representative evaluation set meets the product's minimum,
- the new model is not worse than the one currently deployed,
- quality holds for the subgroups that matter, not only on average,
- the evaluation numbers someone shows you were measured on this artifact.

The last point sounds pedantic until a pipeline rebuilds the image at deploy time, or an evaluation report from yesterday's candidate is still lying in a bucket under a generic name.

## Three questions for a release gate

I want the gate to answer three questions, and to block if any answer is "no" or "unknown".

**Is it good enough?** An absolute floor per metric. Below it, the candidate does not ship even if it is better than the current model. If production is already below the floor, that is a separate incident, not a reason to accept another weak model.

**Is it not worse than what is running?** A regression tolerance against a baseline: the currently deployed model evaluated on the same corpus. A floor alone lets quality drift down release by release, as long as it stays above the line.

**Is the evidence about this exact thing?** The evaluation report must name the artifact it measured (by content digest), the evaluation corpus and its version, and the policy version it was produced for. Otherwise I am comparing numbers that may describe something else.

Both checks — floor and regression — run per slice, not only on the aggregate. A slice is any subgroup the product cares about: language, channel, customer tier, document type. The aggregate is a weighted average of slices, so a large slice that improves can hide a small one that collapses.

The numbers I use below are fixture values for a synthetic example. They are not benchmark results and not a recommendation. A floor comes from what the product can tolerate; a regression tolerance must be wider than the measurement noise you can expect at your slice sizes. Neither has a universal value.

## The example: synthetic counts and one policy

The scenario is a support-ticket intent classifier. The evaluation produces counts — correct predictions and total examples — for the whole corpus and for four slices. Counts are better than pre-computed rates: they are integers, they carry the sample size, and the gate computes the rate itself.

The policy is a versioned JSON file:

```json
{
  "policy_version": "support-intent-release/3",
  "eval_corpus": {"id": "support-intents-synthetic", "version": "3"},
  "overall_floor": 0.85,
  "critical_floor": 0.8,
  "critical_slices": ["lang=pl", "channel=phone"],
  "max_regression": 0.02,
  "min_slice_total": 100
}
```

The baseline report holds counts for the current production model on the same corpus. The candidate report adds the artifact digest and the policy version it was evaluated for.

Here is the interesting fixture, `candidate_slice_fail.json`, against the baseline:

| Slice | Baseline | Candidate |
|---|---|---|
| overall | 860 / 1000 = 0.860 | 885 / 1000 = 0.885 |
| lang=en | 520 / 600 = 0.867 | 545 / 600 = 0.908 |
| lang=pl (critical) | 340 / 400 = 0.850 | 340 / 400 = 0.850 |
| channel=email | 600 / 700 = 0.857 | 650 / 700 = 0.929 |
| channel=phone (critical) | 260 / 300 = 0.867 | 235 / 300 = 0.783 |

The aggregate went up by 2.5 points. A gate that looks only at the overall number lets this candidate through. Phone tickets — a critical slice — got markedly worse, and the email improvement more than paid for it in the average.

## The gate

The whole gate is one file using only the standard library. The core is `evaluate`, which returns a list of reasons to block; an empty list means the candidate is eligible.

```python
def evaluate(policy, candidate, baseline, artifact_sha256):
    """Return blocking reasons. An empty list means: eligible for release."""
    check_policy(policy)
    check_identity(candidate, "candidate", require_policy_version=True)
    check_identity(baseline, "baseline")
    cand = slice_counts(candidate, "candidate")
    base = slice_counts(baseline, "baseline")
    reasons = []
    if candidate.get("artifact_sha256") != artifact_sha256:
        reasons.append("report does not describe this artifact digest")
    if candidate.get("policy_version") != policy["policy_version"]:
        reasons.append("report was produced for another policy version")
    for label, report in (("candidate", candidate), ("baseline", baseline)):
        if report.get("eval_corpus") != policy["eval_corpus"]:
            reasons.append(f"{label} evaluated on {report.get('eval_corpus')}, "
                           f"policy requires {policy['eval_corpus']}")
    critical = set(policy["critical_slices"])
    for name in sorted({"overall", *critical, *base}):
        if name not in cand:
            reasons.append(f"{name}: missing from candidate report")
            continue
        correct, total = cand[name]
        rate = correct / total
        if total < policy["min_slice_total"]:
            reasons.append(f"{name}: {total} examples, policy needs "
                           f"{policy['min_slice_total']}")
        floor = (policy["overall_floor"] if name == "overall"
                 else policy["critical_floor"] if name in critical else None)
        if floor is not None and rate < floor:
            reasons.append(f"{name}: {rate:.4f} below floor {floor}")
        if name not in base:
            reasons.append(f"{name}: no baseline to compare against")
        elif rate < base[name][0] / base[name][1] - policy["max_regression"]:
            reasons.append(f"{name}: {rate:.4f} regressed from baseline "
                           f"{base[name][0] / base[name][1]:.4f}")
    return reasons
```

A few decisions are deliberate.

The `artifact_sha256` passed in is not taken from the report. The command-line wrapper computes it from the artifact file on disk. The report only *claims* which artifact it measured; the gate checks that claim against the thing that is about to be released.

The set of slices to check is the union of `overall`, the critical slices and every slice in the baseline. A slice that exists in the baseline but is missing from the candidate report blocks. So does a critical slice without a baseline: a comparison that cannot run is a failed comparison, not a skipped one.

Floors apply to the aggregate and to critical slices. The regression tolerance applies to every baseline slice, critical or not, so a non-critical slice cannot quietly fall apart either.

## Missing and malformed evidence must block

The most dangerous gate is one that passes when it receives nothing useful. Python makes a few of these failures easy to write.

`json.loads` accepts `NaN`, `Infinity` and `-Infinity` by default, although they are not valid JSON. A NaN then flows into comparisons where it is always false: `nan < 0.8` is `False`, so a check written as `if rate < floor: block()` lets NaN pass. A number literal such as `1e999` is valid JSON syntax and parses to `float("inf")` without going through the NaN/Infinity hook at all. With duplicate keys, the last value silently wins.

Booleans are the other trap. `bool` is a subclass of `int` in Python, so `True >= 0.8` is `True`, and `isinstance(True, int)` holds. A policy with `"critical_floor": true` or a report with `"correct": true` would be treated as numbers.

So the loader is strict:

```python
def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate JSON key in {keys}")
    return dict(pairs)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_reject_constant,
                      object_pairs_hook=_reject_duplicates)
```

And so are the type checks for counts and policy values:

```python
def _is_count(x):
    return type(x) is int and x >= 0  # type() excludes bool


def _is_rate(x):
    return type(x) is float and math.isfinite(x) and 0.0 <= x <= 1.0


def _is_identifier(x):
    return isinstance(x, str) and bool(x.strip())


def check_identity(report, label, require_policy_version=False):
    if not isinstance(report, dict):
        raise ValueError(f"{label} must be a JSON object")
    corpus = report.get("eval_corpus")
    if (not isinstance(corpus, dict)
            or not all(_is_identifier(corpus.get(key)) for key in ("id", "version"))):
        raise ValueError(f"{label}: eval_corpus needs nonempty string id and version")
    if require_policy_version and not _is_identifier(report.get("policy_version")):
        raise ValueError(f"{label}: policy_version must be a nonempty string")
```

`type(x) is int` rejects `True`, where `isinstance` would not. `math.isfinite` catches the `1e999` case. Every slice must contain exactly `correct` and `total`, with `total > 0` and `correct <= total`. The policy itself is validated before anything is compared: a policy with a missing field is not a lenient policy.

The policy, candidate and baseline must each identify the corpus with a nonempty string `id` and `version`. The policy and candidate must also have a nonempty string `policy_version`; the baseline supplies comparison counts on the same corpus, not a release decision under the candidate policy. Blank strings, missing fields, booleans and other non-string identifiers are malformed evidence. Matching empty dictionaries or matching empty version labels cannot establish identity. These checks run before identity comparisons, so malformed identifiers produce exit code `2`.

The command-line wrapper turns this into exit codes: `0` for eligible, `1` for a candidate that violates the policy, `2` when the gate cannot evaluate at all — a missing file, unparsable JSON, a malformed policy. Any non-zero code must stop the release; the distinction between `1` and `2` is only for the person reading the log.

This validator is a toy. It checks that a report is well-formed and consistent with the policy. It cannot tell whether the numbers were produced by an honest evaluation run — anyone who can write the JSON file can write passing numbers. Which brings me to the digest.

## A digest proves integrity, not authenticity

Binding the report to the artifact's SHA-256 digest guarantees one thing: the bytes being released are the bytes the report names. It prevents mix-ups — evaluating one build and shipping another, or reusing an old report.

It does not prove who produced the report or that the evaluation really happened. For that, the evaluation result must come from a producer you trust: a job whose identity you can verify, writing to a store that ordinary contributors cannot write to, ideally with a signature over the digest, corpus version and result. In this example I deliberately stop at integrity, and that is a real limitation, not a detail.

The same applies to the corpus and the policy. Pinning `support-intents-synthetic` version `3` only helps if version `3` is immutable. If someone can edit the corpus in place, the version label is a name, not evidence.

## Running it

The downloadable files contain the gate, the tests, the fixtures and a `pyproject.toml` with no dependencies. They need Python 3.12 and nothing else.

On the eligible candidate:

```text
$ python3.12 release_gate.py --policy fixtures/policy.json --candidate fixtures/candidate_ok.json --baseline fixtures/baseline.json --artifact fixtures/candidate.bin
ELIGIBLE: sha256=266f5df09eb725d6fa9e50ca383be21ced270c14985795a4cf7dbd219790cf6a
$ echo $?
0
```

On the candidate whose aggregate improved:

```text
$ python3.12 release_gate.py --policy fixtures/policy.json --candidate fixtures/candidate_slice_fail.json --baseline fixtures/baseline.json --artifact fixtures/candidate.bin
BLOCKED: channel=phone: 0.7833 below floor 0.8
BLOCKED: channel=phone: 0.7833 regressed from baseline 0.8667
$ echo $?
1
```

The overall rate passes its floor and does not regress, so it does not appear. The phone slice fails both rules.

The test suite (`python3.12 -m unittest -v test_release_gate`) has 33 tests. They check that:

1. A candidate that meets every rule is eligible.
2. An aggregate pass does not hide a failing critical slice.
3. A regression on a non-critical slice blocks, and so does a slice below the minimum size.
4. A wrong artifact digest, a wrong corpus id or version (in the candidate or the baseline) and a wrong policy version block.
5. A missing digest field, a missing critical slice, a critical slice without a baseline, a report without slices and a policy without a required field all block.
6. `NaN`, `-Infinity`, an overflowing `1e999`, a NaN count, boolean floors and counts, `correct > total` and duplicate JSON keys are rejected. Empty, blank, missing, malformed and boolean corpus IDs/versions and policy versions are rejected, including when the policy and reports share the same invalid labels.
7. The CLI exits with `0`, `1` and `2` for an eligible candidate, a blocked one and a missing report; malformed identifiers also exit with `2`.

They test this gate's logic on fixtures. They say nothing about whether a classifier at 0.88 is good enough for anyone's support queue.

## Wiring the gate into CI

A gate script only matters if its failure stops the deployment. In GitHub Actions the basic mechanism is `needs`: a job with `needs` waits for the listed jobs, and if one of them fails or is skipped, the dependent job is skipped too — unless it has a condition that makes it run anyway, such as `always()`.

Here is a fragment. It shows dependency structure, not a working pipeline: triggers, permissions, checkout, Python setup, the artifact registry and credentials are deliberately missing, and the `ci/*.sh` scripts are placeholders for your own build, evaluation and deploy steps.

```yaml
# Fragment only. Not a complete workflow.
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: ./ci/build_and_push.sh   # build once, push, record the sha256 digest
  quality:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: ./ci/evaluate.sh         # pull the artifact BY DIGEST, write candidate.json
      - run: python3.12 release_gate.py --policy policy.json --candidate candidate.json --baseline baseline.json --artifact candidate.bin
  deploy:
    needs: [build, quality]
    runs-on: ubuntu-latest
    steps:
      - run: ./ci/deploy.sh           # deploy the SAME digest; never rebuild here
```

What matters is what is absent:

- **No `continue-on-error` on the gate step or the quality job.** It turns a failed gate into a successful job, and the deploy job no longer sees a failure.
- **No `always()` (or any status-overriding condition) on `deploy`.** That is exactly the escape hatch the documentation describes for running a job whose dependencies did not succeed.
- **No gate that only prints a warning.** An annotation nobody is required to act on is a report, not a gate. If you want an advisory check while you calibrate a new metric, name it as advisory and keep it out of the release path.
- **No rebuild in `deploy`.** The deploy job must release the digest that `quality` evaluated. Rebuilding — even from the same commit — can produce different bytes: a floating base image, an unpinned dependency, a different download of the weights. Pass the digest between jobs and deploy by digest.

## Merge enforcement is not release enforcement

There are two different places where a check can be enforced, and they protect different things.

**Merge enforcement** is branch protection with required status checks. It controls what can be merged into a branch. It is useful, but it has caveats that matter for gates:

- By default, branch protection rules do not apply to people with admin permissions on the repository unless you enable the option that applies them to administrators as well.
- A required check is satisfied not only by a successful status but also by a skipped one; some versions of GitHub's documentation list neutral as well. A job skipped because of a condition reports success. When a job fails, jobs that depend on it are skipped rather than failed. Requiring a downstream job — say, `deploy` or a summary job — can therefore leave a failing gate with a skipped, merge-permitting status.
- In the fragment above, require **both `build` and `quality`** as branch-protection checks. Requiring `quality` alone is insufficient because its `needs: build` means it is skipped if `build` fails. Keep both required jobs free of conditions that can skip them. Alternatively, require an always-running reporting job with `needs: [build, quality]` and `always()` that explicitly fails unless **both dependency results are `success`**. Failure, cancellation, skipped or unknown results must block. Test those cases in your workflow; this fragment does not implement or execute that reporting job. Never put `always()` on `deploy`.
- If a workflow is not triggered at all (path or branch filters), its required checks stay pending and block the merge. That is safe but annoying, and it tempts people to loosen the rule.

**Release enforcement** is about who can change production. A required check on `main` does not stop a deployment from another branch, another workflow, a manual script or someone's laptop. The decision is actually enforced wherever the production credentials live. If only the deploy job of one workflow can obtain the credentials, and that job cannot start unless the gate succeeded, the gate is enforced. If a long-lived deploy token is available to every workflow or to developers, the gate is a convention.

The strongest form moves the check closer to production: the registry or deployment system accepts only digests for which a trusted evaluation record exists. The CI dependency is then a second line, not the only one.

## What the gate does not cover

**Freshness.** An evaluation is evidence about an artifact, a corpus version, a baseline and a policy at a point in time. When the baseline changes because something else was deployed, or the policy version changes, the old result should not count. Requiring the report's policy version to match is a minimal version of that; a real system also needs an expiry and a link to the exact baseline used.

**Flaky checks.** An evaluation that fails intermittently trains people to rerun until green or to disable it. The honest options are to fix it, or to quarantine it explicitly: recorded, owned, with an expiry date, and visible in the release record. "Temporarily skipped" without an owner tends to become permanent.

**Exceptions.** Sometimes a release must go out despite a failing check — a security fix, for example. An exception should be an auditable record: who approved it, for which digest, which rule it overrides, and when it expires. Not an edited threshold and not an `always()` added in a hurry.

**Statistical uncertainty.** With 300 phone tickets and accuracy around 0.87, the standard error of the estimate is about 0.02 — the same size as the regression tolerance in my fixture policy. Differences that small can be noise. `min_slice_total` is only a crude guard. For real decisions, consider confidence intervals, paired comparisons on the same examples, or a tolerance that depends on slice size. And the more slices and metrics you test, the more often one will fail by chance.

**Holdout leakage through repeated tuning.** If people iterate on a model until it passes the gate on the same evaluation corpus, the corpus gradually becomes part of the training signal. The gate then measures how well the model was fitted to the gate. Keep a separate set for the final decision, limit how often it is used, and refresh it.

**Production safety.** The gate compares a candidate with a baseline on a fixed corpus. Production traffic drifts, latency and cost change, and LLM outputs can fail in ways no fixed metric captures. A passed gate is permission to start a staged rollout — a small share of traffic, monitored, with a tested rollback to the previous digest — not a guarantee.

**Scope of this example.** The code was tested with Python 3.12 on synthetic fixtures. It checks one metric (accuracy from counts). It does not verify the baseline's own digest against the deployed artifact, and a slice that appears only in the candidate is checked only if it is declared critical. The GitHub behaviour described above comes from the documentation listed below; platforms and their defaults change, so check the documentation for the version you run.

## Where I would start

Not with the threshold values. First I would make the gate fail closed: missing report, malformed numbers, wrong artifact or wrong corpus all block. Then I would make sure the deploy job cannot run without the gate, and that the deploy credentials are not available anywhere else. Only after that would I spend time arguing about floors and tolerances — because a well-chosen threshold in a gate that can be skipped protects nothing.

### Downloads

- [`release_gate.py`]({{ '/assets/examples/ci-quality-gates/release_gate.py' | relative_url }}) — the gate and its CLI
- [`test_release_gate.py`]({{ '/assets/examples/ci-quality-gates/test_release_gate.py' | relative_url }}) — 33 `unittest` tests
- [`pyproject.toml`]({{ '/assets/examples/ci-quality-gates/pyproject.toml' | relative_url }}) — Python 3.12, no dependencies
- Fixtures: [`policy.json`]({{ '/assets/examples/ci-quality-gates/fixtures/policy.json' | relative_url }}), [`baseline.json`]({{ '/assets/examples/ci-quality-gates/fixtures/baseline.json' | relative_url }}), [`candidate_ok.json`]({{ '/assets/examples/ci-quality-gates/fixtures/candidate_ok.json' | relative_url }}), [`candidate_slice_fail.json`]({{ '/assets/examples/ci-quality-gates/fixtures/candidate_slice_fail.json' | relative_url }}), [`candidate.bin`]({{ '/assets/examples/ci-quality-gates/fixtures/candidate.bin' | relative_url }})

Keep the files in the same layout (`fixtures/` next to the scripts) so the tests find them.

### Sources

- [GitHub Docs: Using jobs in a workflow](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/using-jobs-in-a-workflow) — `needs`, skipping of dependent jobs after a failure or skip, and `always()` to run regardless.
- [GitHub Docs: About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) — required status checks, accepted statuses, and admin bypass by default.
- [GitHub Docs: Troubleshooting required status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks) — skipped jobs reporting success, dependent jobs of a failed job, workflows skipped by path or branch filters.
- [Python 3.12: json](https://docs.python.org/3.12/library/json.html) — `parse_constant`, `object_pairs_hook`, acceptance of `NaN`/`Infinity` and duplicate-key behaviour.
- [Python 3.12: hashlib](https://docs.python.org/3.12/library/hashlib.html) — `hashlib.file_digest`.
