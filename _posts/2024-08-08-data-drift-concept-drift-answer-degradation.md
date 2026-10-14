---
layout: post
lang: en
title: "Data Drift, Concept Drift, and Answer Degradation Are Different Problems"
date: 2024-08-08 09:00:00 +0200
permalink: /blog/data-drift-concept-drift-answer-degradation/
author: Rafał Prońko
categories: [mlops, llmops]
description: "P(X) shift, P(Y|X) shift and a drop in answer quality need different diagnoses. Exact synthetic examples in Python 3.12 plus a stubbed RAG regression."
image: /assets/images/data-drift-concept-drift-answer-degradation.svg
---

# Data Drift, Concept Drift, and Answer Degradation Are Different Problems

![Three examples: data drift changes the input mix while the rule stays the same; harmful concept drift changes the rule with identical inputs; a retrieval regression changes the generator's context and lowers answer quality for fixed external queries and references.]({{ '/assets/images/data-drift-concept-drift-answer-degradation.svg' | relative_url }})

Editorial series date: <time datetime="2024-08-08">8 August 2024</time>. Written and examples executed on <time datetime="2026-10-12">12 October 2026</time>.

A monitoring dashboard turns red. Someone writes "the model drifted" in the incident channel, and the next sentence is usually "let's retrain".

That sentence hides three different claims. The inputs may look different from before. The relationship between inputs and the correct answer may have changed. Or users may simply be getting worse answers. These can happen together, separately, or not at all — and each one points to a different fix.

In this article I separate them on small, exact, synthetic populations, where I know the ground truth because I wrote it. Then I show a stubbed retrieval-augmented pipeline where external queries and reference answers stay fixed, but a retrieval configuration regression changes the generator's context and lowers answer quality.

## Three definitions I will hold to

Let `X` be the input at the stage being monitored and `Y` the correct outcome. In the classifier lab, `X` is the integer feature. In the RAG lab, distinguish the external query from the generator's retrieved context: a stable query distribution does not establish stable inputs at every stage.

**Data drift** here means a change in `P(X)`: the inputs arrive in different proportions. Gama et al. call drift that affects only `P(X)` *virtual drift*.

**Concept drift** here means a change in `P(Y|X)`: the distribution of correct outcomes for the same input changes. Gama et al. call this *real concept drift*. Their formal definition of concept drift uses the joint `P(X, Y)`, before distinguishing real from virtual drift. I use the narrower conditional meaning here. Such a change can harm a frozen model, but its effect must be measured: accuracy can stay the same or improve, and the optimal hard classification can remain unchanged.

**Answer degradation** is an *observation*: a quality metric got worse. It is not a diagnosis. Data drift, concept drift, a bug, a configuration change, an outage or a change in how you measure quality can all produce it.

Three consequences follow directly:

- Data drift and concept drift can coexist, but neither implies the other.
- Data drift does not have to lower quality. If the model is equally good on every kind of input, reshuffling the inputs changes nothing.
- Concept drift can be invisible in `X`. If only the rule changed, every input statistic stays flat.

Two more things are often mixed in.

**A base-rate change** is a change in the marginal `P(Y)`. It can follow from data drift (more inputs from a region where positives are common) or concept drift (the conditional rule changed). On its own it does not tell you which. **Strict label shift**, as defined by Lipton, Wang and Smola, requires `P(Y)` to change while `P(X|Y)` stays fixed. None of the changed populations below satisfy that assumption; the lab monitors label prevalence, not strict label shift.

**A change in how labels are produced is not automatically a change in the world.** If the definition of "fraud" or "resolved ticket" changed, if an annotation guideline was rewritten, or if an event stopped being logged, then `P(Y|X)` in your *data* moved while reality may not have. That is a measurement incident first.

## The lab: exact populations, a frozen model

The examples are deliberately small. A single integer feature `x` in `0..9`, a binary label, and a population written as counts per value of `x`: `{x: (positives, negatives)}`. There is no sampling and no random seed. Every number is an exact `Fraction`, so there are no confidence intervals to report and no significance to pretend.

The model is a fixed threshold, `predict 1 if x >= 5`. It is never retrained. In the reference population the labels are noisy: in slices `0..3` 10% are positive, in `6..9` 90%, and the slices next to the threshold are hard (40% positive at `x = 4`, 60% at `x = 5`). So the model is right 90% of the time on easy slices and 60% on hard ones, 84% overall.

Four changed populations are compared against this reference:

- **A** — more inputs from the upper easy tail and fewer from the lower easy tail, same `P(Y|X)`: pure data drift (covariate shift). The combined share of the easy tails stays at 4/5.
- **B** — identical `P(X)`, but at `x = 5` and `x = 6` the rule moved (10% and 40% positive): pure concept drift.
- **C** — more inputs near the threshold, same `P(Y|X)`: data drift that hurts an imperfect model.
- **D** — identical `P(X)`, but at `x = 2` and `x = 3` items became 90% positive: concept drift where the model predicts negative.

For each population the script reports five instruments:

- `TV P(X)` — total variation distance between the reference and current input distributions (0 = identical, 1 = disjoint).
- `P(Y=1)` — the label base rate, i.e. what a base-rate monitor sees.
- `acc` — accuracy on the whole current population, labels fully known.
- `fixed-mix` — current *per-slice* accuracy re-weighted to the *reference* `P(X)`. It answers: "would quality have changed if the input mix had not?"
- `flagged` — accuracy measured only on items the model predicted positive, imitating a system where labels arrive only for cases somebody reviewed.

The calculation has two parts: ordinary accuracy weights each slice by its current count; fixed-mix accuracy weights the same slice scores by the reference proportions. TV is half the sum of absolute differences in input probabilities. The complete Python 3.12 script is expandable after the results.

## What the numbers say

Output of `python drift_lab.py` (values are exact fractions printed to three decimals; `0.857` is 6/7):

```text
scenario                       TV P(X)  P(Y=1)    acc  fixed-mix  flagged
reference                        0.000   0.500  0.840      0.840    0.840
A: P(X) moves, P(Y|X) fixed      0.200   0.660  0.840      0.840    0.857
B: P(Y|X) moves, P(X) fixed      0.000   0.400  0.740      0.740    0.640
C: P(X) moves to hard slices     0.400   0.500  0.720      0.840    0.720
D: P(Y|X) moves, unlabelled      0.000   0.660  0.680      0.680    0.840
```

**A — data drift without degradation.** The input distribution moved (TV 0.2) and the label rate jumped from 0.50 to 0.66, yet accuracy is exactly 0.84. The model is equally good on the slices that grew and the slices that shrank. An input-drift alert here is true, and it is still not a reason to touch the model. Note also the base-rate change: `P(Y=1)` changed with no change in `P(Y|X)`. This is not strict label shift: for example, `P(X=0|Y=1)` changes from 1/50 to 1/132.

**B — concept drift invisible in X.** TV distance is exactly zero, because `P(X)` is identical. Accuracy fell from 0.84 to 0.74. No statistic computed on inputs alone can detect this; you need outcomes.

Conditional drift need not hurt this classifier. Keeping the reference input counts, changing the positive rates at `x = 0` and `x = 9` to 20% and 100% leaves overall accuracy at 0.84: one slice loses what the other gains. Changing only `x = 0` to 0% improves accuracy to 0.85. Both change `P(Y|X)` while leaving every optimal hard classification unchanged. The tests check these counterexamples as well as the harmful scenarios.

**C — the aggregate metric moved, the model did not get worse anywhere.** Accuracy fell to 0.72, the same direction as in B. But `fixed-mix` stays at 0.84: on every slice the model is exactly as good as before (the test suite checks this slice by slice). The traffic moved toward inputs that were always hard. This is the case where comparing a *monitored mixture* with *fixed-slice* performance matters: same headline drop as B, different cause, different response. Notice that `P(Y=1)` did not move at all.

**D — real concept drift hidden by selective labels.** True accuracy fell to 0.68, the largest drop in the table. But if labels arrive only for items the model flagged, the measured accuracy stays at 0.84, because the drift happened where the model says "no" and nobody checks. Compare with A: both show `P(Y=1) = 0.66`, one is harmless and one is the worst scenario in the table.

No single column separates all five rows. That is the point.

<details markdown="1">
<summary>Full drift_lab.py — runnable Python 3.12</summary>

```python
"""Data drift vs concept drift on exact, synthetic, finite populations.

Deterministic: no sampling, no randomness, exact fractions and a frozen
threshold model. Standard library only; Python 3.12+ (PEP 695 syntax).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

type Population = Mapping[int, tuple[int, int]]  # x -> (pos, neg)
type Model = Callable[[int], int]


def _is_count(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value >= 0)


def size(pop: Population) -> int:
    return sum(pos + neg for pos, neg in pop.values())


def validate_population(pop: Population) -> None:
    if not pop:
        raise ValueError("population must not be empty")
    for x, counts in pop.items():
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"feature value must be an int, got {x!r}")
        if not (isinstance(counts, tuple) and len(counts) == 2
                and all(_is_count(c) for c in counts)):
            raise ValueError(f"counts for x={x} must be two ints >= 0")
    if size(pop) == 0:
        raise ValueError("population must contain at least one item")


def p_x(pop: Population) -> dict[int, Fraction]:
    validate_population(pop)
    n = size(pop)
    return {x: Fraction(pos + neg, n) for x, (pos, neg) in pop.items()}


def p_y1(pop: Population) -> Fraction:
    validate_population(pop)
    return Fraction(sum(pos for pos, _ in pop.values()), size(pop))


def _check_distribution[K](dist: Mapping[K, Fraction]) -> None:
    if (not dist or any(p < 0 for p in dist.values())
            or sum(dist.values()) != 1):
        raise ValueError("expected a distribution summing to 1")


def total_variation[K](p: Mapping[K, Fraction],
                       q: Mapping[K, Fraction]) -> Fraction:
    """Half the L1 distance: 0 if identical, 1 if disjoint."""
    _check_distribution(p)
    _check_distribution(q)
    diffs = (abs(p.get(k, 0) - q.get(k, 0)) for k in p.keys() | q.keys())
    return sum(diffs, Fraction(0)) / 2


def predict(model: Model, x: int) -> int:
    y = model(x)
    if y not in (0, 1):
        raise ValueError(f"model must return 0 or 1, got {y!r} for x={x}")
    return y


def _correct(counts: tuple[int, int], prediction: int) -> int:
    positives, negatives = counts
    return positives if prediction == 1 else negatives


def slice_accuracy(pop: Population, model: Model) -> dict[int, Fraction]:
    validate_population(pop)
    return {x: Fraction(_correct(c, predict(model, x)), sum(c))
            for x, c in pop.items() if sum(c) > 0}


def _always(x: int) -> bool:
    return True


def accuracy(pop: Population, model: Model,
             labelled: Callable[[int], bool] = _always) -> Fraction:
    """Accuracy over the items whose label is actually observed."""
    validate_population(pop)
    seen = {x: c for x, c in pop.items() if labelled(x)}
    n = sum(sum(c) for c in seen.values())
    if n == 0:
        raise ValueError("no labelled items in this window")
    hits = sum(_correct(c, predict(model, x)) for x, c in seen.items())
    return Fraction(hits, n)


def fixed_mix_accuracy(pop: Population, model: Model,
                       reference: Population) -> Fraction:
    """Current per-slice accuracy, re-weighted to the reference P(X)."""
    weights = p_x(reference)
    current = slice_accuracy(pop, model)
    missing = sorted(x for x, w in weights.items()
                     if w and x not in current)
    if missing:
        raise ValueError(f"no current data for reference slices {missing}")
    return sum((w * current[x] for x, w in weights.items() if w),
               Fraction(0))


def bucket(n: int, positive_percent: int) -> tuple[int, int]:
    if not (_is_count(n) and _is_count(positive_percent)
            and positive_percent <= 100):
        raise ValueError("need n >= 0 and 0 <= positive_percent <= 100")
    positives, remainder = divmod(n * positive_percent, 100)
    if remainder:
        raise ValueError(f"{positive_percent}% of {n} is not whole")
    return positives, n - positives


def build(sizes: Mapping[int, int],
          percents: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    if sizes.keys() != percents.keys():
        raise ValueError("sizes and percents need the same x values")
    pop = {x: bucket(sizes[x], percents[x]) for x in sorted(sizes)}
    validate_population(pop)
    return pop


def threshold_model(x: int) -> int:
    """Frozen model: predicts 1 for x >= 5. Never retrained here."""
    return int(x >= 5)


REFERENCE_PERCENT = ({x: 10 for x in range(4)} | {4: 40, 5: 60}
                     | {x: 90 for x in range(6, 10)})
EVEN = dict.fromkeys(range(10), 100)


def tabular_scenarios() -> dict[str, dict[int, tuple[int, int]]]:
    ref = REFERENCE_PERCENT
    tails = {x: 50 if x < 4 else 150 if x > 5 else 100 for x in range(10)}
    hard = {x: 300 if x in (4, 5) else 50 for x in range(10)}
    return {
        "reference": build(EVEN, ref),
        "A: P(X) moves, P(Y|X) fixed": build(tails, ref),
        "B: P(Y|X) moves, P(X) fixed": build(EVEN, ref | {5: 10, 6: 40}),
        "C: P(X) moves to hard slices": build(hard, ref),
        "D: P(Y|X) moves, unlabelled": build(EVEN, ref | {2: 90, 3: 90}),
    }


@dataclass(frozen=True)
class Row:
    name: str
    tv_px: Fraction
    p_y1: Fraction
    accuracy: Fraction
    fixed_mix: Fraction
    flagged_only: Fraction


def tabular_rows(model: Model = threshold_model) -> list[Row]:
    scenarios = tabular_scenarios()
    ref = scenarios["reference"]

    def flagged(x: int) -> bool:  # labels arrive only for flagged items
        return predict(model, x) == 1

    return [Row(name, total_variation(p_x(ref), p_x(pop)), p_y1(pop),
                accuracy(pop, model), fixed_mix_accuracy(pop, model, ref),
                accuracy(pop, model, flagged))
            for name, pop in scenarios.items()]


def main() -> None:
    print(f"{'scenario':<30}{'TV P(X)':>8}{'P(Y=1)':>8}{'acc':>7}"
          f"{'fixed-mix':>11}{'flagged':>9}")
    for r in tabular_rows():
        cells = (r.tv_px, r.p_y1, r.accuracy, r.fixed_mix, r.flagged_only)
        widths = (8, 8, 7, 11, 9)
        print(f"{r.name:<30}" + "".join(
            f"{float(v):>{w}.3f}"
            for v, w in zip(cells, widths, strict=True)))


if __name__ == "__main__":
    main()
```

</details>

## Labels arrive late, and not at random

The `flagged` column is a simplified version of a common reality. Outcomes for credit decisions, fraud alerts, churn or support resolution often arrive weeks later, and frequently only for the cases where the system took an action. Then:

- A metric computed on "labels received so far" describes an older window than the dashboard date suggests.
- If labelling depends on the model's own decision, the labelled set is a biased sample. In D, the region that drifted is exactly the region that is never labelled.
- Where the target can be independently observed, a small, random audit sample of unflagged items, labelled on the same definition as before, can reduce the blind spot.

Missing annotations and unavailable outcomes are different problems. An audit can annotate an unreviewed fraud case if independent evidence exists. It cannot reveal repayment on a loan that was never issued; withheld actions can make the target outcome unavailable, requiring a different data-collection design.

## Windows, sample sizes and comparisons

The lab uses exact populations so the arithmetic is unambiguous. Production monitoring adds problems the lab intentionally avoids:

- **Sample size.** A drift statistic computed on a small window moves with sampling noise alone. Before alerting, know how much the statistic varies between two windows drawn from an unchanged population — for example by comparing past windows with each other.
- **Window choice.** A weekly window hides an incident that lasted a day; an hourly window turns daily seasonality into a stream of "drift". Pick the window from how fast you can act and how fast the underlying process changes.
- **Reference choice.** "Compared with the training data", "with last month" and "with the same week last year" answer different questions. Write the comparison into the alert name.
- **Many features, many tests.** With hundreds of monitored features, some will cross a threshold in every window without anything having changed. Rank by impact on slices you care about instead of paging on every crossing.
- **Missing slices.** `fixed_mix_accuracy` refuses to run if a reference slice has no current data. A monitor that silently skips empty slices reports a number that means something else.

## Answer degradation in LLM and RAG applications

In an application built around a language model, answer quality depends on much more than the external query distribution. Several component changes can alter answers while queries stay fixed; some also change the generator's inputs or the reference answers:

- **Retrieval**: index rebuilds, chunking changes, a filter that excludes a source, an embedding model swapped without re-indexing.
- **Freshness**: the corpus stopped updating, or stale documents were never deleted and now outrank new ones. The correct answer may also have changed, so freshness failures and target drift can coexist.
- **Prompts, model versions and generation settings**: a prompt edit, a provider-side model update, a changed temperature or token limit.
- **Outages and fallbacks**: timeouts that trigger a cheaper fallback model, or an empty retrieval result that the prompt does not handle.
- **The evaluator**: a new grading prompt, a different judge model or a changed reference set moves the score without any change in what users receive.

Retrieval-augmented generation, as described by Lewis et al., combines a parametric model with a non-parametric retrieved memory, with provenance and updatable knowledge among the motivations. That architecture is exactly why the retrieval and corpus side deserves its own monitoring: it is a separate, updatable component that can break on its own.

There is also no general mapping from "the distribution of query tokens or query embeddings moved" to "answers became less correct". Distribution statistics over queries are useful for noticing new topics or new languages. They are not a quality metric, and they say nothing about cases where queries stayed the same and the pipeline changed.

## Lab: a regression with fixed external queries

The second script is a stub on purpose. There is no LLM, no embedding model and no network call. Retrieval is keyword overlap after a crude normalisation, and the "generator" copies the top-ranked passage. It is deterministic so the cause of every difference is visible.

The evaluation set has four fixed queries, each with the ID of the relevant document and a fixed reference answer. The only change between runs is the configuration: the regressed configuration no longer includes the `changelog` source — the kind of change an index-filter edit or a connector rename can produce. Query-term TV monitors the external input; context TV compares the distributions of retrieved document IDs, showing that the generator receives different evidence. Here the corpus text is fixed, so document IDs identify its passages.

Retrieval ranks passages by keyword overlap and breaks ties by document ID. `hit@2` asks whether a relevant document reached the generator. `ref-match` checks only whether the reference substring appears in its answer. This is a proxy, not semantic correctness: it does not handle negation or validate numbers. For a reference of `50000 rows`, both `150000 rows` and `do not have a limit of 50000 rows` pass. A correct paraphrase can fail. The tests demonstrate those boundaries.

The baseline allows handbook, changelog and forum passages. The regression drops changelog passages before ranking; the generator still copies the highest-ranked remaining passage, or returns `NO_ANSWER`. The full listing follows the results and needs `drift_lab.py` beside it.

Output of `python rag_stub_lab.py`:

```text
baseline: query-term TV=0.000 context TV=0.000 hit@2=1 ref-match=1
  How many days do customers have to request refunds?
    -> Customers can request refunds within 30 days of delivery. ['kb-1', 'forum-2']
  When is support available?
    -> Support is available on weekdays from 8 to 18 CET. ['kb-2']
  What is the row limit for exports since release 2.4?
    -> Since release 2.4, exports have a limit of 50000 rows. ['kb-3', 'kb-4']
  When do API tokens expire since release 2.4?
    -> Since release 2.4, API tokens expire after 90 days. ['kb-4', 'kb-3']
regressed: query-term TV=0.000 context TV=0.571 hit@2=1/2 ref-match=1/2
  How many days do customers have to request refunds?
    -> Customers can request refunds within 30 days of delivery. ['kb-1', 'forum-2']
  When is support available?
    -> Support is available on weekdays from 8 to 18 CET. ['kb-2']
  What is the row limit for exports since release 2.4?
    -> Forum archive: exports had a row limit of 10000 rows. ['forum-1']
  When do API tokens expire since release 2.4?
    -> NO_ANSWER []
```

External queries, references and the query-term distribution are identical (TV 0.000), while the retrieved-context distribution changes (TV 4/7, printed as 0.571). Retrieval hit rate and the reference-match proxy both fall from 1 to 1/2. Inspecting the fixed cases confirms that one answer quotes an outdated forum limit (10000 rows instead of 50000), and another has no answer at all. The cause is the source-filter regression, which changes the generator's input; retraining would not restore the excluded evidence. Retrieval hit rate points at the retriever, while the query monitor stays flat.

<details markdown="1">
<summary>Full rag_stub_lab.py — runnable Python 3.12</summary>

```python
"""Application regression in a stubbed, deterministic RAG pipeline.

No LLM, no embeddings, no network: retrieval is keyword overlap and the
"generator" copies the top passage. External queries and references
never change; only the pipeline configuration does, which changes the
context the generator sees. Python 3.12+, standard library.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from drift_lab import total_variation

STOPWORDS = frozenset(
    "a an the is are of to for do does what when how many have has can on"
    .split())
NO_ANSWER = "NO_ANSWER"


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Case:
    query: str
    relevant_id: str
    reference: str


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    sources: frozenset[str]

    def __post_init__(self) -> None:
        if (isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
                or self.top_k < 1):
            raise ValueError("top_k must be a positive int")
        if not self.sources:
            raise ValueError("at least one source must be enabled")


CORPUS = (
    Doc("kb-1", "handbook",
        "Customers can request refunds within 30 days of delivery."),
    Doc("kb-2", "handbook",
        "Support is available on weekdays from 8 to 18 CET."),
    Doc("kb-3", "changelog",
        "Since release 2.4, exports have a limit of 50000 rows."),
    Doc("kb-4", "changelog",
        "Since release 2.4, API tokens expire after 90 days."),
    Doc("forum-1", "forum",
        "Forum archive: exports had a row limit of 10000 rows."),
    Doc("forum-2", "forum",
        "Forum archive: refunds took 14 days to arrive."),
)
CASES = (
    Case("How many days do customers have to request refunds?",
         "kb-1", "30 days"),
    Case("When is support available?", "kb-2", "8 to 18"),
    Case("What is the row limit for exports since release 2.4?",
         "kb-3", "50000 rows"),
    Case("When do API tokens expire since release 2.4?",
         "kb-4", "90 days"),
)
BASELINE = RagConfig(2, frozenset({"handbook", "changelog", "forum"}))
# The regression: an index filter change silently drops one source.
REGRESSED = RagConfig(2, frozenset({"handbook", "forum"}))


def retrieve(query: str, corpus: Sequence[Doc],
             config: RagConfig) -> list[Doc]:
    wanted = set(tokens(query))
    scored = [(len(wanted & set(tokens(d.text))), d) for d in corpus
              if d.source in config.sources]
    ranked = sorted((s for s in scored if s[0] > 0),
                    key=lambda s: (-s[0], s[1].doc_id))
    return [doc for _, doc in ranked[:config.top_k]]


def answer(retrieved: Sequence[Doc]) -> str:
    """Stub generator: copies the top passage verbatim. Not an LLM."""
    return retrieved[0].text if retrieved else NO_ANSWER


def reference_match(reference: str, text: str) -> bool:
    """Proxy: reference substring present. Not semantic correctness."""
    return reference.lower() in text.lower()


@dataclass(frozen=True)
class Result:
    query: str
    retrieved_ids: tuple[str, ...]
    answer: str
    hit: bool
    ref_match: bool


def run_case(case: Case, corpus: Sequence[Doc],
             config: RagConfig) -> Result:
    docs = retrieve(case.query, corpus, config)
    text = answer(docs)
    ids = tuple(d.doc_id for d in docs)
    return Result(case.query, ids, text, case.relevant_id in ids,
                  reference_match(case.reference, text))


def evaluate(cases: Sequence[Case], corpus: Sequence[Doc],
             config: RagConfig) -> list[Result]:
    if not cases:
        raise ValueError("evaluation set must not be empty")
    return [run_case(c, corpus, config) for c in cases]


def rate(flags: Sequence[bool]) -> Fraction:
    if not flags:
        raise ValueError("cannot compute a rate of nothing")
    return Fraction(sum(flags), len(flags))


def distribution(items: Iterable[str]) -> dict[str, Fraction]:
    counts = Counter(items)
    n = counts.total()
    if n == 0:
        raise ValueError("nothing to build a distribution from")
    return {item: Fraction(c, n) for item, c in counts.items()}


def query_terms(results: Sequence[Result]) -> dict[str, Fraction]:
    """X at the entrance: terms of the external queries."""
    return distribution(t for r in results for t in tokens(r.query))


def context_ids(results: Sequence[Result]) -> dict[str, Fraction]:
    """X at the generator: which passages it was given."""
    return distribution(i for r in results for i in r.retrieved_ids)


def main() -> None:
    base = evaluate(CASES, CORPUS, BASELINE)
    for name, config in (("baseline", BASELINE), ("regressed", REGRESSED)):
        res = evaluate(CASES, CORPUS, config)
        q_tv = total_variation(query_terms(base), query_terms(res))
        c_tv = total_variation(context_ids(base), context_ids(res))
        print(f"{name}: query-term TV={float(q_tv):.3f} "
              f"context TV={float(c_tv):.3f} "
              f"hit@{config.top_k}={rate([r.hit for r in res])} "
              f"ref-match={rate([r.ref_match for r in res])}")
        for r in res:
            print(f"  {r.query}\n    -> {r.answer} {list(r.retrieved_ids)}")


if __name__ == "__main__":
    main()
```

</details>

## From alert to intervention

When a quality signal moves, I walk through the questions in this order. Each step is cheaper than the next, and each one can end the investigation.

1. **Did the measurement change?** Evaluator, judge prompt, reference set, label definition, logging, or the denominator of the metric. If yes, fix or re-baseline the measurement first.
2. **Did the system change?** Deployments, configuration, prompts, provider model versions, index rebuilds, corpus freshness, dependency outages. Check the change log for the window. If yes, roll back or fix that component and re-run a fixed evaluation set.
3. **Did the input mix change?** Define `X` for each pipeline stage, then compare its distribution and per-slice quality. If per-slice quality is stable and only the mix moved (scenario C), the response may be capacity, routing, product changes, or improving the model on the hard slices — not necessarily retraining on recent data.
4. **Did `P(Y|X)` change?** Check fresh, comparable labels, including a random audit outside the flagged region where outcomes can be independently observed. Measure the effect on the frozen model: conditional drift can leave accuracy and optimal decisions unchanged or improve accuracy. If a harmful change is expected to persist, evaluate an update to the model, the rule or the features on the new regime.
5. **None of the above?** Keep the alert, widen the window, and collect labels before acting.

## Retraining is not the default fix

Retraining on recent data can be appropriate when a persistent change in the relationship between inputs and outcomes harms the frozen model and a candidate model performs better on the new regime. Conditional drift alone does not establish either requirement. Retraining does not restore an excluded retrieval source or repair a new evaluator or a mislabelled week, and it can make things worse:

- If the new labels come from a broken definition, retraining learns the bug.
- If labels are selective (scenario D), the retraining set under-represents exactly the region that drifted.
- If the drop was a mix shift (scenario C), a retrained model may trade accuracy on easy slices for little gain on hard ones, and the comparison against the old model becomes difficult to read.

Retraining is one intervention among several, chosen after diagnosis, and evaluated like any other model change.

## Reproducing the examples

The two expandable listings above are complete. Save them as `drift_lab.py` and `rag_stub_lab.py` in one empty directory; the RAG script imports `total_variation` from the drift script. They need Python 3.12 or later and nothing outside the standard library.

The accompanying `unittest` suite (not reproduced in full here) checks exact fractions for every table row, unchanged slice accuracy in C, conditional drift with unchanged or improved accuracy, both RAG configurations, fixed external queries with changed context, and the substring proxy's limits. It also covers the threshold, empty slices, TV distance of 0 and 1, bucket edges, ranking ties and invalid inputs. The correction run completed all 31 tests with `OK` on CPython 3.12.11.

A reproduction note. The commands below use uv, modern reproduction tooling to pin the interpreter; it is not part of the method, and plain `python3.12` works the same way. The printed outputs were generated with CPython 3.12.11 on 12 October 2026. The series date, 8 August 2024, is an editorial label, not a claim of historical publication or execution.

```bash
uv venv --python 3.12
uv run --no-project --python .venv/bin/python drift_lab.py
uv run --no-project --python .venv/bin/python rag_stub_lab.py
```

## Limits of these examples

- One feature, binary labels, a threshold model. Real systems have many features, and the interaction between slices is more complex than a single axis.
- Exact populations remove sampling noise. The section on windows and sample sizes is a reminder that production statistics do not get that luxury.
- The RAG stub has no language model. It shows the *shape* of a pipeline regression, not how a real model would phrase a wrong answer or whether it would abstain.
- `ref-match` is a case-insensitive reference-substring proxy. It does not establish semantic correctness, resolve negation or validate numeric values, and it can reject correct paraphrases. Context TV measures document IDs here; it would miss edits to text stored under unchanged IDs.
- The scenarios are constructed to separate the concepts. They are not a benchmark and say nothing about how often each case occurs in practice.

### Sources

- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M. and Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys 46(4). [doi:10.1145/2523813](https://doi.org/10.1145/2523813). Joint-distribution definition and distinction between real (`P(y|X)`) and virtual (`P(X)`) drift, section 2.1 of the [accepted manuscript](https://mpechen.win.tue.nl/publications/pubs/Gama_ACMCS_AdaptationCD_accepted.pdf).
- Lipton, Z. C., Wang, Y.-X. and Smola, A. J. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors.* [arXiv:1802.03916](https://arxiv.org/abs/1802.03916), submitted 12 February 2018, revised 26 July 2018. Abstract definition: changing `P(Y)` with fixed `P(X|Y)`.
- Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401), submitted 22 May 2020, revised 12 April 2021 — parametric and non-parametric memory, provenance and updating knowledge as motivation.
- [Python 3.12.0 release](https://www.python.org/downloads/release/python-3120/), 2 October 2023 — `type` statement and PEP 695 generic syntax used in the listings.
