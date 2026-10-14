---
layout: post
lang: en
title: "Evaluate retrieval, answer quality, and evidence support separately"
date: 2024-05-30 09:00:00 +0200
permalink: /blog/rag-evaluation-three-layers/
author: Rafał Prońko
categories: [llm-evaluation]
description: "Three separate checks for a RAG system: labelled retrieval metrics, answer correctness and completeness, and support of each claim by the context the model actually received."
image: /assets/images/rag-evaluation-three-layers.svg
---

# Evaluate retrieval, answer quality, and evidence support separately

![Diagram of three separate checks for a RAG answer: retrieval against labelled chunks, answer claims against a reference, and the same claims against the context actually provided to the model.]({{ '/assets/images/rag-evaluation-three-layers.svg' | relative_url }})

<time datetime="2024-05-30">30 May 2024</time>

> **Editorial note.** The date above is an editorial label for this article, not a record of when the code was run. The example was reproduced on 13 October 2026 with CPython 3.12.11, uv 0.7.8 and pytest 9.1.1. The papers it cites and the Python 3.12 standard library it relies on were available before May 2024.

A retrieval-augmented generation (RAG) system answers a question in two steps: it retrieves some text, then it generates an answer from that text. When the answer is bad, the first question is not “how bad?”, but “which step failed?”.

A single quality score cannot answer that. A score of 0.6 may mean that retrieval missed the right document. It may mean that the right document was retrieved but cut off before it reached the model. It may also mean that the model had everything it needed and still made something up.

Each of these failures needs a different fix. So I want to measure three things separately:

1. **Retrieval.** Did the top-k results contain the chunks needed to answer?
2. **Answer quality.** Is the answer correct, complete and on topic, compared with a reference?
3. **Evidence support.** Is every claim in the answer backed by the context the model actually received, and do the citations point to that backing?

These checks look similar, but they compare different things. Retrieval compares a ranked list with labelled chunks. Answer quality compares claims with a reference. Evidence support compares claims with the provided context. Mixing them hides the cases that matter most.

## Two answers that a single score gets wrong

Two cases show why the layers must stay apart.

**A correct answer can be unsupported.** The user asks how long EU shipping takes. Retrieval returns the page about shipping fees, the store opening hours and a job advert. The model answers “3–5 business days”, which happens to be correct. Nothing in the context says so. The answer came from the model's parameters, or from a lucky guess. If I only check correctness, this case passes, and the system teaches me nothing about the retriever that missed the shipping-time page.

**A supported answer can be wrong.** The user asks about the return window. Retrieval returns the old refund policy (30 days), which is still in the index after the policy changed to 14 days. The model faithfully repeats “30 days” and cites the old page. Faithfulness is perfect, the citation is perfect, and the customer gets the wrong answer. Evidence support says nothing about whether the source is current.

Faithfulness is a statement about the relationship between an answer and a context. Correctness is a statement about the answer and the world, as represented by a reference. Neither implies the other.

## What the published metrics measure

It helps to read what the well-known metrics actually define, rather than what their names suggest.

The RAGAS paper (Es et al., EACL 2024 system demonstrations) describes a standard RAG setting: a question *q*, a retrieved context *c(q)* and a generated answer. It deliberately focuses on **reference-free** metrics, because human-annotated datasets or reference answers are often unavailable. Section 3 defines three aspects:

- **Faithfulness**: an LLM splits the answer into statements, then decides for each statement whether it can be inferred from the context. The score is the fraction of supported statements.
- **Answer relevance**: an LLM generates questions from the answer, and the score is the mean embedding similarity between those questions and the original one. The paper says explicitly that this assessment *does not take factuality into account*; it penalises incomplete or redundant answers.
- **Context relevance**: the fraction of context sentences that an LLM extracts as needed to answer the question.

These are useful signals when no labels exist. But they are proxies. None of them checks whether the answer is factually correct, and all of them depend on the judging LLM; the paper's own limitations section says so.

FActScore (Min et al., EMNLP 2023) contributes two ideas that I use directly. First, the unit of evaluation is an **atomic fact**: a short statement conveying one piece of information, because a single sentence often mixes supported and unsupported facts. Second, support is defined **relative to a given knowledge source**, not as global truth. Section 3.1 also states the limitation clearly: FActScore measures *precision*, not *recall*, so a model that abstains often or produces fewer facts can score higher, “even if these are not desired”. Its score is averaged only over prompts where the model responded.

That last point is the trap I want to avoid. Any precision-style metric — faithfulness, factual precision, the share of relevant claims — rewards short answers. One true, supported sentence scores 1.0 even when the question needed three facts. A refusal contains no claims at all, so if it is scored as “no unsupported claims”, it looks perfectly faithful. Completeness has to be measured separately, and abstention has to be handled as its own outcome.

## What to label

To score the three layers without guessing, I need labels. In the example below they are written by hand, in the shape a human annotator would produce. For each question:

- **`relevant`**: chunk IDs that contain information needed for the answer. This drives recall@k and precision@k.
- **`evidence_sets`**: one or more sets of chunks, each of which is *sufficient* on its own to answer. A warranty question may be answerable from one combined page, or from two separate FAQ entries. Recall over all relevant chunks would punish a retriever that found the second path; a sufficiency check accepts either.
- **`required_claims`**: IDs of the atomic claims a complete answer must contain. This drives completeness.
- **`answerable`**: an explicit flag, checked against the evidence sets, so that an unanswerable question is a deliberate label and not a forgotten field.
- For each claim in the generated answer: whether it is **correct** with respect to the reference, whether it is **relevant** to the question, which provided chunks **support** it, and which chunks it **cites**.

Two policies need to be explicit.

**Unanswerable questions** have no evidence sets, no relevant chunks and no required claims. Retrieval recall is then undefined, not zero and not one. The right behaviour is abstention, and that is scored as an outcome of its own.

**Labels are scoped to a snapshot.** “Correct” means correct according to the reference documents of a particular corpus version. When the refund policy changes, the label for “30 days” changes from correct to wrong, even though the answer text and its supporting chunk stay the same. That is why a supported-but-wrong answer can only be detected if the reference is maintained separately from the index.

## Freeze what the model actually saw

Faithfulness must be measured against the context the model actually received. That is often not the retrieved list.

Between retrieval and generation there is usually a reranker, a deduplication step and a token budget. Chunks get reordered, merged or dropped. If I log only the top-k list from the retriever and later compute faithfulness against it, I may credit the model for using a chunk it never saw — or blame retrieval for a chunk that was retrieved and then truncated.

So each evaluation record stores:

- the corpus and index version (here a snapshot label, `acme-help@2024-05-20`),
- the retrieved list with its order, before any truncation,
- the provided context after assembly, exactly as passed to the model,
- the answer and the labels.

With both lists stored, “the evidence was retrieved but not provided” becomes a separate, visible failure.

## A small harness

The example is a deterministic synthetic harness. There is no retriever, no model and no automatic entailment. A JSON file contains a tiny made-up help-centre corpus, eleven cases (ten with a list of retrieved chunk IDs, all eleven with the IDs of the chunks provided as context), and hand-written labels for every claim. The Python module validates those labels and aggregates them into metrics and failure tags.

That is a deliberate restriction. The goal is to make the metric definitions and their failure modes inspectable. Deciding whether a sentence is entailed by a chunk is a separate problem — done by humans or by an LLM judge — and its errors would only blur the point here.

The module uses only the Python 3.12 standard library. The tests use pytest; `pyproject.toml` pins only pytest (9.1.1). The recorded run is a later reproduction, made on 13 October 2026 with CPython 3.12.11, uv 0.7.8 and pytest 9.1.1. None of the tooling is needed to understand the code.

The complete source is available for download:

- [`rag_eval.py`]({{ '/assets/examples/rag-evaluation-three-layers/rag_eval.py' | relative_url }}) — validation, metrics, tags and the report,
- [`labels.json`]({{ '/assets/examples/rag-evaluation-three-layers/labels.json' | relative_url }}) — the synthetic corpus and labelled cases,
- [`tests/test_rag_eval.py`]({{ '/assets/examples/rag-evaluation-three-layers/tests/test_rag_eval.py' | relative_url }}) — the tests,
- [`pyproject.toml`]({{ '/assets/examples/rag-evaluation-three-layers/pyproject.toml' | relative_url }}),
- [`README.md`]({{ '/assets/examples/rag-evaluation-three-layers/README.md' | relative_url }}) — layout, commands and the full label contract.

A labelled case looks like this:

```json
{
  "id": "context-dropped",
  "question": "Can I return a gift card, and how long do I have for unopened items?",
  "relevant": ["refund-2024", "refund-exceptions"],
  "evidence_sets": [["refund-2024", "refund-exceptions"]],
  "required_claims": ["no-gift-card-returns", "return-14d"],
  "answerable": true,
  "retrieved": ["refund-exceptions", "shipping-fee", "refund-2024"],
  "provided": ["refund-exceptions", "shipping-fee"],
  "abstained": false,
  "claims": [
    {"id": "no-gift-card-returns", "text": "Gift cards cannot be returned.",
     "correct": true, "relevant": true,
     "supported_by": ["refund-exceptions"], "cited": ["refund-exceptions"]}
  ]
}
```

The retriever found both relevant chunks, but the assembled context kept only two of the three, and `refund-2024` was the one that was dropped.

### The metrics

The core of the module is one function per case:

```python
def score(case: Case, k: int) -> Result:
    relevant = set(case.relevant)
    if case.retrieved is None or not case.answerable:
        recall = precision = sufficient_topk = None
    else:
        top = set(case.retrieved[:k])
        recall = len(relevant & top) / len(relevant)
        precision = len(relevant & top) / k
        sufficient_topk = any(set(s) <= top for s in case.evidence_sets)
    sufficient_context = (any(set(s) <= set(case.provided)
                              for s in case.evidence_sets)
                          if case.answerable else None)

    claims = case.claims
    n = len(claims)
    good = {c.id for c in claims if c.correct}
    completeness = (None if not case.required_claims else
                    sum(r in good for r in case.required_claims)
                    / len(case.required_claims))
    pairs = [(c, cid) for c in claims for cid in c.cited]
    provided = set(case.provided)

    if not case.answerable:
        abstention = "correct" if case.abstained else "should_abstain"
    elif case.abstained:
        abstention = "missed_answer" if sufficient_context else "no_evidence"
    else:
        abstention = "answered"

    result = Result(
        id=case.id,
        recall=recall,
        precision=precision,
        sufficient_topk=sufficient_topk,
        sufficient_context=sufficient_context,
        correctness=_ratio(len(good), n),
        completeness=completeness,
        relevance=_ratio(sum(c.relevant for c in claims), n),
        faithfulness=_ratio(sum(bool(c.supported_by) for c in claims), n),
        citation_coverage=_ratio(sum(bool(c.cited) for c in claims), n),
        citation_entailment=_ratio(
            sum(cid in c.supported_by for c, cid in pairs), len(pairs)),
        invalid_citations=sum(cid not in provided for _, cid in pairs),
        abstention=abstention,
        tags=(),
    )
    return _with_tags(result, case)
```

`_ratio` returns `None` when the denominator is zero. That one rule carries most of the policy: an answer without claims has no correctness, no faithfulness and no citation scores. They are **n/a**, never 0 and never 1. Completeness is the exception: if the question had required claims and the model refused, completeness is 0.

The metrics fall into the three layers:

| Layer | Metric | Compared with |
|---|---|---|
| Retrieval | recall@k, precision@k | labelled relevant chunks in the top k |
| Retrieval | sufficient top-k | any full evidence set in the top k |
| Context assembly | sufficient context | any full evidence set in the provided context |
| Answer | correctness, relevance | per-claim labels: share of correct / on-topic claims |
| Answer | completeness | share of required claims answered correctly |
| Evidence | faithfulness | share of claims supported by the *provided* context |
| Evidence | citation coverage | share of claims with at least one citation |
| Evidence | citation entailment | share of (claim, cited chunk) pairs where that chunk supports the claim |

Precision@k divides by k, so a single relevant chunk in the top three gives 0.33 even for a perfect retrieval. That is the usual definition, and a reminder that precision@k is mainly useful for comparing retrievers at a fixed k, not as an absolute quality bar.

**Citation coverage and citation entailment are different questions.** Coverage asks whether a claim cites anything. Entailment asks whether what it cites actually supports it. A system can put a citation on every sentence and still point to the wrong chunks. A cited ID that was not in the provided context is counted separately as an invalid citation: the model cited something it never saw.

### Failure tags

Numbers are averaged; diagnoses should not be. Each case gets tags that name the failing layer:

```python
def _with_tags(r: Result, case: Case) -> Result:
    tags = []
    if r.abstention in ("should_abstain", "missed_answer", "no_evidence"):
        tags.append(r.abstention)
    if r.sufficient_topk is False:
        tags.append("retrieval_miss")
    elif r.sufficient_topk and r.sufficient_context is False:
        tags.append("context_dropped")
    if any(c.supported_by and not c.correct for c in case.claims):
        tags.append("supported_but_wrong")
    if any(not c.supported_by for c in case.claims):
        tags.append("unsupported_claim")
    if any(not c.cited for c in case.claims):
        tags.append("uncited_claim")
    entailment = r.citation_entailment
    if r.invalid_citations or (entailment is not None and entailment < 1.0):
        tags.append("bad_citation")
    if any(not c.relevant for c in case.claims):
        tags.append("off_topic_claim")
    if (not case.abstained and r.completeness is not None
            and r.completeness < 1.0):
        tags.append("incomplete")
    return Result(**{**r.__dict__, "tags": tuple(tags) or ("ok",)})
```

`retrieval_miss` and `context_dropped` are mutually exclusive on purpose. If no labelled sufficient evidence set reached the top k, the retriever is the first suspect. If one did but did not reach the model, context assembly is the first suspect. A tag is a pointer for one labelled sample, not a verdict on the component. This relies on one contract: the retrieved list may be logged beyond k, but the provided chunks must come from its top k, and `parse` rejects anything else.

### Running it

The tests import `rag_eval` from the project root and read `labels.json` from there, so the downloaded files need this layout:

```
rag-evaluation-three-layers/
├── README.md
├── pyproject.toml
├── rag_eval.py
├── labels.json
└── tests/
    └── test_rag_eval.py
```

Then:

```
mkdir -p rag-evaluation-three-layers/tests
cd rag-evaluation-three-layers
# save README.md, pyproject.toml, rag_eval.py and labels.json here,
# and test_rag_eval.py in tests/
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pytest==9.1.1
.venv/bin/python -m pytest -q
.venv/bin/python rag_eval.py labels.json
```

In the 13 October 2026 reproduction, all 76 tests passed and the report was exactly this:

```
snapshot=acme-help@2024-05-20 k=3
case                   R@k   P@k   topk  ctx  corr  comp  rel   faith  c.cov  c.ent  bad  abstention      tags
ok-refund              1.00  0.33  yes   yes  1.00  1.00  1.00  1.00   1.00   1.00   0    answered        ok
correct-unsupported    0.00  0.00  no    no   1.00  1.00  1.00  0.00   0.00   n/a    0    answered        retrieval_miss,unsupported_claim,uncited_claim
context-dropped        1.00  0.67  yes   no   1.00  0.50  1.00  1.00   1.00   1.00   0    answered        context_dropped,incomplete
context-dropped@gold   n/a   n/a   n/a   yes  1.00  1.00  1.00  1.00   1.00   1.00   0    answered        ok
stale-but-supported    0.00  0.00  no    no   0.00  0.00  1.00  1.00   1.00   1.00   0    answered        retrieval_miss,supported_but_wrong,incomplete
short-but-precise      0.33  0.33  yes   yes  1.00  0.50  1.00  1.00   1.00   1.00   0    answered        incomplete
two-evidence-paths     0.67  0.67  yes   yes  1.00  1.00  1.00  1.00   1.00   1.00   0    answered        ok
bad-citations          1.00  0.67  yes   yes  1.00  1.00  0.67  1.00   1.00   0.50   1    answered        bad_citation,off_topic_claim
unanswerable-abstains  n/a   n/a   n/a   n/a  n/a   n/a   n/a   n/a    n/a    n/a    0    correct         ok
unanswerable-guesses   n/a   n/a   n/a   n/a  0.00  n/a   1.00  0.00   1.00   0.00   0    should_abstain  should_abstain,unsupported_claim,bad_citation
answerable-refuses     1.00  0.33  yes   yes  n/a   0.00  n/a   n/a    n/a    n/a    0    missed_answer   missed_answer
isolate context-dropped: gold context fixes it and the actual context lacked evidence: suspect context assembly (a hint, not proof)
```

## Reading the table

Each row is built to demonstrate one behaviour of the metrics. Four rows are controls that should come out clean — `ok-refund`, `context-dropped@gold`, `two-evidence-paths` and `unanswerable-abstains` — and `unanswerable-guesses` deliberately combines three failures. These are synthetic labels, so the rows prove what the metric definitions do, not how any real system behaves.

**`correct-unsupported`.** Correctness and completeness are 1.00, faithfulness is 0.00, citation entailment is n/a because there are no citations. The labelled shipping-time chunk never reached the top k, so retrieval is the first place to look, even though the user got the right answer. An evaluation based only on correctness would mark this case as a success.

**`context-dropped`.** Recall@k is 1.00: for this question the retriever found every labelled relevant chunk. That says nothing about the retriever on other questions. Sufficient context is “no”: the refund-window chunk was retrieved and then cut. Faithfulness is 1.00, because the model said only what it saw. Completeness is 0.50. The reranker or the token budget is the first place to look; the labels give no reason to suspect the retriever here, though they cannot rule out a weak prompt.

**`stale-but-supported`.** Faithfulness and citation entailment are 1.00, correctness is 0.00. The tags say `retrieval_miss`, `supported_but_wrong` and `incomplete`: the current policy was not retrieved, and the superseded one was. Evidence metrics alone would call this answer perfect. The labels point to the corpus: the superseded page is still indexed, and removing or marking such documents is the obvious first fix to try. In this harness the problem is visible only because a human labelled the claim as incorrect against the current reference; nothing in the context itself reveals that the page is outdated.

**`short-but-precise`.** Correctness, relevance and faithfulness are all 1.00. Completeness is 0.50: the answer stated the warranty length and skipped how to file a claim. This is, in miniature, the precision limitation that the FActScore paper states for its own score, reproduced here with this harness's claim-level ratios. The harness does not implement or measure RAGAS answer relevance, which, as described above, is designed to penalise incomplete answers. A metric that only looks at the claims present cannot see the claims that are missing.

**`two-evidence-paths`.** Recall@k is only 0.67, because one of the three relevant chunks was not retrieved. But the two FAQ chunks together form a sufficient evidence set, and the answer is complete and supported. Recall flags a non-problem here; the sufficiency check does not. With several valid evidence paths, I treat recall as a diagnostic and sufficiency as the pass condition.

**`bad-citations`.** Citation coverage is 1.00: every claim cites something. Citation entailment is 0.50: of four (claim, citation) pairs, two point to a chunk that supports the claim. One claim cites the shipping-time page for the shipping fee; another adds `doc-999`, which was never in the context. Relevance is 0.67 because the answer also mentions that the company is hiring. The context held the right chunks, so the generation prompt or the post-processing of citations is the first place to look.

**`unanswerable-abstains`.** Every ratio metric and both sufficiency checks (`topk`, `ctx`) are n/a, the count of invalid citations (`bad`) is 0, and the abstention is `correct`. This is not a perfect score; it is a different outcome, reported separately. Averaging it as 1.0 into faithfulness would inflate the system's score every time it refused.

**`unanswerable-guesses`.** The model answered a question the corpus cannot answer, and cited the refund page as if it supported a price-matching promise. The three tags name each problem: it should have abstained (`should_abstain`), the claim has no support in the context (`unsupported_claim`), and its only citation points to a chunk that does not support it (`bad_citation`).

**`answerable-refuses`.** The context contained the answer, and the model refused. Faithfulness is n/a, not 1.00; completeness is 0.00; the abstention is `missed_answer`. Had the evidence been missing from the context, the outcome would have been `no_evidence`: a reasonable refusal by the model that points to a failure upstream. The tests cover that outcome on a modified copy of this case.

## Isolating the failure with gold context

Tags point to a suspect. A second, controlled run adds evidence for or against it; it does not prove it.

For a failing case, I run generation again with a **gold context**: a context that contains a full labelled evidence set (usually just that set), passed directly to the model, bypassing retrieval and assembly. It is rarely the same context as the actual one: it usually has fewer distractors, a different order, and sometimes a different evidence path. In the label file this is a separate case that refers to its original:

```json
{
  "id": "context-dropped@gold",
  "gold_of": "context-dropped",
  "provided": ["refund-exceptions", "refund-2024"],
  ...
}
```

It inherits the question and reference labels unchanged, has no retrieved list, so its retrieval metrics are n/a, and gets its own claim labels for the new answer. `parse` enforces this: a gold case may not set `question`, `relevant`, `evidence_sets`, `required_claims` or `answerable`, may not carry a retrieved list, and must refer to an answerable case. Its provided context must contain a full evidence set — a “gold” context that could not answer the question would make the comparison meaningless.

The comparison only means something if both runs use the same model, prompt and decoding settings. Even then, generation is not deterministic, so one pair of samples is weak evidence; repeating each run a few times is better. With that in mind, `isolate` reads the pair like this:

```python
def isolate(actual: Result, gold: Result) -> str:
    """Compare a case with its gold-context rerun: a hint, not a proof.

    Both runs should use the same model, prompt and decoding settings, and
    one pair of samples can still differ by chance.
    """
    if actual.tags == ("ok",):
        return "actual run passes: nothing to isolate"
    if gold.tags != ("ok",):
        return ("gold context does not fix it: suspect generation, the prompt "
                "or the labels; upstream stages may be failing as well")
    if actual.sufficient_context is not False:
        # Sufficient only means a labelled evidence set was present. The two
        # contexts can still differ in distractors, order or evidence path.
        return ("inconclusive: both contexts held a sufficient evidence set; "
                "the gold pass may come from context composition or order, "
                "generation or run-to-run variability")
    stage = ("context assembly" if "context_dropped" in actual.tags
             else "retrieval")
    return ("gold context fixes it and the actual context lacked evidence: "
            f"suspect {stage} (a hint, not proof)")
```

- **The actual context was insufficient and the gold run passes.** That suggests an upstream cause, in retrieval or context assembly, but does not prove it: the second answer is a new sample, and it might have improved for other reasons.
- **The actual context was already sufficient and the gold run passes.** The result is inconclusive. “Sufficient” only means that a labelled evidence set was present in both contexts. The contexts can still differ in distractors, order or evidence path, and those differences, which retrieval and assembly control, can change the answer. So context composition or order, generation and run-to-run variability all remain possible causes. `short-but-precise` is such a case: if a gold run with only the warranty page produces a complete answer, I cannot tell whether removing the shipping and store-hours distractors helped, or the model simply sampled a better answer. `isolate` does not compare the two contexts, so it keeps this conservative reading even when they are identical.
- **Both runs fail.** The generator, the prompt or the labels need attention. That does not rule out an upstream failure at the same time: a case can lose evidence in assembly *and* have a weak prompt, and fixing one of them will not make it pass.

In the report, the gold run of `context-dropped` is clean, its actual context lacked the refund-window chunk, and `isolate` points to context assembly as the suspect. Here I wrote the gold answer's labels myself; in a real evaluation they would come from labelling a second, actual model output. The tests cover all three readings; the inconclusive one is tested with an identical context, with the distractors removed, and with the other evidence path in a different order.

With several sufficient evidence sets, I would run the gold test with each of them, or at least with the smallest. A model that answers well from one combined page may still fail to combine two FAQ entries.

## Validate the labels before trusting the numbers

Metrics computed from broken labels look exactly as precise as metrics computed from good ones. The `parse` function rejects:

- duplicate IDs among cases, claims, retrieved or provided chunks, citations and evidence;
- chunk IDs that are not in the corpus;
- `correct`, `relevant`, `abstained` or `answerable` values that are not JSON booleans — a string `"false"` would otherwise be truthy and silently count as correct;
- an `answerable` flag that disagrees with the evidence sets, an unanswerable case with relevant chunks or required claims, and an answerable case without required claims;
- an evidence set that is empty or contains chunks not labelled relevant;
- an ordinary (non-gold) case whose `retrieved` field is missing, `null` or not a JSON list — otherwise a lost retrieval log would silently look like a gold run, with retrieval metrics n/a and no top-k check; an empty list is valid and means retrieval returned nothing;
- provided chunks that are not in the top k of the retrieved list;
- `supported_by` pointing to a chunk that was not in the provided context — support must come from what the model saw;
- an abstention that also contains claims;
- a claim that uses a required claim ID but is labelled incorrect;
- a non-positive `k`;
- a gold case that refers to a missing, gold or unanswerable case, overrides the inherited reference fields, carries a retrieved list, or lacks a full evidence set in its context.

A citation to an unknown chunk is deliberately *not* a label error. The model produced it, and the evaluation must count it. The tests check both sides: invalid labels raise `ValueError`, and a hallucinated `doc-404` citation appears as one invalid citation and halves citation entailment.

Duplicate retrieved IDs deserve a note. The metrics here work on sets, so a duplicate in the top k would not be counted twice in recall. It would still take up one of the k slots, though, and the set arithmetic would hide the fact that the pipeline returned the same chunk twice. In a real pipeline duplicates usually come from overlapping chunks or several indexes; I would rather reject them in the evaluation input and fix the deduplication explicitly than let the metrics absorb them silently.

## Limits of this approach

The harness is honest about one thing only: given the labels, the numbers follow. Everything hard is in the labels.

- **Claim decomposition is a judgement.** What counts as one atomic claim affects every per-claim ratio: split one sentence into three claims and its weight triples. A team needs written guidelines and agreement checks between annotators.
- **Support is a judgement too.** Here `supported_by` is written by hand. In practice it comes from annotators or from an LLM judge, which brings its own error rate and needs its own validation against human labels.
- **Support is per chunk.** `supported_by` lists chunks that *each* support the claim on their own, and citation entailment checks each (claim, cited chunk) pair separately. A claim that only follows from two chunks together cannot be expressed: joint multi-chunk support is not modelled. Such claims would have to be split, or the schema extended with support sets.
- **All claims weigh the same.** A wrong price and an irrelevant pleasantry count equally. For high-stakes domains I would add a severity label instead of hiding it in the average.
- **Labels are per snapshot.** When the corpus or the reference changes, labels need review. Old scores are comparable only with the same snapshot.
- **Synthetic cases prove definitions, not systems.** Eleven made-up rows show how the metrics behave in each failure mode. They say nothing about the failure rates of any real retriever or model.

Reference-free metrics such as those in RAGAS remain useful when labels are expensive: to monitor production traffic, to triage cases for labelling, to spot regressions. I just would not report them as correctness. They estimate whether an answer is grounded in its context and on topic, not whether it is true.

## Where to start

Not with a single quality score.

First I log, for every evaluated question, the snapshot, the retrieved list and the context actually provided. Then I label a small set of questions with relevant chunks, sufficient evidence sets and required claims, including a few that the corpus cannot answer.

With that, every failing case gets a suspected layer: retrieval, assembly, generation, citations or the corpus itself. A gold-context run under the same settings then adds evidence for or against that suspicion. It is a hint, not proof: a pass is inconclusive when the actual context was already sufficient, because the two contexts may differ in composition or order and the model may simply have sampled a better answer; a failure does not rule out an upstream problem at the same time.

### Sources

- [Es, James, Espinosa-Anke, Schockaert: RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/) — EACL 2024 system demonstrations, March 2024; section 3 defines faithfulness, answer relevance and context relevance as reference-free, LLM-estimated metrics.
- [Min et al.: FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://aclanthology.org/2023.emnlp-main.741/) — EMNLP 2023, December 2023; section 3.1 defines atomic facts, support relative to a knowledge source, and notes that the score measures precision, not recall.
- [Python 3.12.0 release](https://www.python.org/downloads/release/python-3120/) — released 2 October 2023; the example uses only its standard library.
