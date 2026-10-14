---
layout: post
lang: en
title: "Prompting, RAG, or Fine-Tuning: Start with the Failure"
date: 2024-06-27 09:00:00 +0200
permalink: /blog/prompting-rag-fine-tuning/
author: Rafał Prońko
categories: [llm, evaluation]
level: intermediate
series: "LLM evaluation and adaptation"
prerequisites: ["Basic Python", "Prompting basics"]
description: "How to choose between prompting, retrieval and fine-tuning by diagnosing failing cases first: oracle context, grouped holdouts and guardrails enforced in code."
image: /assets/images/prompting-rag-fine-tuning.svg
---

# Prompting, RAG, or Fine-Tuning: Start with the Failure

![Simplified decision flow. Collect failing cases and keep a grouped holdout, then re-run each case with the gold passage. An answer that cannot be read, or has the right content in the wrong shape, is an output-contract problem. Wrong even with gold context: add instructions and examples from development groups; if that fixes it, suspect behaviour, otherwise suspect base-model capability. Right with gold context: if the end-to-end answer is also correct and contract-valid, the case passes; if not, check whether retrieval put the gold passage in the top k. If it did not, fix retrieval; if it did, fix what reaches the model. A band at the bottom lists checks recommended outside the model; the example implements the schema, citation and retired-version checks, but not access control.]({{ '/assets/images/prompting-rag-fine-tuning.svg' | relative_url }})

*The diagram is a simplified view and a recommended architecture. The exact rule the example implements is the `diagnose` function below.*

<time datetime="2024-06-27">27 June 2024</time>

*Editorial note: the date above is an editorial label for this post's place in the series, not the day it was written. The papers and the Python version it relies on predate it. The downloadable example was written and last verified in October 2026; details are in “Run it yourself”.*

“Should we use RAG or fine-tune?” is usually asked before anyone has looked at what the model actually gets wrong.

I think that order is backwards. Prompting, retrieval and fine-tuning are not three levels of seriousness. They change three different things: what the model sees for a single request, what extra information a system fetches for it, and what the model's parameters do on every request. Which one helps depends on why the answer was wrong.

So I start with a small set of failing cases and ask one question per case: *what was missing?* The fact? The behaviour? Or the ability to use the fact even when it is right there in the context?

In this post I go through the three mechanisms, a diagnostic that separates those failures, and a small runnable harness on a synthetic company policy. The harness does not call a model and does not train one. I explain below exactly what it does and what it cannot show.

## Three mechanisms, three places where a change lives

### Prompting: conditioning one request

A prompt is everything the model reads before it answers: instructions, the question, maybe a few worked examples.

Brown et al. (2020) defined the few-shot setting precisely: the model receives K demonstrations of the task as conditioning at inference time, and **no weight updates are allowed**. Zero-shot uses only a natural-language instruction. In their experiments K was typically between 10 and 100, because that is what fitted into a 2,048-token context window.

Two consequences matter in practice.

First, the change lives only in the request. The next request does not remember it. Every example you add is paid for again on every call, in input length and in processing time.

Second, the authors themselves were careful about what in-context learning is. In the limitations section they note it is unclear whether few-shot learning learns a new task at inference time or recognises a task already learned during pre-training, and that this may differ from task to task. I read that as a warning: examples are very good at *specifying* a task the model can already do. They are not a guaranteed way to teach it a new skill.

### Retrieval: an external memory you can replace

Lewis et al. (2020) described retrieval-augmented generation (RAG) as combining two memories. The **parametric memory** is the generator's weights (in the paper, BART). The **non-parametric memory** is a dense vector index of Wikipedia, searched by a neural retriever (DPR). Retrieved passages are concatenated with the input and the generator conditions on them.

The paper lists the problems this addresses: parametric-only models cannot easily expand or revise their memory, cannot straightforwardly show where an answer came from, and may hallucinate. With an external memory, knowledge can be revised directly and inspected.

The most practical experiment in the paper is index hot-swapping. The authors built an index from a December 2016 Wikipedia dump and compared it with their December 2018 index, asking “Who is {position}?” about 82 world leaders who changed in between. With the matching index the model answered 70% (2016) and 68% (2018) correctly; with the mismatched index, 12% and 4%. They updated what the system knew by replacing the index, without retraining.

One detail is often lost when the term travels: in the original paper RAG is itself **fine-tuned end to end**. The query encoder and the generator are trained jointly; the document encoder and index are kept fixed. Much of what is called RAG today is retrieval plus a prompt, with no training at all. Both are legitimate, but they are different systems.

### Fine-tuning: changing the parameters

Fine-tuning updates weights on task-specific examples. The change applies to every request, without spending context on it.

Hu et al. (2021) proposed LoRA to make this cheaper. Take a pre-trained weight matrix W₀ of shape d_out × d_in (the paper writes d × k). LoRA freezes it and constrains the update to a low-rank product: W₀ + ΔW = W₀ + BA, where B has shape d_out × r, A has shape r × d_in, and the rank r is much smaller than min(d_out, d_in). The update ΔW therefore has rank at most r. Only A and B are trained. The paper also scales the update by a constant α/r; I leave that out here because it does not change the argument. Three properties from the paper are relevant for a decision:

- **No extra inference latency by construction**, because BA can be merged into W₀ before serving.
- **Cheap task switching**: subtract one BA and add another. For GPT-3 175B with r = 4 on the query and value projections, the authors report the checkpoint shrinking from roughly 350 GB to 35 MB. You still need the full base model to serve it.
- **A stated limitation**: if the adapters are merged into the weights, batching requests for different tasks in one forward pass is not straightforward.

The LoRA paper also reminds us why prompting is not the end of the story. In its Appendix A, GPT-3 175B validation accuracy on MNLI-matched is 40.6% few-shot (six in-context examples) against 89.5% after full fine-tuning, and on RTE 69.0% against 85.4%. The MNLI fine-tuning figure uses the full training set of about 392,000 examples. The paper's low-data table (Table 16) shows how that depends on data: fine-tuning on 1,000 and 10,000 MNLI examples gives 85.8% and 88.9%. These are reported results for those tasks and that model. How many curated examples a task needs depends on the task, the model and the method, and I would not read a general recipe out of these numbers.

### They are not alternatives

These mechanisms compose. Every RAG system has a prompt. The original RAG model was fine-tuned. A LoRA-adapted model can read retrieved passages. The useful question is not “which one” but “which change, at which layer, fixes *this* failure, and what does it cost to keep it working”.

## Three failures that look the same from outside

From the user's point of view, every failure is “the answer was wrong”. Underneath there are at least three different situations.

**Missing knowledge.** The model does not have the fact, or has an outdated one. A company's hotel cap changed in April; the model has never seen the new policy. Better wording will not produce a number the model has never seen.

**Wrong behaviour.** The fact is available, but the output does not follow the convention you need: the wrong label, a missing citation, broken JSON, answering instead of abstaining. The model *could* do it; it has not been told clearly enough, or the convention is unusual.

**Insufficient ability.** The fact is in the context, the convention is clear, and the model still fails: it misses a negation, does not combine two clauses, or confuses which rule applies. This is a limit of the base model on this task, at least as currently posed.

There is a tempting slogan here: “fine-tuning cannot teach facts, use RAG for knowledge”. I avoid it, because it is not accurate. Lewis et al. start from the observation that pre-trained models *do* store factual knowledge in their parameters, and that parametric-only models need *further training* to update it. Training changes what a model says about facts.

What makes weights an awkward home for changing facts is operational. You cannot point to the passage behind an answer. Removing a retired rule means another training run and a check that the old answer does not come back through a paraphrase. Per-user access to documents cannot be expressed in weights. And you need evidence that the new fact generalises beyond the exact phrasing in the training data. For a policy that changes every quarter and must be cited, those costs usually decide the matter. For stable domain vocabulary or a house style, they may not.

## The oracle-context diagnostic

To tell these failures apart, I re-run each failing case under controlled conditions:

1. **Closed book**: the question alone. Does the model already know?
2. **Retrieved**: the question with what the retrieval pipeline actually returned. This is the end-to-end system.
3. **Oracle**: the question with the *gold* passage, the one a human marked as containing the answer. This removes retrieval from the picture.
4. **Oracle + examples**: the same, plus instructions and a few demonstrations. Only needed when the oracle run fails.

Separately, I check a property of the retriever that does not involve the model at all: was the gold passage in the top k?

The oracle run is the pivot, provided the gold annotation is complete: the marked passage, or passages, really contain everything needed to answer. If the model is right with the gold passage and wrong end to end, the generator is probably not the first thing to change. If it is wrong even with the gold passage, retrieval is unlikely to be the first lever. Evidence selection can still matter (another passage, different chunking), so check the annotation before blaming the model. The labels point to the first intervention to try; they are not proof of cause.

The decision rule fits in a few lines. This is an excerpt from `harness.py` in the downloadable example (see “Run it yourself”); the constants and data classes are defined earlier in the same file:

```python
def diagnose(r: CaseResult) -> str:
    oracle = r.conditions["oracle"]
    if not oracle.readable or (oracle.correct and not oracle.valid):
        return OUTPUT_CONTRACT
    if not oracle.correct:
        fewshot = r.conditions.get("oracle_fewshot")
        if fewshot is None or not fewshot.readable:
            return GOLD_CONTEXT_FAILS
        return BEHAVIOR if fewshot.correct else CAPABILITY_SUSPECTED
    rag = r.conditions["retrieved"]
    if rag.ok:
        return PASS
    if not rag.readable or rag.correct:
        return OUTPUT_CONTRACT
    if r.answerable and not r.gold_in_context:
        return RETRIEVAL_MISS
    return CONTEXT_USE
```

`readable` means the decision and the limit could be read at all. `correct` means they are readable and match the reference. `valid` means the answer satisfies the output contract described later. `ok` means correct and valid. An answer with the right content in the wrong shape, or one that cannot be read, is an output-contract problem at whichever stage it appears. Once examples have fixed the content, a remaining contract error is reported separately; it does not turn the case into a capability problem. `GOLD_CONTEXT_FAILS` is the honest label when the oracle run failed and nobody has yet produced a readable run with examples. Cases without a gold passage, the abstention cases, are not `answerable` and cannot have a retrieval miss.

## A synthetic scenario: a policy assistant

The scenario is an internal assistant that answers expense-policy questions for a fictional company. The corpus is nine short passages I wrote for this example: hotel caps, meals, client entertainment, home-office equipment, training budget, rail travel and supplier gifts. One passage is the *retired* version 2 of the hotel policy, still sitting in the index next to the current version 3. That is a realistic situation, not a contrived one.

Each question has a reference answer in a fixed shape:

```json
{"decision": "allowed", "limit_eur": 220, "citations": ["P-TRV-3"]}
```

The decision is one of `allowed`, `not_allowed`, `needs_approval` or `insufficient_context`.

**An important caveat about the answers.** For each case I wrote, by hand, the answer string for each condition. They are **not outputs of any language model**. They illustrate failure patterns so the diagnostic has something to classify. The harness tells you nothing about how well prompting, RAG or fine-tuning would perform; it shows how the rule separates failures once you have real outputs.

What the harness does compute for real:

- a small BM25 retriever over the synthetic corpus, deterministic, with ties broken by passage id;
- whether the gold passage is in the top k;
- parsing and validating every answer string against the output contract;
- grading against the reference, assigning labels and checking the development/holdout split.

## Run it yourself

The complete example is one download: [prompting-rag-fine-tuning.zip]({{ '/assets/examples/prompting-rag-fine-tuning.zip' | relative_url }}). It contains the source (`harness.py`, `demo.py`), the fixtures, the tests, a README and the project metadata (`pyproject.toml`, `uv.lock`, `.python-version`). The individual files can also be opened directly, for example [harness.py]({{ '/assets/examples/prompting-rag-fine-tuning/harness.py' | relative_url }}) and [cases.json]({{ '/assets/examples/prompting-rag-fine-tuning/fixtures/cases.json' | relative_url }}).

It uses only the Python standard library and needs Python 3.11 or newer:

```text
unzip prompting-rag-fine-tuning.zip
cd prompting-rag-fine-tuning
python3 --version
python3 demo.py
python3 -m unittest discover -s tests -v
```

If `python3 --version` reports something older than 3.11, call a newer interpreter explicitly, for example `python3.11 demo.py`. `python3 demo.py --k 3` changes the retrieval depth; any k from 1 to 9 works.

These commands were last run on 13 October 2026, from a fresh unzip of the archive, with CPython 3.11.12, 3.12.8 and 3.14.4. On each, all 29 tests pass and the table below is copied verbatim from the output. That verification is recent; it is not a record of a run in 2024.

## What the run shows

The main part of the report (k = 2):

```text
case split   group         retrieved top-k       gold@k  closed   oracle   +ex      rag      label
C01  dev     hotel         P-TRV-3,P-TRV-2       yes     wrong    ok       -        wrong    CONTEXT_USE
C02  dev     hotel         P-TRV-3,P-TRV-2       yes     wrong    ok       -        ok       PASS
C03  dev     meals         P-ENT-1,P-EQP-1       no      wrong    ok       -        wrong    RETRIEVAL_MISS
C04  dev     meals         P-MEAL-1,P-TRV-3      yes     wrong    ok       -        ok       PASS
C05  holdout equipment     P-EQP-1,P-TRV-3       yes     wrong    wrong    ok       wrong    BEHAVIOR
C06  holdout equipment     P-EQP-1,P-RAIL-1      yes     wrong    ok       -        ok       PASS
C07  holdout training      P-TRN-1,P-ENT-1       yes     wrong    wrong    wrong    wrong    CAPABILITY_SUSPECTED
C08  dev     rail          P-RAIL-1,P-TRN-1      yes     format   format   -        unparsed OUTPUT_CONTRACT
C09  dev     gifts         P-GIFT-1,P-ENT-1      yes     format   ok       -        ok       PASS
C10  holdout uncovered     P-TRV-3,P-HTL-DISC    n/a     ok       ok       -        wrong    CONTEXT_USE
C11  dev     entertainment P-ENT-1,P-MEAL-1      yes     wrong    ok       -        ok       PASS
```

`format` means the content was right but the answer broke the contract. `wrong` means the content was wrong. `unparsed` means the decision and limit could not be read at all, so the content is not graded. `gold@k` is `n/a` for C10, which has no gold passage; over the ten answerable cases, the gold passage is in the top two for nine. The retrieval columns are computed; the answer columns come from the hand-authored fixtures.

Going through the failures one by one shows why a single “accuracy” number would hide the decision.

**C03, retrieval miss.** “Can I expense the wine I ordered at dinner with a customer?” The policy says “alcoholic drinks are never reimbursable, including at client meals”. The question says *wine* and *customer*; the passage says *alcoholic* and *client*. A lexical retriever does not connect them, and the gold passage is not in the top two. With the gold passage, the illustrative answer is right. The first lever is retrieval: query expansion, a dense or hybrid retriever, or different chunking. Fine-tuning the generator would not put the passage in front of it.

**C01, context use.** The question asks about a 200 EUR hotel in a capital city. Both the current and the retired hotel policy are retrieved. The illustrative end-to-end answer follows the retired one: “not allowed, 150 EUR”. The gold passage *was* in the context, so this is not a retrieval miss in the narrow sense. It is a problem with what reaches the model: two versions of the same rule. The first fix to try is metadata, not a better prompt. With a `status = current` filter, the computed top two becomes `P-TRV-3, P-HTL-DISC`. I did not re-observe the answer after that change; a real run would have to.

**C10, context use, but different.** A question about a pet sitter has no answer in the policy, so there is no gold passage to retrieve. The closed-book and oracle answers correctly abstain. End to end, the retrieved hotel passage pulls the illustrative answer towards “allowed, 180 EUR”. This is a case for an explicit abstention rule, tested with unanswerable questions in the evaluation set.

**C05, behaviour (label `BEHAVIOR`).** Buying a 250 EUR monitor: the policy allows up to 300 EUR but requires approval *before* the purchase. With the gold passage the illustrative answer says “allowed”. With two demonstrations from development groups, one of which maps a prior-approval rule to `needs_approval`, it is right. This pattern suggests testing the output convention before adding knowledge. The fixture condition combines instructions and demonstrations; real outputs under that condition alone would not tell you which change helped. Keeping the examples in the prompt is a first fix to test. If that convention has to hold across many request types, and paying for the examples on every call becomes a cost or latency problem, this is the kind of failure where an adapter trained on curated examples is worth evaluating.

**C07, capability suspected.** Can conference flights go on the training budget? The passage says the budget covers registration fees and that travel “is reimbursed under the travel policy and not charged to the training budget”. The illustrative answers miss that clause both with the gold passage and with examples. When instructions and demonstrations do not help, the next steps are decomposing the question, trying a different base model on the same oracle set, or rewriting the source passage. Fine-tuning *might* help, but only an experiment with a separate holdout can show it. That is why the label says *suspected*.

**C08, output contract.** The oracle answer has the right decision and limit but no `citations` field. The end-to-end answer is prose rather than JSON, so the harness leaves its content ungraded. These observations point to the output contract as the first intervention to test. Validate the output in code first; then improve instructions or examples so the validator rarely has to reject.

**C09** is worth a sentence too. The closed-book answer to “May I accept a cash gift?” is right; it is a common norm. But it carries no citation, so the contract rejects it. “The model already knows” is not the same as “the system can show why”.

## Evaluate the pieces separately

The diagnostic only works if the measurements are not mixed together.

**Retrieval** is evaluated without the generator: is the gold passage in the top k? This needs labelled gold passages, which is real annotation work, but it turns “the answers got worse” into “recall at k dropped after the re-index”. Questions without a gold passage are left out of that number and evaluated separately, as abstentions.

**Answer quality given the evidence** is evaluated with oracle context. Holding the evidence constant helps compare the generator, the prompt or the adapter without retrieval variation.

**End to end** is what users experience. It is the final check, not the place to start debugging.

The split matters as much as the metric. Policy questions come in families: several paraphrases of the same hotel question, several variants of the equipment rule. If paraphrases of one question land on both sides of a random row split, the holdout rewards memorising wording, not handling the rule. The harness splits by **policy area**: hotel, meals, entertainment, rail and gifts for development; equipment, training and an “uncovered” group for holdout. A test checks that a row-by-row split would leak the `hotel` group across the boundary, and another rejects any few-shot exemplar taken from a holdout group:

```python
dev, holdout = h.grouped_split(self.data["cases"], self.data["holdout_groups"])
with self.assertRaisesRegex(ValueError, "holdout group 'equipment'"):
    h.check_exemplars(["C02", "C06"], dev, holdout)
```

That is a fragment of `tests/test_harness.py` in the download. The same rule applies to fine-tuning data: training examples come from development groups only. Otherwise the holdout measures how well the adapter remembers what it was trained on.

Brown et al. faced a large-scale version of this problem. Their training data came from the web, so benchmark test sets could have been in it. They built conservatively “clean” subsets by removing examples with 13-gram overlap with the training data and compared scores. For a policy assistant the source of leakage is smaller and closer to home: your own exemplars, your own tuning set and your own paraphrases.

## Governance does not belong in the prompt

Several requirements in this scenario are not model problems at all:

- the answer must be valid JSON with exactly the agreed fields;
- the decision must be one of four values;
- every decision except an abstention must cite at least one passage;
- citations must point to passages that were actually in the context;
- a retired policy must never be cited;
- a user must only see passages they are allowed to see.

A prompt can *ask* for all of that. It cannot *enforce* it. A fine-tuned model is no different: it makes the right format more likely, not guaranteed. So these rules belong in code that runs on every answer, whichever method produced it.

The harness implements the first five, in `parse_answer`. In the run above they flag the end-to-end answers for C01 (cites a retired passage) and C08 (not JSON). Two limits are worth stating plainly. The citation check is narrow: it confirms that cited passages were in the context and are current, not that they support the answer, so an unrelated current passage would pass. And the harness only reports these errors; a serving system has to act on them, by blocking the answer, retrying or falling back to an abstention.

The sixth rule, access control, is not implemented in the example; the synthetic corpus has no users or permissions. Here it is an architecture recommendation. It belongs in the retrieval layer, as a filter on what can be fetched for this user. Passages the user cannot see should never reach the context, so neither the model nor the prompt has to decide. The band at the bottom of the diagram is meant in the same sense: what a production system should do, of which the example shows only the answer checks.

## Cost, latency and maintenance: the same questions for every option

I do not have benchmark numbers for this scenario, and I won't invent them. I can, however, ask the same questions of every option and answer them from how each mechanism works.

| Question | Prompt and examples | Retrieval | Fine-tuning / adapter |
|---|---|---|---|
| What does every request pay for? | The instructions and examples, every time | Retrieved passages in the input, plus the retrieval step | Nothing extra in the input; serving the adapted model |
| What is the up-front work? | Writing and testing prompts | Ingestion, chunking, index, gold passages for evaluation | Curating and labelling training data, training runs |
| How is a fact updated? | Edit the prompt (only for small, stable content) | Re-index the document; retire the old one | New data and another training run |
| What breaks quietly? | Prompt edits shift other cases | Stale or duplicated documents, retriever drift | Old behaviour returning through paraphrases; base-model upgrades |
| What must be re-evaluated after a change? | The whole answer set | Retrieval recall and end to end | Oracle set, holdout groups and end to end |

Two properties from the papers belong here. LoRA adds no inference latency once merged and lets many tasks share one base model, but mixing differently adapted requests in one batch needs care. RAG lets you change what the system knows by swapping the index, and the knowledge it uses can be inspected. Neither property is free: the index has to be kept current, and the adapter has to be retrained and re-evaluated whenever the base model or the convention changes.

## Where this example ends

The fixtures are hand-authored. The labels show that the decision rule separates failure patterns as designed; they do not show that any method works on real data. Eleven cases are not a sample for estimating anything.

The retriever is a small lexical BM25. Its misses are real for this corpus, but a dense or hybrid retriever would behave differently, which is exactly why retrieval is measured on its own.

The oracle condition assumes you know the gold passage, and that the annotation is complete and sufficient. For many questions that is annotation work, and some questions need several passages. The harness requires *all* gold passages to be in the top k.

With real model outputs, the diagnostic needs more care than these fixtures suggest:

- **Repeat each condition.** The fixtures have one answer per condition. With a real model, one observation can flip a label by chance; run each case several times and compare rates, not single answers.
- **Change one thing at a time.** Use the same model version, decoding settings, instructions and output requirements in every condition, so that only the context differs.
- **Temperature zero is not determinism.** It removes most sampling variation, but many serving stacks do not guarantee identical output for identical input. Repetitions are still needed.
- **Separate instructions from demonstrations.** The oracle + examples condition changes both at once. To know which one helped, run oracle + instructions and oracle + examples as separate ablations.
- **Follow up end to end.** A fix found with oracle context has to be checked again with retrieved context plus the same instructions and examples, because that is what users get.
- **Freeze the inputs.** Keep snapshots of the corpus and index, the exact contexts sent to the model and the group assignments, so that a later run compares like with like.

“Capability suspected” is a hypothesis. A different base model, a decomposed question or a clearer source passage might remove it. The label tells you which experiment to run next.

I did not train a model. The next part of this series is a small LoRA experiment with a baseline and an independent held-out evaluation, where those claims can be tested rather than illustrated.

## Where I start

Not with the method.

I collect failing cases, split them by group rather than by row, and run them with the gold passage. That one comparison usually tells me which intervention to try first: retrieval, what reaches the model, behaviour, or a closer look at the model's limits. It is a hypothesis to test, not a verdict. The guardrails go into code before any of these choices, because none of the three methods can enforce them.

Prompting, retrieval and fine-tuning then become answers to specific failures, and the next question is what each fix will cost to keep working.

### Sources

- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020, arXiv v4 of 22 July 2020): §2 definitions of zero-, one- and few-shot without weight updates; §2.4 evaluation and the 2,048-token context; §4 contamination analysis; §5 limitations, including the open question of what in-context learning learns.
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020, arXiv v4 of 12 April 2021): §1–2 parametric and non-parametric memory, DPR retriever, BART generator and end-to-end training; §4.5 index hot-swapping.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021, arXiv v2 of 16 October 2021): §4.1 low-rank update with matrix shapes and α/r scaling; §4 merging without added latency, task switching and the batching limitation; Appendix A, Table 8, few-shot versus fine-tuned GPT-3; Appendix F.3, Table 16, fine-tuning on MNLI subsets of different sizes.
