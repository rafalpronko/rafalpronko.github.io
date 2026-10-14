---
layout: post
lang: en
title: "Graph RAG and multimodal documents: when the extra complexity pays off"
date: 2024-10-03 09:00:00 +0200
permalink: /blog/graph-rag-multimodal-documents/
author: Rafał Prońko
categories: [llm, rag]
description: "Two independent RAG failure modes — missing relationships or global coverage, and evidence lost during ingestion — and the decision rules, costs and ablations that justify Graph RAG or layout-aware document models."
image: /assets/images/graph-rag-multimodal-documents.svg
---

# Graph RAG and multimodal documents: when the extra complexity pays off

![Diagram of two independent failure modes. Left: finding a candidate quality hold for pump P-200 needs a path across two documents, from P-200 to alloy KA-7 in a bill of materials and from KA-7 to hold QH-22 in a separate notice; every edge carries document, version, page and bounding box. The hold covers only heat lots 31–34 and the P-200 heat lot is unknown, so QH-22 is a candidate hold, and the candidate path is visible only to the quality team. Right: an unordered bag of tokens built from a table has the same token counts when two pressure values are swapped, so it cannot say which value belongs to which pump, while the table with row and column coordinates can.]({{ '/assets/images/graph-rag-multimodal-documents.svg' | relative_url }})

<time datetime="2024-10-03">3 October 2024</time>

> **Editorial note.** The date above is an editorial label in this blog's series, not proof of when the text was first published. The method descriptions rely on two primary sources: the GraphRAG paper (arXiv 2404.16130v1, 24 April 2024) and the LayoutLMv3 paper (arXiv 2204.08387v3, 19 July 2022). The sources were read again and the example code was written and executed in October 2026, with CPython 3.11.12 and `uv`. I describe the method as published in the April 2024 paper, not any later GraphRAG library release or its features.

A RAG system gives a wrong answer about a document collection. The usual next step is to reach for something more advanced: a knowledge graph, Graph RAG, or a multimodal model that "sees" the page.

Before I do that, I want to know **where the information was lost**. In a document system, information can be lost in two different places, and they need different fixes:

1. **Retrieval and coverage.** The facts are in the index, but the answer needs a relationship across several documents, or a view of the whole corpus. Top-k chunk retrieval doesn't bring all of that into the context.
2. **Ingestion.** The facts never got into the index in a usable form. A table became a stream of words, a value lost its column, a checkbox or a stamp disappeared, the page order got mixed up.

These failures are independent. A graph won't restore a column that was lost when the PDF was converted to text. A layout-aware model won't answer "what are the main themes across ten thousand reports?". You can have one problem, the other, both, or neither.

This article is about how to tell them apart, what Graph RAG and LayoutLMv3 actually do according to their papers, and what I need to see before I accept the extra cost.

## Failure mode 1: the answer is spread out

Take a question from a synthetic manufacturing scenario:

> Is there a candidate quality hold for pump P-200?

The bill of materials says that the P-200 impeller is made of alloy KA-7. A separate quality notice says that hold QH-22 applies to alloy KA-7, heat lots 31–34. Neither document mentions both P-200 and QH-22.

Similarity search for the question will probably find the BOM, because it mentions P-200. The quality notice may not rank high enough, because it doesn't mention the pump at all. Finding the candidate needs **a path**: P-200 → KA-7 → QH-22.

The path is not the final answer. The hold is restricted to specific heat lots, and neither document says which heat lot the P-200 impeller came from. So QH-22 is a *candidate* hold, and whether it applies is **unknown** until there is evidence of the P-200 heat lot. A path can only show that two facts are connected through a shared entity. It can't add a condition that no document states, so it can't establish applicability. A system answering this question should say "candidate hold, heat lot unknown", not "yes".

There is a second, different variant of the same family:

> What are the main recurring quality problems across all supplier reports from the last two years?

There is no single passage containing the answer. This isn't multi-hop lookup, it's summarisation of the whole corpus with respect to a question. The GraphRAG paper calls this kind of task query-focused summarization (QFS) and states plainly that RAG "fails on global questions directed at an entire text corpus".

These two variants need different mechanisms, so I separate them:

- **Graph-assisted retrieval** — a general idea: store entities and relationships, and when answering, follow edges from the entities in the question to collect evidence. Many systems do this in different ways. It helps with explicit relationships between named things.
- **Microsoft's Graph RAG** as described in the April 2024 paper — a specific pipeline, built mainly for global questions. It builds an LLM-derived entity graph, partitions it into communities, pre-generates summaries of those communities, and answers a question with a map-reduce over the summaries.

I also want to name what is *not* Graph RAG. If my data is already in relational tables with known keys, a SQL join from `pumps` to `bom_lines` to `quality_holds` answers the first question deterministically. That's an ordinary, very good database query. It doesn't need an LLM-derived graph, and calling it Graph RAG only blurs the decision.

## What the Graph RAG paper actually describes

The paper, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Edge et al., arXiv 2404.16130v1), describes the pipeline in steps. I'll summarise them, because each step has a cost and a place where errors enter.

**Chunking.** Source text is split into chunks, and each chunk is sent to LLM prompts that extract graph elements. The paper notes the trade-off: longer chunks mean fewer LLM calls, but worse recall. On a HotPotQA sample, a 600-token chunk size extracted almost twice as many entity references as 2400 tokens, with a single extraction round.

**Extraction.** A multipart prompt identifies entities (name, type, description) and relationships between "clearly-related" entities. An optional second prompt extracts *claims* as covariates. Extraction can run several rounds of "gleanings": the LLM is asked whether it missed entities, and if so, it is asked to extract more.

**Element summaries.** Many instances of the same entity or relationship are summarised again into one description per graph element. The authors write openly that the LLM may not extract the same entity in a consistent text format, which leads to duplicate nodes. They consider their approach resilient to that, *provided* there is enough connectivity between the variants and a shared set of related entities. They also say that their index, with rich text descriptions on a potentially noisy graph, differs from typical knowledge graphs that rely on concise, consistent triples.

That sentence matters for practice. A graph built for summarising themes can tolerate noise that a graph used to find candidate holds for P-200 cannot.

**Communities.** The graph is treated as undirected and weighted, and partitioned with the Leiden algorithm into a hierarchy of communities. Each level covers all nodes in a mutually exclusive, collectively exhaustive way.

**Community summaries.** For each community, the LLM writes a report-like summary. At the leaf level, edges are ordered by the combined degree of their source and target nodes, and the descriptions of both nodes, the edge and related claims are added to the context in that order until the token limit is reached. At higher levels, sub-community summaries replace their elements when everything doesn't fit.

**Query time.** For a chosen community level, the summaries are shuffled and divided into chunks. The map step generates a partial answer for each chunk in parallel, along with a helpfulness score from 0 to 100; answers scored 0 are filtered out. The reduce step adds partial answers to a new context in descending order of score until the token limit is reached, and generates the final global answer.

### What the evaluation does and doesn't show

The authors evaluated on two corpora of about 1 million tokens (podcast transcripts) and 1.7 million tokens (news articles), with 125 LLM-generated "global sensemaking" questions per dataset. There was no gold answer. Answers were compared head-to-head by an LLM evaluator on comprehensiveness, diversity, empowerment, and directness as a control metric, with each comparison repeated five times.

Six conditions were compared: Graph RAG at four community levels (C0–C3), map-reduce summarisation of the source text without a graph (TS), and naïve semantic-search RAG (SS). Context window size and prompts were the same for all conditions, except for minor modifications to reference styles.

The results the paper reports:

- all global approaches beat naïve RAG on comprehensiveness (72–83% win rate on podcasts, 72–80% on news) and diversity (75–82% and 62–71%);
- naïve RAG gave the most direct answers, as expected for the control metric;
- community summaries gave a "small but consistent" improvement over map-reduce on source text, except at the root level;
- low-level community summaries needed 26–33% fewer context tokens than source-text map-reduce, root-level summaries over 97% fewer;
- empowerment results were mixed, and the authors note that specific examples, quotes and citations seemed key for that metric.

In the discussion they add two caveats I treat as decision inputs. First, the evaluation covers only one class of sensemaking questions on two corpora of about 1 million tokens. Second, "in many cases the graph-free approach to global summarization of source texts performed competitively", and the decision to build a graph index depends on compute budget, the expected number of queries per dataset, and value from other uses of the graph.

So the paper supports a narrow claim: for global questions over a corpus of that size, a graph-based global approach gives more comprehensive and diverse answers than naïve RAG, as judged by an LLM. It does not show better factual accuracy on specific questions, better behaviour with tables, or that the graph beats a simpler global map-reduce by a wide margin. It also doesn't test documents with complex layout. The pipeline starts from "text extraction and chunking".

## Failure mode 2: ingestion lost the evidence

Now a different question about the same BOM:

> What is the maximum pressure of P-200?

The page has a table:

| Pump  | Max pressure (bar) | Impeller alloy |
|-------|-------------------:|----------------|
| P-200 | 16                 | KA-7           |
| P-300 | 10                 | KB-2           |

If text extraction reads this table column by column, or merges cells into one line, the chunk may look like: `Pump P-200 P-300 Max pressure (bar) 16 10 Impeller alloy KA-7 KB-2`. The words are all there. Whether the assignment of a value to a row and a column survived depends on what happens next. In an ordered string it is still *encoded* in the positions: if you know the reading order was column by column and every cell was non-empty, you can rebuild the table. Ordered text with delimiters or labelled rows (`P-200 | 16 | KA-7`) keeps it outright. So a column-major string alone doesn't prove the structure is lost.

It gets lost when the representation stops carrying that information. The example below tests the extreme case: an unordered bag of tokens (a `Counter`), the view that a bag-of-words keyword index works from. **In the bag, the assignment of a value to a row and a column is gone.** Swap the two pressures and the token counts are exactly the same. Real pipelines sit somewhere between the two: merged cells, empty cells, wrapped lines and multi-column pages make the reading order ambiguous, and then position no longer tells you which cell a value came from.

No retrieval improvement fixes a representation that lost the assignment. Neither does a graph built from it: an LLM extracting "P-200 has max pressure X" from such a chunk can only guess.

The same applies to forms (which label a value belongs to), checkboxes, signatures and stamps, headers repeated on every page, footnotes attached to a specific cell, and reading order in multi-column layouts. The text might be correct while its *structure* is wrong.

### What LayoutLMv3 does

LayoutLMv3 (Huang et al., arXiv 2204.08387v3, ACM Multimedia 2022) is one well-documented approach to this class of problems. It's worth understanding at the level of the paper, not the marketing summary.

**Input.** The model takes three kinds of information:

- *text* — words from the page, obtained with an off-the-shelf OCR toolkit, embedded with word embeddings initialised from RoBERTa;
- *layout* — for every text token, a 1D position in the sequence and a 2D position, i.e. bounding box coordinates normalised by image size. LayoutLMv3 uses segment-level positions: words in the same segment share one box;
- *image* — the page image is resized to 224×224 and cut into 16×16 patches (196 patches), each linearly projected into a vector. No CNN backbone or object detector is used for image features.

Text and image embeddings are concatenated and processed by one multimodal Transformer. The base model has 12 layers and hidden size 768; the large model has 24 layers and hidden size 1024. The maximum text sequence length is 512 tokens.

**Pre-training objectives.** The model is pre-trained on about 11 million document images from IIT-CDIP with three objectives:

- *masked language modelling* — about 30% of text tokens are masked in spans, and the model predicts them from the remaining text, layout and image;
- *masked image modelling* — about 40% of image patches are masked in blocks, and the model predicts discrete image tokens produced by a separate image tokenizer;
- *word-patch alignment* (WPA) — for each unmasked text word, the model predicts whether the image patches corresponding to that word were masked.

WPA is the conceptual centre. On a document page, a word has a place on the image. MLM and MIM mask text and image independently, so nothing forces the model to learn which patch belongs to which word. WPA adds exactly that signal. In the paper's ablation, adding WPA improved all evaluated tasks compared with the variant without it. The same ablation shows a text-and-layout-only model already doing well on some tasks, and the image modality being critical for layout analysis on PubLayNet.

**Use.** LayoutLMv3 is fine-tuned per task: labelling words in forms and receipts (FUNSD, CORD), classifying document images (RVL-CDIP), extractive question answering on document images (DocVQA), and layout analysis on PubLayNet, where it serves as the feature backbone of a Cascade R-CNN detector without text input and predicts boxes for regions such as text, title, list, figure and table. Note that a "table" box there is a region, not the rows and columns inside it. In the DocVQA setup the authors used Microsoft Read API to get text and boxes and predicted the start and end of the answer in that text.

### What LayoutLMv3 is not

It's not a general vision-language model that reads any PDF and answers in free text. It's an encoder that produces representations; you add a task head and fine-tune it on labelled data. It depends on OCR or other text-and-box extraction for its text input, so OCR errors happen before the model sees anything. It has a 512-token limit on the text side and a 224×224 image input, so dense multi-page documents need a page or window strategy.

And it isn't magic PDF parsing. If I want a table with rows and columns as output, I still need a task that produces that structure, labels to train or evaluate it, and a representation to store it in.

The paper also has a lesson about comparability: the authors say themselves that their FUNSD result uses segment-level layout positions, which may benefit that task, so it is "not directly comparable" with models using word-level positions. The same care applies to any comparison of pipeline variants.

## Two failures, two axes

I find a simple grid useful:

| | Ingestion keeps structure | Ingestion loses structure |
|---|---|---|
| **Answers are local** | Plain RAG with good chunking and hybrid search; fix ranking first. | Fix ingestion: layout-aware extraction, table structure, provenance. |
| **Answers need relationships or the whole corpus** | Consider graph-assisted retrieval or global summarisation. | Fix ingestion **first**, then consider the graph. |

The order in the bottom-right cell matters. Graph extraction consumes the output of ingestion. If the table lost its row and column assignment, the extracted edges and summaries are built on top of an already-broken representation, and become harder to debug, because the error is now hidden behind an LLM-written description.

## A small, honest example

To make both failure modes concrete, I wrote a tiny offline example in the Python 3.11 standard library. It is **not** a Graph RAG implementation. It doesn't call an LLM, doesn't run OCR or a vision model, doesn't compute embeddings, and doesn't benchmark anything. All relationships are curated by hand in the code.

What it does show:

- traversal across *separate* synthetic documents, where each edge carries document id, version, page, bounding box and quoted text — the path to QH-22 is reported as a candidate, because the quote limits the hold to heat lots 31–34 and nothing in the index gives the P-200 heat lot;
- rejecting an edge without evidence, or whose quote doesn't name both ends;
- refusing an ambiguous name ("Kestrel" could be two companies) instead of guessing;
- returning "no path" as an abstention when this curated index has no evidence, rather than treating it as proof that no hold applies;
- computing the reader set of a derived result as the intersection of readers of every piece of evidence it uses, and refusing to compute it when an edge has no evidence;
- removing a source document and losing the path built on it;
- table lookup by row and column with cell coordinates, and the fact that an unordered bag of tokens can't tell two different tables apart, while the ordered cell values still can.

The project has three files: `pyproject.toml`, `docgraph.py` and `test_docgraph.py`. It has no dependencies outside the standard library. The complete files are available for download, and they are the exact files that were tested: [`pyproject.toml`]({{ '/assets/examples/graph-rag-multimodal-documents/pyproject.toml' | relative_url }}), [`docgraph.py`]({{ '/assets/examples/graph-rag-multimodal-documents/docgraph.py' | relative_url }}), [`test_docgraph.py`]({{ '/assets/examples/graph-rag-multimodal-documents/test_docgraph.py' | relative_url }}).

The core data model is two frozen dataclasses. `Evidence` holds a document id, version, page, bounding box, quoted text and the set of readers allowed to see that document. `Edge` holds a source entity, a relation, a target entity and an optional `Evidence`. Everything else is a small function over these: `resolve` maps a name to one entity or refuses, `validate` checks an edge's evidence, `find_path` runs a breadth-first search over edges with allowed relations, `drop_document` removes edges from one source, and `cell` and `bag_of_tokens` work on the table.

**Two excerpts from `docgraph.py`** (not standalone programs) — a guard for missing evidence, followed by the reader-set intersection. The complete file supplies the imports and dataclasses:

```python
def evidence_of(edge: Edge) -> Evidence:
    if edge.evidence is None:
        raise UnsupportedEdge(
            f"{edge.source} -{edge.relation}-> {edge.target}: no evidence"
        )
    return edge.evidence
```

```python
def readers_of(path: Sequence[Edge]) -> frozenset[str]:
    """Intersect evidence readers. A data operation, not caller auth."""
    if not path:
        raise ValueError("no evidence, no readers")
    return frozenset.intersection(*(evidence_of(e).readers for e in path))
```

The tests cover entity resolution (spelling variants, an ambiguous name, an unknown name), edge validation (curated edges, missing evidence, a quote that doesn't name both ends), traversal (the two-hop path with its provenance, the relation filter, "no path", the reader intersection, `readers_of` refusing an edge without evidence or an empty path, and deleting a source document), and the table (cell lookup with its exact bounding box, a missing cell, the bag that can't tell two tables apart, and the ordered values that can).

I run it from the directory with the three files:

```bash
uv run --python 3.11 python -m unittest -v
uv run --python 3.11 python docgraph.py
```

In the verification run on 12 October 2026 (CPython 3.11.12, `uv`), the test command ran 17 tests and all 17 passed. The script printed exactly:

```text
pump:P-200 -uses-> alloy:KA-7 | bom-pumps v5 p.4 bbox=(72, 310, 290, 324)
alloy:KA-7 -subject_of-> hold:QH-22 | quality-hold v2 p.1 bbox=(56, 120, 520, 150)
candidate path visible to: ['quality']
QH-22 applicability: UNKNOWN; P-200 heat lot is not indexed
P-300 -> QH-22: None
None means abstention: no indexed evidence, not proof of no hold
ambiguous: 'Kestrel' could be ['org:kestrel-alloys', 'org:kestrel-logistics']
P-200 max pressure: 16 bar | bom-pumps v5 p.4 bbox=(300, 310, 410, 324)
bags equal for different tables: True
```

QH-22 is a candidate hold because its notice covers only heat lots 31–34. The P-200 heat lot is absent from the evidence, so applicability stays unknown. The reader intersection concerns the candidate path; it does not establish that the hold applies.

For P-300, `None` means that no path was found in this curated index. It is an **abstention**, not proof that no hold applies. A production index can miss documents or extracted edges; interpreting its abstentions requires measuring extraction recall on labelled evidence.

### What the example deliberately does not exercise

I want to be precise here, because a small green test suite can make a design look more proven than it is.

- **No extraction.** The edges and the table are typed into the code. In a real system an LLM or a parser produces them, and that's where most errors come from. The tests check what I do *with* edges, not whether the edges are right.
- **Weak validation.** `validate` only checks that the quote names both ends. It doesn't check that the quote supports the *relation*. "KA-7 is not subject to QH-22" would pass. Checking the relation itself needs a separate, evaluated step.
- **No communities, summaries or map-reduce.** Nothing here resembles the global part of Graph RAG.
- **No OCR, no layout model, no image.** The bounding boxes are made-up numbers in page points. The table is already structured; the example only shows what discarding ordering and cell associations destroys. Ordered or delimited text can retain those associations.
- **No embeddings and no ranking.** There is nothing to compare it with in terms of retrieval quality.
- **Tiny scale.** A linear scan over four edges says nothing about performance on a large graph.

## Provenance: every derived thing needs a way back

In both failure modes, I want every claim in an answer to point to a place in a source. For a document system, that means at least:

- document id and **version** (or content hash) — not just the file name;
- page number and **bounding box** of the evidence region;
- the quoted text, or a span in the extracted text;
- the version of the extractor or model that produced the derived element.

For graph edges, the edge itself needs evidence, not only its endpoints. "P-200 uses KA-7" without a pointer to a BOM line can't be checked, and can't be removed when the BOM changes.

Community summaries in the Graph RAG pipeline are two steps removed from the source: element descriptions are LLM summaries of extracted instances, and community reports are summaries of those. If a reviewer is to verify a global answer, I need the lineage from the summary to the elements and from the elements to source chunks. The paper's own empowerment observation — that examples, quotes and citations helped judges — points in the same direction.

## Extraction errors and entity resolution

An LLM-built graph has its own error modes, and they compound:

- **missed entities and edges** — the paper's chunk-size observation shows that extraction settings alone change how much is found;
- **invented or wrong edges** — the paper notes that extraction already involves abstractive summarisation and may capture relationships that are implied rather than stated. For sensemaking that may be a feature; for a compliance question it's a risk;
- **duplicates** — "KA7", "KA-7", "alloy KA 7";
- **false merges** — two different companies called "Kestrel" merged into one node, which silently connects unrelated facts.

False merges worry me more than duplicates. A duplicate loses a path; a false merge creates a path that doesn't exist. In the example, `resolve` refuses an ambiguous name instead of choosing one. In a real system I'd store merges as explicit, reviewable decisions with their evidence, and measure precision and recall of edges on a manually labelled sample of documents before trusting multi-hop answers.

## Permissions propagate into derived artifacts

A RAG system usually filters retrieved chunks by the user's permissions. Derived artifacts make this harder.

An edge, an entity description or a community summary may combine content from several documents with different readers. If a summary includes a sentence from a document visible only to the quality team, showing that summary to procurement leaks it — even if procurement can't retrieve the original document.

My working rule, which is my design choice rather than something from either paper: **the readers of a derived artifact are the intersection of the readers of all evidence it was derived from**. That's what `readers_of` computes in the example: the candidate path from P-200 to QH-22 uses the BOM (engineering, quality) and the quality notice (quality only), so its calculated reader set is `{quality}`.

Weak quote validation and reader-set intersection are teaching data operations, not caller authorization enforcement. The example has no authenticated caller and does not check permissions before traversal or output. Candidate identifiers themselves can leak restricted information: `resolve` includes both company IDs in its ambiguity error. A production system must permission-filter candidates, error details, paths and evidence before exposing them, and enforce the caller's access to derived artifacts.

For community summaries this has an uncomfortable consequence. A summary built across many documents with different access lists may end up visible to almost nobody. The alternatives are to partition the graph by access domain before building communities, or to build separate indexes per audience. Both cost more. If I can't compute the readers of a derived artifact, I don't serve it.

## Refresh, delete and rebuild

Documents change. A BOM goes from version 5 to version 6, a quality hold is lifted, a supplier record is deleted for legal reasons.

For plain chunk RAG, updating a document means replacing its chunks. For a derived graph it means more:

1. remove or re-extract edges whose evidence came from the old version;
2. re-summarise the entity and relationship descriptions those edges contributed to;
3. recompute communities if the structure changed — and Leiden on a modified graph may produce a different partition;
4. regenerate the community summaries affected, at every level of the hierarchy.

Deletion is the strict case. If a document has to disappear, its content must disappear from every derived summary too, not just from the chunk index. Without lineage from summaries back to sources, the only safe option is a full rebuild. The example's `drop_document` only shows the first step: the edge goes away, and the path that depended on it goes away with it.

For layout-aware ingestion the update is per page, which is simpler. But a new version of the extraction model is effectively a new version of every derived page representation, and the downstream index has to be rebuilt or at least re-evaluated.

## Costs I write down before deciding

I don't have measured costs for these approaches in this article, and I'm not going to invent them. I write the cost structure down instead, and then measure it on my own corpus.

**Graph RAG, indexing time:**

- one initial extraction call per chunk with the multipart entity-and-relationship prompt — this call happens even with zero gleanings;
- optionally, one claim-extraction call per chunk with the separate covariate prompt, counted on its own;
- for additional gleanings, the yes/no checks and continuation calls actually performed, up to the configured maximum (the paper used 1 gleaning for podcasts and 0 for news);
- one summarisation call per graph element that has several instances;
- community detection;
- one summarisation call per community at every level of the hierarchy.

**Graph RAG, query time:** for a global question, one map call per chunk of community summaries at the chosen level, plus one final reduce call, as in the paper's pipeline. Reducing across several stages or community levels is a variation the paper proposes as future work, not part of what it evaluated. The paper's token comparison is relative to source-text map-reduce, not to naïve RAG, which uses a single generation over retrieved chunks.

**Layout-aware ingestion:** OCR or text-and-box extraction per page, model inference per page (typically on a GPU), labelled data for fine-tuning and evaluation, and annotation work to create it.

**Operations for both:** storage of derived artifacts, lineage, rebuilds on updates and deletes, re-evaluation after any model or prompt change, and people's time to review extraction errors.

The paper's own conclusion frames the trade-off well: the decision depends on compute budget and on the **expected number of lifetime queries per dataset**. An expensive index for a corpus that gets ten global questions is a different investment from the same index for one that gets ten thousand.

## Evaluation: ablations, not a single demo

A single impressive answer doesn't justify a pipeline. For local and named-relationship questions, I want a **2×2 ablation** of layout preservation and graph traversal:

| Condition | Layout preservation | Graph traversal |
|---|---|---|
| A | Off: naïve extracted text, without explicit cell associations or boxes | Off: chunk retrieval |
| B | On: tables and forms retain associations and coordinates | Off: chunk retrieval |
| C | Off: the same ingestion as A | On: graph-assisted retrieval |
| D | On: the same ingestion as B | On: graph-assisted retrieval |

Use the same source corpus, questions, generation model, generation prompt and retrieved-context budget in all four conditions. Keep chunking and extraction settings fixed where the representation permits; record necessary differences. Within each layout setting, the graph and graph-free conditions consume identical ingestion output. Disclose graph-extraction prompts, added indexing calls and traversal settings. Compare B with A and D with C for the layout effect, C with A and D with B for the graph effect, and report their interaction. This includes the graph built from naïve ingestion, rather than assuming its performance.

For global questions, run a separate comparison: **community summaries versus graph-free map-reduce over source text**, like TS in the paper. Both must use identical ingestion, source corpus, questions, model and stated context budget at the map and reduce stages. Match prompts wherever possible and disclose differences in input format, reference style, extraction and summarization prompts, number of calls and total tokens. A per-call context limit is not an equal total compute budget; report the measured cost of each condition. The paper shows that this graph-free global baseline is competitive.

These are evaluation requirements, not experiments executed by the small example.

The questions should be split **by type**, because each mechanism helps a different type:

- local fact in running text;
- value in a table cell or form field;
- multi-hop relationship across documents;
- global or thematic question;
- unanswerable question (abstain with "I don't know"; report "no path" only as a retrieval observation, not as a negative factual answer);
- question whose evidence the user isn't allowed to see.

I split by document family before tuning, keeping related templates, suppliers and document versions in the same split. The **held-out families** remain outside prompt, chunk-size, entity-type and few-shot tuning. I report results separately for every question type and can additionally hold out question types to probe generalization. Otherwise I'm measuring how well I fitted the pipeline to my test set.

Metrics, per question type:

- evidence recall: is the correct document, page and region among what was given to the generator?
- answer correctness against a reference, where a reference exists;
- citation correctness: does the cited region actually support the claim?
- abstention rate on unanswerable questions;
- permission leaks: zero is the only acceptable number;
- tokens, latency and indexing cost per condition.

If I use an LLM as a judge, as the paper did for questions without gold answers, I check its agreement with human labels on a sample first, and I randomise answer order. The paper repeated each comparison five times to account for stochasticity. That is its experimental choice, not a universal minimum; I choose repetitions for the variability and precision required by my evaluation and report uncertainty.

## Decision rules

These are the rules I apply. They're not universal, but they're concrete enough to argue with.

1. **Find the failure first.** Take 30–50 wrong answers and classify each one: was the evidence missing from the index, present but not retrieved, retrieved but misused, or did the question need the whole corpus? Without that, I don't choose a mechanism.
2. **If the evidence was lost during ingestion, fix ingestion.** Start with the cheapest tool that keeps structure: native PDF text with coordinates, a table extractor, a form template. Use a layout-aware model like LayoutLMv3 when a labelled sample shows that simpler extraction loses the fields you need, and you can afford labels for fine-tuning and evaluation.
3. **If the data has a known schema, extract it into tables and use SQL.** Joins over known keys are cheaper, deterministic and easier to audit than an LLM-derived graph.
4. **Add graph-assisted retrieval when** questions repeatedly need explicit relationships across documents, entity identity can be resolved with measured precision, and every edge can store evidence.
5. **Consider Graph RAG-style community summaries when** there's a recurring volume of global, sensemaking questions over a corpus that changes slowly — and only after comparing against map-reduce over source text.
6. **Don't build either if you can't evaluate it.** Without per-type questions, references and a baseline, the extra complexity can't be distinguished from noise.
7. **Don't build either if you can't delete.** If a data removal request would require an untraceable rebuild of derived summaries, solve lineage first.

## Where I start

I don't start with the graph or with a model that sees the page. I start with a sample of wrong answers and a sample of pages.

I check whether the evidence for each answer survived ingestion with its structure and coordinates. Then I check whether retrieval brought it into the context. If the evidence survives ingestion but retrieval misses the required relationships or corpus coverage, I test traversal or global summarisation. If sufficient evidence already reaches the generator, I investigate how it is used.

Graph RAG and multimodal document models solve real problems. They're just different problems, and each one comes with a bill for indexing, permissions, updates and evaluation. I want to know which problem I have before I pay it.

### Sources

- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", arXiv:2404.16130v1, 24 April 2024](https://arxiv.org/abs/2404.16130v1) — pipeline (chunking, extraction, gleanings, element and community summaries, Leiden, map-reduce), evaluation setup, results, limitations and trade-offs. Read from the arXiv HTML version of v1.
- [Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking", arXiv:2204.08387v3, 19 July 2022 (ACM Multimedia 2022)](https://arxiv.org/abs/2204.08387v3) — inputs (OCR text, 2D layout positions, image patches), MLM, MIM and word-patch alignment objectives, fine-tuning tasks and ablation. Read from the arXiv HTML version of v3.
