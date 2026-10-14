---
layout: post
lang: en
title: "Chunking, hybrid search and reranking: a controlled comparison"
date: 2023-12-28 09:00:00 +0100
permalink: /blog/chunking-hybrid-search-reranking/
author: Rafał Prońko
categories: [rag, information-retrieval]
description: "A small deterministic Python experiment: sentence vs window chunking, BM25, a toy concept channel, RRF and a rule reranker, scored on evidence rather than documents."
image: /assets/images/chunking-hybrid-search-reranking.svg
---

# Chunking, hybrid search and reranking: a controlled comparison

![Diagram of the experiment: one corpus chunked two ways, BM25 and a concept channel fused with RRF at depth five, a reranker that can only reorder those five candidates, and evidence labels kept apart from retrieval and used only for metrics.]({{ '/assets/images/chunking-hybrid-search-reranking.svg' | relative_url }})

<time datetime="2023-12-28">28 December 2023</time>

*Editorial note: the date above places this post in the blog's series. The code and every number below come from runs executed in October 2026 with Python 3.14.4 and Python 3.11.12, using uv as the reproduction harness. The ideas and standard-library APIs used here all predate December 2023; the measurements do not.*

When a retrieval-augmented system gives a wrong answer, three knobs get turned almost by reflex: smaller or larger chunks, "add hybrid search", "add a reranker". Each of them can help. Each of them can also hide a problem that the next one then gets the credit for fixing.

I wanted to see the mechanisms separately, on a corpus small enough that I can read every candidate list. So this post is not a leaderboard. It is a controlled experiment on a toy problem: one corpus, one set of queries, one change at a time, and labels that say which **sentences** a reader actually needs.

## What this experiment is, and what it is not

Everything runs on the Python standard library. No model downloads, no API calls.

- The lexical channel is BM25, written out in full.
- The "semantic" channel is a **hand-written synonym table**: words map to concepts (`reimbursement` and `refund` both map to `REFUND`), and chunks are compared by cosine similarity of concept counts.
- The two channels are combined with Reciprocal Rank Fusion.
- The reranker is **three hand-written rules** that look at the query and one passage together.

This is *not* a benchmark of learned dense retrieval such as DPR, and *not* a benchmark of neural cross-encoders. The concept channel and the reranker imitate the *shape* of those components — different words with the same meaning can match; a second stage looks at query and passage jointly and reorders a short list — but none of their quality. Real models fail in other, less predictable ways.

The corpus, the queries, the synonym table and the labels were all written by me, for this post. That is a strong bias, and I come back to it in the limitations. The numbers say what happens on these ten queries. They say nothing about your data.

## The corpus and the labels

The corpus is eight short help-centre pages for a fictional SaaS product, 30 sentences in total: a bulk export API, a search API, error codes, billing, data retention, an FAQ page, token security and a team plan.

Each query has an evidence label: the set of `(document, sentence index)` pairs a reader needs to answer it.

```python
EVIDENCE = {
    # 'It accepts at most 500 rows' is useless without the sentence naming the endpoint.
    "q01": frozenset({("export-api", 0), ("export-api", 1)}),
    "q03": frozenset({("errors", 2)}),
    # 'Its price is 12 euros' needs the plan name, four sentences earlier.
    "q07": frozenset({("team-plan", 0), ("team-plan", 4)}),
    # ...
}
```

Labelling sentences rather than chunks matters for a chunking comparison. If I labelled chunk IDs, every chunking change would need new labels, and I could never be sure the two label sets mean the same thing. With sentence-level evidence, the same labels score both chunkings.

The labels live in their own module. The retrieval code never imports it, and a test enforces that by parsing the imports of every retrieval module.

The ten queries were written to probe specific mechanisms:

| Query | What it probes |
|---|---|
| q01 "What is the row limit of the bulk export endpoint?" | answer split across two adjacent sentences, plus an FAQ sentence that restates the question without answering it |
| q02 "How many rows can the search endpoint accept per call?" | the same split; the export endpoint has an almost identical "It accepts at most 500 rows per call." |
| q03 "What does error ERR-4013 mean?" | exact identifier next to a near-identical one (ERR-4012) |
| q04 "How soon can I expect a reimbursement for a duplicate charge?" | synonyms, no shared content word with the answer |
| q05, q08, q09 | partial overlap or an anchor word plus a synonym |
| q06 "When do API tokens expire?" | easy control |
| q07 "What is the price per seat of the team plan?" | antecedent four sentences away, plus a competing price for another plan |
| q10 "When are dormant workspaces purged?" | vocabulary that neither channel knows |

Two of these questions were reworded during technical review of this post. q02 first asked how many rows the endpoint *returns*, but its evidence only says how many it *accepts*; q04 first asked *how to get* a refund, but its evidence only says *when* one is issued. In both cases the label pointed at sentences that did not answer the question as written. I changed the questions, not the labels, so that the evidence answers them, and a test now pins both wordings and checks that the labelled sentences contain the answer. The effect on the results is reported below; it is one reason this query set is not a holdout.

## Two chunkings

**Sentence chunks**: one sentence per chunk, 30 chunks.

**Window chunks**: three sentences per chunk with a stride of two, so neighbouring windows share one sentence; 13 chunks. With this size and stride every pair of adjacent sentences lands together in at least one window. Sentences two positions apart share a window only when one starts at the first of them, which depends on alignment: sentences 0 and 2 do, sentences 1 and 3 do not. Sentences three or more positions apart never share a window.

```python
def window_chunks(docs, size=3, stride=2):
    chunks = []
    for doc_id, sentences in docs.items():
        for start in range(0, len(sentences), stride):
            ids = tuple(range(start, min(start + size, len(sentences))))
            text = " ".join(sentences[i] for i in ids)
            chunks.append(Chunk(f"{doc_id}#w{ids[0]}-{ids[-1]}", doc_id, ids, text))
            if start + size >= len(sentences):
                break
    return chunks
```

Overlap creates a counting problem: the same evidence sentence can appear in two retrieved chunks. I count evidence by distinct sentences, so the second copy adds nothing, and I log it separately as a redundant chunk.

## The first stage: BM25 and a toy concept channel

BM25 comes in several variants, mainly differing in the idf term. I use the form given in Manning, Raghavan and Schütze's *Introduction to Information Retrieval*, with natural logarithms:

```
score(d, q) = Σ over distinct query terms t:
              log(N / df_t) · (k1 + 1) · tf_td / (k1 · ((1 − b) + b · L_d / L_ave) + tf_td)
```

`tf_td` is the term count in the chunk, `L_d` the chunk length in tokens, `L_ave` the average length. `k1` controls how quickly extra repetitions stop adding score (saturation); `b` controls how strongly long chunks are penalised. I use `k1 = 1.2` and `b = 0.75`, values within the range the book suggests when nothing has been tuned. One consequence of this idf: a term present in every chunk has weight zero.

```python
def term_score(self, term, chunk_id):
    tf = self.tf[chunk_id][term]
    if tf == 0:
        return 0.0
    idf = math.log(self.n / self.df[term])
    norm = self.k1 * ((1 - self.b) + self.b * self.length[chunk_id] / self.avg_length)
    return idf * (self.k1 + 1) * tf / (norm + tf)
```

Tokenisation is deliberately simple: lowercase, a short stop-word list, a trailing plural `s` stripped from longer words, and hyphenated identifiers such as `err-4013` kept as one token.

The concept channel maps tokens through the synonym table and compares concept-count vectors with cosine similarity. Any word not in the table is invisible to it — including every identifier. I left gaps on purpose: `workspace`, `dormant` and `purged` are not in the table, which is how q10 becomes a query no channel can answer.

Both channels return only chunks with a positive score, sorted by score and then by chunk ID. That last rule matters more than I expected; more on ties below.

## Fusion: Reciprocal Rank Fusion

RRF, from Cormack, Clarke and Büttcher (SIGIR 2009), ignores raw scores and uses only positions:

```
RRF(d) = Σ over rankings r: 1 / (c + rank_r(d))      rank is 1-based
```

The constant `c` is called `k` in the original paper; I rename it only because `k` is the cut-off here. The paper uses 60; I use the same, as a convention, not as a tuned value. Truncation happens twice. Each channel's list is cut at depth 5 *before* fusion, so a chunk missing from a truncated list simply gets no contribution from that list. The fused list is cut to 5 again *after* fusion (the `top(..., depth)` call below), so the reranker never sees more than five candidates, whichever first stage feeds it.

```python
def rrf(rankings, c=60, depth=5):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (c + rank)
    return top(dict(scores), depth)
```

Because RRF only sees ranks, it does not care that BM25 scores range roughly from 1 to 11 here while cosine scores sit between 0 and 1. That is its main practical attraction. The cost is visible in the logs: a chunk ranked first by one channel and second by the other gets exactly the same score as one ranked second and first. On a small corpus, such ties are common.

## The second stage: a rule-based reranker

Nogueira and Cho's BERT reranker feeds the query and one passage into the model *together*, scores the pair, and reranks only the candidates from a first-stage retriever; inputs are truncated to at most 512 tokens. My stand-in keeps the first two properties and replaces the model with rules:

- **coverage**: the share of distinct query terms found in the passage, literally or through the same concept;
- **+1** if the passage contains the query's exact identifier;
- **−1** if it contains a different identifier and not the query's one.

```python
def score(query_terms, passage, table):
    tokens = set(tokenize(passage.text))
    concepts = {table[t] for t in tokens if t in table}
    matched = [t for t in query_terms if t in tokens or table.get(t) in concepts]
    value = len(matched) / len(query_terms) if query_terms else 0.0
    wanted = {t for t in query_terms if is_identifier(t)}
    present = {t for t in tokens if is_identifier(t)}
    if wanted & present:
        value += 1.0
    elif wanted and present:
        value -= 1.0
    return value
```

It reorders the five first-stage candidates; ties keep the first-stage order. It never reads labels.

The important property is shared with any real reranker: **it cannot add anything that the first stage did not return.**

## The experimental design

Everything that is not under test stays fixed:

- the same 8 documents, the same 10 queries, the same evidence labels;
- candidate depth 5 for every first stage, and for the reranker's input;
- final `k = 3` chunks for every configuration;
- BM25 `k1 = 1.2`, `b = 0.75`, RRF `c = 60`, none of them tuned.

The baseline is sentence chunks with BM25. From there I change one factor at a time: chunking (sentence → window), first-stage method (BM25 → concept → RRF), and the reranker (on or off). Both chunkings are crossed with all five methods, ten runs in total.

For each query I record:

- **evidence recall@3**: the share of evidence sentences present in the top three chunks combined;
- **full support@3**: whether *all* evidence sentences are present;
- **MRR (complete evidence)**: the reciprocal of the rank at which the evidence first becomes complete (0 if never within `k`). This is not the usual MRR, which takes the rank of the first relevant document. Here the rank is that of the shortest prefix of the list whose chunks together cover all evidence sentences, so the value is never higher than the conventional one. For q01 with sentence chunks and BM25, for example, the first evidence sentence is at rank 2 (conventional reciprocal rank 0.5), but the evidence is complete only at rank 3 (0.33). Everywhere below, including the tables, "MRR" is short for this complete-evidence variant;
- **self-contained@3**: whether a *single* chunk holds all the evidence;
- **parent hit@3**: whether any chunk comes from the right document — a much weaker signal;
- **ceiling**: full support over all five first-stage candidates, the most any reranker could reach;
- **context**: how many distinct sentences the top three chunks hand over.

The report logs every candidate list and every per-query metric, so each number in the tables can be traced to a ranking.

## Results

Means over ten queries, depth 5, `k = 3`:

| Chunking | Method | Evidence recall | Full support | MRR (complete evidence) | Self-contained | Parent hit | Ceiling | Sentences handed over |
|---|---|---|---|---|---|---|---|---|
| sentence | BM25 (baseline) | 0.80 | 0.80 | 0.600 | 0.50 | 0.80 | 0.80 | 2.1 |
| sentence | concept | 0.85 | 0.80 | 0.617 | 0.60 | 0.90 | 0.90 | 2.3 |
| sentence | RRF | 0.90 | 0.90 | 0.650 | 0.60 | 0.90 | 0.90 | 2.4 |
| sentence | BM25 + rerank | 0.80 | 0.80 | 0.600 | 0.50 | 0.80 | 0.80 | 2.1 |
| sentence | RRF + rerank | 0.90 | 0.90 | 0.700 | 0.60 | 0.90 | 0.90 | 2.4 |
| window | BM25 | 0.80 | 0.80 | 0.683 | 0.70 | 0.80 | 0.80 | 5.0 |
| window | concept | 0.90 | 0.90 | 0.833 | 0.80 | 0.90 | 0.90 | 5.2 |
| window | RRF | 0.90 | 0.90 | 0.833 | 0.80 | 0.90 | 0.90 | 5.3 |
| window | BM25 + rerank | 0.80 | 0.80 | 0.683 | 0.70 | 0.80 | 0.80 | 5.0 |
| window | RRF + rerank | 0.90 | 0.90 | 0.783 | 0.80 | 0.90 | 0.90 | 5.3 |

With ten queries, one query flipping a binary metric (full support, self-contained, parent hit, ceiling) between 0 and 1 moves its mean by 0.1. Reciprocal rank moves in fractions: q03 going from 0.5 to 1 moves the MRR mean by 0.05, q02 going from 0 to 0.33 by about 0.033. Read the table as a pointer to the query-level logs, not as a ranking of methods.

What the one-factor changes show:

- **Sentence → window, with BM25**: recall and full support stay at 0.80; MRR goes from 0.600 to 0.683 and self-contained from 0.50 to 0.70. The price is visible in the last column: 5.0 sentences handed over instead of 2.1.
- **BM25 → concept, sentence chunks**: the ceiling rises from 0.80 to 0.90 (q04 becomes reachable), but full support stays at 0.80, because q02 gets *worse* (below).
- **BM25 → RRF, sentence chunks**: full support rises from 0.80 to 0.90 and MRR from 0.600 to 0.650. RRF keeps q04 from the concept channel and q02 from BM25, but inherits the concept channel's tie on q03 (below).
- **Adding the reranker to BM25**: no change in any mean, in either chunking.
- **Adding the reranker to RRF**: MRR goes from 0.650 to 0.700 with sentence chunks, and *down* from 0.833 to 0.783 with windows. The same component helps in one setting and hurts in the other.

No configuration reaches more than 0.90 full support, and none can: q10 fails everywhere, and its ceiling is 0 in every run.

## What happened, query by query

The means hide almost everything interesting. These are the queries that moved.

**q02 — missing boundary context.** The answer is "The search endpoint serves interactive queries…" followed by "It accepts at most 50 rows per call." The export API page contains "It accepts at most 500 rows per call." For a query without numbers, those two second sentences are indistinguishable. With sentence chunks and BM25 they tie on score (8.366192 each), and the tie goes to `export-api#s1` by chunk ID. The evidence still completes at rank 3 (MRR 0.33). The concept channel puts the endpoint sentence first (0.816497), because only it carries both `search` and `endpoint`. But `accept`, `many` and `call` are not in my table, so the query becomes the concept vector `{ROW, SEARCH, ENDPOINT}` and each "It accepts…" sentence becomes just `{ROW}`. The FAQ sentence ("…the row limit of the bulk export endpoint") becomes `{ROW, LIMIT, EXPORT, ENDPOINT}`. Every count in these vectors is 1, so the dot product is the number of shared concepts and each norm is the square root of the vector's size. Each limit sentence shares one concept: 1/(√3 × 1). The FAQ sentence shares two, `ROW` and `ENDPOINT`, but its norm is √4 = 2: 2/(√3 × 2). The vectors differ, yet all three scores equal 1/√3 and tie at 0.57735; ID order puts the right one fourth, outside `k`. Result: evidence recall 0.50, MRR 0, although the right document is ranked first. RRF repairs this, narrowly: BM25 has `search-api#s1` second and the concept channel fourth (1/62 + 1/64 = 0.031754), which beats the FAQ sentence, fourth and third (1/64 + 1/63 = 0.031498). That is a margin, not a tie, so sentence-level RRF completes the evidence at rank 3 (MRR 0.33). The reranker gives both "It accepts…" sentences 0.5 and keeps them in first-stage order; it moves `search-api#s1` from third to second, ahead of `search-api#s0`, but the same three chunks remain, so it changes no metric here. With windows, the endpoint name and its limit sit in the same chunk, and every method puts `search-api#w0-2` first (MRR 1.0).

That is the mechanism behind "add more context to chunks": the sentence with the number has no idea which endpoint it belongs to.

**q07 — boundary context that windows do not fix.** "The team plan is aimed at groups of up to twenty people" and "Its price is 12 euros per seat per month" are four sentences apart. No three-sentence window contains both, so self-contained is false in all ten runs. In every run the first chunk is the one containing the billing page's "The business plan price is 30 euros per seat per month" — the wrong plan, but a better lexical and concept match. Full support is reached at rank 3 everywhere, by combining two separate chunks. Whether a generator would connect a detached "Its price is 12 euros" to the right plan is exactly the question this metric cannot answer. Here full support is the optimistic number and self-contained the pessimistic one; I would not report either alone.

**q03 — exact identifier.** BM25 puts the ERR-4013 sentence first, because `err-4013` is a rare token with high idf. The concept channel cannot see identifiers at all: both error sentences map to the same `ERROR` concept and tie, and ERR-4012 wins on chunk ID. RRF inherits the tie (0.032522 for both: first plus second rank versus second plus first), so sentence-level RRF has MRR 0.5 on this query. The reranker's identifier rule fixes it: sentence RRF + rerank puts `errors#s2` first. This is the whole sentence-level MRR gain from reranking (0.650 → 0.700).

With windows, the ERR-4012 and ERR-4013 sentences share a chunk, so every method "succeeds". It is also the only query where overlap produced a duplicate: both `errors#w0-2` and `errors#w2-3` carry the ERR-4013 sentence. The metric counts it once; a naive chunk-level count would have reported two relevant hits.

**q04 — synonyms.** The query asks how soon a "reimbursement for a duplicate charge" arrives; the page says "Refunds for double billing are issued within five business days." After stop-word removal none of the query's terms (`soon`, `expect`, `reimbursement`, `duplicate`, `charge`) occurs anywhere in the corpus, so BM25 returns nothing; a test checks the zero overlap. The ceiling is 0, and BM25 + rerank stays at 0: the reranker has nothing to reorder. The concept channel finds the billing sentence at rank 1 (cosine 1.0 on `REFUND`, `DUPLICATE` and `BILLING`), and RRF keeps it there. It finds the sentence through the subject, not through the question: `soon` and `expect` are in no table, so nothing here knows the question is about timing; the right sentence simply contains the answer. This is what hybrid search is for — but note why it works here: I put `reimbursement`, `duplicate` and `charge` in the table.

**q01 — a reranker preferring the question to the answer.** The FAQ page says "Customers often ask about the row limit of the bulk export endpoint." It contains every query term and answers nothing. BM25 ranks it first in both chunkings. With windows, the concept channel and RRF give `export-api#w0-2` and `faq#w0-2` identical scores (0.816497 and 0.032522), and the export window wins only on alphabetical chunk ID — MRR 1.0 by a tie-break. The reranker then breaks the tie the other way: the FAQ window covers all five query terms, the export window four (it says "at most", not "limit"). MRR drops to 0.5. That is the window-level loss in the table (0.833 → 0.783).

I find this the most useful case. A rule that rewards query-term coverage rewards text that repeats the question. A trained reranker is not this crude, but whether yours prefers restated questions over answers is something to check on your own FAQ-style pages, not to assume.

**q10 — the ceiling.** "Dormant workspaces purged" shares nothing with "Inactive projects are deleted after 90 days" — not in BM25, not in my synonym table. BM25 returns one candidate, `errors#s3` ("Ask a workspace administrator…"), which is not even the right document. The concept channel returns nothing. No reranker, fusion constant or `k` changes that: the ceiling is 0 in all ten runs.

## Ties decide more than they should

On this corpus, several outcomes were decided by the chunk-ID tie-break rather than by retrieval:

- q01, window chunks, concept and RRF: the 1.0 MRR comes from `export-api` sorting before `faq`;
- q02, sentence chunks: with BM25, `export-api#s1` beats `search-api#s1` on ID (harmless here: the evidence completes at rank 3 either way); with the concept channel, the export and FAQ sentences beat `search-api#s1` on ID and push it out of the top three;
- q03, sentence chunks, concept and RRF: ERR-4012 beats ERR-4013 on ID.

Rename one document and the table changes. Larger corpora have fewer exact ties, but near-ties behave similarly. The lesson I take is procedural: make tie-breaking deterministic and documented, and look for exact or near-equal scores at decisive positions before interpreting a difference.

## Context budget is a confounder

Holding `k` fixed is not the same as holding the input to the generator fixed. Three windows hand over about five sentences; three sentence chunks about two.

To get a rough sense of how much of the window advantage is budget, I ran a supplementary configuration with depth 10 and `k = 6`. This changes two settings at once, so it sits outside the controlled grid. Sentence-chunk RRF does not move: 0.90 full support and 0.650 MRR, as at `k = 3`, now handing over 3.3 sentences instead of 2.4 (lists are shorter than six whenever fewer chunks score above zero). The budget shows only in the concept channel alone, which recovers q02 at rank 4 (MRR 0.25 on that query), taking full support from 0.80 to 0.90 and MRR from 0.617 to 0.642. Window RRF at `k = 3` still has the higher MRR (0.833) and the higher self-contained rate (0.80 against 0.60). In this corpus, a bigger budget helps sentence chunks find the pieces; it does not put them together.

## Limitations

- **Hand-designed bias.** I wrote the corpus, the queries, the labels and the synonym table. The concept channel looks as good as it does because its author knew the queries. A learned dense model brings no such guarantee and fails differently, for example on rare identifiers or domain jargon it did not see in training.
- **No holdout.** Ten queries, all written alongside the corpus. Nothing here estimates quality on unseen queries. Two questions, q02 and q04, were reworded during technical review, after results had been seen, because their labelled evidence did not answer them as first written. The change was made for meaning, not for scores, but it did move the scores: the new q02 shares `accept` with both "It accepts…" sentences, which lifts sentence-level RRF from 0.80 to 0.90 full support. I did not tune `k1`, `b` or `c`. None of this makes ten hand-written, author-revised queries a test set; they illustrate mechanisms.
- **No neural quality or latency claims.** Nothing here was timed, and nothing here says how DPR-style retrievers or BERT-style rerankers perform.
- **Sentence labels are a simplification.** Real evidence can be a table cell, a heading, or half a sentence. Splitting is trivial here because I wrote one fact per sentence.
- **Generation is not evaluated.** Full support says the evidence was handed over. It does not say a model would use it correctly, especially when the pieces arrive detached, as in q07.

## A decision guide

What I would take from this into a real system:

1. **Label evidence, not chunks.** Store answers as spans in source documents. Then chunking can change without relabelling, and duplicated overlap is easy to count once.
2. **Report parent hit and evidence support separately.** In q02 with sentence chunks and the concept channel, the right document was ranked first, but the sentence with the actual limit was not in the top three. A document-level metric would have scored that as a success. Getting the page is not getting the answer.
3. **Measure the candidate ceiling before adding a reranker.** If complete evidence is not in the candidates, reranking cannot help (q04 with BM25, q10 everywhere). Work on recall first: synonyms, query rewriting, a second channel, larger depth.
4. **Keep a lexical channel when queries contain identifiers.** Error codes, SKUs, ticket numbers and function names are where exact matching is reliable and where similarity-based channels can confuse near neighbours (q03).
5. **Check whether facts depend on their neighbours.** Pronouns, "it", "this limit", values under a heading: these need context in the chunk (q02). A window helps when the dependency is short; it does not help when it is long (q07). Prepending a document or section title to each chunk is another option I did not test here.
6. **Compare chunkings at a matched context budget,** or at least report sentences or tokens handed over next to the metric.
7. **Test the reranker on text that restates questions.** FAQs, ticket titles and search logs are full of it (q01).
8. **Read per-query results and look for ties** before trusting a difference in means.

## Run it yourself

**[Download the complete example as a ZIP]({{ '/assets/examples/chunking-hybrid-search-reranking.zip' | relative_url }})** — package, tests, `pyproject.toml` and `uv.lock`, 26 files in the right directory layout. Then:

```bash
unzip chunking-hybrid-search-reranking.zip
cd chunking-hybrid-search-reranking
uv run pytest -q
uv run python -m retrieval_lab.report out
```

`out/report.md` contains the summary, the per-query table and every candidate list. The code uses only the standard library and makes no network calls; uv needs pytest from `uv.lock`, which it fetches on first use unless it is already cached.

The individual files can also be read online. The ZIP additionally contains the empty package marker `retrieval_lab/__init__.py`, which is not linked here because the site build skips files whose names start with an underscore; if you copy the files one by one, create that empty file yourself.

- [README with the module overview and metric definitions]({{ '/assets/examples/chunking-hybrid-search-reranking/README.md' | relative_url }})
- [`pyproject.toml`]({{ '/assets/examples/chunking-hybrid-search-reranking/pyproject.toml' | relative_url }}) and [`uv.lock`]({{ '/assets/examples/chunking-hybrid-search-reranking/uv.lock' | relative_url }})
- Retrieval: [`text.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/text.py' | relative_url }}), [`chunking.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/chunking.py' | relative_url }}), [`bm25.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/bm25.py' | relative_url }}), [`concepts.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/concepts.py' | relative_url }}), [`fusion.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/fusion.py' | relative_url }}), [`rerank.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/rerank.py' | relative_url }}), [`ranking.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/ranking.py' | relative_url }})
- Data and evaluation: [`corpus.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/corpus.py' | relative_url }}), [`labels.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/labels.py' | relative_url }}), [`metrics.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/metrics.py' | relative_url }}), [`experiment.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/experiment.py' | relative_url }}), [`report.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/retrieval_lab/report.py' | relative_url }})
- Tests: [`test_bm25.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_bm25.py' | relative_url }}), [`test_fusion.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_fusion.py' | relative_url }}), [`test_rerank.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_rerank.py' | relative_url }}), [`test_metrics.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_metrics.py' | relative_url }}), [`test_isolation.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_isolation.py' | relative_url }}), [`test_chunking.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_chunking.py' | relative_url }}), [`test_concepts.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_concepts.py' | relative_url }}), [`test_text.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_text.py' | relative_url }}), [`test_experiment.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_experiment.py' | relative_url }}), [`test_queries.py`]({{ '/assets/examples/chunking-hybrid-search-reranking/tests/test_queries.py' | relative_url }})

**Verification scope:** the 35 tests passed on Python 3.11.12 and on Python 3.14.4 (macOS, arm64), both from the working copy and from a fresh unzip of the downloadable archive, and the report generator ran successfully; every number in this post comes from that report. The tests check the BM25 formula and its saturation and length-normalisation behaviour, the RRF formula including missing items, reranker reordering, evidence de-duplication, the separation of labels from retrieval code, the fixed depth and `k`, determinism, and that the wording of q02 and q04 matches what their labelled evidence answers (including q04's zero lexical overlap with the corpus). They do not check anything about neural retrieval, latency, or quality on other data.

## Sources

- G. V. Cormack, C. L. A. Clarke, S. Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), SIGIR 2009 — the RRF formula and the constant 60 used in the paper.
- C. D. Manning, P. Raghavan, H. Schütze, [Introduction to Information Retrieval, "Okapi BM25: a non-binary model"](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html), Cambridge University Press 2008 — the BM25 form used here and the untuned parameter range.
- R. Nogueira, K. Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), arXiv:1901.04085, first version January 2019 — joint query–passage scoring of first-stage candidates, 512-token input limit.
- V. Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), arXiv:2004.04906, first version April 2020 — the kind of learned dense retriever this toy experiment does not evaluate.
