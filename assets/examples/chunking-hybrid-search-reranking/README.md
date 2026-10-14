# Retrieval lab: chunking, hybrid search and reranking on a toy corpus

A small, deterministic, standard-library experiment that compares two chunking
policies and five retrieval pipelines on the same synthetic corpus and queries.

**What this is not.** It is not a benchmark of learned dense retrieval or neural
cross-encoders. The "semantic" channel is a hand-written synonym-to-concept table
with cosine similarity, and the reranker is three hand-written rules. Corpus,
queries, synonym table and labels were all written by the same author, so the
numbers describe mechanisms on these ten queries and nothing more. There is no
holdout set, no latency measurement and no claim about neural model quality.

## Components

| Module | Role | Reads labels? |
|---|---|---|
| `corpus.py` | 8 synthetic help-centre pages (30 sentences) and 10 query texts | no |
| `chunking.py` | sentence chunks; 3-sentence windows with stride 2 (1 sentence overlap) | no |
| `bm25.py` | BM25, `log(N/df)` idf variant, `k1=1.2`, `b=0.75` (formula in docstring) | no |
| `concepts.py` | toy semantic channel: synonym table -> concept counts -> cosine | no |
| `fusion.py` | Reciprocal Rank Fusion, `1/(60 + rank)`, missing items contribute 0 | no |
| `rerank.py` | rule scorer over (query, passage): term/concept coverage +/- identifier match | no |
| `labels.py` | evidence as `(doc_id, sentence_index)` sets, independent of chunking | — |
| `metrics.py` | evidence recall, full support, MRR (complete evidence), self-contained, parent hit, context size | yes |
| `experiment.py` | 2 chunkings x 5 methods, fixed depth=5 candidates, k=3 final chunks | yes |

`tests/test_isolation.py` parses the retrieval modules and fails if any of them
imports `labels`. `tests/test_queries.py` pins the wording of q02 and q04 and checks
that their labelled sentences answer the question as asked (q02 asks how many rows
the search endpoint *accepts*; q04 asks *how soon* a refund arrives). Both questions
were reworded during technical review; see the article's limitations section.

## Run

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 or newer. The code uses only the
standard library, calls no API and downloads no model; pytest is the only dependency,
pinned in `uv.lock` (uv fetches it on first use unless it is already in uv's cache).

```bash
unzip chunking-hybrid-search-reranking.zip
cd chunking-hybrid-search-reranking
uv run pytest -q
uv run python -m retrieval_lab.report out
```

`out/report.md` has the summary table, a per-query table and every candidate
list; `out/report.json` has the same data in machine-readable form.
`out/report-budget.*` is a supplementary run with depth=10 and k=6.

## Metrics (per query, then averaged over 10 queries)

- `evidence_recall` — share of labelled evidence sentences present in the union of the top-k chunks.
- `full_support` — 1 if all evidence sentences are present in that union.
- `reciprocal_rank` — 1 / rank at which the union first contains all evidence; 0 if not within k.
  This is not the conventional MRR (rank of the first relevant item): the whole
  evidence set must be covered, so the value is never higher than the conventional one.
  The article labels its mean as "MRR (complete evidence)"; wherever the article or
  this README says "MRR" for short, it means this complete-evidence variant.
- `self_contained` — 1 if one single chunk in the top-k holds all evidence sentences.
- `parent_hit` — 1 if any top-k chunk comes from a document that contains evidence (weaker than support).
- `ceiling_full_support` — `full_support` measured over all first-stage candidates (depth); a reranker cannot exceed it.
- `context_sentences` — distinct sentences handed over in the top-k (a cost proxy).
- `redundant_evidence_chunks` — chunks whose evidence sentences were already covered by a higher chunk; such duplicates are not counted again.

Ties are broken in two different ways:

- **First stage** (BM25, concept channel, RRF): equal scores are ordered by chunk id,
  alphabetically (`ranking.top`).
- **Reranker**: equal reranker scores keep the order of the first-stage list it was
  given (a stable sort), so they inherit whatever that list did. Example: q02 with
  sentence chunks and BM25 + rerank gives `export-api#s1` and `search-api#s1` the same
  reranker score (0.5); they stay in BM25's order, which itself came from an
  alphabetical tie.

On a corpus this small, ties decide several rankings; the article lists where.
