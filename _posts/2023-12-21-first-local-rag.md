---
layout: post
lang: en
title: "Your first local RAG: public documents and traceable citations"
date: 2023-12-21 09:00:00 +0100
permalink: /blog/first-local-rag/
author: Rafał Prońko
categories: [rag]
description: "A small retrieval-augmented generation pipeline on a CPU: RFC 9110 extracts, a TF-IDF retriever, flan-t5-small and citations the application controls."
image: /assets/images/first-local-rag.svg
---

# Your first local RAG: public documents and traceable citations

![Diagram of a local RAG pipeline: a question and three hash-verified RFC 9110 extracts go into a TF-IDF retriever; without a match the application refuses, otherwise the top passages go into a 512-token prompt, flan-t5-small generates an answer on the CPU and the application attaches citations from the retrieved passages.]({{ '/assets/images/first-local-rag.svg' | relative_url }})

<time datetime="2023-12-21">21 December 2023</time>

*The date places this post in the series by the technology it uses: every library version and the model revision existed by 21 December 2023. The example itself was executed in October 2026 with modern tooling, as described in "Running it" below.*

Most RAG tutorials start with a vector database, an embedding API and a hosted model. That is three new moving parts before you have seen what the pattern actually does.

I want to start smaller. Everything in this post runs on a laptop CPU: a few paragraphs of a public specification, a retriever written with the Python standard library and a language model with about 77 million parameters (76,961,152 when I counted them).

The model will be bad. That is fine — and useful. A weak model makes the parts of the system that do not depend on the model much easier to see: where the evidence comes from, what the application refuses to answer and who decides what counts as a citation.

## What "retrieval-augmented generation" means here

The term comes from a 2020 paper by Lewis et al. ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)). Its abstract describes a sequence-to-sequence model combined with a dense vector index of Wikipedia accessed by a neural retriever. What I build below is not that architecture. It keeps only the general idea:

1. **Retrieve** a few passages relevant to the question from a corpus you control.
2. **Augment** the prompt with those passages.
3. **Generate** an answer with a language model that reads them.

The retriever here is lexical TF-IDF, not dense embeddings, and nothing is trained. I am using RAG as the name of the pattern, not claiming to reproduce the paper.

## The corpus: three sections of RFC 9110

I need documents that are public, stable and precise enough to check answers by hand. RFC 9110, *HTTP Semantics*, was published as an Internet Standard in June 2022. I take three sections:

- [9.2.2. Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)
- [9.3.1. GET](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3.1)
- [9.3.2. HEAD](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3.2)

That is the whole corpus: 19 paragraphs. Not a PDF, not a crawl, no OCR.

A small script downloads the plain-text RFC and copies whole lines from each section heading to the line before the next heading. The files are **unmodified extracts**, and a `manifest.json` records for each one the section URL, the line range in `rfc9110.txt`, the SHA-256 of the extract and the IETF attribution. The IETF Trust Legal Provisions allow publishing unmodified portions of an RFC as long as they are attributed to the IETF and identify the RFC — which is one more reason not to edit them.

Before using the corpus, the program checks every file against its recorded hash:

```python
def read_verified(corpus_dir: pathlib.Path, record: dict) -> str:
    raw = (corpus_dir / record["file"]).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != record["sha256"]:
        raise IntegrityError(f"{record['file']}: expected {record['sha256']}, got {actual}")
    return raw.decode("utf-8")
```

This sounds pedantic for three text files. It is the cheapest possible version of a question every RAG system has to answer: *is the document I am citing the document I think it is?* If someone "fixes" a `SHOULD NOT` to a `MAY` in the local copy, the application refuses to start instead of quoting a standard that does not exist.

## Chunking: one paragraph per chunk

Each extract is split on blank lines. The heading is dropped and the RFC's line wrapping is collapsed into single spaces in memory. The words stay the same; the files on disk stay identical to the RFC.

A paragraph is a natural unit in an RFC: requirements are usually stated within one. In other documents it might be too small (a table split from its caption) or too large (a 3-page section). Chunking decides what the retriever can possibly return, so it is a design decision, not a preprocessing detail. A later post compares chunking strategies; here I want the simplest one that respects the source structure.

Each chunk carries its identity with it: `rfc9110-9.2.2#p1` is the first paragraph of section 9.2.2, together with the URL and the extract hash.

## Retrieval: TF-IDF from the standard library

The retriever lowercases the text, drops a short list of stop words, weights each term by TF-IDF and ranks paragraphs by cosine similarity with the question:

```python
    def search(self, question: str, k: int = 3, min_score: float = 0.1) -> list[Hit]:
        query = self._weigh(Counter(tokenize(question)))
        hits = []
        for chunk, vector in zip(self.chunks, self.vectors):
            shared = sorted(set(query) & set(vector))
            score = sum(query[t] * vector[t] for t in shared)
            if score >= min_score:
                hits.append(Hit(chunk, round(score, 4), tuple(shared)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:k]
```

Two things matter more than the formula.

First, every hit records the terms it matched. When retrieval goes wrong, I can see *why* a paragraph was picked.

Second, this is keyword matching. A question about "safely repeatable calls" does not find section 9.2.2, because the word "repeatable" never appears in it. One of the tests pins that blind spot down on purpose. Dense embeddings would address some of these misses and introduce their own; for nineteen paragraphs, a transparent baseline is the better place to start.

The `min_score` threshold of 0.1 is a demo parameter chosen for this corpus, not a recommended value.

## The prompt and its token budget

The model's configuration lists `n_positions = 512`, so I treat 512 input tokens as a hard budget. (T5 uses relative position embeddings, so this is a sensible limit rather than an absolute one.) Passages go into the prompt in rank order; a passage that would overflow the budget is dropped as a whole, never cut in the middle:

```python
    used, dropped = [], []
    for hit in hits:
        candidate = render(question, [h.chunk.text for h in used + [hit]])
        if count_tokens(candidate) <= max_tokens:
            used.append(hit)
        else:
            dropped.append(hit)
```

Cutting a passage at an arbitrary token is how "the server MUST NOT send content" becomes "the server MUST". Dropping whole passages loses recall instead of changing meaning — a trade-off I prefer to make explicitly. The tokens are counted with the model's own tokenizer, because word counts and token counts diverge quickly in technical text.

The template asks the model to answer only from the numbered passages, says that the passages are quoted material rather than instructions, and asks for `unknown` when the answer is not there.

## Generation: a small local model

The generator is `google/flan-t5-small` — an instruction-tuned T5 model with 76,961,152 parameters (counted locally), published on the Hugging Face Hub in October 2022 under Apache-2.0. I pin the exact Hugging Face revision, so the weights cannot change under the same name:

```python
MODEL_ID = "google/flan-t5-small"
# Commit "Adding `safetensors` variant of this model (#14)", 2023-10-10.
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
```

Decoding is greedy (`do_sample=False`): no random sampling. In my runs two consecutive calls with the same prompt returned identical text, and a test checks that. I treat it as a useful property for spotting regressions, not as a guarantee across hardware or library versions — and it does not make the answers correct.

I chose this model because it is small, runs on a CPU and needs no account or API key. It is an architecture demo, not a recommendation. A larger model may answer better — I have not measured that here — and it costs more memory and latency and, if hosted, money and data exposure.

## Citations belong to the application

This is the part I care about most:

```python
def answer(question: str, retriever: TfidfRetriever, generator: Generator,
           k: int = 3, min_score: float = 0.1, max_tokens: int = 512) -> Answer:
    hits = retriever.search(question, k=k, min_score=min_score)
    if not hits:
        # Refuse before generation: a model with an empty context still answers.
        return Answer(question, "no_evidence", NO_EVIDENCE, [], 0, [])
    prompt = build_prompt(question, hits, generator.count_tokens, max_tokens)
    if not prompt.used:
        return Answer(question, "no_evidence", NO_EVIDENCE, [], prompt.tokens,
                      [h.chunk.chunk_id for h in prompt.dropped])
    text = generator.generate(prompt.text)
    status = "model_said_unknown" if text.strip().lower() in {"unknown", ""} else "answered"
    return Answer(
        question=question,
        status=status,
        text=text,
        citations=[cite(h) for h in prompt.used],
        prompt_tokens=prompt.tokens,
        dropped_for_budget=[h.chunk.chunk_id for h in prompt.dropped],
    )
```

The model never writes the citations. They are built from the passages that actually went into the prompt: chunk ID, section title, URL, paragraph number, retrieval score, matched terms and the hash of the extract. If the model writes "see RFC 1234, Section 99", that string stays in the answer text and has no effect on the citation list — a unit test checks exactly that.

What a citation here does **not** mean is equally important: it says *which passages the model was given*, not *that every claim in the answer is supported by them*. A small model can read the right paragraph and still answer something else. Checking support claim-by-claim is a separate step that this demo does not do.

The refusal comes before generation. When no paragraph passes the retrieval threshold, the model is not called at all. A language model with an empty context does not say "I don't know" reliably; it generates something plausible.

## Running it

The stack is pinned to packages that existed by the date of this post: Python 3.11, `transformers==4.35.2`, `torch==2.1.2`, `sentencepiece==0.1.99` and `numpy==1.26.2`. The run itself, however, is not from 2023: I executed the example in October 2026 with uv 0.7.8 and CPython 3.11.12 on macOS arm64. To keep transitive dependencies (tokenizers, safetensors, huggingface-hub and others) consistent with that date as well, `pyproject.toml` sets uv's `exclude-newer = "2023-12-21T00:00:00Z"`. uv is the modern execution tool; the code uses only APIs from the 2023 stack.

The complete example — package, tests, corpus with provenance, evaluation questions and `uv.lock` — is available as [first-local-rag.zip]({{ '/assets/examples/first-local-rag.zip' | relative_url }}); the same files are also published unpacked under [/assets/examples/first-local-rag/]({{ '/assets/examples/first-local-rag/README.md' | relative_url }}). After unpacking:

```bash
cd first-local-rag
uv sync
uv run python -m unittest discover -s tests -t . -v
RUN_MODEL=1 uv run python -m unittest discover -s tests -t . -v
uv run python scripts/run_eval.py
uv run python -m first_local_rag.ask "Which request methods are idempotent?"
```

The first command without `RUN_MODEL=1` runs 14 tests and skips the three that need the model. Commands that load the model download `google/flan-t5-small` from the Hugging Face Hub on first use. I ran them with `OMP_NUM_THREADS=1`; that only limits CPU threads and is optional.

## What actually happened

`scripts/run_eval.py` runs six questions and prints one JSON line per question. It reports two checks separately: **retrieval** (did the expected RFC section appear among the cited passages?) and **answer** (does the generated text contain an expected phrase — a crude substring test, not a quality metric). These are the outputs of my run, copied from the log:

| Question | Status | Generated text | Top citation (score) | Retrieval | Answer check |
|---|---|---|---|---|---|
| Which request methods are idempotent? | answered | `PUT, DELETE, and safe request methods` | 9.2.2 ¶1 (0.5138) | hit | contains "PUT, DELETE" |
| Can a proxy automatically retry a non-idempotent request? | answered | `[3]` | 9.2.2 ¶7 (0.6985) | hit | phrase missing |
| How is the HEAD method different from GET? | answered | first sentences of 9.3.2 ¶1, cut off after "testing hypertext links" | 9.3.2 ¶1 (0.2795) | hit | contains "content" |
| Is the response to a GET request cacheable? | answered | `[1]` | 9.3.1 ¶7 (0.3998) | hit | phrase missing |
| Which status code tells the client that the resource does not exist? | model_said_unknown | `unknown` | 9.2.2 ¶7 (0.2805) | n/a | n/a |
| Who won the 2018 FIFA World Cup? | no_evidence | refusal, model not called | — | n/a | n/a |

Retrieval put the right section first for all four answerable questions. The answers are a different story.

**The first answer is good.** It is the phrase from 9.2.2 ¶1 almost word for word.

**Two answers are just a passage number.** For the proxy question, the right paragraph — "A proxy MUST NOT automatically retry non-idempotent requests" — was passage `[1]` in the prompt. The model answered `[3]`, which in that prompt was 9.2.2 ¶4, the paragraph about *clients* that "SHOULD NOT automatically retry" non-idempotent requests. Related, but not the answer. For the caching question it answered `[1]`, which did point at the right paragraph, but a passage number is not an answer a user can read. The numbered-passage format I chose for traceability nudged this small model into echoing labels.

**The HEAD answer is copied, not composed.** The model reproduced the beginning of the retrieved paragraph and stopped at the 48-token output limit in the middle of a sentence. The answer check passes because the word "content" is in it. That says more about the check than about the answer.

**The status-code question is answered with `unknown`.** Retrieval returned three passages that share words like "request" and "client" with the question but contain no status code for a missing resource. The threshold did not stop this question; the model declined. In this run that worked, but it is one observation, not a property I can rely on — the retrieval-level refusal is the only one the application controls.

**The World Cup question never reached the model.** No term overlapped with the corpus, so the pipeline refused before generation.

So with retrieval scoring 4 of 4 on this tiny set, the end-to-end result is one good answer, one copied fragment and two answers that are only labels. If I had measured only retrieval, the system would look finished. If I had measured only the answer substring, the HEAD truncation would count as a success. This is why the two are reported separately.

The test suite (`RUN_MODEL=1`) passed 17 tests, including three that load the real model. They assert what I can honestly state: an answer is generated, the first citation is 9.2.2 ¶1, the prompt stays within 512 tokens, two consecutive greedy runs gave the same text, and the unknown question never reaches the model. None of them asserts that an answer is correct.

## What this demo does not handle

**Prompt injection and trust.** RFC text is trusted input here. In a real corpus, a document can contain "ignore previous instructions". Labelling passages as quoted material in the prompt is not a defence a small model reliably respects, and nor is it for large ones. The test I have only checks that injected text stays inside the context block; it says nothing about how the model reacts. The real controls are outside the model: deciding which sources are allowed into the corpus, keeping their provenance, and not giving the model's output any authority — no tool calls, no writes — that a document author should not have.

**Staleness.** The manifest records when the RFC was downloaded and the hash of what was downloaded. That tells me *what* I am citing, not whether it is still current. RFC 9110 is stable, but in a corpus of internal docs or product pages, updating and deleting outdated chunks is a pipeline problem of its own.

**Costs and privacy.** Everything runs locally: no API key, no per-token billing and no question leaves the machine (apart from the one-time model download). The price is quality and the hardware you have. With a hosted model, every retrieved passage is sent to the provider, so the corpus decides what data you are disclosing.

**Evaluation.** Six questions are a smoke test, not a benchmark. They do, however, show the shape of the evaluation I want later: retrieval and answers are scored separately, and questions without an answer in the corpus are part of the set from day one.

## Where to go from here

The useful part of this pipeline is not the model. It is the boundary around it: verified sources, explicit chunk identities, a token budget that does not cut sentences, a refusal path that does not depend on the model, and citations the application owns.

The next steps are better retrieval (chunking, hybrid search, reranking — measured, not assumed) and a proper test set that tells retrieval failures apart from generation failures.

### Sources

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html) — IETF, June 2022; sections 9.2.2, 9.3.1 and 9.3.2 used as the corpus. Copyright (c) 2022 IETF Trust and the persons identified as the document authors.
- [IETF Trust Legal Provisions](https://trustee.ietf.org/assets/trust-legal-provisions/) — TLP 5.0, Section 3.c.iii on reproducing unmodified portions.
- [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — abstract; origin of the term.
- [google/flan-t5-small model card](https://huggingface.co/google/flan-t5-small) — Apache-2.0; T5Tokenizer and T5ForConditionalGeneration usage on CPU.
- [Chung et al., Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — the Flan-T5 paper, October 2022.
- [transformers 4.35.2](https://pypi.org/project/transformers/4.35.2/) (15 November 2023), [torch 2.1.2](https://pypi.org/project/torch/2.1.2/) (14 December 2023), [sentencepiece 0.1.99](https://pypi.org/project/sentencepiece/0.1.99/), [numpy 1.26.2](https://pypi.org/project/numpy/1.26.2/).
- [uv settings: exclude-newer](https://docs.astral.sh/uv/reference/settings/#exclude-newer) — used for the modern, date-bounded installation.
