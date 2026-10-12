# first-local-rag

A deliberately tiny retrieval-augmented generation (RAG) demo that runs on a CPU:

1. three unmodified extracts from RFC 9110 (`corpus/`, see `corpus/PROVENANCE.md`),
2. a TF-IDF retriever written with the Python standard library,
3. a prompt assembled within a 512-token input budget,
4. local generation with `google/flan-t5-small` pinned to revision
   `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`,
5. citations built by the application from the retrieved passages.

It is an architecture demo, not a recommendation of this model or this
retriever. The model is small and gets answers wrong; the citations show which
passages the model was given, not that every generated word is supported.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv can install it).
- Network access when installing dependencies and on the first model run:
  `google/flan-t5-small` is downloaded from the Hugging Face Hub into the
  default Hugging Face cache (`HF_HOME` changes the location).

Pinned versions: `transformers==4.35.2`, `torch==2.1.2`,
`sentencepiece==0.1.99`, `numpy==1.26.2`. `pyproject.toml` sets
`exclude-newer = "2023-12-21T00:00:00Z"`, so uv resolves transitive packages
as they were available on that date. Binary wheels may not exist for every
platform; this was executed on macOS arm64 only.

## Run

From the directory that contains this README (`first-local-rag/` after unpacking
the archive):

```bash
uv sync
uv run python -m unittest discover -s tests -t . -v               # 14 tests run, 3 model tests skipped
RUN_MODEL=1 uv run python -m unittest discover -s tests -t . -v   # all 17 tests, loads the model
uv run python scripts/run_eval.py
uv run python -m first_local_rag.ask "Which request methods are idempotent?"
```

The author's run used `OMP_NUM_THREADS=1` (optional; it only limits CPU
threads). Generated answers can be wrong; `run_eval.py` reports retrieval and
answer checks separately so you can see where a failure comes from.

`scripts/build_corpus.py` re-downloads RFC 9110 and rebuilds `corpus/`. The
extract hashes should match `manifest.json` as long as the RFC text served by
rfc-editor.org does not change.

## Layout

| Path | Purpose |
|---|---|
| `first_local_rag/corpus.py` | Verify extract hashes, split into paragraph chunks |
| `first_local_rag/retriever.py` | Lexical TF-IDF retrieval with a minimum score |
| `first_local_rag/prompt.py` | Prompt template and token budget (whole passages only) |
| `first_local_rag/generator.py` | flan-t5-small on CPU, greedy decoding |
| `first_local_rag/pipeline.py` | Refusal without evidence, generation, citations |
| `first_local_rag/ask.py` | Command line entry point, prints JSON |
| `scripts/run_eval.py` | Retrieval check and answer check, reported separately |
| `eval/questions.json` | Six questions, including two without an answer in the corpus |

## Third-party material

Corpus extracts: IETF Trust, see
`corpus/PROVENANCE.md`. Model: `google/flan-t5-small`, Apache-2.0, downloaded
separately and not included here.
