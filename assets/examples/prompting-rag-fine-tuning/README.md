# Prompting, RAG, or fine-tuning: diagnostic harness

Companion code for the article
"Prompting, RAG, or Fine-Tuning: Start with the Failure"
(https://rafalpronko.github.io/blog/prompting-rag-fine-tuning/).

Standard library only, no third-party packages. Python 3.11 or newer.

## Run

```text
unzip prompting-rag-fine-tuning.zip
cd prompting-rag-fine-tuning
python3 --version                       # must report 3.11 or newer
python3 demo.py                         # report for k = 2
python3 demo.py --k 3                   # any k from 1 to 9
python3 -m unittest discover -s tests -v
```

If `python3` is older than 3.11, call a newer interpreter explicitly,
for example `python3.11 demo.py`. With uv: `uv run python demo.py`.

## Files

- `harness.py`: fixture validation, BM25 retrieval, output contract,
  grading, diagnostic labels, grouped split. One file on purpose: see its
  docstring.
- `demo.py`: prints the report shown in the article.
- `fixtures/corpus.json`: nine synthetic policy passages for a fictional company.
- `fixtures/cases.json`: eleven questions with references and answer strings.
- `tests/test_harness.py`: unit tests for the harness and the demo CLI.
- `pyproject.toml`, `uv.lock`, `.python-version`: optional project metadata
  for uv; not needed for the plain `python3` commands above.

## What this is not

The answer strings in `fixtures/cases.json` were written by hand to illustrate
failure patterns. They are not outputs of any language model. Nothing here calls
or trains a model, and the label counts are not a measurement of prompting, RAG
or fine-tuning. Retrieval, contract checks, grading, labels and split checks are
computed for real.

The harness does not implement access control and does not check that a citation
supports the answer; see the article for what a production system would add.
