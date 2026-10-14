# rag-evaluation-three-layers

Companion code for the article “Evaluate retrieval, answer quality, and
evidence support separately”. A deterministic, label-driven harness that
scores retrieval, answer quality and evidence support as separate layers.

There is no retriever, no model and no automatic entailment here. Every
number is aggregated from hand-written labels in `labels.json`: chunk IDs,
atomic claim IDs and per-claim judgements. The cases are synthetic; they
show how the metric definitions behave, not how any real system performs.

## Layout

Download the files into this layout (the tests import `rag_eval` from the
project root and read `labels.json` from there):

```
rag-evaluation-three-layers/
├── README.md
├── pyproject.toml
├── rag_eval.py
├── labels.json
└── tests/
    └── test_rag_eval.py
```

## Run

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

`rag_eval.py` needs only the Python 3.12 standard library. `pyproject.toml`
pins only pytest (9.1.1), for the tests; uv is just a convenient way to
create the virtual environment and is not pinned.

The recorded reproduction was run on 13 October 2026 with CPython 3.12.11,
uv 0.7.8 and pytest 9.1.1: 76 tests passed, and the report matched the one
printed in the article. The article's date is an editorial label; it does
not mean the code was run with these tools at that date.

## Label contract

`parse` rejects labels that would make the numbers meaningless:

- duplicate IDs among cases, claims, retrieved or provided chunks,
  citations and evidence; chunk IDs that are not in the corpus;
- `correct`, `relevant`, `abstained` and `answerable` that are not JSON
  booleans (strings such as `"false"`, numbers and `null` are errors);
- an `answerable` flag that disagrees with the evidence sets; an
  unanswerable case with relevant chunks or required claims; an answerable
  case without required claims; an empty evidence set, or one with chunks
  not labelled relevant;
- an ordinary (non-gold) case without a `retrieved` field, or with
  `retrieved` that is `null` or not a JSON list (an empty list is valid and
  means retrieval returned nothing; only gold cases bypass retrieval);
- `provided` chunks outside the top k of `retrieved` (`retrieved` may be
  logged beyond k, but the model can only receive chunks from the top k);
- `supported_by` pointing outside the provided context; an abstention with
  claims; a claim that uses a required claim ID but is labelled incorrect;
- a non-positive `k`;
- a gold-context case (`gold_of`) whose base is missing, is itself a gold
  case or is unanswerable; that sets any of `question`, `relevant`,
  `evidence_sets`, `required_claims` or `answerable` (they are inherited
  unchanged); that has a `retrieved` list; or whose provided context does
  not contain a full evidence set.

A citation to an unknown chunk is not a label error: it is a model error and
is counted as an invalid citation.

## Limitations

- Claim decomposition and support labels are human judgements; the harness
  only aggregates them.
- `supported_by` lists chunks that *each* support a claim on their own. A
  claim that needs two chunks together cannot be expressed, so joint
  multi-chunk support is not modelled.
- All claims weigh the same, and labels are valid only for one snapshot of
  the corpus and reference.
- The gold-context comparison (`isolate`) is a hint, not proof. A gold pass
  after an insufficient actual context suggests an upstream cause. A gold
  pass when the actual context was already sufficient is inconclusive:
  sufficiency only means a labelled evidence set was present, and the two
  contexts can differ in distractors, order or evidence path, so context
  composition or order, generation and run-to-run variability all remain
  possible causes. `isolate` does not compare the two contexts and gives
  the same reading even when they are identical. A gold failure does not
  rule out upstream failures at the same time.
- The cases are synthetic. Tags and `isolate` readings describe what the
  labels show for one sample, not a verdict on any real retriever or model.
