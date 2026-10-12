"""Run every question in eval/questions.json and report retrieval and answer checks separately.

Run from the examples directory: python scripts/run_eval.py
The answer check is a crude case-insensitive substring test, not a quality metric.
"""

import json
import pathlib
import platform
import sys

import sentencepiece
import torch
import transformers

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from first_local_rag.corpus import load_chunks  # noqa: E402
from first_local_rag.generator import FlanT5Generator  # noqa: E402
from first_local_rag.pipeline import answer  # noqa: E402
from first_local_rag.retriever import TfidfRetriever  # noqa: E402

QUESTIONS = pathlib.Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def check(case: dict, result) -> dict:
    cited = [c.chunk_id.split("#")[0] for c in result.citations]
    expected_doc, phrase = case["expected_doc"], case["expected_phrase"]
    if expected_doc is None:
        retrieval = "n/a (no supporting passage exists)"
        answer_check = "n/a (expected a refusal or 'unknown')"
    else:
        retrieval = "hit" if expected_doc in cited else "miss"
        found = phrase.lower() in result.text.lower()
        answer_check = "contains expected phrase" if found else "phrase missing"
    return {"retrieval": retrieval, "answer_check": answer_check}


def main() -> None:
    print(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentencepiece": sentencepiece.__version__,
    }))
    generator = FlanT5Generator()
    print(json.dumps({"model": generator.model_id, "revision": generator.revision}))
    retriever = TfidfRetriever(load_chunks())
    for case in json.loads(QUESTIONS.read_text()):
        result = answer(case["question"], retriever, generator)
        row = {
            "question": result.question,
            "status": result.status,
            "answer": result.text,
            "cited": [(c.chunk_id, c.score) for c in result.citations],
            "prompt_tokens": result.prompt_tokens,
            "dropped_for_budget": result.dropped_for_budget,
            **check(case, result),
        }
        print(json.dumps(row))


if __name__ == "__main__":
    main()
