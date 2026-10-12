"""Command line: python -m first_local_rag.ask "Which request methods are idempotent?" """

import json
import sys

from first_local_rag.corpus import load_chunks
from first_local_rag.generator import FlanT5Generator
from first_local_rag.pipeline import answer
from first_local_rag.retriever import TfidfRetriever


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print('usage: python -m first_local_rag.ask "your question"', file=sys.stderr)
        return 2
    retriever = TfidfRetriever(load_chunks())
    result = answer(argv[0], retriever, FlanT5Generator())
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
