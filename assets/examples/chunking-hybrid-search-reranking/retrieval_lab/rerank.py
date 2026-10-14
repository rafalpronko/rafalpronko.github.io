"""A rule-based query-passage scorer used as a toy second stage.

Like a cross-encoder it looks at the query and one candidate passage together and
can only reorder the candidates it is given. Unlike a cross-encoder it is three
hand-written rules, not a trained model:

  coverage   share of distinct query terms found in the passage, either literally
             or through the same concept in the synonym table
  +1         the passage contains the query's exact identifier (e.g. ERR-4013)
  -1         the passage contains a different identifier and not the query's one

Ties keep the first-stage order (Python's sort is stable).
"""

from retrieval_lab.chunking import Chunk
from retrieval_lab.text import tokenize


def is_identifier(token: str) -> bool:
    return "-" in token and any(ch.isdigit() for ch in token)


def score(query_terms: set[str], passage: Chunk, table: dict[str, str]) -> float:
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


def rerank(query: str, candidates: list[Chunk], table: dict[str, str]) -> list[tuple[str, float]]:
    terms = set(tokenize(query))
    scored = [(c.chunk_id, score(terms, c, table)) for c in candidates]
    return sorted(scored, key=lambda item: -item[1])
