"""A lexical TF-IDF retriever built from the standard library only.

This is keyword matching with term weighting, not dense embeddings: a question
that uses different words than the source ("repeatable" vs "idempotent") can
miss the right paragraph entirely.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from first_local_rag.corpus import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from has have how if in "
    "is it its may might of on or should so such that the their then there "
    "this to was were what when which who why will with would".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float              # cosine similarity in [0, 1]
    matched_terms: tuple[str, ...]


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c.text)) for c in chunks]
        document_frequency = Counter(term for c in counts for term in c)
        n = len(chunks)
        self.idf = {
            term: math.log((1 + n) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        self.vectors = [self._weigh(c) for c in counts]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        """TF-IDF weights, L2-normalised; terms unknown to the corpus are dropped."""
        weights = {t: (1 + math.log(f)) * self.idf[t] for t, f in counts.items() if t in self.idf}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        return {t: w / norm for t, w in weights.items()} if norm else {}

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
