"""Okapi BM25 in the form of Manning, Raghavan & Schuetze (2008, IR book section 11.4.3).

RSV(d, q) = sum over query terms t of
    log(N / df_t) * (k1 + 1) * tf_td / (k1 * ((1 - b) + b * L_d / L_ave) + tf_td)

Natural log. Repeated query terms are counted once (no query-tf factor).
With this idf variant a term present in every chunk gets weight zero.
"""

import math
from collections import Counter

from retrieval_lab.chunking import Chunk
from retrieval_lab.ranking import top
from retrieval_lab.text import tokenize


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {c.chunk_id: Counter(tokenize(c.text)) for c in chunks}
        self.length = {cid: sum(tf.values()) for cid, tf in self.tf.items()}
        self.avg_length = sum(self.length.values()) / len(self.length)
        self.df = Counter(term for tf in self.tf.values() for term in tf)
        self.n = len(chunks)

    def term_score(self, term: str, chunk_id: str) -> float:
        tf = self.tf[chunk_id][term]
        if tf == 0:
            return 0.0
        idf = math.log(self.n / self.df[term])
        norm = self.k1 * ((1 - self.b) + self.b * self.length[chunk_id] / self.avg_length)
        return idf * (self.k1 + 1) * tf / (norm + tf)

    def search(self, query: str, depth: int) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        scores = {cid: sum(self.term_score(t, cid) for t in terms) for cid in self.tf}
        return top(scores, depth)
