"""A toy 'semantic' channel: a hand-written synonym -> concept table plus cosine.

This is NOT a learned dense retriever. It only shows the mechanism of matching
different words with the same meaning. Words missing from the table, including
identifiers such as ERR-4013, are invisible to it. The table was written by the
author together with the synthetic corpus, so it is biased towards these queries.
"""

import math
from collections import Counter

from retrieval_lab.chunking import Chunk
from retrieval_lab.ranking import top
from retrieval_lab.text import tokenize

CONCEPTS = {
    "refund": "REFUND", "reimbursement": "REFUND", "repayment": "REFUND",
    "duplicate": "DUPLICATE", "double": "DUPLICATE",
    "charge": "BILLING", "billing": "BILLING", "billed": "BILLING",
    "invoice": "INVOICE", "invoiced": "INVOICE",
    "error": "ERROR", "token": "TOKEN", "credential": "TOKEN",
    "expire": "EXPIRY", "expired": "EXPIRY", "expiry": "EXPIRY",
    "limit": "LIMIT", "maximum": "LIMIT", "row": "ROW", "record": "ROW",
    "export": "EXPORT", "download": "EXPORT", "endpoint": "ENDPOINT",
    "price": "PRICE", "cost": "PRICE", "seat": "SEAT", "license": "SEAT",
    "plan": "PLAN", "tier": "PLAN", "team": "TEAM", "group": "TEAM",
    "delete": "DELETE", "deleted": "DELETE", "deletion": "DELETE", "removed": "DELETE",
    "inactive": "INACTIVE", "idle": "INACTIVE", "project": "PROJECT",
    "restore": "RESTORE", "restored": "RESTORE", "recover": "RESTORE", "recovered": "RESTORE",
    "kept": "RETAIN", "retain": "RETAIN", "retained": "RETAIN", "stored": "RETAIN",
    "audit": "AUDIT", "log": "LOG", "trail": "LOG",
    "search": "SEARCH", "query": "SEARCH", "card": "CARD", "money": "PAYMENT",
    "payment": "PAYMENT", "return": "RETURN", "back": "RETURN",
}


def concepts_of(tokens: list[str], table: dict[str, str]) -> Counter:
    return Counter(table[t] for t in tokens if t in table)


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class ConceptIndex:
    def __init__(self, chunks: list[Chunk], table: dict[str, str] = CONCEPTS):
        self.table = table
        self.vectors = {c.chunk_id: concepts_of(tokenize(c.text), table) for c in chunks}

    def search(self, query: str, depth: int) -> list[tuple[str, float]]:
        q = concepts_of(tokenize(query), self.table)
        scores = {cid: cosine(q, vec) for cid, vec in self.vectors.items()}
        return top(scores, depth)
