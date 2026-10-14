"""Run every (chunking, method) configuration with the same queries, depth and k.

depth = candidates produced by the first stage (and the most a reranker may see)
k     = chunks finally handed to a reader / generator
"""

from retrieval_lab.bm25 import BM25
from retrieval_lab.chunking import Chunk, sentence_chunks, window_chunks
from retrieval_lab.concepts import CONCEPTS, ConceptIndex
from retrieval_lab.corpus import DOCUMENTS, QUERIES
from retrieval_lab.fusion import rrf
from retrieval_lab.labels import EVIDENCE, PURPOSE
from retrieval_lab.metrics import query_metrics
from retrieval_lab.rerank import rerank

K1, B, RRF_C = 1.2, 0.75, 60
CHUNKINGS = {"sentence": sentence_chunks, "window3": lambda d: window_chunks(d, 3, 2)}
METHODS = ["bm25", "concept", "rrf", "bm25+rerank", "rrf+rerank"]


def first_stage(method: str, query: str, bm25: BM25, concept: ConceptIndex, depth: int):
    if method.startswith("bm25"):
        return bm25.search(query, depth)
    if method == "concept":
        return concept.search(query, depth)
    lists = [[cid for cid, _ in index.search(query, depth)] for index in (bm25, concept)]
    return rrf(lists, c=RRF_C, depth=depth)


def run_query(method, qid, by_id, bm25, concept, depth, k) -> dict:
    candidates = first_stage(method, QUERIES[qid], bm25, concept, depth)
    final = candidates
    if method.endswith("+rerank"):
        final = rerank(QUERIES[qid], [by_id[cid] for cid, _ in candidates], CONCEPTS)
    evidence = EVIDENCE[qid]
    ceiling = query_metrics([by_id[cid] for cid, _ in candidates], evidence, depth)
    return {
        "qid": qid,
        "candidates": [{"chunk_id": cid, "score": round(s, 6)} for cid, s in candidates],
        "top_k": [{"chunk_id": cid, "score": round(s, 6)} for cid, s in final[:k]],
        "metrics": query_metrics([by_id[cid] for cid, _ in final], evidence, k),
        "ceiling_full_support": ceiling["full_support"],
    }


def summarise(queries: list[dict]) -> dict:
    n = len(queries)
    mean = lambda key: round(sum(float(q["metrics"][key]) for q in queries) / n, 3)
    keys = ["evidence_recall", "full_support", "reciprocal_rank", "self_contained",
            "parent_hit", "context_sentences"]
    summary = {key: mean(key) for key in keys}
    summary["ceiling_full_support"] = round(sum(q["ceiling_full_support"] for q in queries) / n, 3)
    summary["redundant_evidence_chunks"] = sum(q["metrics"]["redundant_evidence_chunks"] for q in queries)
    return summary


def run_experiment(depth: int = 5, k: int = 3) -> dict:
    runs = []
    for chunking, make in CHUNKINGS.items():
        chunks: list[Chunk] = make(DOCUMENTS)
        by_id = {c.chunk_id: c for c in chunks}
        bm25, concept = BM25(chunks, K1, B), ConceptIndex(chunks, CONCEPTS)
        for method in METHODS:
            queries = [run_query(method, q, by_id, bm25, concept, depth, k) for q in QUERIES]
            runs.append({"chunking": chunking, "method": method, "n_chunks": len(chunks),
                         "summary": summarise(queries), "queries": queries})
    settings = {"depth": depth, "k": k, "bm25_k1": K1, "bm25_b": B, "rrf_c": RRF_C}
    return {"settings": settings, "purpose": PURPOSE, "runs": runs}
