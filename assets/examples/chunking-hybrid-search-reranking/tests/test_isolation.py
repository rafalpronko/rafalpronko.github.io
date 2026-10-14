import ast
from pathlib import Path

from retrieval_lab.chunking import sentence_chunks, window_chunks
from retrieval_lab.corpus import DOCUMENTS, QUERIES
from retrieval_lab.labels import EVIDENCE

PKG = Path(__file__).resolve().parents[1] / "retrieval_lab"
RETRIEVAL_MODULES = ["text", "chunking", "ranking", "bm25", "concepts", "fusion", "rerank", "corpus"]


def imported_names(module):
    tree = ast.parse((PKG / f"{module}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_retrieval_code_never_imports_labels():
    for module in RETRIEVAL_MODULES:
        assert not any("labels" in name for name in imported_names(module)), module


def test_every_query_has_evidence_pointing_at_real_sentences():
    assert set(QUERIES) == set(EVIDENCE)
    for spans in EVIDENCE.values():
        for doc, sid in spans:
            assert 0 <= sid < len(DOCUMENTS[doc])


def test_both_chunkings_cover_every_sentence_exactly_or_with_overlap():
    for chunks in (sentence_chunks(DOCUMENTS), window_chunks(DOCUMENTS, 3, 2)):
        covered = {(c.doc_id, s) for c in chunks for s in c.sentence_ids}
        assert covered == {(d, i) for d, ss in DOCUMENTS.items() for i in range(len(ss))}
