from retrieval_lab.chunking import sentence_chunks, window_chunks

DOCS = {"d": ["s0.", "s1.", "s2.", "s3.", "s4."], "e": ["t0.", "t1."]}


def test_sentence_chunks_one_per_sentence():
    chunks = sentence_chunks(DOCS)
    assert [c.chunk_id for c in chunks] == [
        "d#s0", "d#s1", "d#s2", "d#s3", "d#s4", "e#s0", "e#s1"]
    assert chunks[2].sentence_ids == (2,) and chunks[2].doc_id == "d"


def test_window_chunks_overlap_by_one_sentence():
    chunks = window_chunks(DOCS, size=3, stride=2)
    assert [c.sentence_ids for c in chunks if c.doc_id == "d"] == [(0, 1, 2), (2, 3, 4)]
    assert [c.sentence_ids for c in chunks if c.doc_id == "e"] == [(0, 1)]
    assert chunks[0].text == "s0. s1. s2."


def test_every_adjacent_pair_shares_a_window():
    chunks = window_chunks(DOCS, size=3, stride=2)
    spans = [set(c.sentence_ids) for c in chunks if c.doc_id == "d"]
    for i in range(4):
        assert any({i, i + 1} <= span for span in spans)
