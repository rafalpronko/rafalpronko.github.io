import math

from retrieval_lab.bm25 import BM25
from retrieval_lab.chunking import sentence_chunks


def index(docs):
    return BM25(sentence_chunks(docs), k1=1.2, b=0.75)


def test_matches_the_explicit_formula():
    bm = index({"a": ["apple apple pie"], "b": ["banana pie"]})
    tf, ld, lave = 2, 3, 2.5
    expected = math.log(2 / 1) * (2.2 * tf) / (1.2 * (0.25 + 0.75 * ld / lave) + tf)
    assert bm.search("apple", depth=5) == [("a#s0", expected)]


def test_term_in_every_chunk_has_zero_weight_and_is_not_returned():
    bm = index({"a": ["apple pie"], "b": ["banana pie"]})
    assert bm.search("pie", depth=5) == []


def test_term_frequency_saturates_at_equal_length():
    docs = {"a": ["kiwi pad pad pad"], "b": ["kiwi kiwi pad pad"],
            "c": ["kiwi kiwi kiwi kiwi"], "d": ["pad"]}
    bm = index(docs)
    s = dict(bm.search("kiwi", depth=5))
    assert s["a#s0"] < s["b#s0"] < s["c#s0"]
    assert (s["c#s0"] - s["b#s0"]) / 2 < s["b#s0"] - s["a#s0"]
    assert s["c#s0"] < math.log(4 / 3) * 2.2  # upper bound idf * (k1 + 1)


def test_length_normalisation_prefers_shorter_chunk_at_equal_tf():
    bm = index({"a": ["kiwi"], "b": ["kiwi grape melon lemon"], "c": ["x"]})
    ranked = bm.search("kiwi", depth=5)
    assert [cid for cid, _ in ranked] == ["a#s0", "b#s0"]


def test_depth_truncates_and_ties_break_by_chunk_id():
    bm = index({"b": ["kiwi"], "a": ["kiwi"], "c": ["x"]})
    assert [cid for cid, _ in bm.search("kiwi", depth=1)] == ["a#s0"]
