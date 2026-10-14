import pytest

from docsys.chunking import content_hash, split_paragraphs
from docsys.query import QueryError, to_match_expression


def test_split_paragraphs_drops_blank_blocks_and_normalises_whitespace():
    body = "First  line\ncontinues.\n\n\n   \nSecond paragraph.\n"
    assert split_paragraphs(body) == ["First line continues.", "Second paragraph."]


def test_content_hash_covers_title_and_body():
    assert content_hash("t", "b") == content_hash("t", "b")
    assert content_hash("t", "b") != content_hash("t2", "b")
    assert content_hash("a", "bc") != content_hash("ab", "c")


def test_plain_terms_become_quoted_and_lowercased():
    assert to_match_expression("Restore  SNAPSHOT") == '"restore" "snapshot"'


@pytest.mark.parametrize("raw", ["OR", "NOT backup", "NEAR(a b)", "title: x"])
def test_fts_operators_are_neutralised_or_rejected(raw):
    try:
        expression = to_match_expression(raw)
    except QueryError:
        return
    assert all(part.startswith('"') and part.endswith('"') for part in expression.split(" "))


@pytest.mark.parametrize(
    "raw",
    ["", "   ", 'restore"', "back*", "a+b", "x" * 41, " ".join(["term"] * 9), "zażółć", None, 42],
)
def test_invalid_queries_raise_query_error(raw):
    with pytest.raises(QueryError):
        to_match_expression(raw)
