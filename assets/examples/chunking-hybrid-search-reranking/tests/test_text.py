from retrieval_lab.text import tokenize


def test_lowercases_and_drops_stopwords():
    assert tokenize("What is the row limit?") == ["row", "limit"]


def test_keeps_hyphenated_identifiers_intact():
    assert tokenize("Error ERR-4013 means") == ["error", "err-4013", "mean"]


def test_strips_simple_plural_but_not_double_s():
    assert tokenize("rows access") == ["row", "access"]
