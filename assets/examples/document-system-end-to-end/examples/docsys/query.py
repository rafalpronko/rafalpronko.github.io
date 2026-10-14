"""Restricted query language: plain ASCII terms only.

Binding the query as an SQL parameter prevents SQL injection, but FTS5 still
parses the bound string with its own grammar (OR, NOT, NEAR, column filters,
prefix '*', '+', quotes). We accept only [a-z0-9] terms and emit each one as
a double-quoted FTS5 string, so every term is matched literally and terms are
combined with FTS5's implicit AND.
"""

import re

MAX_TERMS = 8
MAX_TERM_LENGTH = 40
_TERM = re.compile(r"[a-z0-9]+")


class QueryError(ValueError):
    """The user query is outside the supported restricted syntax."""


def parse_terms(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise QueryError("query must be a string")
    terms = raw.lower().split()
    if not terms:
        raise QueryError("query is empty")
    if len(terms) > MAX_TERMS:
        raise QueryError(f"at most {MAX_TERMS} terms are allowed")
    for term in terms:
        if len(term) > MAX_TERM_LENGTH or not _TERM.fullmatch(term):
            raise QueryError(f"unsupported term: {term[:MAX_TERM_LENGTH]!r}")
    return terms


def to_match_expression(raw: object) -> str:
    return " ".join(f'"{term}"' for term in parse_terms(raw))
