"""Tokenisation shared by every channel: lowercase, identifiers kept whole."""

import re

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are at be by can do doe does for from get how i if in is it its "
    "of on or per the this to until what when where which with".split()
)


def normalise(token: str) -> str:
    """Strip a plural 's' from longer words; keep 'access', 'class' etc."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = (normalise(t) for t in TOKEN.findall(text.lower()))
    return [t for t in tokens if t not in STOPWORDS]
