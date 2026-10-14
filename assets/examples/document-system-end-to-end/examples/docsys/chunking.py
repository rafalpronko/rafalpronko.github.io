"""Paragraph chunking and content fingerprinting."""

import hashlib
import re

_BLANK_LINE = re.compile(r"\n\s*\n")


def split_paragraphs(body: str) -> list[str]:
    """Split on blank lines and collapse internal whitespace."""
    paragraphs = (" ".join(block.split()) for block in _BLANK_LINE.split(body))
    return [paragraph for paragraph in paragraphs if paragraph]


def content_hash(title: str, body: str) -> str:
    """Length-prefixed so that ("a", "bc") and ("ab", "c") differ."""
    digest = hashlib.sha256()
    for part in (title, body):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
