"""Load verified local extracts and split them into paragraph chunks."""

import hashlib
import json
import pathlib
from dataclasses import dataclass

DEFAULT_CORPUS = pathlib.Path(__file__).resolve().parent.parent / "corpus"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # e.g. "rfc9110-9.2.2#p3"
    doc_id: str
    title: str
    url: str
    sha256: str        # hash of the whole unmodified extract file
    paragraph: int     # 1-based paragraph number inside the extract
    text: str          # paragraph with line wrapping collapsed to single spaces


class IntegrityError(Exception):
    """A local extract no longer matches the hash recorded in the manifest."""


def load_manifest(corpus_dir: pathlib.Path = DEFAULT_CORPUS) -> list[dict]:
    return json.loads((corpus_dir / "manifest.json").read_text())["documents"]


def read_verified(corpus_dir: pathlib.Path, record: dict) -> str:
    raw = (corpus_dir / record["file"]).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != record["sha256"]:
        raise IntegrityError(f"{record['file']}: expected {record['sha256']}, got {actual}")
    return raw.decode("utf-8")


def split_paragraphs(text: str) -> list[str]:
    """Blank-line separated paragraphs; the section heading line is skipped."""
    blocks = [block for block in text.split("\n\n") if block.strip()]
    body = blocks[1:]  # blocks[0] is the heading, e.g. "9.2.2.  Idempotent Methods"
    return [" ".join(block.split()) for block in body]


def load_chunks(corpus_dir: pathlib.Path = DEFAULT_CORPUS) -> list[Chunk]:
    chunks = []
    for record in load_manifest(corpus_dir):
        text = read_verified(corpus_dir, record)
        for number, paragraph in enumerate(split_paragraphs(text), start=1):
            chunks.append(Chunk(
                chunk_id=f"{record['id']}#p{number}",
                doc_id=record["id"],
                title=record["title"],
                url=record["url"],
                sha256=record["sha256"],
                paragraph=number,
                text=paragraph,
            ))
    return chunks
