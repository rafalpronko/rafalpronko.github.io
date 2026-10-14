"""Two chunking policies over the same sentence-split documents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    sentence_ids: tuple[int, ...]
    text: str


def sentence_chunks(docs: dict[str, list[str]]) -> list[Chunk]:
    return [
        Chunk(f"{doc_id}#s{i}", doc_id, (i,), sentence)
        for doc_id, sentences in docs.items()
        for i, sentence in enumerate(sentences)
    ]


def window_chunks(docs: dict[str, list[str]], size: int = 3, stride: int = 2) -> list[Chunk]:
    """Sliding windows; size 3 / stride 2 puts every adjacent sentence pair in one window."""
    chunks = []
    for doc_id, sentences in docs.items():
        for start in range(0, len(sentences), stride):
            ids = tuple(range(start, min(start + size, len(sentences))))
            text = " ".join(sentences[i] for i in ids)
            chunks.append(Chunk(f"{doc_id}#w{ids[0]}-{ids[-1]}", doc_id, ids, text))
            if start + size >= len(sentences):
                break
    return chunks
