"""Retrieve -> prompt -> generate, with citations owned by the application."""

from dataclasses import asdict, dataclass
from typing import Protocol

from first_local_rag.prompt import build_prompt
from first_local_rag.retriever import Hit, TfidfRetriever

NO_EVIDENCE = "No passage in the local corpus matched this question."


class Generator(Protocol):
    def count_tokens(self, text: str) -> int: ...
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Citation:
    """A retrieved source passage. It shows what the model was given,
    not proof that every word of the generated answer is supported by it."""
    chunk_id: str
    title: str
    url: str
    extract_sha256: str
    paragraph: int
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    question: str
    status: str            # "answered" | "model_said_unknown" | "no_evidence"
    text: str
    citations: list[Citation]
    prompt_tokens: int
    dropped_for_budget: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def cite(hit: Hit) -> Citation:
    c = hit.chunk
    return Citation(c.chunk_id, c.title, c.url, c.sha256, c.paragraph,
                    hit.score, hit.matched_terms)


def answer(question: str, retriever: TfidfRetriever, generator: Generator,
           k: int = 3, min_score: float = 0.1, max_tokens: int = 512) -> Answer:
    hits = retriever.search(question, k=k, min_score=min_score)
    if not hits:
        # Refuse before generation: a model with an empty context still answers.
        return Answer(question, "no_evidence", NO_EVIDENCE, [], 0, [])
    prompt = build_prompt(question, hits, generator.count_tokens, max_tokens)
    if not prompt.used:
        return Answer(question, "no_evidence", NO_EVIDENCE, [], prompt.tokens,
                      [h.chunk.chunk_id for h in prompt.dropped])
    text = generator.generate(prompt.text)
    status = "model_said_unknown" if text.strip().lower() in {"unknown", ""} else "answered"
    return Answer(
        question=question,
        status=status,
        text=text,
        citations=[cite(h) for h in prompt.used],
        prompt_tokens=prompt.tokens,
        dropped_for_budget=[h.chunk.chunk_id for h in prompt.dropped],
    )
