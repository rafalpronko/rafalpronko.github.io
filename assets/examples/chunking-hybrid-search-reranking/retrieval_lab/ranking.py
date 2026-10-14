"""Shared deterministic ordering: score descending, then chunk id ascending."""


def top(scores: dict[str, float], depth: int) -> list[tuple[str, float]]:
    positive = [(cid, s) for cid, s in scores.items() if s > 0]
    return sorted(positive, key=lambda item: (-item[1], item[0]))[:depth]
