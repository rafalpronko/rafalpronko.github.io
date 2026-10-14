"""Synthetic, offline illustration. Not GraphRAG, OCR, a VLM or a benchmark."""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    version: int
    page: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in page points
    text: str
    readers: frozenset[str]


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    evidence: Evidence | None


Cell = tuple[str, str, Evidence]  # row, column, evidence


class UnknownEntity(LookupError):
    """No entity in the index has this name."""


class AmbiguousEntity(LookupError):
    """Several entities share this name; refuse instead of guessing."""


class UnsupportedEdge(ValueError):
    """An edge has no usable evidence, or its quote misses an endpoint."""


def norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


LABELS = {
    "pump:P-200": "P-200",
    "pump:P-300": "P-300",
    "alloy:KA-7": "KA-7",
    "alloy:KB-2": "KB-2",
    "hold:QH-22": "QH-22",
    "org:kestrel-alloys": "Kestrel Alloys",
    "org:kestrel-logistics": "Kestrel Logistics",
}
ALIASES: dict[str, set[str]] = {}
for _id, _label in LABELS.items():
    ALIASES.setdefault(norm(_label), set()).add(_id)
ALIASES[norm("Kestrel")] = {"org:kestrel-alloys", "org:kestrel-logistics"}


def resolve(name: str, aliases: dict[str, set[str]]) -> str:
    ids = aliases.get(norm(name), set())
    if not ids:
        raise UnknownEntity(name)
    if len(ids) > 1:
        # Teaching example: production code must permission-filter these
        # candidate identifiers before showing them to a caller.
        raise AmbiguousEntity(f"{name!r} could be {sorted(ids)}")
    return next(iter(ids))


def ev(
    doc: str,
    version: int,
    page: int,
    bbox: tuple[float, float, float, float],
    text: str,
    readers: Iterable[str],
) -> Evidence:
    return Evidence(doc, version, page, bbox, text, frozenset(readers))


BOM = ("bom-pumps", 5, 4)
BOM_READERS = {"engineering", "quality"}
EDGES = (
    Edge(
        "pump:P-200",
        "uses",
        "alloy:KA-7",
        ev(
            *BOM,
            (72, 310, 290, 324),
            "P-200 impeller: alloy KA-7",
            BOM_READERS,
        ),
    ),
    Edge(
        "pump:P-300",
        "uses",
        "alloy:KB-2",
        ev(
            *BOM,
            (72, 328, 290, 342),
            "P-300 impeller: alloy KB-2",
            BOM_READERS,
        ),
    ),
    # The hold is restricted to heat lots 31-34. Nothing here records the
    # heat lot of the P-200 impeller, so a path through this edge makes
    # QH-22 a candidate hold only; its applicability stays unknown.
    Edge(
        "alloy:KA-7",
        "subject_of",
        "hold:QH-22",
        ev(
            "quality-hold",
            2,
            1,
            (56, 120, 520, 150),
            "Quality hold QH-22 applies to alloy KA-7, heat lots 31-34.",
            {"quality"},
        ),
    ),
    Edge(
        "alloy:KA-7",
        "supplied_by",
        "org:kestrel-alloys",
        ev(
            "supplier-register",
            3,
            2,
            (60, 400, 480, 414),
            "KA-7 bar stock is supplied by Kestrel Alloys.",
            {"procurement", "quality"},
        ),
    ),
)


def mentions(text: str, label: str) -> bool:
    pattern = rf"(?<![\w-]){re.escape(label)}(?![\w-])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def evidence_of(edge: Edge) -> Evidence:
    if edge.evidence is None:
        raise UnsupportedEdge(
            f"{edge.source} -{edge.relation}-> {edge.target}: no evidence"
        )
    return edge.evidence


def validate(edge: Edge, labels: dict[str, str]) -> Edge:
    """Weak check: the quote names both ends, not that it states the edge."""
    e = evidence_of(edge)
    if e.version < 1 or e.page < 1 or len(e.bbox) != 4:
        raise UnsupportedEdge(f"{e.doc_id}: no version, page or box")
    for end in (edge.source, edge.target):
        if not mentions(e.text, labels[end]):
            raise UnsupportedEdge(f"quote in {e.doc_id} does not name {end}")
    return edge


def find_path(
    edges: Iterable[Edge], start: str, goal: str, relations: set[str]
) -> list[Edge] | None:
    """Breadth-first search; None means no path in this index, not absence."""
    edge_list = tuple(edges)
    queue: deque[tuple[str, list[Edge]]] = deque([(start, [])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for e in edge_list:
            if (
                e.source == node
                and e.relation in relations
                and e.target not in seen
            ):
                seen.add(e.target)
                queue.append((e.target, path + [e]))
    return None


def readers_of(path: Sequence[Edge]) -> frozenset[str]:
    """Intersect evidence readers. A data operation, not caller auth."""
    if not path:
        raise ValueError("no evidence, no readers")
    return frozenset.intersection(*(evidence_of(e).readers for e in path))


def drop_document(edges: Iterable[Edge], doc_id: str) -> tuple[Edge, ...]:
    return tuple(
        e
        for e in edges
        if e.evidence is not None and e.evidence.doc_id != doc_id
    )


ROW_HEADER = "Pump"
COLUMNS = ("Max pressure (bar)", "Impeller alloy")


def table(values: dict[str, tuple[str, str]]) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for r, (row, row_values) in enumerate(values.items()):
        for c, (column, value) in enumerate(zip(COLUMNS, row_values)):
            box = (300 + 120 * c, 310 + 18 * r, 410 + 120 * c, 324 + 18 * r)
            cells.append((row, column, ev(*BOM, box, value, BOM_READERS)))
    return tuple(cells)


TABLE = table({"P-200": ("16", "KA-7"), "P-300": ("10", "KB-2")})
SWAPPED_TABLE = table({"P-200": ("10", "KA-7"), "P-300": ("16", "KB-2")})


def cell(cells: Iterable[Cell], row: str, column: str) -> Evidence:
    for r, c, evidence in cells:
        if (r, c) == (row, column):
            return evidence
    raise LookupError(f"no cell ({row!r}, {column!r}) in table")


def bag_of_tokens(cells: Iterable[Cell]) -> Counter[str]:
    """Unordered token counts: which value sat in which cell is discarded."""
    cell_list = tuple(cells)
    rows = [r for r, _, _ in cell_list]
    columns = [c for _, c, _ in cell_list]
    values = [e.text for _, _, e in cell_list]
    words = " ".join([ROW_HEADER, *set(rows), *set(columns), *values])
    return Counter(words.split())


def cite(e: Evidence) -> str:
    return f"{e.doc_id} v{e.version} p.{e.page} bbox={e.bbox}"


if __name__ == "__main__":
    for e in EDGES:
        validate(e, LABELS)
    hops = {"uses", "subject_of"}
    start = resolve("p200", ALIASES)
    # A path makes QH-22 a candidate hold for P-200, not a confirmed one.
    path = find_path(EDGES, start, "hold:QH-22", hops)
    if path is None:
        raise SystemExit("P-200 -> QH-22: no evidence in this index")
    for e in path:
        print(
            f"{e.source} -{e.relation}-> {e.target} | "
            f"{cite(evidence_of(e))}"
        )
    print("candidate path visible to:", sorted(readers_of(path)))
    print("QH-22 applicability: UNKNOWN; P-200 heat lot is not indexed")
    # None: no evidence in this curated index, not proof of no hold.
    print(
        "P-300 -> QH-22:",
        find_path(EDGES, "pump:P-300", "hold:QH-22", hops),
    )
    print("None means abstention: no indexed evidence, not proof of no hold")
    try:
        resolve("Kestrel", ALIASES)
    except AmbiguousEntity as error:
        print("ambiguous:", error)
    found = cell(TABLE, "P-200", "Max pressure (bar)")
    print("P-200 max pressure:", found.text, "bar |", cite(found))
    print(
        "bags equal for different tables:",
        bag_of_tokens(TABLE) == bag_of_tokens(SWAPPED_TABLE),
    )
