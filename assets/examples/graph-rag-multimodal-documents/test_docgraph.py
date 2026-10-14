import unittest
from collections.abc import Iterable

from docgraph import (
    ALIASES,
    EDGES,
    LABELS,
    SWAPPED_TABLE,
    TABLE,
    AmbiguousEntity,
    Cell,
    Edge,
    UnknownEntity,
    UnsupportedEdge,
    bag_of_tokens,
    cell,
    drop_document,
    evidence_of,
    find_path,
    readers_of,
    resolve,
    validate,
)

HOPS = {"uses", "subject_of"}
PRESSURE = "Max pressure (bar)"


class EntityResolution(unittest.TestCase):
    def test_spelling_variants_resolve_to_one_entity(self) -> None:
        self.assertEqual(resolve("KA7", ALIASES), "alloy:KA-7")
        self.assertEqual(resolve("ka-7", ALIASES), "alloy:KA-7")

    def test_ambiguous_name_is_refused(self) -> None:
        with self.assertRaises(AmbiguousEntity):
            resolve("Kestrel", ALIASES)

    def test_unknown_name_is_refused(self) -> None:
        with self.assertRaises(UnknownEntity):
            resolve("P-999", ALIASES)


class EdgeValidation(unittest.TestCase):
    def test_all_curated_edges_are_supported(self) -> None:
        for edge in EDGES:
            self.assertIs(validate(edge, LABELS), edge)

    def test_edge_without_evidence_is_rejected(self) -> None:
        edge = Edge("pump:P-300", "uses", "alloy:KB-2", None)
        with self.assertRaises(UnsupportedEdge):
            validate(edge, LABELS)

    def test_quote_must_name_both_ends(self) -> None:
        hold_edge = next(e for e in EDGES if e.relation == "subject_of")
        hold = evidence_of(hold_edge)
        wrong = Edge("alloy:KB-2", "subject_of", "hold:QH-22", hold)
        with self.assertRaises(UnsupportedEdge):
            validate(wrong, LABELS)


class Traversal(unittest.TestCase):
    def test_candidate_path_keeps_cross_document_provenance(self) -> None:
        path = find_path(EDGES, "pump:P-200", "hold:QH-22", HOPS)
        if path is None:
            self.fail("expected the curated candidate path")
        self.assertEqual([e.relation for e in path], ["uses", "subject_of"])
        self.assertEqual(
            [evidence_of(e).doc_id for e in path],
            ["bom-pumps", "quality-hold"],
        )
        for edge in path:
            evidence = evidence_of(edge)
            self.assertGreater(evidence.version, 0)
            self.assertGreater(evidence.page, 0)
            self.assertEqual(len(evidence.bbox), 4)

    def test_relation_filter_blocks_unrelated_edges(self) -> None:
        self.assertIsNone(
            find_path(EDGES, "pump:P-200", "org:kestrel-alloys", {"uses"})
        )
        path = find_path(
            EDGES, "pump:P-200", "org:kestrel-alloys", {"uses", "supplied_by"}
        )
        if path is None:
            self.fail("expected the curated supplier path")
        self.assertEqual([e.relation for e in path], ["uses", "supplied_by"])

    def test_no_indexed_path_returns_none_for_abstention(self) -> None:
        self.assertIsNone(find_path(EDGES, "pump:P-300", "hold:QH-22", HOPS))

    def test_candidate_path_intersects_evidence_readers(self) -> None:
        path = find_path(EDGES, "pump:P-200", "hold:QH-22", HOPS)
        if path is None:
            self.fail("expected the curated candidate path")
        self.assertEqual(readers_of(path), frozenset({"quality"}))

    def test_readers_of_refuses_an_edge_without_evidence(self) -> None:
        unsupported = Edge("alloy:KA-7", "subject_of", "hold:QH-22", None)
        path = [EDGES[0], unsupported]
        with self.assertRaises(UnsupportedEdge):
            readers_of(path)

    def test_readers_of_refuses_an_empty_path(self) -> None:
        with self.assertRaises(ValueError):
            readers_of([])

    def test_deleting_a_source_removes_paths_built_on_it(self) -> None:
        remaining = drop_document(EDGES, "quality-hold")
        self.assertEqual(len(remaining), len(EDGES) - 1)
        self.assertIsNone(
            find_path(remaining, "pump:P-200", "hold:QH-22", HOPS)
        )


class TableEvidence(unittest.TestCase):
    def test_cell_lookup_returns_value_with_coordinates(self) -> None:
        found = cell(TABLE, "P-200", PRESSURE)
        self.assertEqual(found.text, "16")
        self.assertEqual(
            (found.doc_id, found.version, found.page), ("bom-pumps", 5, 4)
        )
        self.assertEqual(found.bbox, (300, 310, 410, 324))

    def test_missing_cell_is_an_error_not_a_guess(self) -> None:
        with self.assertRaises(LookupError):
            cell(TABLE, "P-999", PRESSURE)

    def test_bag_cannot_tell_two_different_tables_apart(self) -> None:
        self.assertEqual(bag_of_tokens(TABLE), bag_of_tokens(SWAPPED_TABLE))
        self.assertEqual(bag_of_tokens(TABLE)["Pump"], 1)
        self.assertNotEqual(
            cell(TABLE, "P-200", PRESSURE).text,
            cell(SWAPPED_TABLE, "P-200", PRESSURE).text,
        )

    def test_ordered_values_keep_what_the_bag_discards(self) -> None:
        def ordered(cells: Iterable[Cell]) -> list[str]:
            return [e.text for _, _, e in cells]

        self.assertNotEqual(ordered(TABLE), ordered(SWAPPED_TABLE))


if __name__ == "__main__":
    unittest.main()
