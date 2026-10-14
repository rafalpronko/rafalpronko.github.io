"""Tests for the strict IOB2 scorer in ner_eval.py (synthetic data only)."""

import pytest

from ner_eval import (
    DEMO_GOLD,
    DEMO_PRED,
    AlignmentError,
    BioError,
    Ratio,
    Span,
    collapse_subwords,
    diagnose,
    evaluate,
    extract_spans,
    read_columns,
)


def one(tags: list[str]) -> list[list[list[str]]]:
    """Wrap a single sentence as a one-document corpus."""
    return [[tags]]


def counts(report, label: str = "micro") -> tuple[int, int, int]:
    row = report.entity[label]
    return row.tp, row.fp, row.fn


# --- span extraction -------------------------------------------------------


def test_spans_are_half_open_document_offsets_and_respect_sentences():
    corpus = [[["B-PER", "I-PER", "O"], ["B-LOC"]], [["O", "B-ORG"]]]
    assert extract_spans(corpus) == [
        Span(0, 0, 2, "PER"),
        Span(0, 3, 4, "LOC"),
        Span(1, 1, 2, "ORG"),
    ]


def test_adjacent_entities_of_same_type_need_b_tag():
    assert extract_spans(one(["B-PER", "I-PER", "B-PER"])) == [
        Span(0, 0, 2, "PER"),
        Span(0, 2, 3, "PER"),
    ]


def test_entity_cannot_continue_across_sentence_boundary():
    corpus = [[["O", "B-PER"], ["I-PER", "O"]]]
    with pytest.raises(BioError, match="sentence 1"):
        extract_spans(corpus)


@pytest.mark.parametrize(
    "bad", ["", "B-", "I-", "PER", "b-PER", "B_PER", "E-PER", "o", None, 3]
)
def test_malformed_tags_fail_closed(bad):
    with pytest.raises(BioError):
        extract_spans(one(["O", bad]))


@pytest.mark.parametrize(
    "tags",
    [["I-PER"], ["O", "I-PER"], ["B-ORG", "I-PER"], ["B-PER", "O", "I-PER"]],
)
def test_invalid_i_transitions_fail_closed_by_default(tags):
    with pytest.raises(BioError, match="I-PER"):
        extract_spans(one(tags))


def test_invalid_gold_is_rejected_even_with_lenient_prediction_policy():
    with pytest.raises(BioError, match="gold"):
        evaluate(
            one(["O", "I-PER"]), one(["O", "B-PER"]), on_invalid="as-begin"
        )


def test_lenient_prediction_policy_is_explicit_and_counted():
    report = evaluate(
        one(["O", "B-PER", "I-PER"]),
        one(["O", "I-PER", "I-PER"]),
        on_invalid="as-begin",
    )
    assert report.pred_repairs == 1
    assert counts(report) == (1, 0, 0)


def test_strict_prediction_policy_raises_on_orphan_i():
    with pytest.raises(BioError, match="pred"):
        evaluate(one(["O", "B-PER"]), one(["O", "I-PER"]))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="on_invalid"):
        evaluate(one(["O"]), one(["O"]), on_invalid="ignore")


# --- exact entity matching -------------------------------------------------


def test_exact_match():
    tags = ["B-PER", "I-PER", "O", "B-LOC"]
    report = evaluate(one(tags), one(tags))
    assert counts(report) == (2, 0, 0)
    assert report.entity["micro"].f1 == Ratio(4, 4)
    assert report.token_accuracy == Ratio(4, 4)


def test_truncated_entity_is_one_fp_and_one_fn():
    report = evaluate(
        one(["B-ORG", "I-ORG", "I-ORG"]), one(["B-ORG", "I-ORG", "O"])
    )
    assert counts(report) == (0, 1, 1)
    assert report.token_accuracy == Ratio(2, 3)
    assert report.token_micro.f1 == Ratio(4, 5)
    assert report.entity["micro"].f1 == Ratio(0, 2)
    diagnosis = diagnose(report.gold_spans, report.pred_spans)
    assert diagnosis["boundary_same_type"] == 1


def test_wrong_type_with_correct_boundaries():
    report = evaluate(one(["B-PER", "I-PER"]), one(["B-ORG", "I-ORG"]))
    assert counts(report) == (0, 1, 1)
    assert counts(report, "PER") == (0, 0, 1)
    assert counts(report, "ORG") == (0, 1, 0)
    assert diagnose(report.gold_spans, report.pred_spans)["wrong_type"] == 1


def test_split_entity():
    report = evaluate(
        one(["B-ORG", "I-ORG", "I-ORG"]), one(["B-ORG", "I-ORG", "B-ORG"])
    )
    assert counts(report) == (0, 2, 1)
    assert report.token_accuracy == Ratio(2, 3)


def test_merged_entities():
    report = evaluate(
        one(["B-PER", "I-PER", "B-PER"]), one(["B-PER", "I-PER", "I-PER"])
    )
    assert counts(report) == (0, 1, 2)
    assert report.token_accuracy == Ratio(2, 3)


def test_o_majority_inflates_accuracy_but_not_entity_recall():
    gold = one(["O"] * 98 + ["B-LOC", "I-LOC"])
    pred = one(["O"] * 100)
    report = evaluate(gold, pred)
    assert report.token_accuracy == Ratio(98, 100)
    assert report.token_micro.precision == Ratio(0, 0)
    assert report.token_micro.precision.value is None
    assert report.entity["micro"].recall == Ratio(0, 1)
    assert report.entity["micro"].precision.value is None
    assert report.entity["micro"].f1 == Ratio(0, 1)


def test_multiple_documents_do_not_share_offsets():
    gold = [[["B-PER", "O"]], [["O", "O"]]]
    pred = [[["O", "O"]], [["B-PER", "O"]]]
    report = evaluate(gold, pred)
    assert counts(report) == (0, 1, 1)
    assert report.token_accuracy == Ratio(2, 4)


def test_empty_entity_sets_have_explicit_zero_denominators():
    report = evaluate([[["O", "O"]], [[]]], [[["O", "O"]], [[]]])
    micro = report.entity["micro"]
    assert (micro.precision, micro.recall, micro.f1) == (
        Ratio(0, 0),
        Ratio(0, 0),
        Ratio(0, 0),
    )
    assert str(micro.f1) == "undefined (0/0)"
    assert report.token_accuracy == Ratio(2, 2)


def test_per_type_support_counts_gold_entities_only():
    gold = one(["B-PER", "O", "B-PER", "O", "B-LOC"])
    pred = one(["B-PER", "O", "O", "B-ORG", "B-LOC"])
    report = evaluate(gold, pred)
    assert report.entity["PER"].support == 2
    assert report.entity["LOC"].support == 1
    assert report.entity["ORG"].support == 0
    assert report.entity["ORG"].recall.value is None
    assert report.entity["micro"].support == 3


# --- shape checks: no silent truncation ------------------------------------


@pytest.mark.parametrize(
    ("gold", "pred", "where"),
    [
        ([[["O"]]], [[["O"]], [["O"]]], "documents"),
        ([[["O"], ["O"]]], [[["O"]]], "document 0"),
        ([[["O", "O"]]], [[["O"]]], "sentence 0"),
        ([[["O"]]], [[["O", "B-PER"]]], "sentence 0"),
    ],
)
def test_length_mismatch_fails_instead_of_truncating(gold, pred, where):
    with pytest.raises(AlignmentError, match=where):
        evaluate(gold, pred)


def test_flat_string_instead_of_tag_list_is_rejected():
    with pytest.raises(TypeError):
        evaluate([["O O"]], [["O O"]])


# --- subword alignment -----------------------------------------------------


def test_collapse_subwords_takes_first_piece_and_skips_special_and_padding():
    word_ids = [None, 0, 0, 1, 2, 2, 2, None, None, None]
    tags = [
        "O", "B-LOC", "I-LOC", "O", "B-PER", "I-PER",
        "O", "O", "B-PER", "I-PER",
    ]
    words = collapse_subwords(word_ids, tags, n_words=3)
    assert words == ["B-LOC", "O", "B-PER"]


@pytest.mark.parametrize(
    ("word_ids", "n_words"),
    [
        ([None, 0, 2, None], 3),  # word 1 has no pieces (lost to truncation)
        ([None, 0, 1, 0, None], 2),  # non-contiguous pieces
        ([None, 0, 1, None], 3),  # max_length cut the last word
        ([None, 0, 1, 2, None], 2),  # more words than expected
    ],
)
def test_collapse_subwords_rejects_broken_alignment(word_ids, n_words):
    with pytest.raises(AlignmentError):
        collapse_subwords(word_ids, ["O"] * len(word_ids), n_words=n_words)


def test_collapse_subwords_rejects_length_mismatch():
    with pytest.raises(AlignmentError):
        collapse_subwords([None, 0, None], ["O", "O"], n_words=1)


# --- three-column input --------------------------------------------------


def test_read_columns_keeps_documents_and_sentences():
    text = (
        "-DOCSTART-\nAnna B-PER B-PER\n. O O\n\n"
        "Hi O O\n-DOCSTART-\nOslo B-LOC O\n"
    )
    words, gold, pred = read_columns(text)
    assert words == [[["Anna", "."], ["Hi"]], [["Oslo"]]]
    assert gold == [[["B-PER", "O"], ["O"]], [["B-LOC"]]]
    assert pred == [[["B-PER", "O"], ["O"]], [["O"]]]


def test_read_columns_without_docstart_is_one_document():
    words, _, _ = read_columns("a O O\n\n\nb O O\n")
    assert words == [[["a"], ["b"]]]


@pytest.mark.parametrize("line", ["Anna B-PER", "Anna B-PER B-PER extra"])
def test_read_columns_rejects_wrong_column_count(line):
    with pytest.raises(AlignmentError, match="line 2"):
        read_columns("ok O O\n" + line + "\n")


# --- the article's synthetic demonstration ---------------------------------


def test_demo_numbers_quoted_in_article():
    report = evaluate(DEMO_GOLD, DEMO_PRED)
    assert report.token_accuracy == Ratio(27, 33)
    assert report.token_micro.precision == Ratio(10, 15)
    assert report.token_micro.recall == Ratio(10, 15)
    assert counts(report) == (2, 5, 5)
    assert report.entity["micro"].f1 == Ratio(4, 14)
    assert dict(diagnose(report.gold_spans, report.pred_spans)) == {
        "exact": 2,
        "boundary_same_type": 3,
        "boundary_and_type": 1,
        "wrong_type": 1,
        "missed": 0,
        "spurious": 0,
    }
