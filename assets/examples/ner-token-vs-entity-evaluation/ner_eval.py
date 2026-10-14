"""Strict IOB2 scorer for flat, non-overlapping named entities.

Standard library only. It scores the same predictions three ways: token
accuracy, token-level micro F1 over non-O tags, and exact entity-level
micro F1. Run without arguments for the synthetic demo, or pass a file with
"word gold pred" columns (blank line = sentence end, -DOCSTART- = new
document).

This is a single-file teaching download on purpose: the scoring core
(types, parse_tag, the decoder, evaluate and diagnose) comes first, followed
by the subword helper, the three-column reader, the report formatter and the
demo. In a library these would be separate modules; here one file keeps the
example easy to download and run.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

POLICIES = ("raise", "as-begin")
Corpus = Sequence[Sequence[Sequence[str]]]  # documents -> sentences -> tags


class BioError(ValueError):
    """A tag or a tag transition that is not valid IOB2."""


class AlignmentError(ValueError):
    """Gold and predicted sequences (or a subword map) do not line up."""


class Span(NamedTuple):
    """Document index, half-open token range [start, end) and entity type."""

    doc: int
    start: int
    end: int
    type: str


class Ratio(NamedTuple):
    """A fraction that keeps its denominator; 0/0 is undefined, not zero."""

    num: int
    den: int

    @property
    def value(self) -> float | None:
        return None if self.den == 0 else self.num / self.den

    def __str__(self) -> str:
        if self.den == 0:
            return "undefined (0/0)"
        return f"{self.num}/{self.den} = {self.num / self.den:.3f}"


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def support(self) -> int:
        return self.tp + self.fn  # gold items

    @property
    def predicted(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> Ratio:
        return Ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> Ratio:
        return Ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> Ratio:
        # Equals the harmonic mean 2PR/(P+R) when P and R are defined and
        # P + R > 0. With TP = 0 and some errors this is 0, while 2PR/(P+R)
        # would be 0/0. Undefined (0/0) only if all three counts are 0.
        return Ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)


@dataclass(frozen=True)
class Report:
    token_accuracy: Ratio
    token_micro: Counts
    entity: dict[str, Counts]  # entity types in sorted order, then "micro"
    gold_spans: list[Span]
    pred_spans: list[Span]
    pred_repairs: int


def parse_tag(tag: object) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    if (
        isinstance(tag, str)
        and len(tag) > 2
        and tag[0] in "BI"
        and tag[1] == "-"
    ):
        entity_type = tag[2:]
        if entity_type.split() == [entity_type]:
            return tag[0], entity_type
    raise BioError(f"malformed tag {tag!r}")


def _check_policy(on_invalid: str) -> None:
    if on_invalid not in POLICIES:
        raise ValueError(
            f"on_invalid must be one of {POLICIES}, got {on_invalid!r}"
        )


def _check_containers(corpus: Corpus, name: str) -> None:
    def is_list(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, str)

    if not is_list(corpus):
        raise TypeError(f"{name}: expected a list of documents")
    for d, doc in enumerate(corpus):
        if not is_list(doc):
            raise TypeError(
                f"{name} document {d}: expected a list of sentences"
            )
        for s, sentence in enumerate(doc):
            if not is_list(sentence):
                raise TypeError(
                    f"{name} document {d} sentence {s}: "
                    "expected a list of tags"
                )


def check_shapes(gold: Corpus, pred: Corpus) -> None:
    """Fail on any length difference instead of letting zip() truncate."""
    if len(gold) != len(pred):
        raise AlignmentError(
            f"documents: gold has {len(gold)}, pred has {len(pred)}"
        )
    for d, (gold_doc, pred_doc) in enumerate(zip(gold, pred, strict=True)):
        if len(gold_doc) != len(pred_doc):
            raise AlignmentError(
                f"document {d}: gold has {len(gold_doc)} sentences, "
                f"pred has {len(pred_doc)}"
            )
        for s, (g, p) in enumerate(zip(gold_doc, pred_doc, strict=True)):
            if len(g) != len(p):
                raise AlignmentError(
                    f"document {d} sentence {s}: "
                    f"gold has {len(g)} tags, pred has {len(p)}"
                )


def _extract(
    corpus: Corpus, name: str, on_invalid: str
) -> tuple[list[Span], int]:
    spans: list[Span] = []
    repairs = 0
    for d, doc in enumerate(corpus):
        offset = 0  # token offset of the current sentence in the document
        for s, sentence in enumerate(doc):
            # Entity state never crosses a sentence boundary.
            open_type: str | None = None
            start = 0
            for i, tag in enumerate(sentence):
                where = f"{name} document {d} sentence {s} token {i}"
                try:
                    prefix, entity_type = parse_tag(tag)
                except BioError as error:
                    raise BioError(f"{where}: {error}") from None
                if prefix == "I" and entity_type != open_type:
                    if on_invalid == "raise":
                        after = (
                            "sentence start"
                            if i == 0
                            else repr(sentence[i - 1])
                        )
                        raise BioError(
                            f"{where}: {tag} after {after} is not valid IOB2"
                        )
                    # "as-begin": an orphan I-X opens a new X entity.
                    prefix = "B"
                    repairs += 1
                if prefix != "I" and open_type is not None:
                    spans.append(
                        Span(d, offset + start, offset + i, open_type)
                    )
                    open_type = None
                if prefix == "B":
                    open_type, start = entity_type, i
            if open_type is not None:
                spans.append(
                    Span(d, offset + start, offset + len(sentence), open_type)
                )
            offset += len(sentence)
    return spans, repairs


def extract_spans(corpus: Corpus, on_invalid: str = "raise") -> list[Span]:
    _check_policy(on_invalid)
    _check_containers(corpus, "corpus")
    return _extract(corpus, "corpus", on_invalid)[0]


def _entity_counts(gold: set[Span], pred: set[Span]) -> Counts:
    return Counts(
        tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred)
    )


def evaluate(gold: Corpus, pred: Corpus, on_invalid: str = "raise") -> Report:
    """Score predictions against gold.

    Gold is always parsed strictly; on_invalid applies to pred only.
    """
    _check_policy(on_invalid)
    _check_containers(gold, "gold")
    _check_containers(pred, "pred")
    check_shapes(gold, pred)
    gold_spans, _ = _extract(gold, "gold", "raise")
    pred_spans, repairs = _extract(pred, "pred", on_invalid)

    correct = total = tp = fp = fn = 0
    for gold_doc, pred_doc in zip(gold, pred, strict=True):
        for gold_sentence, pred_sentence in zip(
            gold_doc, pred_doc, strict=True
        ):
            for g, p in zip(gold_sentence, pred_sentence, strict=True):
                total += 1
                if g == p:
                    correct += 1
                    tp += g != "O"
                else:
                    fp += p != "O"
                    fn += g != "O"

    gold_set, pred_set = set(gold_spans), set(pred_spans)
    types = sorted({span.type for span in gold_set | pred_set})
    if "micro" in types:
        raise ValueError(
            "entity type 'micro' clashes with the micro-average row"
        )
    entity = {
        t: _entity_counts(
            {x for x in gold_set if x.type == t},
            {x for x in pred_set if x.type == t},
        )
        for t in types
    }
    entity["micro"] = _entity_counts(gold_set, pred_set)
    return Report(
        Ratio(correct, total),
        Counts(tp, fp, fn),
        entity,
        gold_spans,
        pred_spans,
        repairs,
    )


def diagnose(
    gold_spans: Sequence[Span], pred_spans: Sequence[Span]
) -> dict[str, int]:
    """Explain exact-match errors. A diagnostic only: it never changes a score.

    Each distinct predicted span gets exactly one category, the first that
    applies in this order: exact, wrong_type (same start and end as a gold
    span), boundary_same_type (overlaps a gold span of its type),
    boundary_and_type (overlaps only other types), spurious (no overlap).
    This is not one-to-one matching: a merged prediction covering two gold
    entities is counted once, a split produces one category per predicted
    piece, and one gold entity can explain several predictions. "missed"
    counts gold spans that no prediction overlaps.
    """
    out = dict.fromkeys(
        (
            "exact",
            "wrong_type",
            "boundary_same_type",
            "boundary_and_type",
            "spurious",
            "missed",
        ),
        0,
    )
    gold, pred = set(gold_spans), set(pred_spans)

    def overlaps(a: Span, b: Span) -> bool:
        return a.doc == b.doc and a.start < b.end and b.start < a.end

    for p in pred:
        hits = [g for g in gold if overlaps(p, g)]
        if p in gold:
            out["exact"] += 1
        elif any((g.start, g.end) == (p.start, p.end) for g in hits):
            out["wrong_type"] += 1
        elif any(g.type == p.type for g in hits):
            out["boundary_same_type"] += 1
        elif hits:
            out["boundary_and_type"] += 1
        else:
            out["spurious"] += 1
    out["missed"] = sum(
        1 for g in gold if not any(overlaps(g, p) for p in pred)
    )
    return out


def collapse_subwords(
    word_ids: Sequence[int | None], subword_tags: Sequence[str], n_words: int
) -> list[str]:
    """Keep the tag of the first piece of every word.

    word_ids follows the usual fast-tokenizer convention: None marks special
    tokens and padding. Every word 0..n_words-1 must appear exactly once as a
    contiguous run of pieces; anything else is an alignment error.
    """
    if len(word_ids) != len(subword_tags):
        raise AlignmentError(
            f"{len(word_ids)} word ids but {len(subword_tags)} subword tags"
        )
    words: list[str] = []
    previous: int | None = None
    for i, (word_id, tag) in enumerate(
        zip(word_ids, subword_tags, strict=True)
    ):
        if word_id is None:
            previous = None
            continue
        if word_id == previous:
            continue  # a later piece of the same word
        if word_id != len(words):
            raise AlignmentError(
                f"position {i}: expected word {len(words)}, got word {word_id}"
            )
        words.append(tag)
        previous = word_id
    if len(words) != n_words:
        raise AlignmentError(
            f"recovered {len(words)} words, expected {n_words}"
        )
    return words


def read_columns(text: str) -> tuple[list, list, list]:
    """Parse "word gold pred" lines into three document/sentence corpora."""
    corpora: tuple[list, list, list] = ([], [], [])
    sentence: tuple[list, list, list] = ([], [], [])

    def new_document() -> None:
        for corpus in corpora:
            corpus.append([])

    def end_sentence() -> None:
        if not sentence[0]:
            return
        if not corpora[0]:
            new_document()
        for corpus, column in zip(corpora, sentence, strict=True):
            corpus[-1].append(column.copy())
            column.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            end_sentence()
        elif fields[0] == "-DOCSTART-":
            end_sentence()
            new_document()
        elif len(fields) != 3:
            raise AlignmentError(
                f"line {number}: expected 'word gold pred', "
                f"got {len(fields)} fields"
            )
        else:
            for column, value in zip(sentence, fields, strict=True):
                column.append(value)
    end_sentence()
    return corpora


# Synthetic example: invented names, hand-written gold and predicted tags.
DEMO_COLUMNS = """\
-DOCSTART-
Anna B-PER B-PER
Maria I-PER I-PER
Nowak I-PER I-PER
joined O O
Nordwave B-ORG B-ORG
Data I-ORG I-ORG
Labs I-ORG O
in O O
Gdańsk B-LOC B-LOC
. O O

She O O
met O O
Jan B-PER B-PER
Kowalski I-PER I-PER
and O I-PER
Ewa B-PER I-PER
Lis I-PER I-PER
on O O
Monday O O
. O O

-DOCSTART-
The O O
Vistula B-MISC B-LOC
Data I-MISC B-MISC
Summit I-MISC I-MISC
took O O
place O O
in O O
Warsaw B-LOC B-ORG
. O O

Weather O O
was O O
mild O O
. O O
"""
DEMO_WORDS, DEMO_GOLD, DEMO_PRED = read_columns(DEMO_COLUMNS)


def format_report(report: Report, words: Corpus | None = None) -> str:
    def text(span: Span) -> str:
        if words is None:
            return ""
        tokens = [w for sentence in words[span.doc] for w in sentence]
        return " " + repr(" ".join(tokens[span.start : span.end]))

    tok = report.token_micro
    lines = [
        f"Token accuracy (all tags, O included): {report.token_accuracy}",
        "Token micro scores (non-O tags, B-/I- prefix must match):",
        f"  TP={tok.tp} FP={tok.fp} FN={tok.fn}",
        f"  precision {tok.precision} | recall {tok.recall} | F1 {tok.f1}",
        "Entity scores (exact document, start, end and type):",
        f"  {'type':<6}{'support':>8}{'pred':>6}{'TP':>4}{'FP':>4}{'FN':>4}"
        "  precision        recall           F1",
    ]
    for label, c in report.entity.items():
        lines.append(
            f"  {label:<6}{c.support:>8}{c.predicted:>6}"
            f"{c.tp:>4}{c.fp:>4}{c.fn:>4}"
            f"  {str(c.precision):<17}{str(c.recall):<17}{c.f1}"
        )
    if report.pred_repairs:
        lines.append(
            f"Orphan I- tags repaired in predictions: {report.pred_repairs}"
        )
    gold, pred = set(report.gold_spans), set(report.pred_spans)
    lines.append("Span-level errors:")
    for span in sorted(gold - pred):
        lines.append(f"  FN gold {tuple(span)}{text(span)}")
    for span in sorted(pred - gold):
        lines.append(f"  FP pred {tuple(span)}{text(span)}")
    diagnosis = diagnose(report.gold_spans, report.pred_spans)
    lines.append(
        "Diagnosis (does not change the scores): "
        + ", ".join(f"{key}={value}" for key, value in diagnosis.items())
    )
    return "\n".join(lines)


def _demo_extras() -> None:
    gold, pred = [[["O", "B-PER", "I-PER"]]], [[["O", "I-PER", "I-PER"]]]
    print(
        "\n== Orphan I- tag policy: "
        "gold O B-PER I-PER, pred O I-PER I-PER =="
    )
    try:
        evaluate(gold, pred)
    except BioError as error:
        print(f"on_invalid='raise'    -> BioError: {error}")
    lenient = evaluate(gold, pred, on_invalid="as-begin")
    micro = lenient.entity["micro"]
    print(
        f"on_invalid='as-begin' -> repairs={lenient.pred_repairs}, "
        f"entity TP={micro.tp} FP={micro.fp} FN={micro.fn}, "
        f"token accuracy {lenient.token_accuracy}"
    )

    print(
        "\n== Subword alignment: 3 words, 10 positions "
        "with [CLS], [SEP] and padding =="
    )
    word_ids = [None, 0, 0, 1, 2, 2, 2, None, None, None]
    pieces = [
        "O", "B-LOC", "I-LOC", "O", "B-PER", "I-PER",
        "O", "O", "B-PER", "I-PER",
    ]
    print(f"word_ids     {word_ids}")
    print(f"subword tags {pieces}")
    print(f"word tags    {collapse_subwords(word_ids, pieces, n_words=3)}")
    try:
        collapse_subwords([None, 0, 0, 1, None], ["O"] * 5, n_words=3)
    except AlignmentError as error:
        print(f"truncated input -> AlignmentError: {error}")


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print("usage: python ner_eval.py [columns.txt]", file=sys.stderr)
        return 2
    source = (
        Path(argv[1]).read_text(encoding="utf-8")
        if len(argv) == 2
        else DEMO_COLUMNS
    )
    words, gold, pred = read_columns(source)
    n_sentences = sum(len(doc) for doc in gold)
    n_tokens = sum(len(sentence) for doc in gold for sentence in doc)
    print(
        f"Corpus: {len(gold)} documents, {n_sentences} sentences, "
        f"{n_tokens} tokens"
    )
    print(format_report(evaluate(gold, pred), words))
    if len(argv) == 1:
        _demo_extras()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
