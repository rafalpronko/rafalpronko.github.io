"""Score retrieval, answer quality and evidence support as separate layers.

Every number comes from human-style labels in a JSON file: chunk IDs,
atomic claim IDs and per-claim judgements. Nothing here retrieves,
generates or decides entailment; it only aggregates the labels.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reference labels shared by a case and its gold-context rerun.
REFERENCE_FIELDS = ("question", "relevant", "evidence_sets", "required_claims",
                    "answerable")


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    correct: bool
    relevant: bool
    # Chunks that each support the claim on their own. A claim that needs
    # two chunks jointly cannot be expressed in this schema.
    supported_by: tuple[str, ...]
    cited: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    relevant: tuple[str, ...]
    evidence_sets: tuple[tuple[str, ...], ...]
    required_claims: tuple[str, ...]
    answerable: bool
    retrieved: tuple[str, ...] | None  # None: gold context, retrieval bypassed
    provided: tuple[str, ...]
    abstained: bool
    claims: tuple[Claim, ...]
    gold_of: str | None = None


@dataclass(frozen=True)
class Dataset:
    snapshot: str
    k: int
    corpus: dict[str, str]
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Result:
    id: str
    recall: float | None
    precision: float | None
    sufficient_topk: bool | None
    sufficient_context: bool | None
    correctness: float | None
    completeness: float | None
    relevance: float | None
    faithfulness: float | None
    citation_coverage: float | None
    citation_entailment: float | None
    invalid_citations: int
    abstention: str
    tags: tuple[str, ...]


def _ids(values: list[str], what: str, where: str) -> tuple[str, ...]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{where}: duplicate {what} ID {value!r}")
        seen.add(value)
    return tuple(values)


def _flag(raw: dict[str, Any], key: str, where: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key!r} must be a JSON boolean, "
                         f"got {value!r}")
    return value


def _known(ids: tuple[str, ...], corpus: dict[str, str], where: str) -> None:
    for chunk_id in ids:
        if chunk_id not in corpus:
            raise ValueError(f"{where}: unknown chunk ID {chunk_id!r}")


def _claim(raw: dict[str, Any], provided: set[str], where: str) -> Claim:
    supported = _ids(raw["supported_by"], "supported_by", where)
    outside = [c for c in supported if c not in provided]
    if outside:
        raise ValueError(f"{where}: supported_by {outside} "
                         "not in provided context")
    # Cited IDs are not validated: citing a missing chunk is a model error
    # that the metrics must count, not a labelling error.
    cited = _ids(raw["cited"], "cited", where)
    return Claim(raw["id"], raw["text"], _flag(raw, "correct", where),
                 _flag(raw, "relevant", where), supported, cited)


def _case(raw: dict[str, Any], corpus: dict[str, str], k: int) -> Case:
    where = f"case {raw['id']!r}"
    relevant = _ids(raw["relevant"], "relevant", where)
    sets = tuple(_ids(s, "evidence", where) for s in raw["evidence_sets"])
    required = _ids(raw["required_claims"], "required claim", where)
    # Only a gold case (checked in parse) bypasses retrieval. Every other
    # case must log what retrieval returned; [] means it returned nothing.
    retrieved = None
    if raw.get("gold_of") is None:
        if "retrieved" not in raw:
            raise ValueError(f"{where}: ordinary case needs a 'retrieved' "
                             "list")
        if not isinstance(raw["retrieved"], list):
            raise ValueError(f"{where}: 'retrieved' must be a JSON list, "
                             f"got {raw['retrieved']!r}")
        retrieved = _ids(raw["retrieved"], "retrieved", where)
    provided = _ids(raw["provided"], "provided", where)
    for ids in (relevant, retrieved or (), provided, *sets):
        _known(ids, corpus, where)
    answerable = _flag(raw, "answerable", where)
    abstained = _flag(raw, "abstained", where)

    if answerable != bool(sets):
        raise ValueError(f"{where}: answerable flag is {answerable} but "
                         f"{len(sets)} evidence sets are labelled")
    if not sets and (relevant or required):
        raise ValueError(f"{where}: unanswerable case (no evidence sets) "
                         "cannot have relevant chunks or required claims")
    if sets and not required:
        raise ValueError(f"{where}: answerable case needs required claims")
    for evidence in sets:
        if not evidence or not set(evidence) <= set(relevant):
            raise ValueError(f"{where}: evidence set {list(evidence)} must be "
                             "non-empty and labelled relevant")
    # `retrieved` may be logged beyond k, but the model can only be given
    # chunks from the top k; otherwise context_dropped would be meaningless.
    if retrieved is not None and not set(provided) <= set(retrieved[:k]):
        raise ValueError(f"{where}: provided chunks must come from the top k "
                         "retrieved")
    if raw.get("gold_of") is not None and not any(
            set(s) <= set(provided) for s in sets):
        raise ValueError(f"{where}: gold context must contain a full "
                         "evidence set")

    claims = tuple(_claim(c, set(provided), where) for c in raw["claims"])
    _ids([c.id for c in claims], "claim", where)
    if abstained and claims:
        raise ValueError(f"{where}: abstained answer cannot contain claims")
    for claim in claims:
        if claim.id in required and not claim.correct:
            raise ValueError(f"{where}: claim {claim.id!r} uses a required "
                             "claim ID but is labelled incorrect")
    return Case(raw["id"], raw["question"], relevant, sets, required,
                answerable, retrieved, provided, abstained, claims,
                raw.get("gold_of"))


def parse(raw: dict[str, Any]) -> Dataset:
    k = raw["k"]
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    corpus = dict(raw["corpus"])
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw["cases"]:
        if item["id"] in by_id:
            raise ValueError(f"duplicate case ID {item['id']!r}")
        by_id[item["id"]] = item

    cases = []
    for item in raw["cases"]:
        target = item.get("gold_of")
        if target is not None:
            where = f"case {item['id']!r}"
            base = by_id.get(target)
            if (base is None or base.get("gold_of") is not None
                    or base.get("answerable") is not True):
                raise ValueError(f"{where}: gold_of {target!r} must name an "
                                 "answerable, non-gold case")
            # A gold-context run reuses the question and reference labels of
            # its base case unchanged, and skips retrieval: the context is
            # supplied directly.
            clash = [f for f in REFERENCE_FIELDS if f in item]
            if clash:
                raise ValueError(f"{where}: gold case must not set {clash}; "
                                 f"they are inherited from {target!r}")
            if "retrieved" in item:
                raise ValueError(f"{where}: gold case cannot have a "
                                 "retrieved list")
            item = {**{f: base[f] for f in REFERENCE_FIELDS}, **item}
        cases.append(_case(item, corpus, k))
    return Dataset(raw["snapshot"], k, corpus, tuple(cases))


def load(path: str | Path) -> Dataset:
    return parse(json.loads(Path(path).read_text(encoding="utf-8")))


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def score(case: Case, k: int) -> Result:
    relevant = set(case.relevant)
    if case.retrieved is None or not case.answerable:
        recall = precision = sufficient_topk = None
    else:
        top = set(case.retrieved[:k])
        recall = len(relevant & top) / len(relevant)
        precision = len(relevant & top) / k
        sufficient_topk = any(set(s) <= top for s in case.evidence_sets)
    sufficient_context = (any(set(s) <= set(case.provided)
                              for s in case.evidence_sets)
                          if case.answerable else None)

    claims = case.claims
    n = len(claims)
    good = {c.id for c in claims if c.correct}
    completeness = (None if not case.required_claims else
                    sum(r in good for r in case.required_claims)
                    / len(case.required_claims))
    pairs = [(c, cid) for c in claims for cid in c.cited]
    provided = set(case.provided)

    if not case.answerable:
        abstention = "correct" if case.abstained else "should_abstain"
    elif case.abstained:
        abstention = "missed_answer" if sufficient_context else "no_evidence"
    else:
        abstention = "answered"

    result = Result(
        id=case.id,
        recall=recall,
        precision=precision,
        sufficient_topk=sufficient_topk,
        sufficient_context=sufficient_context,
        correctness=_ratio(len(good), n),
        completeness=completeness,
        relevance=_ratio(sum(c.relevant for c in claims), n),
        faithfulness=_ratio(sum(bool(c.supported_by) for c in claims), n),
        citation_coverage=_ratio(sum(bool(c.cited) for c in claims), n),
        citation_entailment=_ratio(
            sum(cid in c.supported_by for c, cid in pairs), len(pairs)),
        invalid_citations=sum(cid not in provided for _, cid in pairs),
        abstention=abstention,
        tags=(),
    )
    return _with_tags(result, case)


def _with_tags(r: Result, case: Case) -> Result:
    tags = []
    if r.abstention in ("should_abstain", "missed_answer", "no_evidence"):
        tags.append(r.abstention)
    if r.sufficient_topk is False:
        tags.append("retrieval_miss")
    elif r.sufficient_topk and r.sufficient_context is False:
        tags.append("context_dropped")
    if any(c.supported_by and not c.correct for c in case.claims):
        tags.append("supported_but_wrong")
    if any(not c.supported_by for c in case.claims):
        tags.append("unsupported_claim")
    if any(not c.cited for c in case.claims):
        tags.append("uncited_claim")
    entailment = r.citation_entailment
    if r.invalid_citations or (entailment is not None and entailment < 1.0):
        tags.append("bad_citation")
    if any(not c.relevant for c in case.claims):
        tags.append("off_topic_claim")
    if (not case.abstained and r.completeness is not None
            and r.completeness < 1.0):
        tags.append("incomplete")
    return Result(**{**r.__dict__, "tags": tuple(tags) or ("ok",)})


def isolate(actual: Result, gold: Result) -> str:
    """Compare a case with its gold-context rerun: a hint, not a proof.

    Both runs should use the same model, prompt and decoding settings, and
    one pair of samples can still differ by chance.
    """
    if actual.tags == ("ok",):
        return "actual run passes: nothing to isolate"
    if gold.tags != ("ok",):
        return ("gold context does not fix it: suspect generation, the prompt "
                "or the labels; upstream stages may be failing as well")
    if actual.sufficient_context is not False:
        # Sufficient only means a labelled evidence set was present. The two
        # contexts can still differ in distractors, order or evidence path.
        return ("inconclusive: both contexts held a sufficient evidence set; "
                "the gold pass may come from context composition or order, "
                "generation or run-to-run variability")
    stage = ("context assembly" if "context_dropped" in actual.tags
             else "retrieval")
    return ("gold context fixes it and the actual context lacked evidence: "
            f"suspect {stage} (a hint, not proof)")


def _fmt(value: float | bool | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.2f}"


def report(data: Dataset) -> str:
    head = ("case", "R@k", "P@k", "topk", "ctx", "corr", "comp", "rel",
            "faith", "c.cov", "c.ent", "bad", "abstention", "tags")
    rows = [head]
    results = {c.id: score(c, data.k) for c in data.cases}
    for r in results.values():
        rows.append((r.id, *map(_fmt, (
            r.recall, r.precision, r.sufficient_topk, r.sufficient_context,
            r.correctness, r.completeness, r.relevance, r.faithfulness,
            r.citation_coverage, r.citation_entailment)),
            str(r.invalid_citations), r.abstention, ",".join(r.tags)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    lines = [f"snapshot={data.snapshot} k={data.k}"]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip()
              for row in rows]
    for case in data.cases:
        if case.gold_of:
            verdict = isolate(results[case.gold_of], results[case.id])
            lines.append(f"isolate {case.gold_of}: {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(load(sys.argv[1] if len(sys.argv) > 1 else "labels.json")))
