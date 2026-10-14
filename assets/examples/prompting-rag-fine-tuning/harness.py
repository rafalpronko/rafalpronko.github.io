"""Deterministic diagnostic harness: prompting vs retrieval vs fine-tuning.

Nothing here calls or trains a model. The "observations" in the fixtures are
hand-authored answer strings that illustrate failure patterns. The harness does
real work on them: it validates inputs, runs a small lexical retriever over a
synthetic corpus, parses and validates answers against an output contract,
grades them and assigns a diagnostic label per case.

Why one file: this is teaching code that accompanies a single article. A reader
can follow the whole path from fixture to label top to bottom, and the article
can quote one path. The sections below (loading, retrieval, output contract,
grading, splitting) are the natural seams for a split if the code grows.

Scope of the output contract: parse_answer enforces the shape of the answer,
the allowed decisions, that citations were in the supplied context, that they
are current, and that a decision cites something. It does not check that a
citation actually supports the answer, and there is no access control here:
the corpus has no users or permissions.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DECISIONS = ("allowed", "not_allowed", "needs_approval", "insufficient_context")
CONDITIONS = ("closed_book", "retrieved", "oracle")
OPTIONAL_CONDITIONS = ("oracle_fewshot",)
STATUSES = ("current", "retired")

PASS = "PASS"
OUTPUT_CONTRACT = "OUTPUT_CONTRACT"
RETRIEVAL_MISS = "RETRIEVAL_MISS"
CONTEXT_USE = "CONTEXT_USE"
BEHAVIOR = "BEHAVIOR"
CAPABILITY_SUSPECTED = "CAPABILITY_SUSPECTED"
GOLD_CONTEXT_FAILS = "GOLD_CONTEXT_FAILS"

NEXT_STEP = {
    PASS: "no change for this case",
    OUTPUT_CONTRACT: "enforce the schema in code; fix the format with instructions/examples",
    RETRIEVAL_MISS: "fix retrieval: query, vocabulary, chunking, index, filters",
    CONTEXT_USE: "fix what reaches the model: versions, distractors, abstention rule",
    BEHAVIOR: "keep the examples, or consider an adapter if the convention must hold cheaply",
    CAPABILITY_SUSPECTED: "decompose the task or try another base model; tune only with evidence",
    GOLD_CONTEXT_FAILS: "run oracle + examples (with a readable answer) to probe behaviour and capability",
}

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from i in is it my of on or "
    "the to what when with how much may me per".split()
)


# ---------------------------------------------------------------- loading


def load_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"fixture file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc.msg}, line {exc.lineno})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    return data


def _require_str(obj: dict, key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return value


def validate_corpus(data: dict) -> dict[str, dict]:
    passages = data.get("passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError("corpus: 'passages' must be a non-empty list")
    corpus: dict[str, dict] = {}
    for i, p in enumerate(passages):
        where = f"corpus passage #{i}"
        if not isinstance(p, dict):
            raise ValueError(f"{where}: must be an object")
        pid = _require_str(p, "id", where)
        for key in ("title", "text"):
            _require_str(p, key, f"passage {pid}")
        if p.get("status") not in STATUSES:
            raise ValueError(f"passage {pid}: status must be one of {STATUSES}")
        if pid in corpus:
            raise ValueError(f"corpus: duplicate passage id {pid}")
        corpus[pid] = p
    return corpus


def validate_cases(data: dict, corpus: dict[str, dict]) -> list[dict]:
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases: 'cases' must be a non-empty list")
    seen: set[str] = set()
    for i, c in enumerate(cases):
        where = f"case #{i}"
        if not isinstance(c, dict):
            raise ValueError(f"{where}: must be an object")
        cid = _require_str(c, "id", where)
        where = f"case {cid}"
        if cid in seen:
            raise ValueError(f"cases: duplicate case id {cid}")
        seen.add(cid)
        _require_str(c, "group", where)
        _require_str(c, "question", where)

        expected = c.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"{where}: 'expected' must be an object")
        if expected.get("decision") not in DECISIONS:
            raise ValueError(f"{where}: expected decision must be one of {DECISIONS}")
        if "limit_eur" not in expected or not _is_limit(expected["limit_eur"]):
            raise ValueError(f"{where}: expected limit_eur must be a non-negative int or null")

        gold = c.get("gold_ids")
        if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
            raise ValueError(f"{where}: 'gold_ids' must be a list of passage ids")
        unknown = [g for g in gold if g not in corpus]
        if unknown:
            raise ValueError(f"{where}: unknown gold passage ids {unknown}")
        if not gold and expected["decision"] != "insufficient_context":
            raise ValueError(f"{where}: empty gold_ids is only valid for insufficient_context")
        retired = [g for g in gold if corpus[g]["status"] != "current"]
        if retired:
            raise ValueError(f"{where}: gold passages must be current, got {retired}")

        obs = c.get("observations")
        if not isinstance(obs, dict):
            raise ValueError(f"{where}: 'observations' must be an object")
        missing = [k for k in CONDITIONS if k not in obs]
        if missing:
            raise ValueError(f"{where}: missing observations {missing}")
        extra = [k for k in obs if k not in CONDITIONS + OPTIONAL_CONDITIONS]
        if extra:
            raise ValueError(f"{where}: unknown observation conditions {extra}")
        for k, raw in obs.items():
            if not isinstance(raw, str):
                raise ValueError(f"{where}: observation '{k}' must be the raw answer string")
    return cases


def load_fixtures(directory: Path) -> tuple[dict[str, dict], dict]:
    directory = Path(directory)
    corpus = validate_corpus(load_json(directory / "corpus.json"))
    data = load_json(directory / "cases.json")
    validate_cases(data, corpus)
    return corpus, data


# -------------------------------------------------------------- retrieval


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]  # crude plural folding, deliberately no real stemming
        tokens.append(t)
    return tokens


def retrieve(
    query: str,
    corpus: dict[str, dict],
    k: int,
    current_only: bool = False,
) -> list[str]:
    """BM25 over title + text. Ties are broken by passage id for determinism."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    docs = {
        pid: tokenize(p["title"] + " " + p["text"])
        for pid, p in corpus.items()
        if not current_only or p["status"] == "current"
    }
    if k > len(docs):
        raise ValueError(f"k={k} exceeds the {len(docs)} searchable passages")
    n = len(docs)
    avg_len = sum(len(d) for d in docs.values()) / n
    df = Counter(t for d in docs.values() for t in set(d))
    k1, b = 1.2, 0.75
    scores = {}
    for pid, doc in docs.items():
        tf = Counter(doc)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            norm = tf[term] + k1 * (1 - b + b * len(doc) / avg_len)
            score += idf * tf[term] * (k1 + 1) / norm
        scores[pid] = score
    ranked = sorted(scores, key=lambda pid: (-scores[pid], pid))
    return ranked[:k]


# --------------------------------------------------------- output contract


def _is_limit(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_answer(
    raw: str,
    context_ids: list[str],
    corpus: dict[str, dict],
) -> tuple[dict | None, list[str]]:
    """Check an answer against the contract. Returns (object or None, errors)."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["not valid JSON"]
    if not isinstance(obj, dict):
        return None, ["answer must be a JSON object"]

    errors = []
    allowed_keys = {"decision", "limit_eur", "citations"}
    missing = sorted(allowed_keys - obj.keys())
    if missing:
        errors.append(f"missing fields {missing}")
    extra = sorted(obj.keys() - allowed_keys)
    if extra:
        errors.append(f"unexpected fields {extra}")
    if "decision" in obj and obj["decision"] not in DECISIONS:
        errors.append(f"decision {obj['decision']!r} not in {DECISIONS}")
    if "limit_eur" in obj and not _is_limit(obj["limit_eur"]):
        errors.append("limit_eur must be a non-negative int or null")

    citations = obj.get("citations")
    if "citations" in obj:
        if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            errors.append("citations must be a list of passage ids")
        else:
            outside = [c for c in citations if c not in context_ids]
            if outside:
                errors.append(f"cites passages that were not in context {outside}")
            retired = [c for c in citations if c in corpus and corpus[c]["status"] != "current"]
            if retired:
                errors.append(f"cites retired passages {retired}")
            if not citations and obj.get("decision") != "insufficient_context":
                errors.append("a decision needs at least one citation")
    return obj, errors


# ---------------------------------------------------------------- grading


@dataclass(frozen=True)
class ConditionResult:
    readable: bool  # decision and limit could be read, so the content can be graded
    correct: bool  # readable, and decision and limit match the reference
    valid: bool  # the answer satisfies the output contract
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.correct and self.valid


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    group: str
    retrieved_ids: tuple[str, ...]
    answerable: bool  # the case has gold passages; abstention cases do not
    gold_in_context: bool  # answerable, and every gold passage is in the top k
    conditions: dict[str, ConditionResult]

    @property
    def needs_external_knowledge(self) -> bool:
        closed = self.conditions["closed_book"]
        return closed.readable and not closed.correct and self.conditions["oracle"].correct


def grade(raw: str, expected: dict, context_ids: list[str], corpus: dict[str, dict]) -> ConditionResult:
    obj, errors = parse_answer(raw, context_ids, corpus)
    readable = (
        obj is not None
        and obj.get("decision") in DECISIONS
        and "limit_eur" in obj
        and _is_limit(obj["limit_eur"])
    )
    correct = (
        readable
        and obj["decision"] == expected["decision"]
        and obj["limit_eur"] == expected["limit_eur"]
    )
    return ConditionResult(readable=readable, correct=correct, valid=not errors, errors=tuple(errors))


def evaluate_case(case: dict, corpus: dict[str, dict], k: int) -> CaseResult:
    retrieved = retrieve(case["question"], corpus, k)
    gold = case["gold_ids"]
    contexts = {
        "closed_book": [],
        "retrieved": retrieved,
        "oracle": gold,
        "oracle_fewshot": gold,  # demonstrations are not citable evidence
    }
    results = {
        name: grade(raw, case["expected"], contexts[name], corpus)
        for name, raw in case["observations"].items()
    }
    return CaseResult(
        case_id=case["id"],
        group=case["group"],
        retrieved_ids=tuple(retrieved),
        answerable=bool(gold),
        gold_in_context=bool(gold) and all(g in retrieved for g in gold),
        conditions=results,
    )


def diagnose(r: CaseResult) -> str:
    oracle = r.conditions["oracle"]
    if not oracle.readable or (oracle.correct and not oracle.valid):
        return OUTPUT_CONTRACT
    if not oracle.correct:
        fewshot = r.conditions.get("oracle_fewshot")
        if fewshot is None or not fewshot.readable:
            return GOLD_CONTEXT_FAILS
        return BEHAVIOR if fewshot.correct else CAPABILITY_SUSPECTED
    rag = r.conditions["retrieved"]
    if rag.ok:
        return PASS
    if not rag.readable or rag.correct:
        return OUTPUT_CONTRACT
    if r.answerable and not r.gold_in_context:
        return RETRIEVAL_MISS
    return CONTEXT_USE


# --------------------------------------------------------------- splitting


def grouped_split(cases: list[dict], holdout_groups: list[str]) -> tuple[list[dict], list[dict]]:
    if not isinstance(holdout_groups, list) or not holdout_groups:
        raise ValueError("holdout_groups must be a non-empty list")
    groups = {c["group"] for c in cases}
    unknown = sorted(set(holdout_groups) - groups)
    if unknown:
        raise ValueError(f"unknown holdout groups {unknown}")
    if set(holdout_groups) == groups:
        raise ValueError("every group is in holdout; nothing left for development")
    dev = [c for c in cases if c["group"] not in holdout_groups]
    holdout = [c for c in cases if c["group"] in holdout_groups]
    return dev, holdout


def leaking_groups(dev: list[dict], holdout: list[dict]) -> set[str]:
    return {c["group"] for c in dev} & {c["group"] for c in holdout}


def check_exemplars(exemplar_ids: list[str], dev: list[dict], holdout: list[dict]) -> None:
    """Few-shot demonstrations (or tuning data) may only come from dev groups."""
    dev_ids = {c["id"] for c in dev}
    holdout_by_id = {c["id"]: c["group"] for c in holdout}
    for eid in exemplar_ids:
        if eid in holdout_by_id:
            raise ValueError(f"exemplar {eid} comes from holdout group {holdout_by_id[eid]!r}")
        if eid not in dev_ids:
            raise ValueError(f"exemplar {eid} is not a known dev case")
