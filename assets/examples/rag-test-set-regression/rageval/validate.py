BEHAVIORS = {"answer", "abstain", "clarify"}
SPLITS = {"dev", "holdout"}
CASE_KEYS = {
    "case_id", "group_id", "split", "kind", "critical", "question",
    "expected_behavior", "required_facts", "evidence_sets",
}
CASE_STRINGS = ("case_id", "group_id", "split", "kind", "question", "expected_behavior")
RESPONSE_KEYS = {
    "case_id", "action", "answer", "cited_passage_ids", "retrieved_passage_ids",
}
VERSION_FIELDS = ("dataset_id", "dataset_version", "corpus_version")


class ValidationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _check_keys(obj, expected, where):
    _require(isinstance(obj, dict), f"{where}: expected an object")
    missing, extra = sorted(expected - set(obj)), sorted(set(obj) - expected)
    _require(not missing and not extra, f"{where}: missing keys {missing}, extra keys {extra}")


def _check_string(value, where):
    # Type before membership: a list or dict here would otherwise raise TypeError
    # in `value in SOME_SET` and escape as a crash instead of invalid input.
    _require(isinstance(value, str) and value.strip(), f"{where} must be a non-empty string")


def _check_strings(value, where):
    ok = isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
    _require(ok, f"{where} must be a list of non-empty strings")


def _check_items(obj, key, where):
    items = obj[key]
    _require(isinstance(items, list) and items, f"{where}: {key} must be a non-empty list")
    return items


def _unique_ids(items, key, label):
    seen = set()
    for item in items:
        _require(item[key] not in seen, f"duplicate {label} {item[key]!r}")
        seen.add(item[key])
    return seen


def validate_corpus(corpus):
    """Return the set of passage IDs after checking they are unique."""
    _check_keys(corpus, {"corpus_id", "corpus_version", "passages"}, "corpus")
    for key in ("corpus_id", "corpus_version"):
        _check_string(corpus[key], f"corpus: {key}")
    for passage in _check_items(corpus, "passages", "corpus"):
        _check_keys(passage, {"passage_id", "text"}, "passage")
        _check_string(passage["passage_id"], "passage: passage_id")
        _check_string(passage["text"], f"passage {passage['passage_id']!r}: text")
    return _unique_ids(corpus["passages"], "passage_id", "passage_id")


def _name(obj, key):
    return obj.get(key) if isinstance(obj, dict) else obj


def _validate_case(case, passage_ids):
    _check_keys(case, CASE_KEYS, f"case {_name(case, 'case_id')!r}")
    for key in CASE_STRINGS:
        _check_string(case[key], f"case {case['case_id']!r}: {key}")
    cid = case["case_id"]
    _require(case["split"] in SPLITS, f"{cid}: split {case['split']!r} not in {sorted(SPLITS)}")
    _require(case["expected_behavior"] in BEHAVIORS, f"{cid}: unknown expected_behavior")
    _require(isinstance(case["critical"], bool), f"{cid}: critical must be true/false")
    _check_strings(case["required_facts"], f"{cid}: required_facts")
    _require(isinstance(case["evidence_sets"], list), f"{cid}: evidence_sets must be a list")
    answers = case["expected_behavior"] == "answer"
    has_refs = bool(case["required_facts"]) and bool(case["evidence_sets"])
    _require(has_refs if answers else not (case["required_facts"] or case["evidence_sets"]),
             f"{cid}: answer cases need facts and evidence; others must not have them")
    for evidence_set in case["evidence_sets"]:
        _check_strings(evidence_set, f"{cid}: evidence set")
        _require(evidence_set, f"{cid}: empty evidence set")
        # A repeated ID would count twice in the recall denominator, but once in the numerator.
        for i, pid in enumerate(evidence_set):
            _require(pid in passage_ids, f"{cid}: unknown passage {pid!r}")
            _require(pid not in evidence_set[:i], f"{cid}: duplicate passage {pid!r} in evidence set")


def _check_group_isolation(cases):
    split_of = {}
    for case in cases:
        first = split_of.setdefault(case["group_id"], case["split"])
        _require(first == case["split"],
                 f"group {case['group_id']!r} appears in both dev and holdout")


def validate_dataset(dataset, corpus):
    """Check the dataset against the corpus it was written for; return its cases."""
    _check_keys(dataset, {"dataset_id", "dataset_version", "corpus_version", "cases"}, "dataset")
    for key in VERSION_FIELDS:
        _check_string(dataset[key], f"dataset: {key}")
    passage_ids = validate_corpus(corpus)
    _require(dataset["corpus_version"] == corpus["corpus_version"],
             f"dataset corpus_version {dataset['corpus_version']!r} != corpus {corpus['corpus_version']!r}")
    cases = _check_items(dataset, "cases", "dataset")
    for case in cases:
        _validate_case(case, passage_ids)
    _unique_ids(cases, "case_id", "case_id")
    _check_group_isolation(cases)
    return cases


def _validate_response(response, passage_ids):
    _check_keys(response, RESPONSE_KEYS, f"response {_name(response, 'case_id')!r}")
    for key in ("case_id", "action"):
        _check_string(response[key], f"response {response['case_id']!r}: {key}")
    cid = response["case_id"]
    _require(response["action"] in BEHAVIORS, f"{cid}: action {response['action']!r} not allowed")
    _require(isinstance(response["answer"], str), f"{cid}: answer must be a string")
    for key in ("cited_passage_ids", "retrieved_passage_ids"):
        _check_strings(response[key], f"{cid}: {key}")
    for pid in response["cited_passage_ids"] + response["retrieved_passage_ids"]:
        _require(pid in passage_ids, f"{cid}: unknown passage {pid!r}")


def validate_run(run, dataset, passage_ids):
    """Return responses keyed by case_id; every dataset case exactly once."""
    _check_keys(run, {"run_id", *VERSION_FIELDS, "responses"}, "run")
    _check_string(run["run_id"], "run: run_id")
    for field in VERSION_FIELDS:
        _require(run[field] == dataset[field],
                 f"run {run['run_id']}: {field} {run[field]!r} != dataset {dataset[field]!r}")
    by_id = {}
    for response in _check_items(run, "responses", f"run {run['run_id']}"):
        _validate_response(response, passage_ids)
        _require(response["case_id"] not in by_id, f"duplicate response for {response['case_id']!r}")
        by_id[response["case_id"]] = response
    _check_coverage(run["run_id"], {case["case_id"] for case in dataset["cases"]}, set(by_id))
    return by_id


def _check_coverage(run_id, expected, actual):
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    _require(not missing, f"run {run_id}: missing responses: {', '.join(missing)}")
    _require(not extra, f"run {run_id}: unexpected responses: {', '.join(extra)}")
