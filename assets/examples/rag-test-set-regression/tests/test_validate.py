import pytest

from rageval.load import load_json
from rageval.validate import (
    ValidationError,
    validate_corpus,
    validate_dataset,
    validate_run,
)


def test_load_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"run_id": "a", "run_id": "b"}', encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate key 'run_id'"):
        load_json(path)


def test_load_json_rejects_malformed_json(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"run_id": ', encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid JSON"):
        load_json(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_json_rejects_non_finite_constants(tmp_path, constant):
    path = tmp_path / "policy.json"
    path.write_text(f'{{"max_slice_drop": {constant}}}', encoding="utf-8")
    with pytest.raises(ValidationError, match=f"non-finite number {constant}"):
        load_json(path)


def test_load_json_rejects_integer_beyond_digit_limit(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(f'{{"max_regressions": {"9" * 4301}}}', encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid JSON: .*4300 digits"):
        load_json(path)


def test_load_json_rejects_too_deeply_nested_json(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid JSON: nesting too deep"):
        load_json(path)


def test_load_json_reports_missing_file_as_invalid_input(tmp_path):
    with pytest.raises(ValidationError, match="cannot read"):
        load_json(tmp_path / "absent.json")


def test_load_json_reports_unreadable_path_as_invalid_input(tmp_path):
    with pytest.raises(ValidationError, match="cannot read"):
        load_json(tmp_path)


def test_valid_inputs_pass(corpus, dataset, good_run):
    passage_ids = validate_corpus(corpus)
    cases = validate_dataset(dataset, corpus)
    responses = validate_run(good_run, dataset, passage_ids)
    assert [c["case_id"] for c in cases] == ["q-logs", "q-sso", "q-price"]
    assert set(responses) == {"q-logs", "q-sso", "q-price"}


def test_dataset_rejects_duplicate_case_ids(corpus, dataset):
    dataset["cases"].append(dict(dataset["cases"][0]))
    with pytest.raises(ValidationError, match="duplicate case_id 'q-logs'"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_unknown_evidence_passage(corpus, dataset):
    dataset["cases"][0]["evidence_sets"] = [["P-missing"]]
    with pytest.raises(ValidationError, match="unknown passage 'P-missing'"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_duplicate_passage_in_evidence_set(corpus, dataset):
    # ["P-a", "P-a"] would make evidence recall 0.5 for a complete retrieval.
    dataset["cases"][0]["evidence_sets"] = [["P-a", "P-a"]]
    with pytest.raises(ValidationError, match="q-logs: duplicate passage 'P-a' in evidence set"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_corpus_version_mismatch(corpus, dataset):
    dataset["corpus_version"] = "2024-04-01"
    with pytest.raises(ValidationError, match="corpus_version"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_group_in_both_splits(corpus, dataset):
    leaked = dict(dataset["cases"][0], case_id="q-logs-para", split="holdout")
    dataset["cases"].append(leaked)
    with pytest.raises(ValidationError, match="group 'g-logs' appears in both"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_case_that_is_not_an_object(corpus, dataset):
    dataset["cases"].append("q-orphan")
    with pytest.raises(ValidationError, match="expected an object"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_facts_given_as_string(corpus, dataset):
    dataset["cases"][0]["required_facts"] = "400 days"
    with pytest.raises(ValidationError, match="q-logs: required_facts must be a list"):
        validate_dataset(dataset, corpus)


def test_dataset_rejects_abstain_case_with_evidence(corpus, dataset):
    dataset["cases"][2]["evidence_sets"] = [["P-a"]]
    with pytest.raises(ValidationError, match="q-price.*must not have"):
        validate_dataset(dataset, corpus)


def test_run_rejects_missing_case(corpus, dataset, good_run):
    good_run["responses"].pop()
    with pytest.raises(ValidationError, match="missing responses: q-price"):
        validate_run(good_run, dataset, validate_corpus(corpus))


def test_run_rejects_extra_case(corpus, dataset, good_run):
    good_run["responses"].append(dict(good_run["responses"][0], case_id="q-new"))
    with pytest.raises(ValidationError, match="unexpected responses: q-new"):
        validate_run(good_run, dataset, validate_corpus(corpus))


def test_run_rejects_duplicate_response(corpus, dataset, good_run):
    good_run["responses"].append(dict(good_run["responses"][0]))
    with pytest.raises(ValidationError, match="duplicate response for 'q-logs'"):
        validate_run(good_run, dataset, validate_corpus(corpus))


def test_run_rejects_dataset_version_mismatch(corpus, dataset, good_run):
    good_run["dataset_version"] = "0.9.0"
    with pytest.raises(ValidationError, match="dataset_version"):
        validate_run(good_run, dataset, validate_corpus(corpus))


def test_run_rejects_unknown_action(corpus, dataset, good_run):
    good_run["responses"][0]["action"] = "guess"
    with pytest.raises(ValidationError, match="action 'guess'"):
        validate_run(good_run, dataset, validate_corpus(corpus))


@pytest.mark.parametrize("cases", [[], None, {}, "q-logs"])
def test_dataset_rejects_empty_or_non_list_cases(corpus, dataset, cases):
    dataset["cases"] = cases
    with pytest.raises(ValidationError, match="dataset: cases must be a non-empty list"):
        validate_dataset(dataset, corpus)


@pytest.mark.parametrize("passages", [[], None, {}])
def test_corpus_rejects_empty_or_non_list_passages(corpus, passages):
    corpus["passages"] = passages
    with pytest.raises(ValidationError, match="corpus: passages must be a non-empty list"):
        validate_corpus(corpus)


@pytest.mark.parametrize("passage_id", [None, ["P-a"], ""])
def test_corpus_rejects_non_string_passage_id(corpus, passage_id):
    corpus["passages"][0]["passage_id"] = passage_id
    with pytest.raises(ValidationError, match="passage_id must be a non-empty string"):
        validate_corpus(corpus)


@pytest.mark.parametrize("responses", [[], None, {}])
def test_run_rejects_empty_or_non_list_responses(corpus, dataset, good_run, responses):
    good_run["responses"] = responses
    with pytest.raises(ValidationError, match="responses must be a non-empty list"):
        validate_run(good_run, dataset, validate_corpus(corpus))


@pytest.mark.parametrize("field", ["case_id", "group_id", "split", "kind",
                                   "question", "expected_behavior"])
@pytest.mark.parametrize("value", [None, ["x"], {"x": 1}, 3, ""])
def test_dataset_rejects_non_string_case_fields(corpus, dataset, field, value):
    dataset["cases"][0][field] = value
    with pytest.raises(ValidationError, match=f"{field} must be a non-empty string"):
        validate_dataset(dataset, corpus)


@pytest.mark.parametrize("field", ["case_id", "action"])
@pytest.mark.parametrize("value", [None, ["answer"], {"answer": 1}, ""])
def test_run_rejects_non_string_response_fields(corpus, dataset, good_run, field, value):
    good_run["responses"][0][field] = value
    with pytest.raises(ValidationError, match=f"{field} must be a non-empty string"):
        validate_run(good_run, dataset, validate_corpus(corpus))
