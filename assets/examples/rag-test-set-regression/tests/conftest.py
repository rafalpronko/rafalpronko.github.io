import copy

import pytest

CORPUS = {
    "corpus_id": "toy-docs",
    "corpus_version": "2024-05-01",
    "passages": [
        {"passage_id": "P-a", "text": "Plan A keeps logs for 400 days."},
        {"passage_id": "P-b", "text": "SSO requires the Enterprise plan."},
        {"passage_id": "P-c", "text": "SSO setup requires the owner role."},
    ],
}

DATASET = {
    "dataset_id": "toy-qa",
    "dataset_version": "1.0.0",
    "corpus_version": "2024-05-01",
    "cases": [
        {
            "case_id": "q-logs",
            "group_id": "g-logs",
            "split": "dev",
            "kind": "single",
            "critical": False,
            "question": "How long does plan A keep logs?",
            "expected_behavior": "answer",
            "required_facts": ["400 days"],
            "evidence_sets": [["P-a"]],
        },
        {
            "case_id": "q-sso",
            "group_id": "g-sso",
            "split": "holdout",
            "kind": "multi_hop",
            "critical": True,
            "question": "Which plan and role do I need for SSO?",
            "expected_behavior": "answer",
            "required_facts": ["enterprise", "owner"],
            "evidence_sets": [["P-b", "P-c"]],
        },
        {
            "case_id": "q-price",
            "group_id": "g-price",
            "split": "holdout",
            "kind": "unanswerable",
            "critical": True,
            "question": "How much does plan A cost?",
            "expected_behavior": "abstain",
            "required_facts": [],
            "evidence_sets": [],
        },
    ],
}

GOOD_RESPONSES = [
    {
        "case_id": "q-logs",
        "action": "answer",
        "answer": "Plan A keeps logs for 400 days.",
        "cited_passage_ids": ["P-a"],
        "retrieved_passage_ids": ["P-a", "P-b"],
    },
    {
        "case_id": "q-sso",
        "action": "answer",
        "answer": "You need the Enterprise plan and the owner role.",
        "cited_passage_ids": ["P-b", "P-c"],
        "retrieved_passage_ids": ["P-b", "P-c"],
    },
    {
        "case_id": "q-price",
        "action": "abstain",
        "answer": "",
        "cited_passage_ids": [],
        "retrieved_passage_ids": ["P-a"],
    },
]


def make_run(run_id, responses):
    return {
        "run_id": run_id,
        "dataset_id": "toy-qa",
        "dataset_version": "1.0.0",
        "corpus_version": "2024-05-01",
        "responses": copy.deepcopy(responses),
    }


@pytest.fixture
def corpus():
    return copy.deepcopy(CORPUS)


@pytest.fixture
def dataset():
    return copy.deepcopy(DATASET)


@pytest.fixture
def good_run():
    return make_run("base", GOOD_RESPONSES)
