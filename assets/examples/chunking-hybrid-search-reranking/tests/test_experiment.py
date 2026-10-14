from retrieval_lab.experiment import run_experiment


def test_grid_is_complete_and_uses_fixed_depth_and_k():
    report = run_experiment(depth=5, k=3)
    assert report["settings"] == {"depth": 5, "k": 3, "bm25_k1": 1.2, "bm25_b": 0.75, "rrf_c": 60}
    names = {(r["chunking"], r["method"]) for r in report["runs"]}
    assert len(names) == 10
    for run in report["runs"]:
        for q in run["queries"]:
            assert len(q["top_k"]) <= 3 and len(q["candidates"]) <= 5


def test_reranked_top_k_is_drawn_from_candidates():
    for run in run_experiment(depth=5, k=3)["runs"]:
        for q in run["queries"]:
            cands = {c["chunk_id"] for c in q["candidates"]}
            assert {c["chunk_id"] for c in q["top_k"]} <= cands


def test_run_is_deterministic():
    assert run_experiment(5, 3) == run_experiment(5, 3)


def test_supplementary_budget_run_changes_only_depth_and_k():
    report = run_experiment(depth=10, k=6)
    assert report["settings"]["depth"] == 10 and report["settings"]["k"] == 6
    assert all(len(q["top_k"]) <= 6 for r in report["runs"] for q in r["queries"])
