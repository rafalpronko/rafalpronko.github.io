import json
import subprocess
import sys
from pathlib import Path

import pytest

from rageval.__main__ import main

DATA = Path(__file__).resolve().parent.parent / "data"


def args(**overrides):
    paths = {
        "corpus": DATA / "corpus.json",
        "dataset": DATA / "dataset.json",
        "baseline": DATA / "baseline.json",
        "candidate": DATA / "candidate.json",
        "policy": DATA / "gate_policy.json",
    }
    paths.update(overrides)
    return [f"--{name}={path}" for name, path in paths.items()]


def test_demo_data_fails_gate_with_exit_code_1(capsys):
    assert main(args()) == 1
    out = capsys.readouterr().out
    assert "overall" in out and "[CRITICAL]" in out and "GATE: FAIL" in out


def test_baseline_against_itself_passes(capsys):
    assert main(args(candidate=DATA / "baseline.json")) == 0


def test_candidate_against_itself_still_fails_absolute_critical_rule(capsys):
    assert main(args(baseline=DATA / "candidate.json")) == 1
    out = capsys.readouterr().out
    assert "critical failures in candidate: 1" in out
    assert "Unchanged failures (both runs fail):\n  [CRITICAL] q-sso-plan-role" in out


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def demo(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_corpus_version_mismatch_exits_with_code_2(tmp_path, capsys):
    run = dict(demo("candidate.json"), corpus_version="2024-04-01")
    assert main(args(candidate=write(tmp_path, "candidate.json", run))) == 2
    assert "corpus_version" in capsys.readouterr().err


def test_missing_input_file_exits_with_code_2(tmp_path, capsys):
    assert main(args(candidate=tmp_path / "absent.json")) == 2
    assert "invalid input: cannot read" in capsys.readouterr().err


def test_directory_as_input_exits_with_code_2(tmp_path, capsys):
    assert main(args(policy=tmp_path)) == 2
    assert "invalid input: cannot read" in capsys.readouterr().err


def test_empty_evaluation_exits_with_code_2(tmp_path, capsys):
    dataset = dict(demo("dataset.json"), cases=[])
    runs = {name: dict(demo(name), responses=[]) for name in ("baseline.json", "candidate.json")}
    code = main(args(dataset=write(tmp_path, "dataset.json", dataset),
                     baseline=write(tmp_path, "baseline.json", runs["baseline.json"]),
                     candidate=write(tmp_path, "candidate.json", runs["candidate.json"])))
    assert code == 2
    assert "cases must be a non-empty list" in capsys.readouterr().err


def test_boolean_thresholds_cannot_turn_the_demo_into_a_pass(tmp_path, capsys):
    policy = dict(demo("gate_policy.json"), max_critical_failures=True, max_critical_regressions=True)
    assert main(args(policy=write(tmp_path, "policy.json", policy))) == 2
    assert "max_critical_failures must be an integer >= 0" in capsys.readouterr().err


def test_nan_threshold_exits_with_code_2(tmp_path, capsys):
    policy = dict(demo("gate_policy.json"), max_slice_drop=float("nan"))
    assert main(args(policy=write(tmp_path, "policy.json", policy))) == 2
    assert "non-finite number NaN" in capsys.readouterr().err


def test_null_cases_exit_with_code_2_not_a_traceback(tmp_path, capsys):
    dataset = dict(demo("dataset.json"), cases=None)
    assert main(args(dataset=write(tmp_path, "dataset.json", dataset))) == 2
    assert "cases must be a non-empty list" in capsys.readouterr().err


def raw_policy(key, raw_number):
    """The demo policy with one value replaced by raw JSON number text."""
    text = (DATA / "gate_policy.json").read_text(encoding="utf-8")
    old = f'"{key}": {json.dumps(demo("gate_policy.json")[key])}'
    assert old in text
    return text.replace(old, f'"{key}": {raw_number}')


@pytest.mark.parametrize("key, raw_number, message", [
    pytest.param("max_slice_drop", "1" + "0" * 309,
                 "max_slice_drop must be a finite number in [0, 1]", id="drop-10**309"),
    pytest.param("max_slice_drop", "-1" + "0" * 309,
                 "max_slice_drop must be a finite number in [0, 1]", id="drop--10**309"),
    pytest.param("max_regressions", "9" * 4301,
                 "invalid JSON: Exceeds the limit (4300 digits)", id="count-4301-digits"),
])
def test_huge_json_integers_exit_with_code_2_in_a_real_process(tmp_path, key, raw_number, message):
    policy = tmp_path / "policy.json"
    policy.write_text(raw_policy(key, raw_number), encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "rageval", *args(policy=policy)],
                         cwd=DATA.parent, capture_output=True, text=True)
    assert (run.returncode, run.stdout) == (2, "")
    assert run.stderr.startswith("invalid input: ") and message in run.stderr
    assert "Traceback" not in run.stderr
