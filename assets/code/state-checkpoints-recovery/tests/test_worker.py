"""Crash/restart tests. Every crash is a real os._exit in a child process."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TICKETS = [
    {"id": "T-3", "text": "  Refund for DOUBLE charge  "},
    {"id": "T-1", "text": "Cannot log in after password reset"},
    {"id": "T-2", "text": "Invoice shows the wrong VAT number"},
]
STEPS = ["normalize", "classify", "summarize", "notify"]
CRASH_EXIT = 77


def write_input(tmp_path, tickets=TICKETS):
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps(tickets), encoding="utf-8")
    return path


def run_worker(tmp_path, crash=None):
    env = {k: v for k, v in os.environ.items() if k != "CRASH_AT"}
    if crash:
        env["CRASH_AT"] = crash
    cmd = [sys.executable, str(ROOT / "worker.py"),
           "--db", str(tmp_path / "state.db"),
           "--receiver", str(tmp_path / "receiver.db"),
           "--input", str(tmp_path / "tickets.json"),
           "--run-id", "run-1"]
    return subprocess.run(cmd, capture_output=True, text=True,
                          env=env, timeout=60)


def executed(proc):
    return [line.split("=", 1)[1] for line in proc.stdout.splitlines()
            if line.startswith("exec step=")]


def query(path, sql):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def state(tmp_path):
    db = tmp_path / "state.db"
    return {
        "cursor": query(db, "SELECT cursor FROM run")[0][0],
        "outputs": dict(query(db, "SELECT name, output FROM step_result")),
        "operations": query(db, "SELECT operation_id, status, receipt "
                                "FROM operation"),
    }


def deliveries(tmp_path):
    return query(tmp_path / "receiver.db",
                 "SELECT operation_id, attempts, receipt FROM delivery")


@pytest.fixture(scope="module")
def clean_state(tmp_path_factory):
    path = tmp_path_factory.mktemp("clean")
    write_input(path)
    proc = run_worker(path)
    assert proc.returncode == 0, proc.stderr
    return state(path), deliveries(path)


def test_clean_run_executes_each_step_once(tmp_path):
    write_input(tmp_path)
    proc = run_worker(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert executed(proc) == STEPS
    current = state(tmp_path)
    assert current["cursor"] == len(STEPS)
    assert json.loads(current["outputs"]["summarize"]) == {
        "access": 1, "billing": 2}
    assert [(op, status) for op, status, _ in current["operations"]] == [
        ("run-1:3:notify", "acknowledged")]
    assert [(op, n) for op, n, _ in deliveries(tmp_path)] == [
        ("run-1:3:notify", 1)]


def test_restart_after_completion_does_nothing(tmp_path):
    write_input(tmp_path)
    assert run_worker(tmp_path).returncode == 0
    again = run_worker(tmp_path)
    assert again.returncode == 0, again.stderr
    assert executed(again) == []
    assert [n for _, n, _ in deliveries(tmp_path)] == [1]


def test_crash_before_commit_rolls_back_and_reexecutes(tmp_path):
    write_input(tmp_path)
    crashed = run_worker(tmp_path, "before_commit:classify")
    assert crashed.returncode == CRASH_EXIT
    assert executed(crashed) == ["normalize", "classify"]
    assert state(tmp_path)["cursor"] == 1
    assert "classify" not in state(tmp_path)["outputs"]
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert executed(resumed) == ["classify", "summarize", "notify"]


def test_crash_after_commit_resumes_at_next_step(tmp_path):
    write_input(tmp_path)
    crashed = run_worker(tmp_path, "after_commit:classify")
    assert crashed.returncode == CRASH_EXIT
    assert state(tmp_path)["cursor"] == 2
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert executed(resumed) == ["summarize", "notify"]


def test_lost_ack_replays_same_operation_id(tmp_path):
    write_input(tmp_path)
    crashed = run_worker(tmp_path, "before_commit:notify")
    assert crashed.returncode == CRASH_EXIT
    assert state(tmp_path)["operations"] == [
        ("run-1:3:notify", "pending", None)]
    assert [n for _, n, _ in deliveries(tmp_path)] == [1]
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    [(op_id, attempts, receipt)] = deliveries(tmp_path)
    assert (op_id, attempts) == ("run-1:3:notify", 2)
    assert state(tmp_path)["operations"] == [
        ("run-1:3:notify", "acknowledged", receipt)]


def test_crash_after_final_commit_sends_nothing_more(tmp_path):
    write_input(tmp_path)
    assert run_worker(tmp_path, "after_commit:notify").returncode == 77
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert executed(resumed) == []
    assert [n for _, n, _ in deliveries(tmp_path)] == [1]


def test_intent_committed_but_not_sent_is_sent_on_restart(tmp_path):
    write_input(tmp_path)
    crashed = run_worker(tmp_path, "after_intent:notify")
    assert crashed.returncode == CRASH_EXIT
    assert not (tmp_path / "receiver.db").exists() or not deliveries(
        tmp_path)
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert [n for _, n, _ in deliveries(tmp_path)] == [1]


CRASH_POINTS = ([f"{when}:{step}" for step in STEPS
                 for when in ("before_commit", "after_commit")]
                + ["after_intent:notify"])


@pytest.mark.parametrize("crash", CRASH_POINTS)
def test_any_single_crash_converges_to_clean_result(
        tmp_path, crash, clean_state):
    write_input(tmp_path)
    assert run_worker(tmp_path, crash).returncode == CRASH_EXIT
    assert run_worker(tmp_path).returncode == 0
    clean, clean_deliveries = clean_state
    assert state(tmp_path) == clean
    assert [(op, r) for op, _, r in deliveries(tmp_path)] == [
        (op, r) for op, _, r in clean_deliveries]


def test_changed_input_fails_closed(tmp_path):
    write_input(tmp_path)
    assert run_worker(tmp_path, "after_commit:normalize").returncode == 77
    before = (tmp_path / "state.db").read_bytes()
    write_input(tmp_path, TICKETS + [{"id": "T-4", "text": "new"}])
    proc = run_worker(tmp_path)
    assert proc.returncode == 3
    assert "input digest" in proc.stderr
    assert executed(proc) == []
    assert (tmp_path / "state.db").read_bytes() == before


@pytest.mark.parametrize("tamper, message", [
    ("UPDATE run SET workflow_version = 'triage/0'", "workflow version"),
    ("PRAGMA user_version = 99", "schema version"),
])
def test_incompatible_checkpoint_fails_closed(tmp_path, tamper, message):
    write_input(tmp_path)
    assert run_worker(tmp_path, "after_commit:normalize").returncode == 77
    con = sqlite3.connect(tmp_path / "state.db")
    con.execute(tamper)
    con.commit()
    con.close()
    proc = run_worker(tmp_path)
    assert proc.returncode == 3
    assert message in proc.stderr
    assert executed(proc) == []


def test_corrupt_state_file_fails_closed(tmp_path):
    write_input(tmp_path)
    garbage = b"this is not an sqlite database" * 200
    (tmp_path / "state.db").write_bytes(garbage)
    proc = run_worker(tmp_path)
    assert proc.returncode == 4
    assert executed(proc) == []
    assert (tmp_path / "state.db").read_bytes() == garbage


def test_input_digest_ignores_key_order():
    import worker
    reordered = [{"text": t["text"], "id": t["id"]} for t in TICKETS]
    assert worker.input_digest(TICKETS) == worker.input_digest(reordered)
    assert worker.input_digest(TICKETS) != worker.input_digest(TICKETS[:2])
