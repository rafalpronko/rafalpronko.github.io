"""Checkpointed four-step workflow in one SQLite file (synthetic demo).

Assumption: exactly one worker process uses a given state file at a time.
There is no lease or fencing token; see the article for why that matters.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import receiver

WORKFLOW_VERSION = "triage/1"
SCHEMA_VERSION = 1
STEPS = ("normalize", "classify", "summarize", "notify")
CRASH_EXIT = 77
SCHEMA = """
CREATE TABLE run (
    run_id TEXT PRIMARY KEY, workflow_version TEXT NOT NULL,
    input_digest TEXT NOT NULL, cursor INTEGER NOT NULL);
CREATE TABLE step_result (
    run_id TEXT NOT NULL, step INTEGER NOT NULL, name TEXT NOT NULL,
    output TEXT NOT NULL, PRIMARY KEY (run_id, step));
CREATE TABLE operation (
    operation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
    step INTEGER NOT NULL, request_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'acknowledged')),
    receipt TEXT);
"""
LABELS = {"billing": ("refund", "charge", "invoice", "vat"),
          "access": ("log in", "login", "password")}


class IncompatibleCheckpoint(Exception):
    """Stored state was written for different code, schema or input."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def input_digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def maybe_crash(point: str, step: int) -> None:
    """Test hook: die without cleanup, like kill -9 or an OOM kill."""
    if os.environ.get("CRASH_AT") == f"{point}:{STEPS[step]}":
        sys.stdout.flush()
        os._exit(CRASH_EXIT)


@contextmanager
def transaction(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


def ensure_schema(con: sqlite3.Connection) -> None:
    version = con.execute("PRAGMA user_version").fetchone()[0]
    tables = con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
    if version == 0 and tables == 0:
        with transaction(con):
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    con.execute(statement)
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif version != SCHEMA_VERSION:
        raise IncompatibleCheckpoint(
            f"schema version {version}, expected {SCHEMA_VERSION}")


def open_run(con: sqlite3.Connection, run_id: str, tickets) -> int:
    """Create the run or check that the checkpoint belongs to this input."""
    digest = input_digest(tickets)
    with transaction(con):
        row = con.execute("SELECT workflow_version, input_digest, cursor "
                          "FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            con.execute("INSERT INTO run VALUES (?, ?, ?, 0)",
                        (run_id, WORKFLOW_VERSION, digest))
            return 0
    if row[0] != WORKFLOW_VERSION:
        raise IncompatibleCheckpoint(
            f"workflow version {row[0]!r}, running {WORKFLOW_VERSION!r}")
    if row[1] != digest:
        raise IncompatibleCheckpoint("input digest differs from checkpoint")
    return row[2]


def load_outputs(con: sqlite3.Connection, run_id: str) -> dict:
    rows = con.execute("SELECT name, output FROM step_result "
                       "WHERE run_id = ?", (run_id,))
    return {name: json.loads(output) for name, output in rows}


def normalize(tickets, outputs):
    return [{"id": t["id"], "text": " ".join(t["text"].lower().split())}
            for t in sorted(tickets, key=lambda t: t["id"])]


def classify(tickets, outputs):
    def label(text):
        hits = [name for name, words in LABELS.items()
                if any(word in text for word in words)]
        return hits[0] if hits else "other"
    return [{"id": t["id"], "label": label(t["text"])}
            for t in outputs["normalize"]]


def summarize(tickets, outputs):
    counts = {}
    for item in outputs["classify"]:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    return dict(sorted(counts.items()))


PURE_STEPS = {"normalize": normalize, "classify": classify,
              "summarize": summarize}


def commit_step(con, run_id: str, step: int, output, ack=None) -> None:
    """Store the step output, any ack and the cursor in ONE transaction."""
    with transaction(con):
        con.execute("INSERT INTO step_result VALUES (?, ?, ?, ?)",
                    (run_id, step, STEPS[step], canonical(output)))
        if ack is not None:
            con.execute("UPDATE operation SET status = 'acknowledged', "
                        "receipt = ? WHERE operation_id = ?",
                        (ack[1], ack[0]))
        moved = con.execute("UPDATE run SET cursor = ? WHERE run_id = ? "
                            "AND cursor = ?", (step + 1, run_id, step))
        if moved.rowcount != 1:
            raise RuntimeError("cursor moved under us; single-worker "
                               "assumption violated")
        maybe_crash("before_commit", step)
    maybe_crash("after_commit", step)


def record_intent(con, run_id: str, step: int, payload: str) -> str:
    """Persist operation identity and request before any external call."""
    operation_id = f"{run_id}:{step}:{STEPS[step]}"
    request = input_digest(payload)
    with transaction(con):
        con.execute("INSERT INTO operation VALUES (?, ?, ?, ?, 'pending', "
                    "NULL) ON CONFLICT (operation_id) DO NOTHING",
                    (operation_id, run_id, step, request))
        stored = con.execute("SELECT request_digest FROM operation "
                             "WHERE operation_id = ?",
                             (operation_id,)).fetchone()[0]
    if stored != request:
        raise IncompatibleCheckpoint(f"request changed for {operation_id}")
    return operation_id


def notify(con, run_id: str, step: int, outputs, receiver_path: str):
    payload = canonical({"run_id": run_id, "summary": outputs["summarize"]})
    operation_id = record_intent(con, run_id, step, payload)
    maybe_crash("after_intent", step)
    # Outcome unknown from here until commit_step: a crash leaves 'pending'.
    receipt = receiver.deliver(receiver_path, operation_id, payload)
    commit_step(con, run_id, step, {"receipt": receipt},
                ack=(operation_id, receipt))


def run(con, run_id: str, tickets, receiver_path: str) -> None:
    cursor = open_run(con, run_id, tickets)
    while cursor < len(STEPS):
        name = STEPS[cursor]
        print(f"exec step={name}", flush=True)
        outputs = load_outputs(con, run_id)
        if name == "notify":
            notify(con, run_id, cursor, outputs, receiver_path)
        else:
            output = PURE_STEPS[name](tickets, outputs)
            commit_step(con, run_id, cursor, output)
        cursor += 1
    print(f"done run={run_id} cursor={cursor}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--db", "--receiver", "--input", "--run-id"):
        parser.add_argument(flag, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tickets = json.loads(Path(args.input).read_text(encoding="utf-8"))
    con = sqlite3.connect(args.db, isolation_level=None)
    try:
        ensure_schema(con)
        run(con, args.run_id, tickets, args.receiver)
    except IncompatibleCheckpoint as exc:
        print(f"refusing to resume: {exc}", file=sys.stderr)
        return 3
    except sqlite3.DatabaseError as exc:
        print(f"storage error, not starting over: {exc}", file=sys.stderr)
        return 4
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
