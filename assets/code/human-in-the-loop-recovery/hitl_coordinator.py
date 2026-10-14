"""Coordinator: human approvals, durable intent, execution and reconciliation.

Single worker model: there is no lease or distributed lock. Two coordinators
running against the same files at the same time are outside this example.
"""
import os
import sqlite3
import uuid
from typing import NamedTuple

from hitl_db import connect, transaction
from hitl_device import VersionMismatch

SCHEMA = """
CREATE TABLE IF NOT EXISTS approval (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, setting TEXT NOT NULL,
    from_value INTEGER NOT NULL, to_value INTEGER NOT NULL,
    expected_version INTEGER NOT NULL, compensates TEXT,
    approver TEXT NOT NULL, expires_at INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS action (
    id TEXT PRIMARY KEY, approval_id TEXT NOT NULL UNIQUE REFERENCES approval(id),
    kind TEXT NOT NULL, setting TEXT NOT NULL, from_value INTEGER NOT NULL,
    to_value INTEGER NOT NULL, expected_version INTEGER NOT NULL, compensates TEXT,
    state TEXT NOT NULL, observed_value INTEGER, observed_version INTEGER);
CREATE TABLE IF NOT EXISTS event (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
    at INTEGER NOT NULL, name TEXT NOT NULL);
"""
SPEC = "kind, setting, from_value, to_value, expected_version, compensates"


class Refused(Exception):
    """A server-side guard said no. Nothing was sent to the device."""


class SimulatedCrash(Exception):
    """The worker dies at this point (in-process stand-in for a crash)."""


class ActionSpec(NamedTuple):
    kind: str  # "change" or "compensate"
    setting: str
    from_value: int
    to_value: int
    expected_version: int
    compensates: str | None = None


class Coordinator:
    def __init__(self, path, device, clock) -> None:
        self.con = connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self.device, self.clock = device, clock

    def close(self) -> None:
        self.con.close()

    # --- proposals: anyone (a script, a model) may propose; nothing runs yet
    def propose_change(self, setting: str, to_value: int) -> ActionSpec:
        value, version = self.device.read(setting)
        return ActionSpec("change", setting, value, to_value, version)

    def propose_compensation(self, action_id: str) -> ActionSpec:
        row = self._action(action_id)
        if row["kind"] != "change" or row["state"] != "APPLIED":
            raise Refused(f"not compensable in state {row['state']}")
        observed = (row["observed_value"], row["observed_version"])
        if self.device.read(row["setting"]) != observed:
            raise Refused("diverged: device changed after our action; a human decides")
        return ActionSpec("compensate", row["setting"], observed[0],
                          row["from_value"], observed[1], action_id)

    # --- approval: a simulated trusted operator; authentication is not modelled
    def approve(self, spec: ActionSpec, approver: str, ttl_seconds: int) -> str:
        approval_id = uuid.uuid4().hex
        with transaction(self.con):
            self.con.execute(
                f"INSERT INTO approval (id, {SPEC}, approver, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (approval_id, *spec, approver, self.clock() + ttl_seconds))
        return approval_id

    # --- execution
    def execute(self, spec: ActionSpec, approval_id: str, fault: str | None = None) -> str:
        action_id = self._record_intent(spec, approval_id)
        return self._send(action_id, spec, fault)

    def _record_intent(self, spec: ActionSpec, approval_id: str) -> str:
        """Guards + intent in one local transaction: all of it commits, or none."""
        action_id = uuid.uuid4().hex
        with transaction(self.con):
            self._check(spec, approval_id)
            self.con.execute("UPDATE approval SET used = 1 WHERE id = ?", (approval_id,))
            self.con.execute(
                f"INSERT INTO action (id, approval_id, {SPEC}, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'INTENT')",
                (action_id, approval_id, *spec))
            self._event(action_id, "intent_recorded")
        return action_id

    def _check(self, spec: ActionSpec, approval_id: str) -> None:
        row = self.con.execute(f"SELECT {SPEC}, expires_at, used FROM approval WHERE id = ?",
                               (approval_id,)).fetchone()
        if row is None:
            raise Refused("unknown approval")
        if row["used"]:
            raise Refused("approval already used")
        if ActionSpec(*tuple(row)[:6]) != spec:
            raise Refused("approval mismatch: a different action was approved")
        if self.clock() >= row["expires_at"]:
            raise Refused("approval expired")
        if self.unresolved(spec.setting):
            raise Refused("unresolved action on this setting: reconcile first")
        if spec.kind == "compensate" and self.state(spec.compensates) != "APPLIED":
            raise Refused("compensation target is not in state APPLIED")
        if self.device.read(spec.setting) != (spec.from_value, spec.expected_version):
            raise Refused("stale approval: device no longer matches what was approved")

    def _send(self, action_id: str, spec: ActionSpec, fault: str | None) -> str:
        if fault == "crash_before_send":
            raise SimulatedCrash("worker died after recording intent")
        if fault == "interfere_before_send":
            self.device.external_write(spec.setting, 999)
        try:
            observed = self.device.apply(action_id, spec.setting,
                                         spec.expected_version, spec.to_value)
            if fault == "crash_after_device_commit":
                raise SimulatedCrash("worker died before recording acknowledgement")
            if fault == "exit_after_device_commit":
                os._exit(70)  # a real process exit: no cleanup, no finally blocks
            if fault == "timeout_after_device_commit":
                raise TimeoutError("no acknowledgement within deadline")
        except VersionMismatch:
            self._finish(action_id, "REJECTED", event="device_rejected")
        except TimeoutError:
            self._finish(action_id, "UNKNOWN", event="timeout")
        else:
            self._finish(action_id, "APPLIED", observed, event="acknowledged",
                         fail=fault == "ack_write_rolls_back")
        return action_id

    def _finish(self, action_id, state, observed=(None, None), *, event, fail=False):
        with transaction(self.con):
            self.con.execute(
                "UPDATE action SET state = ?, observed_value = ?, observed_version = ? "
                "WHERE id = ?", (state, *observed, action_id))
            if state == "APPLIED":  # a confirmed compensation closes its target
                self.con.execute(
                    "UPDATE action SET state = 'COMPENSATED' WHERE id = "
                    "(SELECT compensates FROM action WHERE id = ?)", (action_id,))
            self._event(action_id, f"{event}:{state}")
            if fail:
                raise RuntimeError("simulated failure while recording acknowledgement")

    # --- recovery: compare durable intent with observed device reality
    def reconcile(self) -> dict[str, str]:
        outcome = {}
        for action_id in self.unresolved():
            row = self._action(action_id)
            observed = self.device.applied(action_id)
            if observed is not None:
                state = "APPLIED"
            elif self.device.read(row["setting"]) == (row["from_value"],
                                                      row["expected_version"]):
                state = "NOT_APPLIED"
            else:
                state = "CONFLICT"
            self._finish(action_id, state, observed or (None, None), event="reconciled")
            outcome[action_id] = state
        return outcome

    # --- read helpers
    def _action(self, action_id: str) -> sqlite3.Row:
        return self.con.execute("SELECT * FROM action WHERE id = ?", (action_id,)).fetchone()

    def _event(self, action_id: str, name: str) -> None:
        self.con.execute("INSERT INTO event (action_id, at, name) VALUES (?, ?, ?)",
                         (action_id, self.clock(), name))

    def state(self, action_id: str) -> str:
        return self._action(action_id)["state"]

    def observed(self, action_id: str) -> tuple[int, int]:
        row = self._action(action_id)
        return row["observed_value"], row["observed_version"]

    def unresolved(self, setting: str | None = None) -> list[str]:
        rows = self.con.execute(
            "SELECT id FROM action WHERE state IN ('INTENT', 'UNKNOWN') "
            "AND (? IS NULL OR setting = ?) ORDER BY rowid", (setting, setting))
        return [r["id"] for r in rows]

    def approval_used(self, approval_id: str) -> bool:
        return bool(self.con.execute("SELECT used FROM approval WHERE id = ?",
                                     (approval_id,)).fetchone()["used"])

    def events(self, action_id: str) -> list[str]:
        rows = self.con.execute("SELECT name FROM event WHERE action_id = ? ORDER BY seq",
                                (action_id,))
        return [r["name"] for r in rows]
