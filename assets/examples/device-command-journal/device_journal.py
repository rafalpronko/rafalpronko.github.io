"""A small SQLite command journal with read-after-write verification.

Assumptions: a single writer process, "set desired state" commands (no toggles)
for a lamp with power "on"/"off" and integer brightness 0..100, and a device
adapter that raises WriteTimeout, ReadTimeout or DeviceRejected.
"""
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fake_device import DeviceRejected, ReadTimeout, WriteTimeout

PENDING = "PENDING"                # intent committed, outcome not assessed yet
REJECTED = "REJECTED"              # the device answered "no"
OBSERVED_MATCH = "OBSERVED_MATCH"  # a fresh read showed the desired fields
NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"  # read failed or showed other values

SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    operation_id   TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    device_id      TEXT NOT NULL,
    desired        TEXT NOT NULL,
    status         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at    TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    operation_id   TEXT NOT NULL REFERENCES operations (operation_id),
    attempt_id     TEXT,
    kind           TEXT NOT NULL,
    detail         TEXT
);
"""


class PayloadConflict(Exception):
    """The operation ID is already used for a different command."""


class InvalidCommand(ValueError):
    """The command is outside the lamp contract. Nothing was recorded or sent."""


@dataclass(frozen=True)
class Operation:
    operation_id: str
    correlation_id: str
    device_id: str
    desired: dict
    status: str


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _valid_field(key, value):
    if key == "power":
        return type(value) is str and value in ("on", "off")
    if key == "brightness":
        return type(value) is int and 0 <= value <= 100  # bool is not an int here
    return False


def validate_command(operation_id, correlation_id, device_id, desired):
    for name, value in (("operation_id", operation_id), ("correlation_id", correlation_id),
                        ("device_id", device_id)):
        if not isinstance(value, str) or not value.strip():
            raise InvalidCommand(f"{name} must be a non-empty string, got {value!r}")
    if not isinstance(desired, dict) or not desired:
        raise InvalidCommand(f"desired must be a non-empty dict, got {desired!r}")
    for key, value in desired.items():
        if not _valid_field(key, value):
            raise InvalidCommand(f"unsupported field {key!r}={value!r}")


def _same(observed, desired):
    """A field matches only if it is present with the same type and value."""
    return all(key in observed and type(observed[key]) is type(value)
               and observed[key] == value for key, value in desired.items())


class CommandJournal:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)

    def record_intent(self, operation_id, correlation_id, device_id, desired):
        """Commit the intent. Return False if this exact command is already known."""
        validate_command(operation_id, correlation_id, device_id, desired)
        try:
            with self.con:
                self.con.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, ?, ?)",
                    (operation_id, correlation_id, device_id, _json(desired), PENDING))
                self._event(correlation_id, operation_id, None, "INTENT_RECORDED",
                            {"device_id": device_id, "desired": desired})
            return True
        except sqlite3.IntegrityError:
            existing = self.get(operation_id)
        if (existing.device_id, _json(existing.desired)) != (device_id, _json(desired)):
            raise PayloadConflict(
                f"{operation_id} already means {existing.device_id} {existing.desired}")
        with self.con:
            self._event(correlation_id, operation_id, None, "DUPLICATE_SUBMIT", None)
        return False

    def start_attempt(self, op):
        attempt_id = str(uuid.uuid4())
        with self.con:  # back to PENDING: a crash from here on leaves the outcome open
            self._event(op.correlation_id, op.operation_id, attempt_id, "ATTEMPT_STARTED", None)
            self._set_status(op.operation_id, PENDING)
        return attempt_id

    def record(self, op, attempt_id, kind, detail=None, status=None):
        with self.con:  # the event and the status change commit together or not at all
            self._event(op.correlation_id, op.operation_id, attempt_id, kind, detail)
            if status is not None:
                self._set_status(op.operation_id, status)

    def get(self, operation_id):
        row = self.con.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)).fetchone()
        if row is None:
            raise KeyError(operation_id)
        return Operation(row["operation_id"], row["correlation_id"], row["device_id"],
                         json.loads(row["desired"]), row["status"])

    def attempts(self, operation_id):
        rows = self.con.execute(
            "SELECT attempt_id FROM events WHERE operation_id = ? AND kind = 'ATTEMPT_STARTED'"
            " ORDER BY seq", (operation_id,))
        return [row["attempt_id"] for row in rows]

    def pending(self):
        rows = self.con.execute(
            "SELECT operation_id FROM operations WHERE status = ? ORDER BY rowid",
            (PENDING,)).fetchall()
        return [self.get(row["operation_id"]) for row in rows]

    def events(self, correlation_id):
        rows = self.con.execute(
            "SELECT seq, operation_id, attempt_id, kind, detail FROM events"
            " WHERE correlation_id = ? ORDER BY seq", (correlation_id,))
        return [{**dict(row), "detail": row["detail"] and json.loads(row["detail"])}
                for row in rows]

    def close(self):
        self.con.close()

    def _event(self, correlation_id, operation_id, attempt_id, kind, detail):
        self.con.execute(
            "INSERT INTO events (recorded_at, correlation_id, operation_id, attempt_id,"
            " kind, detail) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), correlation_id, operation_id,
             attempt_id, kind, None if detail is None else _json(detail)))

    def _set_status(self, operation_id, status):
        self.con.execute("UPDATE operations SET status = ? WHERE operation_id = ?",
                         (status, operation_id))


def submit(journal, device, *, operation_id, correlation_id, desired):
    """Record the intent, then send it once. Submitting it again never resends."""
    if not journal.record_intent(operation_id, correlation_id, device.device_id, desired):
        return journal.get(operation_id).status
    return send(journal, device, operation_id)


def send(journal, device, operation_id):
    """One attempt. Allowed for a fresh intent or after an explicit reconciliation."""
    op = journal.get(operation_id)
    fresh = op.status == PENDING and not journal.attempts(operation_id)
    if not (fresh or op.status == NEEDS_RECONCILIATION):
        raise RuntimeError(f"{operation_id} is {op.status}; refusing to send blindly")
    if op.device_id != device.device_id:
        raise ValueError(f"{operation_id} targets {op.device_id}, not {device.device_id}")

    attempt_id = journal.start_attempt(op)  # committed before any I/O
    try:
        device.set_state(op.desired, attempt_id=attempt_id)
    except DeviceRejected as exc:
        journal.record(op, attempt_id, "ATTEMPT_REJECTED", {"reason": str(exc)}, REJECTED)
        return REJECTED
    except WriteTimeout:
        journal.record(op, attempt_id, "ATTEMPT_TIMEOUT")  # unknown, not "rejected"
    else:
        journal.record(op, attempt_id, "ATTEMPT_ACKED")
    return verify(journal, device, op, attempt_id)


def verify(journal, device, op, attempt_id=None):
    """Read the state now and compare only the fields this command asked for."""
    try:
        observed = device.read_state()
    except ReadTimeout:
        journal.record(op, attempt_id, "READ_TIMEOUT", None, NEEDS_RECONCILIATION)
        return NEEDS_RECONCILIATION
    status = OBSERVED_MATCH if _same(observed, op.desired) else NEEDS_RECONCILIATION
    journal.record(op, attempt_id, "READBACK", {"observed": observed}, status)
    return status


def reconcile_pending(journal, device):
    """After a restart: read the state for open intents. Never sends anything."""
    results = {}
    for op in journal.pending():
        if op.device_id == device.device_id:
            journal.record(op, None, "RESTART_RECONCILE")
            results[op.operation_id] = verify(journal, device, op)
    return results
