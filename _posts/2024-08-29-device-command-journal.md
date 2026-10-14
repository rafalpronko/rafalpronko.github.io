---
layout: post
lang: en
title: "After the timeout: a command journal, correlation IDs and read-after-write on a simulated device"
date: 2024-08-29 09:00:00 +0200
permalink: /blog/device-command-journal/
author: Rafał Prońko
categories: [reliability]
description: "A small SQLite command journal in Python: operation, attempt and correlation IDs, intent committed before I/O, read-after-write and restart reconciliation on a synthetic device."
image: /assets/images/device-command-journal.svg
---

# After the timeout: a command journal, correlation IDs and read-after-write on a simulated device

![Diagram of a command journal: four locally committed steps, a terminal rejection at step three, readback only after an acknowledgement or timeout, and a restart rule that reads pending intents without resending them. The synthetic device is outside the SQLite transaction.]({{ '/assets/images/device-command-journal.svg' | relative_url }})

<time datetime="2024-08-29">29 August 2024</time>

**Editorial and reproduction note:** 29 August 2024 is a proposed editorial label, not a historical publication date. This synthetic example was prepared and verified on **12 October 2026**, using CPython 3.11.12, SQLite 3.47.1, pytest 8.3.2 and uv 0.7.8. Primary references were retrieved on that date too. The downloadable `uv.lock` was generated with modern uv for this reproduction. The example and its execution record describe a prepared simulation, not a personal deployment or incident involving the author.

A program sends a command to a device: “set the power to on, brightness to 100”. The call times out.

What do I know at this point? Less than it seems. The command may never have reached the device. It may have been applied, and only the acknowledgement got lost. The device may also have applied something slightly different from what I asked for.

If my only record is a log line that says `timeout`, the next step is a guess. Retry, and maybe apply the command twice. Give up, and maybe leave the device in a state nobody intended.

The [earlier article about idempotent pipelines]({{ '/blog/pipeline-idempotencja/' | relative_url }}) stops at effects outside the database: a transaction rollback does not undo an email or a payment. A device write is the same kind of effect. This example builds the smallest thing that answers three questions after a failure:

1. What was I asked to do?
2. What did I try to send, and when?
3. What did I observe afterwards?

The example uses Python 3.11, the standard library, a file-based SQLite journal and a **purely synthetic fake device**. No real hardware, no network, no vendor protocol. The point is the bookkeeping, not a device integration.

## Three identifiers, three different jobs

I use three identifiers. They are easy to mix up, so I'll name them precisely.

**Correlation ID** groups everything that happened because of one request. A single request may produce several commands — for example, “switch on two lamps”. When I look into the journal later, the correlation ID lets me pull out all related events in order. It does not deduplicate anything.

**Operation ID** identifies one desired command: *this* device, *this* desired state. It is the local deduplication key. If the same operation arrives again with the same payload, I don't send it again. If the same operation ID arrives with a **different** payload, that is not a duplicate. It is a contract violation, and I reject it explicitly instead of silently keeping either version.

**Attempt ID** identifies one send. Every time the code calls the device, it generates a new one. One operation can have several attempts, but only after an explicit decision — more on that below.

For attempt IDs I use `uuid.uuid4()`. The Python 3.11 `uuid` documentation describes `uuid4()` as creating a random UUID and warns that `uuid1()` may compromise privacy, because it contains the computer's network address. A random identifier is all I need here.

## Set the desired state, don't toggle

The commands in this example are **desired-state sets**: “power should be `on`, brightness should be `100`”. I deliberately don't support toggles.

The difference matters as soon as something is repeated:

- Two identical sets leave the device in the same state as one set.
- Two toggles cancel each other out.

There is a second reason. After a set, I know exactly what to look for when I read the state back: the fields I asked for and their values. After a toggle that timed out, the expected state depends on how many times the toggle was applied — which is exactly what I don't know.

A set is not automatically safe to repeat in every context. The world may have moved on since the intent was recorded. But it is a command whose expected result I can describe, and that is the minimum for read-after-write.

## The device is not part of my transaction

SQLite gives me atomic transactions. Its documentation states that all changes within a single transaction either occur completely or not at all, even if writing them to disk is interrupted by a program crash, an operating system crash or a power failure.

That guarantee covers the database file. It does not cover anything my code does while the transaction is open.

It is tempting to write this (a fragment of `demo.py`):

```python
with con:
    con.execute("INSERT INTO note VALUES ('sent')")
    device.set_state(ON, attempt_id="a-1")
    raise RuntimeError("failure before COMMIT")
```

If something fails before the commit, the `INSERT` is rolled back. The device write is not. After the block, the journal says nothing happened, and the device is on. The demo below runs exactly this and prints both sides.

So I do the opposite. I never try to put the device inside the transaction. Instead, I **persist my uncertainty before the I/O**:

1. Commit the intent: operation ID, correlation ID, device, desired state, status `PENDING`.
2. Commit the start of an attempt with a fresh attempt ID. The status stays `PENDING`.

   Then call the device. This happens outside any local transaction.
3. Commit what the device answered: acknowledged, timed out, or rejected. Rejection is final and ends here.
4. After an acknowledgement or timeout, read the device state, compare it with the desired fields, and commit the result.

Each numbered step is its own small transaction. Between any two of them, the process can die. That is acceptable as long as the journal never shows less uncertainty than there really is: before every device call, it already says “I might have sent this”.

In Python 3.11's `sqlite3`, with the default `isolation_level`, an `INSERT` or `UPDATE` implicitly opens a transaction. Using the connection as a context manager (`with con:`) commits when the block succeeds and rolls back when it raises. It does not close the connection. That is what every journal method relies on.

## The fake device

The device is a short Python class with a programmable list of faults. It models no real product. Its `applied` list records what the device actually did — the tests check this list and the device state, not just the journal.

[`fake_device.py`]({{ '/assets/examples/device-command-journal/fake_device.py' | relative_url }}):

```python
"""A purely synthetic device simulator. It models no real product or protocol."""


class WriteTimeout(Exception):
    """No answer to a write. The write may or may not have been applied."""


class ReadTimeout(Exception):
    """No answer to a read. Nothing is known about the current state."""


class DeviceRejected(Exception):
    """This simulator's contract: a rejected write is not applied."""


class SimulatedCrash(BaseException):
    """Stands in for the process dying. Journal code does not catch it."""


class FakeDevice:
    def __init__(self, device_id, state, *, write_faults=(), read_faults=()):
        self.device_id = device_id
        self._state = dict(state)
        self.write_faults = list(write_faults)  # consumed one per set_state()
        self.read_faults = list(read_faults)    # consumed one per read_state()
        self.applied = []  # (attempt_id, fields) the device actually applied
        self.reads = 0

    def set_state(self, desired, *, attempt_id):
        fault = self.write_faults.pop(0) if self.write_faults else None
        if fault == "reject":
            raise DeviceRejected("value outside the simulated allowed range")
        if fault == "crash_before_send":
            raise SimulatedCrash("process died before the write left")
        if fault == "timeout_before_apply":
            raise WriteTimeout("write lost on the way to the device")
        applied = dict(desired)
        if fault == "clamp" and "brightness" in applied:
            applied["brightness"] = min(applied["brightness"], 80)
        self._state.update(applied)
        self.applied.append((attempt_id, applied))
        if fault == "timeout_after_apply":
            raise WriteTimeout("write applied, acknowledgement lost")
        if fault == "crash_after_apply":
            raise SimulatedCrash("process died after the device applied the write")

    def toggle(self, field):
        self._state[field] = "off" if self._state[field] == "on" else "on"
        self.applied.append((None, {field: self._state[field]}))

    def read_state(self):
        self.reads += 1
        fault = self.read_faults.pop(0) if self.read_faults else None
        if fault == "timeout":
            raise ReadTimeout("no answer to the state read")
        return dict(self._state)
```

Three exceptions deserve a comment.

`WriteTimeout` means “no answer”. In the simulator, it is raised either before or after the write is applied. The calling code cannot tell which — just like in the real situation it imitates.

`DeviceRejected` is defined by this simulator's contract: a rejected write is not applied. For a real device, that meaning has to come from its documentation, not from my assumptions. And under no circumstances do I treat a timeout as a rejection.

`SimulatedCrash` derives from `BaseException`, so the journal code doesn't handle it, and even a broad `except Exception` wouldn't catch it. It stands in for the process disappearing in the middle of the flow.

## The journal

[`device_journal.py`]({{ '/assets/examples/device-command-journal/device_journal.py' | relative_url }}):

```python
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
```

A few decisions are worth explaining.

**The lamp command has a narrow, validated contract.** `desired` must be a nonempty dictionary containing `power`, `brightness`, or both. `power` is the string `"on"` or `"off"`; `brightness` is an integer from 0 through 100. Booleans, floats, null values and unknown fields are rejected. Operation, correlation and device IDs must be nonblank strings. `InvalidCommand`, a `ValueError` subclass, is raised before inserting any operation or event and before device I/O. In particular, `{"brightness": True}` is invalid even though Python considers `True == 1`.

**The operation ID is a primary key.** `record_intent` validates first, then tries to insert. If the primary key already exists, it loads the stored operation and compares the device and canonical JSON payload. The same valid command is logged as `DUPLICATE_SUBMIT`, and nothing is sent. A different valid command raises `PayloadConflict`. An invalid payload raises `InvalidCommand` even when its operation ID already exists. The primary key prevents two rows with the same operation ID, and the stored command is never updated afterwards.

**The attempt start is committed before the device call.** One of the tests opens a separate SQLite connection from inside the fake device's `set_state` and checks that it can already see the `PENDING` operation and the `ATTEMPT_STARTED` row. A second connection sees only committed data, so this is a direct check of “committed before I/O”.

**`start_attempt` sets the status back to `PENDING`.** This matters for an explicit resend from `NEEDS_RECONCILIATION`. If the process dies during that resend, the operation must look unresolved again, not like a finished case that is safe to send once more.

**`send` refuses to send blindly.** It allows exactly two cases: a fresh intent with no attempts, or an operation that a previous read marked as `NEEDS_RECONCILIATION`. A `PENDING` operation with an attempt already started has an unknown outcome. Sending again would be a guess, so it raises.

**`verify` requires each requested field to be present with exactly the same type and value.** A missing `brightness`, a null, `True` or `1.0` cannot match the requested integer `1`. The comparison covers only the command's fields; the device may report additional fields. Invalid observations produce `NEEDS_RECONCILIATION`, not a matching read.

**Status names describe observations, not causes.** `OBSERVED_MATCH` means: a fresh read showed the desired values. It doesn't say that this write produced them. The device may already have been in that state, or something else may have changed it. In this example there is a single writer, which makes the match easy to interpret, but the status name doesn't claim more than the read showed. Likewise, the read returns what the simulated device reports about itself. Nothing here checks a physical effect, a sensor or the world outside the program.

**`reconcile_pending` never sends.** After a restart it reads the state for each `PENDING` operation and records either `OBSERVED_MATCH` or `NEEDS_RECONCILIATION`. Anything beyond that is a decision for a person or for a separate, explicit policy.

## Running it

The complete example, including tests, is available as downloadable files:

- [`pyproject.toml`]({{ '/assets/examples/device-command-journal/pyproject.toml' | relative_url }}) and [`uv.lock`]({{ '/assets/examples/device-command-journal/uv.lock' | relative_url }})
- [`fake_device.py`]({{ '/assets/examples/device-command-journal/fake_device.py' | relative_url }}), [`device_journal.py`]({{ '/assets/examples/device-command-journal/device_journal.py' | relative_url }}), [`demo.py`]({{ '/assets/examples/device-command-journal/demo.py' | relative_url }})
- [`tests/test_device_journal.py`]({{ '/assets/examples/device-command-journal/tests/test_device_journal.py' | relative_url }})

The runtime uses only the standard library. pytest is a development dependency. The prepared example was verified with uv 0.7.8 on 12 October 2026, using an isolated `.venv`. The commands below are for that modern reproduction:

```bash
uv sync --locked --python 3.11
uv run --no-sync python demo.py
uv run --no-sync python -m pytest
```

The configuration uses `[tool.uv] dev-dependencies`, present in the [uv 0.3.0 schema](https://github.com/astral-sh/uv/blob/0.3.0/uv.schema.json), and `package = false`, documented in the [uv 0.4.0 release of 28 August 2024](https://github.com/astral-sh/uv/releases/tag/0.4.0). Those settings support the proposed editorial date. The included lockfile was generated with uv 0.7.8 on 12 October 2026; it is a modern reproduction artifact and was not tested with a 2024 uv binary. `exclude-newer` limits package upload dates, not the age of the interpreter, SQLite library or lockfile format.

`demo.py` deletes an existing `demo-journal.db` in the current directory and creates a new one on every run. Run it in the downloaded example's own directory; use a different path for any journal you want to keep. The following output was observed while verifying the synthetic example on 12 October 2026 with CPython 3.11.12 and SQLite 3.47.1. Attempt IDs are random, so yours will differ:

```text
== single attempts ==
success                      status=OBSERVED_MATCH        device writes=1
same op, same payload        status=OBSERVED_MATCH        device writes=1
same op, other payload       PayloadConflict: op-1 already means sim-lamp-1 {'brightness': 100, 'power': 'on'}
applied, then timeout        status=OBSERVED_MATCH        device writes=1
timeout before apply         status=NEEDS_RECONCILIATION  device writes=0
rejected                     status=REJECTED              device writes=0
acked, read timeout          status=NEEDS_RECONCILIATION  device writes=1
acked, different state       status=NEEDS_RECONCILIATION  device writes=1

== explicit resend of op-3 after reconciliation ==
status: OBSERVED_MATCH | device writes: 1
   10 INTENT_RECORDED   attempt=-        {'desired': {'brightness': 100, 'power': 'on'}, 'device_id': 'sim-lamp-1'}
   11 ATTEMPT_STARTED   attempt=01247c9b 
   12 ATTEMPT_TIMEOUT   attempt=01247c9b 
   13 READBACK          attempt=01247c9b {'observed': {'brightness': 0, 'power': 'off'}}
   25 ATTEMPT_STARTED   attempt=739c39db 
   26 ATTEMPT_ACKED     attempt=739c39db 
   27 READBACK          attempt=739c39db {'observed': {'brightness': 100, 'power': 'on'}}

== process dies after the device applied the write ==
crash: process died after the device applied the write
pending after restart: ['op-7']
send refused: op-7 is PENDING; refusing to send blindly
reconcile: {'op-7': 'OBSERVED_MATCH'} | device writes: 1

== a local rollback does not undo the device write ==
journal rows: 0 | device state: {'power': 'on', 'brightness': 100}

== retrying a toggle versus retrying a set ==
toggle twice: off | set twice: on
```

Let me read it the way I'd read a journal after an incident.

- **success**: one device write, a read showing the desired values, `OBSERVED_MATCH`.
- **same op, same payload**: still one device write. The duplicate is logged, not sent.
- **same op, other payload**: rejected with `PayloadConflict` before anything reaches the device.
- **applied, then timeout**: the device applied the write and the acknowledgement was lost. The journal records `ATTEMPT_TIMEOUT`, not `ATTEMPT_ACKED`, and the read shows the desired state. The status is `OBSERVED_MATCH` — an observation, not a claim about what the timeout meant.
- **timeout before apply**: same exception for the caller, zero device writes, a read showing `off`, status `NEEDS_RECONCILIATION`. The two timeout cases look identical from the call site; only the read tells them apart.
- **rejected**: zero device writes, status `REJECTED`. No read is needed, because in this simulator's contract a rejection means “not applied”.
- **acked, read timeout**: the write went through, but the read did not confirm it. `NEEDS_RECONCILIATION`, not success.
- **acked, different state**: the device acknowledged, but reports brightness 80 instead of 100. An acknowledgement is not a read.

The event list for `op-3` shows the attempt IDs at work: the first attempt timed out and the read showed `off`; the explicit resend got a new attempt ID, an acknowledgement and a matching read. Event numbers are global for the journal file, which is why they jump from 13 to 25.

## Crash windows and what a restart sees

The interesting failures happen between the numbered steps. Here is each window, what the journal contains and what the device may have done.

| Process dies… | Journal after restart | Device may have | Restart action |
|---|---|---|---|
| before step 1 commits | nothing | nothing (not sent) | caller may resubmit the same operation ID |
| after step 1, before step 2 | `PENDING`, no attempts | nothing (not sent) | read state, record result |
| after step 2, during or after the call | `PENDING`, `ATTEMPT_STARTED` without an answer | applied or not | read state, record result |
| after step 3, before step 4 (ACKED or TIMEOUT only) | `PENDING`, answer stored | applied (`ATTEMPT_ACKED`); unknown (`ATTEMPT_TIMEOUT`) | read state, record result |
| after step 4 | final status | — | nothing |

The `REJECTED` path ends at step 3: the rejection and the final status are committed together.

The first row is why a caller that never got an answer should repeat its request with the same operation ID. If the first copy did reach the journal, the repeat is recognised as a duplicate and returns the stored status instead of sending again. If it didn't, the repeat is simply the first recorded intent.

In the other rows, `pending()` returns the operation and `reconcile_pending()` reads the state without sending anything. In rows three and four there is also an attempt without a final outcome, so `send()` refuses to touch the operation. In row two nothing has been sent yet; an explicit `send()` would be allowed, but the restart path still doesn't do it on its own.

The demo shows this with an in-process `SimulatedCrash`. After the “crash”, the journal is closed and opened again from the same file, the way a restarted process would open it. The pending list contains `op-7`, `send` refuses, reconciliation records `OBSERVED_MATCH`, and the device still has exactly one write.

An exception is a polite crash. To check the journal side with an actual process exit, one test starts a child Python process that submits a command to a fake device that calls `os._exit(70)` right after applying the write. According to the Python documentation, `os._exit()` ends the process without calling cleanup handlers or flushing stdio buffers. The parent process then opens the journal file and finds the operation `PENDING`, with exactly two events: `INTENT_RECORDED` and `ATTEMPT_STARTED`.

That test does not simulate an operating system crash or a power failure. For those, I rely on SQLite's documented transaction guarantees for committed data, not on a simulated process-exit test.

## Why not resend automatically?

After a restart, the tempting move is “it's a set, sets are idempotent, just send it again”. I don't do that, for three reasons.

First, the intent may be stale. It was recorded before the crash; it is being executed after an unknown delay. Whether the command still makes sense is a separate question, and a journal shouldn't answer it implicitly.

Second, idempotency is a property of the device's semantics, not of my code. HTTP gives a useful vocabulary here. RFC 9110 (June 2022) calls a method idempotent when the intended effect of several identical requests is the same as for one, and says such requests can be repeated automatically after a communication failure. My fake device is not HTTP and I'm not claiming it follows these rules. But the definition shows what I would need to know about a real device before automatic retries became a reasonable policy.

Third, even a successful write doesn't guarantee a later read sees it. In RFC 9110's description of `PUT`, a successful request suggests that a later `GET` would return an equivalent representation, but there is no guarantee, because other clients or the server itself may act on the resource in between. That is the reason a read is recorded as an observation and nothing more.

So the journal stops at `NEEDS_RECONCILIATION`. A new attempt is an explicit call to `send()`, and it gets its own attempt ID. If that attempt also dies halfway, the operation is `PENDING` again and the cycle repeats: read first, decide second.

## What the tests check

The test file has **22 test functions, producing 60 test cases** after parametrization. Both the example and a fresh isolated copy of the downloadable files passed all 60 cases on 12 October 2026. They cover:

1. success: one device write, a fresh read and `OBSERVED_MATCH`;
2. the same operation ID with the same payload is not sent again;
3. the same operation ID with a different payload or a different device raises `PayloadConflict`, the original device gets no new write and the stored command stays unchanged;
4. a write applied before a timeout is confirmed by the read;
5. a timeout without a write ends in `NEEDS_RECONCILIATION`, not `REJECTED`;
6. a rejection leaves the device unchanged and does not trigger a read;
7. a read timeout after an acknowledgement ends in `NEEDS_RECONCILIATION`;
8. an acknowledged write that produced different values ends in `NEEDS_RECONCILIATION`;
9. the intent and the attempt start are visible to a separate connection *during* the device call;
10. a crash after the write leaves a `PENDING` intent, a restart refuses to send and reconciliation doesn't write to the device;
11. a crash before the send ends in `NEEDS_RECONCILIATION` after a restart, with no device write;
12. a real child-process exit after the write leaves a `PENDING` intent in the file;
13. an explicit resend gets a new attempt ID, and the device records that ID;
14. a crash during an explicit resend makes the operation `PENDING` again and blocks another send;
15. the correlation ID groups events of several operations, and only those;
16. a local rollback doesn't undo a device write;
17. a repeated toggle undoes itself, a repeated set does not.

The correction regressions add these cases:

18. 13 invalid desired payloads are refused before recording an operation or event, consuming a device fault or reading state;
19. empty, whitespace-only and nonstring values for each of the three IDs are refused before recording or I/O (18 cases);
20. reusing an integer-brightness operation with boolean brightness raises `InvalidCommand`, without a duplicate event, new write or changed stored command;
21. absent, null, boolean and float observations do not match a requested integer after a timeout before application (5 cases);
22. supported fields and brightness boundaries match even when the device reports additional, unrequested fields (6 cases).

Most assertions look at `device.applied` and `device.read_state()`, not only at the journal. A journal that says the right thing while the device did something else would be the worst outcome here.

The preparation evidence records 16 original tests failing against stubs, then passing against the implementation. The crash-during-resend test came afterwards and was checked by a mutation that removed the status reset from `start_attempt`; that test, and only that test, failed. Five other deliberate changes also broke tests. These are the original 17-case suite's mutation results, not a claim that mutations were rerun on the corrected version.

The command-validation and readback regressions were recorded separately: 19 failures against the original implementation with an exception-class stub, then 38 passing cases after correction. An intermediate GREEN attempt failed because a test compared `sqlite3.Row` with a tuple; that failed log was retained and the assertion corrected. The final expansion produced 3 failures and 57 passes before whitespace-only IDs were rejected, followed by **60 passes**. These execution records describe the preparation of this synthetic example, not the author's personal project history.

## Where this guarantee ends

This is a small model with explicit assumptions. Outside them, the conclusions don't carry over.

- **One writer.** Nothing coordinates two processes writing to the same journal or the same device. `send()` checks the status and starts an attempt in two separate steps, so two processes could both pass the check and both send. Another writer would also make `OBSERVED_MATCH` weaker evidence about the journal's own command.
- **A synchronous, fresh read.** The fake device returns its current state immediately. Real devices may report cached values, update their state with a delay, or need time to settle. A read taken too early can produce a false mismatch; a cached read can produce a false match.
- **Reported state, not physical effect.** The read tells me what the device reports. It doesn't tell me that a lamp actually emits light or that a valve physically moved. That would need an independent measurement, which this example doesn't have.
- **No exactly-once delivery.** The operation ID deduplicates locally, in my journal. The fake device receives the attempt ID but doesn't deduplicate on it. An explicit resend after an ambiguous attempt can apply the command twice; for a set, that ends in the same state, for other commands it may not.
- **Rejection is a contract.** “Rejected means not applied” is true in the simulator because I defined it that way.
- **A crash in tests is not a power cut.** One test kills a process; the others raise an exception. Durability against power loss comes from SQLite's documented guarantees and the storage underneath, which these tests do not exercise.
- **Timestamps are for humans.** `recorded_at` uses the local clock. The order of events comes from `seq`, not from time.
- **No performance claims.** No performance measurements were made. Four small commits per successful command is a deliberate trade for clarity, not a tuned design.

## Where I start

Not with a retry count.

First I decide what a command means: a desired state I can describe, or an action whose effect depends on history. Then I give it an identity that survives a restart and reject reuse of that identity for anything else. I write down the intent and the attempt before the device call, because after the call I may not get another chance.

Only then does a read-after-write make sense, and only as what it is: a fresh observation, recorded next to the command, that tells me whether I can stop or whether someone has to look.

### Sources

- [SQLite: “SQLite Is Transactional”](https://www.sqlite.org/transactional.html) — atomicity of a single transaction even if writing is interrupted by a program crash, an operating system crash or a power failure.
- [SQLite: Transaction](https://www.sqlite.org/lang_transaction.html) — implicit and explicit transactions, one write transaction at a time.
- [Python 3.11: sqlite3](https://docs.python.org/3.11/library/sqlite3.html) — implicit transactions with the default `isolation_level`, the connection context manager, `executescript()` committing pending transactions.
- [Python 3.11: uuid](https://docs.python.org/3.11/library/uuid.html) — `uuid4()` as a random UUID, the privacy note on `uuid1()`.
- [Python 3.11: os](https://docs.python.org/3.11/library/os.html#os._exit) — `os._exit()` exits without calling cleanup handlers or flushing stdio buffers.
- [Python 3.11.0 release page](https://www.python.org/downloads/release/python-3110/) — release date 24 October 2022.
- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110), sections 9.2.2 (idempotent methods) and 9.3.4 (`PUT`) — used only as a vocabulary for idempotency and observability; the fake device is not HTTP.

- [Astral: uv project tooling announcement, 20 August 2024](https://astral.sh/blog/uv-unified-python-packaging) — dated primary evidence for `uv run`, `uv lock` and `uv sync`.
- [uv 0.3.0 schema](https://github.com/astral-sh/uv/blob/0.3.0/uv.schema.json) — development dependencies and package-date constraints.
- [uv 0.4.0 release](https://github.com/astral-sh/uv/releases/tag/0.4.0) — 28 August 2024 release documenting `package = false`.
