---
layout: post
lang: en
title: "Human-in-the-loop state machines: rollback and reconciliation after a crash"
date: 2024-09-12 09:00:00 +0200
permalink: /blog/human-in-the-loop-recovery/
author: Rafał Prońko
categories: [reliability]
description: "A human approves a device change, the worker crashes after the device commits. Durable intent, reconciliation and approved compensation in Python and SQLite."
image: /assets/images/human-in-the-loop-recovery.svg
---

# Human-in-the-loop state machines: rollback and reconciliation after a crash

![State machine: proposal, human approval bound to an exact value and version, durable intent, conditional send branching to four possible outcomes, reconciliation against the device log, and compensation as a new approved action.]({{ '/assets/images/human-in-the-loop-recovery.svg' | relative_url }})

<time datetime="2024-09-12">12 September 2024</time>

*Editorial note: the date above places this post in a series. It does not claim the post was published or the code was tested on that day. The example was written and run in October 2026; exact versions are in the verification section at the end.*

An operator approves a change: lower the heating setpoint in a zone from 20.5 °C to 19.0 °C. A worker sends the command. The device writes the new value. Then the worker process dies before it records that the device acknowledged.

After a restart the coordinator has an approved action and no idea what happened to it.

There are three tempting reactions, and all three are wrong:

- **Send a fresh command again.** A retry with a new command ID, or a generic unconditional "set value", can overwrite someone who changed the setting after the first try. Retrying the same command ID in this example is different: the device deduplicates it and returns the historical receipt.
- **Mark it failed.** The device holds 19.0 °C. The record says nothing changed.
- **Roll it back.** Rollback of what? The coordinator's database never contained the device's state.

This post builds a small state machine that treats a crash as a normal event. It runs on Python's standard library and two SQLite files. One file belongs to the coordinator, the other plays the role of the device. They never share a transaction, because a real coordinator and a real device never do.

## Three pairs of words that must not be confused

Most of the design falls out of three distinctions.

**Local transaction rollback versus compensation.** `ROLLBACK` undoes writes inside one database, before they are committed. It cannot touch anything outside that database. Once the device has committed a change, the only way back is a *new* forward action that sets the old value. That is compensation. It can fail, it can be refused, and it can itself be interrupted by a crash.

**Timeout versus failure.** A failure is a definite answer: the device said "no" and changed nothing. A timeout is the absence of an answer. The command may have been lost before it arrived, or applied a millisecond before the connection dropped. A timeout leads to `UNKNOWN`, and the next step is to find out, not to assume.

**Approved intent versus observed reality.** The coordinator's records describe what a human allowed, what the worker tried to do, and which command receipts were observed. The device's store describes what is true now. `APPLIED` proves that this command took effect at that point in history; it does not prove the device still has that value later. When the two stores disagree, the device wins. The coordinator's job after a crash is to find out which of its intents became reality.

## The model, and what is left out

The example is deliberately small:

- one reversible setting, `setpoint`, an integer in tenths of a degree;
- `device.db`: the synthetic device, a value, a version that increases on every write, and a log of applied command IDs;
- `coordinator.db`: approvals, actions with their states, and an event journal;
- **one worker**. There is no lease, no distributed lock and no fencing token. Two coordinators running against the same files at the same time are outside the model.

The states of an action:

| State | Meaning | Who moves it on |
|---|---|---|
| `INTENT` | Approved, guards passed, recorded on disk, send not confirmed | worker or reconciliation |
| `APPLIED` | The device confirmed this exact command; observed value and version are a historical receipt | compensation may be proposed after a fresh read |
| `REJECTED` | The device refused it; nothing changed | terminal |
| `UNKNOWN` | Timeout; the outcome must be checked | reconciliation |
| `NOT_APPLIED` | Reconciliation found the command did not take effect | terminal |
| `CONFLICT` | Terminal record: the command did not take effect, and the device changed anyway | fresh proposal and approval if action is still wanted |
| `COMPENSATED` | A later approved compensation was confirmed; not a guarantee of current device state | terminal historical record |

Files to download: [`hitl_db.py`]({{ '/assets/code/human-in-the-loop-recovery/hitl_db.py' | relative_url }}), [`hitl_device.py`]({{ '/assets/code/human-in-the-loop-recovery/hitl_device.py' | relative_url }}), [`hitl_coordinator.py`]({{ '/assets/code/human-in-the-loop-recovery/hitl_coordinator.py' | relative_url }}) and [`hitl_demo.py`]({{ '/assets/code/human-in-the-loop-recovery/hitl_demo.py' | relative_url }}). Put them in one directory and run `python3 hitl_demo.py` with Python 3.12. The demo uses a temporary directory and deletes it at the end.

All business transactions are explicit. The connection is opened with `isolation_level=None`, so the `sqlite3` module does not start transactions on its own. `executescript` appears only during start-up schema creation, outside the action and approval transactions; every business `BEGIN`, `COMMIT` and `ROLLBACK` in the example is visible in the code:

```python
def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("PRAGMA synchronous = FULL")
    return con


@contextmanager
def transaction(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
```

`BEGIN IMMEDIATE` takes the write lock at the start, so a check and the write that depends on it happen without another writer in between.

## The device: a conditional, deduplicated write

The device does not trust the coordinator to be up to date. It applies a command only if its own version still matches the version the command expects, and it remembers every command ID it has applied:

```python
def apply(self, command_id: str, name: str, expected_version: int,
          value: int) -> tuple[int, int]:
    """Conditional, deduplicated write. Returns the observed (value, version)."""
    with transaction(self.con):
        done = self.applied(command_id)
        if done is not None:
            return done
        _, version = self.read(name)
        if version != expected_version:
            raise VersionMismatch(f"{name}: expected v{expected_version}, is v{version}")
        self.con.execute("UPDATE setting SET value = ?, version = ? WHERE name = ?",
                         (value, version + 1, name))
        self.con.execute("INSERT INTO applied VALUES (?, ?, ?, ?)",
                         (command_id, name, value, version + 1))
        return value, version + 1
```

Two properties matter here.

The deduplication check comes first. If the same command arrives twice, the second copy returns the recorded result instead of failing on a version the first copy produced.

The applied log is what makes recovery possible at all. Without a way to ask "did command X take effect?", a timeout cannot be resolved. You could only compare values and guess. Many real devices and APIs offer nothing like this log. In that case the design has to change, and I come back to this in the limits section.

## Approval binds to one exact action

An approval in this example is not "the operator said yes". It is a row that fixes every field of the action: kind, setting, the value it expects to find, the value it will write, the device version, and an expiry time. It can be used once.

Anything may *propose* an action: a script, a scheduler, a language model. In this demo, that means using `propose_change()` or `propose_compensation()` to build a well-formed specification from the current device state. Approval is a separate step, and the guards that enforce it run on the coordinator, in code. A model that drafts a proposal never gets a permission that bypasses them.

The operator in the example is simulated. `approve()` takes a free-text name such as `"operator-1 (simulated, not authenticated)"`. There is no login, no signature and no role check. In a real system the approval would come from an authenticated identity with an audit trail. The example only shows what the approval must be *bound to*.

The guards run inside the same transaction that records the intent:

```python
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
```

Each line closes a specific gap:

- **mismatch**: an approval for 19.0 °C does not authorise 15.0 °C;
- **expired**: a human agreed to a change under the conditions of one particular moment, not forever;
- **stale**: if the device moved since the human looked at it, the human approved a situation that no longer exists;
- **unresolved**: while an earlier action on this setting has an unknown outcome, nothing new is sent.

The last check reads the device before sending, but between that read and the send someone can still change the setting. That is why the device has its own version condition. The coordinator's check gives a clear refusal early. The device's check holds when the coordinator's information is already out of date.

Expiry is checked only at intent authorization time. Once `_record_intent()` passes, the approval is immediately marked used and the action row becomes the durable authority for recovery. `_send()` does not re-check expiry, and the synthetic device does not enforce an expiry time on the command ID.

## Durable intent before the send

The intent is written to disk before anything leaves the coordinator. Guards, marking the approval as used, and inserting the action happen in one local transaction:

```python
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
```

This is where local rollback is the right tool. If a guard refuses, the transaction rolls back: no action row, approval still unused, nothing sent. A test checks exactly that for an expired approval.

The action ID doubles as the command ID sent to the device. That single identifier links the coordinator's intent with the device's applied log. Retrying that same identifier is a deduplicated lookup of the original command receipt, not a new unconditional write.

Then the send. The example injects faults at the points that matter:

```python
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
```

`VersionMismatch` and `TimeoutError` lead to different states on purpose. The first is a definite answer from the device. The second is no answer at all.

`exit_after_device_commit` is not a simulation inside one Python process. The demo and one test start a child process that calls `os._exit(70)` right after the device commits. No `finally` block runs and no in-memory state survives. The parent then opens both files with fresh objects, as a restarted service would.

The fault `ack_write_rolls_back` shows the first distinction in practice. The device commits, then the transaction that records the acknowledgement fails and rolls back. Afterwards the coordinator's database says `INTENT` and the device holds the new value. The local rollback worked as designed and did not undo anything on the device.

## Reconciliation: ask the device, then decide

On restart, every action in `INTENT` or `UNKNOWN` is compared with the device:

```python
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
```

Reconciliation never sends anything. It only turns unknown outcomes into known records. Whether to try again is a new decision, and in this model it needs a new approval: the old one is already used.

An `APPLIED` record is still a command receipt, not a current-state assertion. The observed value and version tell us what the device returned for this command. A later external write can make compensation unsafe, so compensation reads the device again before it proposes anything.

Each action is closed in its own transaction, so a crash in the middle of reconciliation leaves the remaining actions unresolved, and a second run picks them up. A test checks that a repeated run changes nothing and writes no second `reconciled` event. A crash *during* reconciliation is not tested separately; the claim rests on reconciliation being read-only towards the device and finishing each action atomically.

The recovery table:

| Coordinator says | Device log has command ID | Device version | Result | Next step |
|---|---|---|---|---|
| `INTENT` / `UNKNOWN` | yes | any current version | `APPLIED` | Record the historical receipt; optionally reread and propose compensation |
| `INTENT` / `UNKNOWN` | no | as expected | `NOT_APPLIED` | Nothing changed; a retry needs a fresh proposal and approval |
| `INTENT` / `UNKNOWN` | no | moved | `CONFLICT` | Stop this attempt. Someone else changed the device. A human decides |
| `APPLIED` | — | equals observed | compensable | New proposal, new approval |
| `APPLIED` | — | moved since | `Refused: diverged` | Do not overwrite. A human decides |

`CONFLICT`, `REJECTED`, `NOT_APPLIED` and `COMPENSATED` are terminal records, not persistent blocking interlocks. The guard that refuses new work only checks `INTENT` and `UNKNOWN`. After a terminal record, a fresh proposal and a new approval can move onward against the current device version.

`NOT_APPLIED` relies on the single-worker model. After a restart there is no other copy of the command still travelling to the device. With a real network that is not guaranteed: a delayed copy could arrive after reconciliation. The device's version condition limits the damage — a copy arriving after any other change is refused — but if nothing else has changed, the late copy would still apply. Finite polling cannot prove that no arbitrary delayed command exists. A production design would need cancellation, fencing, device-side command expiry, or a proven delivery bound; without one of those, the honest state is still unknown.

## Compensation is a new action

Undoing an applied change means proposing a new action whose expected state is what our change produced:

```python
def propose_compensation(self, action_id: str) -> ActionSpec:
    row = self._action(action_id)
    if row["kind"] != "change" or row["state"] != "APPLIED":
        raise Refused(f"not compensable in state {row['state']}")
    observed = (row["observed_value"], row["observed_version"])
    if self.device.read(row["setting"]) != observed:
        raise Refused("diverged: device changed after our action; a human decides")
    return ActionSpec("compensate", row["setting"], observed[0],
                      row["from_value"], observed[1], action_id)
```

This specification goes through the same path as any other action: a new approval, the same guards, a durable intent, a conditional send. The original approval cannot be reused. It is already marked as used, and it describes a different action anyway.

The version condition is what keeps a well-formed compensation from causing damage. Suppose someone at the wall panel sets 21.5 °C after our change. A naive "undo" would write 20.5 °C and erase a deliberate human decision. Here, proposing compensation is refused when the device has diverged. If the interference happens after the compensation is approved, the same freshness check that refuses stale change approvals refuses it; the tests exercise that check on a change, not separately on a compensation. If the interference happens between that check and the send, the device rejects the command and the external value stays. That case is tested for a compensation.

A crash during compensation is not a special case. The compensating action has its own intent and its own command ID. The demo and a test crash the worker after the device commits the compensation. After a restart, reconciliation finds the command ID in the device log, marks the compensation `APPLIED` and, in the same local transaction, marks the original change `COMPENSATED`. That is historical closure: it says a compensation command was confirmed, not that the device cannot drift again afterwards.

A test also checks that a second compensation for the same change is refused, both when the first one is still unresolved and when it has completed.

## Safe stop conditions

The system stops and waits, without sending anything, when:

1. any action on the setting is `INTENT` or `UNKNOWN`;
2. the current action was recorded as `CONFLICT`;
3. compensation is proposed and the device has diverged from what our change left behind;
4. an approval is unknown, used, mismatched, expired or stale;
5. the device rejected a command because of a version mismatch.

In each case, continuing automatically would mean acting on information that is known to be outdated or incomplete. Except for `INTENT` and `UNKNOWN`, these stop conditions do not create a permanent hold. A later action starts with a fresh proposal and a fresh approval against the device state that exists then.

## Where the human decides and where the code decides

The code decides what can be determined mechanically: whether a command took effect, whether an approval still matches reality, whether a compensation would overwrite someone else's change. It does not decide whether that change was right.

A human decides:

- whether the original change should happen at all;
- whether to retry after `NOT_APPLIED` — conditions may have changed since the first approval;
- what to do after `CONFLICT`: the external change may be correct, or it may be the problem;
- whether to compensate an applied change, and with what value if the device has diverged.

The human does not override the guards. An operator who wants to force a value approves a new action against the current device version. Nothing in the example records an intent without a matching, unexpired, unused approval at authorization time. After that point, the approval is used; the send and any later recovery are tied to the durable action ID and the device's version check.

## Runbook

After a restart or an alert about a stuck action:

1. **Do not send a fresh retry.** Look up unresolved actions: `coordinator.unresolved()`. In this example, only `INTENT` and `UNKNOWN` are unresolved.
2. **Run reconciliation.** It reads the device and never writes to it. Running it twice is safe.
3. **`APPLIED`**: this command has a historical receipt. If it should be undone, reread the device by proposing compensation and get a new approval.
4. **`NOT_APPLIED`**: nothing changed. If the change is still wanted, make a fresh proposal from the current device state and get a new approval.
5. **`CONFLICT`**: read the device's current value, find out who changed it, and decide. The record is terminal, not a persistent lock; any further change goes through a new approval against the current version.
6. **Compensation refused as `diverged`**: the same as `CONFLICT` — someone acted after us. Do not try to restore our "before" value automatically.
7. **Check the journal**: `coordinator.events(action_id)` shows intent, acknowledgement or timeout, and reconciliation for each action.

## What the tests check

The test suite has 19 cases. They cover:

1. a successful change consumes its approval, and the approval cannot be used again;
2. an intent recorded before a crash survives the restart; the device is unchanged and reconciliation says `NOT_APPLIED`;
3. a crash after the device commit is reconciled to `APPLIED`, with the observed value and version;
4. the same, with a real child process that exits through `os._exit` after the device commit;
5. a timeout leads to `UNKNOWN`, not to a failure, and is then reconciled;
6. a rolled-back acknowledgement write leaves the device change in place;
7. a new action is refused while an earlier outcome is unresolved;
8. an expired approval is refused and the refusal leaves no action and an unused approval;
9. a stale approval is refused after an external change, which stays in place;
10. an approval for one value does not authorise another value;
11. the device rejects a command after interference between the check and the send;
12. compensation needs its own approval and restores the original value;
13. compensation is refused when the device has diverged;
14. an approved compensation does not overwrite interference;
15. a crash during compensation is reconciled and the original change is marked `COMPENSATED`;
16. a completed compensation cannot be repeated;
17. a pending compensation blocks a second compensation;
18. not applied plus a moved device version gives `CONFLICT`;
19. reconciliation can be repeated without new effects.

These are behavioural tests on one machine. They do not test disk failures, concurrent workers, network partitions or a real device.

## Where this guarantee ends

**No exactly-once.** The example gives at most one *confirmed* effect per approval, because the device deduplicates by command ID and checks the version. It does not give exactly-once delivery, and nothing here should be read as a production guarantee of that kind.

**One worker.** There is no lease or fencing. A second worker started while the first is still sending could reconcile an action that is about to complete. Running more than one coordinator needs a lease with a fencing token checked by the device, or an equivalent mechanism.

**API boundary.** The demo assumes callers use the proposal methods to build `ActionSpec` values. It does not harden the lower-level `execute()` method against arbitrary hand-built inconsistent specs. In a production API, the server would derive or validate those fields again at the boundary.

**The device must cooperate.** The design relies on a device that stores applied command IDs and supports a conditional write. If a real device offers only "set value", reconciliation can compare values but cannot prove which command produced them, and the recovery table becomes a set of guesses. In that case I would make that uncertainty explicit and send more cases to a human.

**The command log is not forever.** A real device keeps a bounded history. Reconciliation has to run within that window.

**Delayed commands and expiry.** Approval expiry is enforced by the coordinator before it records intent. The synthetic device does not enforce expiry, cancellation or fencing. If an environment cannot bound delayed delivery, `NOT_APPLIED` after finite polling is not enough to prove a command will never arrive.

**Clocks.** Expiry uses the coordinator's clock, injected as a function so that tests can control it. A real system needs a clear rule about which clock decides, especially if command expiry moves to the device side.

**Storage.** SQLite makes a single transaction atomic inside one database file. How durable a commit is depends on the file system and hardware actually honouring sync requests. Two files means two independent transactions, by design. The schema includes a declarative `REFERENCES approval(id)`, but the example does not enable `PRAGMA foreign_keys = ON`; none of the article's guarantees rely on foreign-key enforcement.

**Simulated authority.** Approvals are not authenticated. The example shows binding and server-side checking, not identity.

## Verification

Run on 12 October 2026 on macOS (arm64), with environments created by uv 0.7.8:

- Python 3.12.11, SQLite library 3.47.1, pytest 8.3.2: 19 tests passed; `hitl_demo.py` finished with exit code 0.
- Python 3.12.5, SQLite library 3.46.0, pytest 8.3.2, in a separate virtual environment: 19 tests passed; the demo finished with exit code 0.

These runs happened in 2026, not on the date in the header. The `pyproject.toml` dependency group and `uv.lock` are modern reproduction tooling, not evidence that this exact workspace existed in 2024. The code uses only the standard library: `sqlite3` with `isolation_level=None`, `BEGIN IMMEDIATE`, start-up `executescript`, `sqlite3.Row`, plus `uuid`, `subprocess` and `typing.NamedTuple`. It does not use the `autocommit` connection attribute added in Python 3.12; transactions are controlled only through `isolation_level=None` and explicit statements. I ran it only on the two versions above.

### Sources

- [Python 3.12: sqlite3](https://docs.python.org/3.12/library/sqlite3.html) — `isolation_level=None` and explicit transaction control. This is a rolling documentation page; the historical 3.12.5 documentation source is tagged at [cpython v3.12.5 sqlite3.rst](https://raw.githubusercontent.com/python/cpython/v3.12.5/Doc/library/sqlite3.rst).
- [Python 3.12.5 release](https://www.python.org/downloads/release/python-3125/) — released 6 August 2024.
- [pytest 8.3.2](https://pypi.org/project/pytest/8.3.2/) — published to PyPI on 25 July 2024; the 2026 runs above used this version.
- [SQLite: Atomic Commit In SQLite](https://www.sqlite.org/atomiccommit.html) — atomicity of a transaction within one database and its dependence on storage behaviour.
- [Azure Architecture Center: Compensating Transaction pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/compensating-transaction) — compensation as a separate, possibly failing step rather than a rollback. A current conceptual reference, not a historical source.
