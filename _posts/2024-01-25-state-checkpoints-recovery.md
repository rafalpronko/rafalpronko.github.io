---
layout: post
lang: en
title: "State, checkpoints and recovery: what must survive a restart"
date: 2024-01-25 09:00:00 +0100
permalink: /blog/state-checkpoints-recovery/
author: Rafał Prońko
categories: [agents]
description: "What an agent workflow must persist to resume safely: cursor, step results and operation journal committed in one SQLite transaction, tested with real process crashes."
image: /assets/images/state-checkpoints-recovery.svg
---

# State, checkpoints and recovery: what must survive a restart

![Diagram: a workflow step commits its result and cursor in one SQLite transaction; a crash before COMMIT re-runs the step, a crash after COMMIT resumes at the next one; a step with an external effect records a pending operation, sends it with a stable id and only then commits the receipt.]({{ '/assets/images/state-checkpoints-recovery.svg' | relative_url }})

<time datetime="2024-01-25">25 January 2024</time>

> **About the date and the code.** The date above is an editorial label that places this article in the series. It is not a claim that the text or the code existed on that day. The example was written and executed in October 2026, on Python 3.11.12 linked against SQLite 3.47.1, and uses only standard library features documented for Python 3.11 (released 24 October 2022) and SQLite syntax available since SQLite 3.24.0 (June 2018). Python 3.11's `sqlite3` module accepts SQLite 3.7.15 or newer, so the SQLite library your Python is linked against must itself be 3.24.0 or newer; `sqlite3.sqlite_version` shows which one you have. The `uv` commands are present-day reproduction tooling; I don't claim they were available in January 2024.

A workflow that runs for more than a few seconds will be interrupted. A deploy replaces the container, the kernel kills the process for using too much memory, someone presses Ctrl-C, a laptop goes to sleep. An agent that calls a model several times and then a few tools is exactly this kind of workflow.

The interesting question isn't whether the process dies. It's what the *next* process finds on disk, and whether that is enough to continue without repeating external effects or using a different basis from the one the earlier steps used.

I'll answer it with a small, synthetic, single-worker example: Python 3.11, the standard library, one SQLite file, and tests that really kill the process with `os._exit()` at chosen points and start it again.

## Four things that get called "state"

When people say "the agent's state", they often mean four different things. They need different treatment, so I separate them first.

| Thing | What it answers | Source of truth for "what has been done"? |
|---|---|---|
| **Workflow state** | Which run is this, at which step, on which input, with which code version? | Yes, together with the journal |
| **Chat transcript** | What did the model and the tools say to each other? | No |
| **Checkpoint** | What durable snapshot can the next process resume from? | Yes, at the moment it was committed |
| **Operation journal** | Which operations with external effects were started, with what identity and input, and what came back? | Yes, for those operations |

The transcript deserves a separate remark, because it is tempting to treat it as the state. It records that the model *asked* for a tool call. It doesn't prove that the call ran, that it succeeded, or that its result was stored. Feeding the transcript back into a model after a restart is not guaranteed to reproduce the earlier decisions; I return to that below. A transcript is useful as context and as an audit record. It is a poor answer to "may I send this email again?".

The checkpoint and the journal are where the actual decisions live. In the example they sit in the same SQLite file and, crucially, change in the same transaction.

## The example: a four-step ticket triage

The input is a JSON list of three synthetic support tickets. The workflow has a fixed, finite list of steps:

1. `normalize` — sort tickets by id, lowercase and collapse whitespace.
2. `classify` — assign a label with keyword rules. In an agent this would be a model call; I use rules so that the tests are deterministic and free.
3. `summarize` — count labels.
4. `notify` — send the summary to a "receiver", which stands in for an external system.

The position in the workflow is an integer **cursor**: 0 before `normalize`, 4 when the run is finished. The step list doesn't depend on any output, so the cursor alone tells the next process what to do. That is a deliberate simplification. In an agent where the model chooses the next tool, the cursor is replaced by the list of committed decisions, and the next decision is read from storage, not recomputed.

The receiver opens its own connection to a second SQLite file with a single table keyed by operation id. What keeps its commits out of the worker's checkpoint transaction is that independent connection, not the second file. One connection can `ATTACH` several database files and modify them in one transaction; SQLite's atomic commit document describes how that commit is made atomic across files in rollback-journal mode, the default, and notes that settings such as `PRAGMA synchronous=OFF` or `journal_mode=MEMORY` skip the step that protects it against power loss. I don't rely on that guarantee for other journal modes such as WAL. The demo deliberately uses independent connections, because the receiver stands in for an external API that no local transaction can include.

**One explicit assumption: exactly one worker process uses a given state file at a time.** There is no lease and no fencing token. I come back to what breaks without that assumption.

The full source is small enough to read in one sitting:
[`worker.py`]({{ '/assets/code/state-checkpoints-recovery/worker.py' | relative_url }}),
[`receiver.py`]({{ '/assets/code/state-checkpoints-recovery/receiver.py' | relative_url }}),
[`tests/test_worker.py`]({{ '/assets/code/state-checkpoints-recovery/tests/test_worker.py' | relative_url }}),
[`demo.sh`]({{ '/assets/code/state-checkpoints-recovery/demo.sh' | relative_url }}),
[`pyproject.toml`]({{ '/assets/code/state-checkpoints-recovery/pyproject.toml' | relative_url }}),
[`uv.lock`]({{ '/assets/code/state-checkpoints-recovery/uv.lock' | relative_url }}) and
[`README.md`]({{ '/assets/code/state-checkpoints-recovery/README.md' | relative_url }}).
The listings below are excerpts from those files.

## What the state file contains

```sql
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
```

- `run` is the workflow state: the cursor, plus what the cursor is *relative to* — the workflow version and a SHA-256 digest of the canonical JSON input.
- `step_result` stores every committed step output. A resumed process reads earlier outputs from here; it never recomputes them.
- `operation` is the journal for effects outside the file: a stable operation id, a digest of the exact request, a status and the receipt.

The storage schema has its own version in SQLite's `PRAGMA user_version`. That is a separate thing from the workflow version, and both are checked.

## One step, one transaction

Python's `sqlite3` module opens transactions implicitly by default. The Python 3.11 documentation describes the alternative: with `isolation_level=None`, the module opens no transactions on its own, and you control them with explicit SQL. I use that, because I want to see exactly where each transaction begins and which kind of `BEGIN` it uses.

```python
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

`BEGIN IMMEDIATE` starts a write transaction immediately instead of upgrading a read transaction at the first write. According to the SQLite documentation it can fail with `SQLITE_BUSY` if another connection already holds a write transaction. With a single worker that shouldn't happen; if it does, the worker exits with an error instead of half-writing something.

The central function stores the step output, an optional acknowledgement and the new cursor together:

```python
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
```

There are two design decisions here.

The first is that the result and the cursor can't disagree. If they were written in two transactions, a crash between them would leave either a result whose step will run again, or a cursor that points past a step whose result is missing. In one transaction there are only two possible local states after a crash: the step's result and the advanced cursor are both committed, or neither is. That is a statement about the stored record, not about the work. A step can compute its output, or cause an effect outside the file, and still die before `COMMIT`; the record then says the step didn't happen. For a pure step that only means it runs again. For an external effect it is the gap the operation journal below has to cover.

The second is the condition `AND cursor = ?`. The update only succeeds if the cursor is still where this process thinks it is. That catches some forms of misuse, but it is not fencing: by the time this check fails, a second worker may already have done the step's work. It's a guard rail for the single-worker assumption, not a replacement for it.

`maybe_crash` is the test hook. If the environment variable `CRASH_AT` names this point, it calls `os._exit(77)`. The Python documentation describes `os._exit()` as exiting the process "without calling cleanup handlers, flushing stdio buffers, etc." Unlike `sys.exit()`, it raises no exception, so no `finally` block or context manager exit runs either. The open transaction is neither committed nor rolled back by Python; the process just disappears, which is what an OOM kill or `kill -9` looks like to the database.

The run loop reads earlier outputs from storage before each step:

```python
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
```

## Crash before COMMIT, crash after COMMIT

Here is the beginning of the demo script's output. The worker is killed inside the `classify` transaction, after its writes and before `COMMIT`:

```text
== 1. crash inside the classify transaction, before COMMIT
exec step=normalize
exec step=classify
exit=77
files: run-a.db run-a.db-journal tickets.json 
[(1,)]
[('normalize',)]
== restart
exec step=classify
exec step=summarize
exec step=notify
done run=run-a cursor=4
exit=0
```

After the crash the cursor is still 1 and only `normalize` has a stored result. The restart runs `classify` again and finishes the run.

Note the `run-a.db-journal` file. SQLite in its default rollback-journal mode writes the original content of pages to the journal before changing them in the database file. The SQLite documentation on atomic commit explains that a journal left behind by a crashed writer can be a "hot journal", which the next connection plays back to restore the database before using it. Whether a journal is actually hot depends on what had already been written before the crash; I don't inspect that. The tests check the observable outcome instead: after any single crash and one restart, the state is identical to that of a run without crashes.

When the crash happens right after `COMMIT`, the restart starts with the next step. The test asserts that `classify` is not executed again:

```python
def test_crash_after_commit_resumes_at_next_step(tmp_path):
    write_input(tmp_path)
    crashed = run_worker(tmp_path, "after_commit:classify")
    assert crashed.returncode == CRASH_EXIT
    assert state(tmp_path)["cursor"] == 2
    resumed = run_worker(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert executed(resumed) == ["summarize", "notify"]
```

For a deterministic step, running it twice is harmless. For a model call it is not quite the same, and that's worth spelling out. If the process dies before `COMMIT`, the first model answer was never stored and nothing downstream saw it. Asking again is fine — but you may get a different answer. If the process dies after `COMMIT`, you must not ask again: later steps already depend on the stored answer. "Resume" means reading committed outputs, not replaying the conversation and hoping the model says the same thing. I don't rely on any model API returning identical output for identical input, even with sampling settings that suggest it.

## The step with an external effect

In this example, `notify` spans independent commits: the receiver opens its own connection, outside the worker's transaction. Attaching the receiver's file to the worker's connection could, in rollback-journal mode, put both writes into one transaction, but a real external API cannot join a local SQLite transaction. Holding a write lock on the state file while waiting for a network call would also be a poor idea. So the step is three phases:

```python
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
```

1. **Record the intent.** The operation id and the digest of the exact request are committed as `pending` *before* anything leaves the process.
2. **Send** with that id.
3. **Commit the receipt, the step result and the cursor** in one transaction.

The operation id is derived from the position in the workflow: `run-1:3:notify`. It is not a random UUID generated at send time. A random id created after the restart would be new, and the receiver would have no way to recognise the second attempt as a repeat. This scheme requires `run_id` to be unique across all state files and workflows sharing the receiver's operation-id namespace. Keep the same id for retries of that run, and never reuse it for a distinct run; a new local state file does not create a new receiver namespace. The example relies on the caller to satisfy this requirement.

The request digest guards the other direction. If a restart would send *different* content under the same operation id — because the input or the code changed — the worker refuses instead of reusing the id for something else.

Between phase 2 and phase 3 there is a window where the effect may already exist but the worker has no record of the result. This is the classic lost acknowledgement. The demo kills the worker exactly there:

```text
== 2. effect delivered, crash before the ack is committed
exec step=normalize
exec step=classify
exec step=summarize
exec step=notify
exit=77
[('run-b:3:notify', 'pending', None)]
== restart replays the same operation id
exec step=notify
done run=run-b cursor=4
exit=0
[('run-b:3:notify', 2, 'rcpt-f41316924f15')]
[('run-b:3:notify', 'acknowledged', 'rcpt-f41316924f15')]
```

The local journal says `pending`, so the restarted worker can't know whether the delivery happened. It sends again with the same id. The receiver has seen two attempts, applied the effect once and returned the same receipt both times. The worker stores that receipt and the cursor together.

The word doing the work in that paragraph is *receiver*. The demo receiver deduplicates because I wrote it to:

```python
def _apply(con: sqlite3.Connection, operation_id: str, payload: str) -> str:
    row = con.execute(
        "SELECT payload, receipt FROM delivery WHERE operation_id = ?",
        (operation_id,)).fetchone()
    if row is None:
        digest = hashlib.sha256(operation_id.encode()).hexdigest()
        receipt = f"rcpt-{digest[:12]}"
        con.execute("INSERT INTO delivery VALUES (?, ?, 1, ?)",
                    (operation_id, payload, receipt))
        return receipt
    if row[0] != payload:
        raise PayloadConflict(operation_id)
    con.execute("UPDATE delivery SET attempts = attempts + 1 "
                "WHERE operation_id = ?", (operation_id,))
    return row[1]
```

A real external system may not offer idempotency keys at all. If it does, the key may be valid only for a limited time, scoped to one endpoint, or compared against the request body in ways you need to read in its documentation. Without such a guarantee, a `pending` operation after a restart is genuinely ambiguous, and "just send it again" may double the effect. The honest options are then to reconcile — ask the external system whether an operation with your reference exists — or to stop and hand the decision to a person. Neither is implemented here.

This is also where the outbox pattern from the [idempotency article]({{ '/blog/pipeline-idempotencja/' | relative_url }}) fits. In an outbox, the intent is committed together with the state change that caused it, so the intent can't be lost. Here the intent gets its own transaction, but it is built only from outputs that are already committed, so a crash before it is recorded just recreates the same intent with the same id. Either way, the best this gives is at-least-once delivery, and only if the run is eventually resumed and a retry reaches the receiver and succeeds. Recording the intent doesn't make anything get delivered by itself: if nobody restarts the worker, or the receiver keeps failing, the operation simply stays `pending`. And it isn't exactly once: the sender can always die after sending and before recording the acknowledgement. The example demonstrates exactly that case, and the duplicate is absorbed only because the receiver cooperates.

## Refusing to resume on a different basis

A checkpoint is only meaningful relative to the code and the input it was written for. `open_run` checks both before doing anything:

```python
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
```

The digest is computed over JSON with sorted keys, so reordering keys inside a ticket doesn't count as a change, but adding a ticket does. The last part of the demo changes the input under the same run id:

```text
== 3. same run id, different input: refuse to resume
refusing to resume: input digest differs from checkpoint
exit=3
```

Failing closed is a choice, not the only correct behaviour. The alternatives are an explicit migration of old checkpoints, or abandoning the old run and starting a new one under a new id — while remembering that the old run may have `pending` operations. What I want to avoid is the silent option: new code picks up an old cursor and continues as if the first half of the run had been produced by the same logic.

In an agent, "the code" is more than Python. The things that can make a stored checkpoint inconsistent with the process reading it include:

- the workflow or tool-dispatch code;
- the prompt templates;
- the model identifier;
- the tool input and output schemas;
- the input data;
- the storage schema.

The example records three of these: a workflow version string, the input digest and `PRAGMA user_version`. In a model-driven workflow I would either fold prompt, model and tool-schema versions into the workflow version, or store them as separate fields and decide per field whether a change blocks a resume. A model version change, for example, might be acceptable for steps that haven't run yet but not for interpreting steps that already have.

## Approvals and credentials

The example has no human approval step, so this section is a design note, not a tested property.

If a workflow pauses for approval — "send this refund?" — the approval is part of the state that must survive a restart. What should be stored is the *decision*: who approved, what exactly (a digest of the action, like the request digest above), when, and until when it is valid. On resume, the worker should check that the approval has not expired and re-evaluate whether the approver and the worker are still authorised under the current policy. Hours may pass between the approval and the restart, and permissions change.

What should not be stored are credentials. Serialising an access token into a checkpoint leaves a copy of the secret on disk after the process that obtained it is gone, exposing it to anyone who can read the state file. Persistence does not extend the lifetime set by the issuer, but a resumed run may attempt to reuse a token that has expired or been revoked. The resumed process should obtain credentials the same way a fresh process does.

## What breaks with two workers

The single-worker assumption is not a footnote. Suppose a supervisor thinks a worker has died and starts a second one, while the first is merely stalled. Both read cursor 3 and both call the receiver for `notify`.

In this example a few things limit the damage: `BEGIN IMMEDIATE` serialises the writers, and the primary key on `step_result` together with the compare-and-set on the cursor makes the second commit fail. But both checks happen *after* the external call. The duplicate send has already happened, and it is harmless only because the receiver deduplicates by operation id.

The usual remedy is a lease with an expiry plus a fencing token: a number that increases each time the lease changes hands, which every write carries and which storage — ideally also the external system — rejects if it is older than the newest one seen. Without it, a worker that pauses long enough to lose its lease can wake up and act on stale authority. None of this is implemented in the example, which is why the assumption appears in the module docstring and in the first lines of the README.

## Storage failure, corruption and retention

The worker treats an unreadable state file as a stop condition:

```text
storage error, not starting over: file is not a database
```

That is what the worker prints when `state.db` holds garbage bytes instead of a database; it exits with code 4. The corresponding test checks the exit code, that no step ran and that the file is byte-for-byte unchanged. Starting from scratch would be the dangerous reaction. A fresh run would repeat every completed step, including effects, and would forget any `pending` operation, which is the only evidence that something may need reconciling.

Some failure modes are outside what the example can show:

- **Power loss and lying disks.** SQLite's atomic commit document describes its assumptions about storage, including that a flush or `fsync` doesn't return until the data has been written. If the hardware or the operating system breaks that promise, a power failure during commit can corrupt the database. The tests kill a process; they don't cut power, and they don't prove durability on any particular disk.
- **Backups.** SQLite's page on how databases get corrupted warns that a backup copied while a transaction is in progress may contain some old and some new content. Python's `sqlite3` exposes SQLite's backup API as `Connection.backup()`; the example doesn't use it.
- **Disk full or busy database.** These surface as `sqlite3` exceptions. The worker reports them and stops; it doesn't retry.
- **Retention.** A finished run's state can't be deleted the moment the cursor reaches the end. The operation journal is needed at least as long as someone may ask "was this sent?", and at least as long as the receiver's deduplication window if you rely on one. Transcripts and step outputs may contain personal data and need their own retention rule, which is often shorter. These two requirements pull in different directions, which is one more reason to keep the journal separate from the transcript.

## What the tests show, and what they don't

The suite has 21 tests. Each crash test starts the worker as a child process with `subprocess.run`, lets it die via `os._exit(77)` at a chosen point and starts a new process on the same files.

Tested behaviour:

| Failure point | What is durable | Expected restart | Tested |
|---|---|---|---|
| Before `COMMIT` of a pure step | Previous steps only | Step runs again; final state equals a clean run | Yes, for each step |
| After `COMMIT` of a pure step | Step result and cursor | Resume at the next step; the step does not run again | Yes |
| `notify`: intent committed, nothing sent | `pending` operation | Send once; receiver sees one attempt | Yes |
| `notify`: sent, ack not committed (lost ack) | `pending` operation; effect at receiver | Same operation id resent; receiver applies once, returns same receipt | Yes |
| `notify`: after final `COMMIT` | Receipt, result, cursor = 4 | Nothing runs; receiver still sees one attempt | Yes |
| Input changed under the same run id | Old checkpoint | Exit 3, nothing executed, file unchanged | Yes |
| Different workflow or schema version | Old checkpoint | Exit 3, nothing executed | Yes |
| State file is not SQLite | Garbage | Exit 4, file unchanged | Yes |
| Two workers at once | — | Undefined; outside the assumption | No |
| Power loss, `fsync` not honoured | — | Depends on hardware and OS | No |
| Receiver without deduplication | — | Duplicate effect possible | No |

One parametrised test runs every single crash point from the table, restarts once, and compares the whole stored state, plus the operation ids and receipts held by the receiver, with those of a run that never crashed. All points converge. (The receiver's attempt counter is deliberately left out of that comparison: in the lost-ack case it is 2.)

What the suite does not establish: behaviour after two crashes in a row at different points, behaviour under concurrent workers, durability under power loss, or anything about a real external API. It shows that this code, on this machine, recovers from a single process death at each of the instrumented points. That is a narrower claim than "crash-safe", and it's the one I can back with output.

The executed run, on macOS with Python 3.11.12, SQLite 3.47.1 and pytest 7.4.4:

```text
$ uv run --python 3.11 --extra test pytest -q
.....................                                                    [100%]
21 passed in 71.82s (0:01:11)
```

## Reproducing it

Download the files linked above into one directory, keeping `test_worker.py` under `tests/`. Then:

```bash
uv run --python 3.11 --extra test pytest -v
bash demo.sh
```

The worker itself needs nothing but Python 3.11. From the directory containing `worker.py`, create `tickets.json` with the same synthetic input used by `demo.sh`, then run it:

```bash
cat > tickets.json <<'JSON'
[{"id": "T-3", "text": "  Refund for DOUBLE charge  "},
 {"id": "T-1", "text": "Cannot log in after password reset"},
 {"id": "T-2", "text": "Invoice shows the wrong VAT number"}]
JSON
python3.11 worker.py --db state.db --receiver receiver.db \
    --input tickets.json --run-id run-1
```

The crash tests start a few dozen Python processes, so the suite takes longer than its size suggests; on the machine above it took a little over a minute.

## What must survive a restart

Reduced to a list, for a single-worker workflow:

1. The position in the workflow, and the version of code and the input digest it is relative to.
2. Every committed step output that later steps depend on — especially model outputs, whose reproducibility is not guaranteed.
3. For every operation with an external effect: a stable identity derived from the workflow, a digest of the request, its status and its outcome, with `pending` recorded before the call.
4. Approval decisions with their scope and expiry — not the credentials used to act on them.

And the rule that ties them together: a step's result, its acknowledgement and the cursor move in the same transaction. Everything else — retries, reconciliation, leases — builds on that.

### Sources

- [Python 3.11: `sqlite3`](https://docs.python.org/3.11/library/sqlite3.html) — `isolation_level=None` and explicit transaction control; the connection context manager commits or rolls back but does not close the connection; `Connection.backup()`. Also checked against the [3.11.7 source of that page](https://raw.githubusercontent.com/python/cpython/v3.11.7/Doc/library/sqlite3.rst).
- [What's New in Python 3.11](https://docs.python.org/3.11/whatsnew/3.11.html) — Python 3.11 released on 24 October 2022. [Python 3.11.7](https://www.python.org/downloads/release/python-3117/) was released on 4 December 2023.
- [SQLite: UPSERT](https://www.sqlite.org/lang_upsert.html) — `ON CONFLICT ... DO NOTHING`, available since SQLite 3.24.0 (2018-06-04).
- [SQLite: Transaction](https://www.sqlite.org/lang_transaction.html) — `BEGIN IMMEDIATE` starts a write transaction at once and may fail with `SQLITE_BUSY`.
- [SQLite: Atomic Commit In SQLite](https://www.sqlite.org/atomiccommit.html) — rollback journals, hot journals, multi-file commits for attached databases in rollback-journal mode, and the assumptions about `fsync` and hardware behind atomic commit.
- [SQLite: How To Corrupt An SQLite Database File](https://www.sqlite.org/howtocorrupt.html) — backups copied during an active transaction.
- [Python 3.11: `os._exit()`](https://docs.python.org/3.11/library/os.html#os._exit) — exit without cleanup handlers or flushing stdio buffers.
