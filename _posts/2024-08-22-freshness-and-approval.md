---
layout: post
lang: en
title: "Freshness and approval: why a valid command can become unsafe"
date: 2024-08-22 09:00:00 +0200
permalink: /blog/freshness-and-approval/
author: Rafał Prońko
categories: [reliability, security]
description: "A command can pass validation and get a human's approval and still be wrong by the time it runs. A small Python gate for state freshness, scoped single-use approvals and conditional writes."
image: /assets/images/freshness-and-approval.svg
---

# Freshness and approval: why a valid command can become unsafe

![Diagram: observe the target at version 7, approve a command bound to that version, check again at dispatch, then write only if the target is still at version 7. Meanwhile someone else moves it to version 8, so the write is refused and nothing is applied.]({{ '/assets/images/freshness-and-approval.svg' | relative_url }})

*Editorial date: <time datetime="2024-08-22">22 August 2024</time>. That is the date this post is filed under in the archive, not a record of when it was published or when its code was run. The example and its tests were verified on 13 October 2026 with current tools; the details are in [Running the example](#running-the-example).*

An automation proposes a change: set `level` on some target to 3. It might be a device setting, a feature flag, the replica count of a service, a threshold in a pipeline. The proposal comes from a rule engine or an LLM tool call; it doesn't matter much which.

The command passes schema validation. `level` is an integer in the allowed range, the target exists, the action is on the allowlist. A person looks at it in a review screen and clicks “approve”.

Forty seconds later the dispatcher sends it.

In those forty seconds someone else changed the same target. Maybe an operator did it by hand, maybe another job ran, maybe the target changed on its own schedule. The command is still valid. The approval is still in the database. But the person approved “go from 1 to 3”, and what will actually happen is “go from 5 to 3”. Nobody approved that.

This post is about that gap. I'll build a small gate in Python that refuses to act on stale state, binds an approval to exactly what was approved and lets it be used once. Then I'll show why even a correct gate isn't enough, and what has to happen on the target's side.

The example is synthetic and runs entirely in memory. It doesn't touch a real device, service or database, and it isn't a production design. Its value is that every rule is small enough to test.

## Valid is not the same as safe right now

Validation answers a question about the command alone: is it well formed, are the parameters within range, is the action allowed at all.

Safety is a question about the command **together with** the state it will be applied to, the time at which it is applied and the authority behind it. All three of those can change after validation, while the command itself stays exactly the same.

So I split the decision into three separate checks, and I want each of them to fail closed: if I can't prove it's fine, the answer is no.

1. **Freshness** — is my view of the target recent enough, and measured in a way I can trust?
2. **Approval** — did someone with authority approve *this exact* command against *this exact* state, and is that approval still alive?
3. **The write itself** — does the target refuse the change if its state moved since I looked?

The first two live in my gate. The third can only live on the target.

## What exactly did the person approve?

A common shortcut is to store approvals as a flag on a request: `request 42: approved = true`. Then anything that later carries request ID 42 is treated as approved.

That flag doesn't say what was approved. If the parameters are edited after approval, if the request is retargeted, or if the state underneath changed, the flag still says `true`.

I want the approval to be bound to everything the person actually saw when deciding:

- the target,
- the action,
- the parameters, copied and frozen at approval time,
- the version of the target state that was on the screen,
- the policy under which the decision was made: its name and its limits.

In the example a command is an immutable value. `Command.create()` copies the caller's parameters into a sorted tuple and accepts only JSON scalars, rejecting `NaN` and infinity. Changing the original dictionary afterwards has no effect on the command.

The binding is a hash of a canonical JSON form. A small helper, `_sha256_json()`, serialises the payload with sorted keys, no whitespace and `allow_nan=False`, and hashes it with SHA-256. The binding itself, from `freshness_gate.py`:

```python
def binding_digest(command: Command, state_version: int, policy: Policy) -> str:
    return _sha256_json({
        "target": command.target,
        "action": command.action,
        "params": [list(p) for p in command.params],
        "state_version": state_version,
        "policy": asdict(policy),  # every field, not only policy_id
    })
```

At dispatch I recompute the digest from the command I'm about to send, the state I currently see and the policy the gate runs. If anything differs — another level, another target, a newer state version, another policy, or a changed limit under the same policy name — the digest differs and the gate says `approval_binding_mismatch`.

The policy goes in whole. If the digest carried only the policy's name, someone could shorten the approval TTL of a policy called `demo-low-risk`, and approvals issued under the old, longer TTL would still match. Binding every field means that any policy change, even a stricter one, invalidates the approvals issued before it, and people have to approve again. For this kind of gate I think that's the right default. The alternative is a contract that a policy ID always names one immutable version, so `risk-v3` never changes and a change becomes `risk-v4`. That works too, but the code can't check it: a limit changed under the same ID would go unnoticed.

There's one thing the digest can't carry: *whose* version it is. The snapshot contributes only a number, and two targets can both be at version 7. So besides the gate's own check that the snapshot belongs to the command's target, the target is checked in three more places, each with its own test. `issue_approval()` refuses a snapshot of a different target than the command's. The dispatcher refuses a command for a target it isn't connected to, before it spends the approval. And the fake target refuses a write addressed to another target, on its conditional and its unconditional path alike.

One thing this digest is **not**: proof that someone approved anything. Anyone can compute a SHA-256. In the example the gate never accepts an approval from the caller; it looks it up by ID in a ledger that the requester can't write to. In a real system that ledger, or a signature from the approving service, is the source of authority. The digest only makes sure the authority is applied to the right thing.

## Freshness: how old is my view, and on whose clock?

A state snapshot in the example carries the target, a version, the ID of the clock it was stamped with and the time it was received. From `freshness_gate.py`:

```python
@dataclass(frozen=True)
class Snapshot:
    """Target state as received by the gate, stamped with the gate's own clock.

    received_at bounds the age since *receipt*, not since the source measured
    the state. That is only a freshness bound if the read is a synchronous,
    authoritative read from the owner of the state, as FakeTarget.read() is.
    """

    target: str
    version: int | None
    clock_id: str
    received_at: float
```

The time is the moment *my* process received the state, read from *my* clock. I deliberately don't use a timestamp reported by the target. That timestamp comes from another machine's clock, and comparing it with mine mixes two clock domains.

That choice has a limit worth stating. `received_at` bounds the age since my process *received* the state, not the age of the measurement itself. The example can treat the two as the same only because `FakeTarget.read()` is a synchronous, authoritative read from the owner of the state: the version it returns is current at the moment of the call. (The dispatcher stamps the time when the read returns; with a real network read, stamping before sending the request would be the conservative choice.) If the state came through a cache, a queue or an event feed, it could already be old when it arrives, and a recent receipt time would make it look fresh. Re-reading such a source doesn't help either. Then freshness has to come from the source's own semantics, such as a version or sequence number that says which measurement this is, and the gate has to check that instead of my receipt time.

Even on one machine, the choice of clock matters. Python's documentation is explicit that `time.time()` can return a lower value than a previous call if the system clock was set back. An age computed from it can be negative, or suddenly small. `time.monotonic()` cannot go backwards and isn't affected by system clock updates, but its reference point is undefined: only the difference between two calls is meaningful.

That last sentence is easy to read past, and easy to over-read. Since Python 3.5 the documentation describes `time.monotonic()` as system-wide (on macOS since 3.10). Another process on the same machine, or my own process after a restart, reads the same clock, so a process restart alone doesn't break the arithmetic. What the contract doesn't give me is portability. With an undefined reference point, a value can't be assumed to mean anything on another machine, and nothing in the contract makes it comparable across an OS boot.

In the example each `MonotonicClock` instance gets a fresh random `clock_id`, and the gate rejects any snapshot or approval stamped with a different ID. That's a policy of mine, deliberately stricter than the clock requires: after a process restart, nothing old is trusted and everything is read again. It costs a re-read, and possibly a new approval, after every restart. In exchange I never have to work out whether a persisted value came from the current boot.

If I need a timestamp for an audit log, I store wall-clock time separately. I just don't make safety decisions with it.

## The gate, check by check

Here is the whole decision function, `evaluate()` from `freshness_gate.py`, in full. It doesn't read a clock, a ledger or the target; the caller passes everything in. That makes it a plain function I can test against a table of cases.

```python
def evaluate(*, command: Command, snapshot: Snapshot | None, approval: Approval | None,
             approval_status: Status | None, policy: Policy, clock_id: str,
             now: float) -> Decision:
    """Every branch that is not positively proven safe returns a denial."""
    if not math.isfinite(now):
        return _deny("nonfinite_time")

    # State: present, known, ours, measured on our clock, not from the future, fresh.
    if snapshot is None:
        return _deny("missing_state")
    if snapshot.target != command.target:
        return _deny("state_target_mismatch")
    if snapshot.version is None:
        return _deny("unknown_state")
    if snapshot.clock_id != clock_id:
        return _deny("clock_domain_mismatch")
    if not math.isfinite(snapshot.received_at):
        return _deny("nonfinite_time")
    if snapshot.received_at > now:
        return _deny("state_from_future")
    if now - snapshot.received_at > policy.max_state_age_s:
        return _deny("stale_state")

    # Approval: present, known to the ledger, still active, on our clock, in its window.
    if approval is None:
        return _deny("missing_approval")
    if approval_status is None:
        return _deny("unknown_approval")
    if approval_status is Status.REVOKED:
        return _deny("approval_revoked")
    if approval_status is Status.USED:
        return _deny("approval_used")
    if approval_status is not Status.ACTIVE:
        return _deny("unknown_approval")
    if approval.clock_id != clock_id:
        return _deny("clock_domain_mismatch")
    if not (math.isfinite(approval.issued_at) and math.isfinite(approval.expires_at)):
        return _deny("nonfinite_time")
    if approval.issued_at > now:
        return _deny("approval_from_future")
    if now >= approval.expires_at:
        return _deny("approval_expired")

    # Binding: exactly this command, against exactly this state, under this policy.
    if approval.digest != binding_digest(command, snapshot.version, policy):
        return _deny("approval_binding_mismatch")
    return Decision(True, "ok")
```

A few of these checks look paranoid until you see what happens without them.

**Missing and unknown are different from “fine”.** A snapshot that is `None`, or a version that is `None` because the read timed out, is a denial. The tempting default is to treat “no information” as “no change”. That's exactly the case where I know the least.

**Non-finite times.** `NaN` compares false with everything. A naive `if age > max_age: deny` lets a `NaN` age straight through, because `NaN > 5.0` is `False`. So I check `math.isfinite()` explicitly before any comparison. Infinity is rejected for the same reason.

**Times from the future.** A snapshot stamped later than “now” on the same clock means something is broken: a bug, a mixed-up clock, a forged value. A negative age would pass a `>` check, so I deny it explicitly.

**The order of the result codes is for humans.** Every failing branch denies. The order only decides which reason is reported first, and I put state before approval because a stale view is usually the more actionable message.

## Two limits, chosen per risk

`Policy` has two numbers and no defaults. From `freshness_gate.py`, without the validation in `__post_init__()`:

```python
@dataclass(frozen=True)
class Policy:
    """Limits chosen per risk class by the owner of the action; no defaults on purpose.

    Approvals bind every field, not just policy_id: changing a limit under the
    same id invalidates approvals issued before the change.
    """

    policy_id: str
    max_state_age_s: float
    approval_ttl_s: float
```

Its constructor refuses zero, negative and non-finite values, and stores both limits as floats, so `5` and `5.0` give the same policy and the same digest. It doesn't refuse “unusual” ones, because I don't know what's unusual for your system.

The two limits answer different questions.

`max_state_age_s` is about how fast the world drifts. How quickly does this target change on its own or through other actors? What does it cost to act on a view that is that old?

`approval_ttl_s` is about how long a human decision stays meaningful. A person approved something in a particular context. Ten minutes later, would they approve it again without looking?

In the tests I use 5 seconds and 60 seconds. Those are illustrative numbers for a test, not recommendations. I'd set real values from measurements: how often the state actually changes, how long reviews actually take, whether the action can be undone. A reversible change to a dashboard setting and an irreversible change to a production system shouldn't share a policy, and that's why the whole policy, not only its ID, is part of the approval binding: a command approved under a lenient policy doesn't pass a gate that runs a strict one, even if both carry the same name.

The difference between the two limits shows up in a test in `tests/test_dispatch.py`. The person approves, then the dispatcher waits 30 seconds. The approval is still inside its 60-second TTL. The state isn't:

```python
def test_human_delay_makes_state_stale_even_with_valid_approval():
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    clock.advance(30)  # approval still inside its TTL, state is not
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "stale_state"
    assert target.read()[0] == 7
```

And if I simply re-read the state to make it fresh, the gate still refuses when the version moved, because the approval was given for the old version. A second test from the same file:

```python
def test_fresh_reread_after_change_needs_new_approval():
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    target.external_change(level=5)  # someone else acted: version 8
    clock.advance(1)
    fresh = d.observe()
    assert d.dispatch(set_level(3), fresh, "ap-1", "key-1") == "approval_binding_mismatch"
```

That's the behaviour I want. Fresh state doesn't make an old approval valid for a new situation. It means asking again.

## Expiry, revocation and single use

An approval in the example has a life cycle in the ledger: `ACTIVE`, then either `REVOKED` or `USED`. The gate allows only `ACTIVE`. An unknown approval ID is a denial, not an error to be retried.

Single use is enforced by the ledger, not by the gate. The gate only reads a status; between reading it and acting, another dispatcher could read the same `ACTIVE`, and the clock keeps moving. So the ledger has one atomic operation that reads the clock, runs the whole gate and spends the approval, `ApprovalLedger.authorize_and_consume()` in `fake_target.py`:

```python
    def authorize_and_consume(self, approval_id: str, *, command: Command,
                              snapshot: Snapshot | None, policy: Policy,
                              clock: ManualClock | MonotonicClock) -> Decision:
        """The authorization cutoff: read the clock, run the whole gate and flip
        ACTIVE -> USED in one step under the ledger lock. A denial changes nothing.
        """
        with self._lock:
            approval, status = self._entries.get(approval_id, (None, None))
            decision = evaluate(command=command, snapshot=snapshot, approval=approval,
                                approval_status=status, policy=policy,
                                clock_id=clock.clock_id, now=clock.now())
            # The gate already denies a missing approval; repeated for type checkers.
            if decision.allowed and approval is not None:
                self._entries[approval_id] = (approval, Status.USED)
            return decision
```

The dispatcher calls it **before** sending the write and doesn't run the gate on its own. `Dispatcher.dispatch()` from `fake_target.py`:

```python
    def dispatch(self, command: Command, snapshot: Snapshot | None, approval_id: str,
                 idempotency_key: str) -> str:
        with self._local_lock:
            if command.target != self.target.name:  # before the approval is spent
                return "wrong_target"
            if self.before_consume is not None:
                self.before_consume()
            # Spend the approval before acting: a failed write needs a new approval.
            # The gate runs inside the ledger's atomic step, on a clock reading
            # taken there. After a successful consume, revocation can't recall the
            # write and expiry is not checked again.
            decision = self.ledger.authorize_and_consume(
                approval_id, command=command, snapshot=snapshot,
                policy=self.policy, clock=self.clock)
            if not decision.allowed:
                return decision.reason
            # The gate already denied these; repeated for type checkers.
            if snapshot is None or snapshot.version is None:
                return "unknown_state"
            if self.before_write is not None:
                self.before_write()
            result = self.target.write_if_match(expected_version=snapshot.version,
                                                command=command,
                                                idempotency_key=idempotency_key)
            return result.status
```

Spending the approval first is a deliberate trade-off. If the write then fails, the approval is gone and a person has to approve again. The alternative — consume only after success — means that a crash between the write and the consume leaves an approval that can be used a second time. I'd rather pay with an occasional extra approval than with a replay.

## Where authorization ends

The authorization cutoff in this example is one clock reading: the one `authorize_and_consume()` takes while it holds the ledger lock. At that instant the gate checks everything together: the approval's status and its window, the state's age, both clock domains, and the binding to the command, the state version and the policy. If the gate allows, the approval becomes `USED` before the lock is released. If it denies, nothing changes: the approval keeps its status, the denial reason is returned as it is, and no write is sent.

A tempting split is to run the gate first and then call a consume that only flips `ACTIVE` to `USED`. Those are two moments. A scheduling pause or a wait for the lock between them can let the approval expire, or the state grow too old, and the consume still succeeds. The regression for exactly that, from `tests/test_dispatch.py`:

```python
def test_approval_expiring_before_consume_is_denied_and_not_spent():
    """A pause before the consume: the approval expires before it is spent."""
    target, ledger, clock, d = setup()
    d.approve("ap-1", set_level(3), d.observe())  # expires at 160.0
    clock.advance(59)
    snap = d.observe()  # refreshed at 159.0, still version 7
    clock.advance(0.9)  # dispatch starts at 159.9
    d.before_consume = lambda: clock.advance(0.2)  # the consume runs at 160.1
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_expired"
    assert target.read() == (7, {"level": 1}) and target.lookup("key-1") is None
    assert ledger.get("ap-1")[1] is Status.ACTIVE  # denied, so not spent
```

Against the split version this test got `applied`. The same pause with the state at 4.9 seconds old gives `stale_state`, and a revocation inside the pause gives `approval_revoked`; in every case the approval stays unspent and the target untouched. One more test uses a clock that records whether the ledger lock was held each time it was read, and fails if `dispatch()` reads the clock anywhere else. Running the gate a second time just before a status-only consume would pass the first three tests and fail that one.

The contract is about one instant, not a stretch of time:

- **Revocation is serialized with the consume.** `revoke()` takes the same lock. Either it gets there first and the gate answers `approval_revoked`, or the consume does and `revoke()` returns `False`. There's no third outcome.
- **The reading is a sample, not a guarantee that lasts.** Evaluating the gate and releasing the lock take time, and the write is sent after that. The decision is correct for the moment of the reading; nothing about the approval or the state's age is checked again.
- **Revocation after consume can't recall the write.** `revoke()` flips only an `ACTIVE` approval and returns `False` for one that is already `USED`. A revocation that arrives while the write is in flight comes too late, and the write goes ahead.
- **Expiry after consume is allowed.** If the write is delayed after the consume and lands after `expires_at`, it is still applied, as long as the target's version hasn't moved.

The last one, from `tests/test_dispatch.py`:

```python
def test_expiry_is_not_rechecked_after_consume():
    """Known limit: the version condition checks state, not the approval deadline."""
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)  # expires at 160.0
    d.before_write = lambda: clock.advance(61)  # the write lands at 161.0
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "applied"
    assert target.read() == (8, {"level": 3})
```

In the example the gap after the cutoff is one in-process call, with no queue and no retry in between. The next section is about what happens when the gap between the check and the change gets longer.

## The check happens at dispatch. The change happens later.

Everything so far runs in my process. Even a perfect gate checks the state *at the moment of dispatch*. The change happens when the target applies it. Between those two moments there's a network call, maybe a queue, maybe a retry. This is a classic time-of-check to time-of-use (TOCTOU) gap, and re-checking just before sending only makes it narrower.

The example has a test seam for exactly that window: `before_write` runs after the gate allowed and before the write is sent. In this test from `tests/test_dispatch.py`, someone else changes the target there:

```python
def test_toctou_conditional_write_rejects_change_after_check():
    """Gate passes, then the target changes before the write lands."""
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.before_write = lambda: target.external_change(level=5)
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "precondition_failed"
    assert target.read() == (8, {"level": 5})
```

The gate passed. The change still didn't happen, because the condition travelled *with* the write: `expected_version=7`. The fake target evaluates it under its own lock, at the moment of applying the change. A fragment of `FakeTarget.write_if_match()` in `fake_target.py`:

```python
            if self._version != expected_version:
                return WriteResult("precondition_failed", self._version)
            self._apply(command)
```

Just before that, the same method compares the command's target with its own name and answers `wrong_target` if they differ. A correct version number on the wrong target is still the wrong write.

This is the same idea HTTP standardises as conditional requests. RFC 9110 describes `If-Match` as the way to prevent the “lost update” problem, says the origin server must evaluate the precondition just before performing the method, and must not perform it if the condition is false; it may answer `412 Precondition Failed`. The important part isn't the header. It's who evaluates the condition and when: the party that owns the state, at the moment it changes it.

Without that, the same race silently overwrites someone else's change. The companion test does exactly that with `write_unconditional()`: the version I saw was 7, the target is at 9 afterwards, and the other actor's `level=5` is gone without any error. That method still refuses a command addressed to another target: what it demonstrates is the race on state, not a routing error.

The same reasoning applies to queues, with one more trap. If a command can sit in a queue, a freshness check at enqueue time says nothing about the moment of execution. A version condition evaluated by the target covers the *state* half of that: if the target moved, the write is refused. It does nothing for the *authorization* half. An approval that expired or was revoked while the command waited still passes a version check if nobody touched the target. A command whose target is unchanged isn't approved just because the condition holds.

So for a delayed command, whoever actually executes it needs both checks at execution time. Is the approval still alive: not expired, not revoked, and consumed at this moment rather than at enqueue time? And does the target's state still match the approved version? If the executor is a separate service, it needs its own clock for the deadline, or a deadline the target can evaluate in its own clock domain. A deadline in my monotonic clock is meaningless on another machine. The example has no queue. Its executor is the dispatcher, which is why its cutoff is an atomic check-and-consume immediately before the write.

## A local lock is not a target-side condition

A natural reaction to races is to add a lock. In the example each `Dispatcher` has a `threading.Lock`. It serialises dispatches within that one dispatcher, and that's all.

The test below uses two `Dispatcher` objects in one process to stand in for two separate actors. It isn't a multi-process test, but separate processes wouldn't share these locks either. From `tests/test_dispatch.py`:

```python
def test_local_locks_do_not_serialize_two_dispatchers():
    """Two dispatcher objects in one process stand in for two separate actors.

    Not a multi-process test: it shows that two local locks, one per dispatcher,
    don't serialize writes to a shared target; only the target's condition does.
    """
    target = FakeTarget("demo-actuator-1", level=1, version=7)
    ledger = ApprovalLedger()
    a = Dispatcher(target, ledger, POLICY, ManualClock("boot-A", 100.0))
    b = Dispatcher(target, ledger, POLICY, ManualClock("boot-B", 100.0))
    snap_a, snap_b = a.observe(), b.observe()
    a.approve("ap-a", set_level(3), snap_a)
    b.approve("ap-b", set_level(9), snap_b)
    assert a.dispatch(set_level(3), snap_a, "ap-a", "key-a") == "applied"
    assert b.dispatch(set_level(9), snap_b, "ap-b", "key-b") == "precondition_failed"
    assert target.read() == (8, {"level": 3})
```

Both dispatchers saw version 7. Both had a valid, fresh, correctly bound approval. Both gates said yes. Each held its own lock. Only the target's version check stopped the second write.

In a real system the “second dispatcher” is often not a copy of my code. It's a person in another UI, a cron job, a vendor's app, the target's own scheduler. None of them takes my lock. A distributed lock widens the circle of participants, but it still covers only the writers that cooperate, and it brings its own failure modes around expiry. The one check every writer passes through is the one the target makes.

If the target can't do a conditional write at all, I can't close this window. I can make it smaller, and I should say plainly that it's still there. For some actions, that's a reason not to automate them.

## Idempotency is not authorization

There's one more failure that looks similar and needs a different tool. The dispatcher sends the write, the target applies it, and the response is lost. Did it happen?

In an earlier post on [idempotent pipelines]({{ '/blog/pipeline-idempotencja/' | relative_url }}) I argued that a retry has to be safe when you can't tell. Here that is done with an idempotency key. The fake target remembers each key together with a fingerprint of the command:

- the same key with the same command is a `duplicate`, and nothing is applied again,
- the same key with a different command is a `key_conflict`,
- a new key goes through the version check.

RFC 9110 allows the same trade-off for `If-Match`: when a state-changing request appears to have already been applied, the server may respond with success instead of `412`, since the client may simply have lost the earlier response.

What I don't want is to let the idempotency key and the approval stand in for each other. They answer different questions. The approval says “is this allowed, now?”. The key says “has this already happened?”.

So after a lost response the right move is to ask the target, not to replay the approval. From `tests/test_dispatch.py`:

```python
def test_lost_response_is_resolved_by_lookup_not_by_reusing_approval():
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.dispatch(set_level(3), snap, "ap-1", "key-1")  # pretend the reply was lost
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_used"
    assert target.lookup("key-1").status == "applied"
    assert target.read()[0] == 8  # applied exactly once
```

The retry is stopped by the gate, because the approval is spent. The answer comes from `lookup()`. The opposite direction is covered too: a request with a fresh idempotency key and no approval is denied with `missing_approval`, and the target never sees the key. A key protects against doing something twice. It never grants permission to do it once.

## Running the example

The full code is available to download:

- [`freshness_gate.py`]({{ '/assets/examples/freshness-and-approval/freshness_gate.py' | relative_url }}) — the pure gate,
- [`fake_target.py`]({{ '/assets/examples/freshness-and-approval/fake_target.py' | relative_url }}) — the in-memory target, ledger, clocks and dispatcher,
- [`tests/test_gate.py`]({{ '/assets/examples/freshness-and-approval/tests/test_gate.py' | relative_url }}) and [`tests/test_dispatch.py`]({{ '/assets/examples/freshness-and-approval/tests/test_dispatch.py' | relative_url }}),
- [`pyproject.toml`]({{ '/assets/examples/freshness-and-approval/pyproject.toml' | relative_url }}) and [`README.md`]({{ '/assets/examples/freshness-and-approval/README.md' | relative_url }}).

It needs Python 3.12 and only the standard library; pytest runs the tests. Put the files in one directory, keeping `tests/` as a subdirectory, and run from that directory. These commands use current tooling, uv and a pinned pytest 9.1.1, and were checked on 13 October 2026. They are not a claim about the tools that existed on this post's editorial date:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python 'pytest==9.1.1'
.venv/bin/python -m pytest
```

The verification run on 13 October 2026 (uv 0.7.8, CPython 3.12.11, pytest 9.1.1, macOS) reported `65 passed`. Durations vary between runs and machines, so none is quoted here.

Most tests were written before the code they test and seen failing before they passed. A few deliberately pin down existing behaviour, such as the expiry limit above, so they passed from the start. That doesn't make the example complete. It means the behaviour described in this post is the behaviour the code actually has.

## Where this guarantee ends

It's worth being precise about what the example does not do.

- **It's one process.** The target, the ledger and the dispatchers share memory and use `threading.Lock`; the “two dispatchers” in the tests are two objects in that process. A real ledger needs durable storage with one atomic step that checks the deadline on its own clock and consumes only if the approval is active, and a real target needs its own conditional write.
- **The read is assumed authoritative.** Ages are measured from receipt. That equals the age of the state only because `FakeTarget.read()` is a synchronous read from the owner of the state.
- **Authorization ends at one clock reading.** The gate runs on the reading taken inside the ledger's atomic step. Revocation and expiry after it don't stop the write, and nothing is guaranteed continuously between that reading and the moment the target applies the change. Delayed or queued execution needs both checks again where the command actually runs.
- **Accepted is not done.** `applied` means the fake target accepted the write. With a real target, accepting a command and the effect being observable are separate events. Confirming the effect by reading back state is a topic of its own.
- **The approval isn't authenticated.** The gate trusts the ledger. Who may approve what, and how that identity is checked, is outside this example.
- **Clock domains are per clock instance, by choice.** Each `MonotonicClock` object gets its own ID. Dispatchers given the same clock object share its domain, as the two dispatchers in the routing tests do. The monotonic clock itself is system-wide. Invalidating everything on restart, when a new clock instance is created, is my stricter policy: safe, but it can be inconvenient. Sharing a clock domain across machines is a different and harder problem, and I avoid it by keeping every time comparison inside one process.
- **A policy change invalidates approvals.** The whole policy is bound, so changing any limit means asking people again. That's intended, and it's worth knowing before you change a limit.
- **The numbers are illustrative.** 5 seconds and 60 seconds were chosen for the tests. Your limits have to come from your system.
- **No safety claims beyond software state.** The example models a version number and a level. It says nothing about the physical or business consequences of a particular action.

## What I check before an automated command runs

When an automated system can change something real, I go through these questions:

1. Is the approval bound to the exact target, action, parameters, state version and policy (its limits, not just its name), or just to a request ID?
2. Is the target checked again where the command is issued, dispatched and written?
3. Is “missing” or “unknown” state a denial everywhere, including timeouts?
4. Is the state I read authoritative, or could it be a cached or queued value that only *arrived* recently?
5. Are ages measured on one monotonic clock in my process, with `NaN`, infinity and future timestamps rejected explicitly?
6. Do I refuse monotonic values from another machine or another boot, and have I decided explicitly whether a process restart invalidates them?
7. Are the state-age limit and approval TTL set per risk, from measurements, rather than copied from somewhere?
8. Can an approval be revoked, and is it consumed atomically, exactly once, in the same step that checks its deadline? Do I know the point after which revocation no longer helps?
9. If a command can wait, does whatever executes it check both the approval and the state condition at that moment?
10. Does the target evaluate a condition at the moment of the change, or am I relying on a local lock?
11. After a lost response, do I look the result up by idempotency key instead of re-using the approval?

The command at the start of this post was valid the whole time. What changed was the world around it. Validation can't see that, so the gate has to check it again just before dispatch, and the target has to check it once more at the moment of the change.

### Sources

- [RFC 9110: HTTP Semantics, section 13.1.1 If-Match and section 13.2 Evaluation of Preconditions](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1.1) — June 2022; conditional requests against the “lost update” problem, evaluation by the origin server just before performing the method, `412 Precondition Failed`, and the option to return success when a change appears already applied.
- [Python 3.12: time.monotonic()](https://docs.python.org/3.12/library/time.html#time.monotonic) — a clock that cannot go backwards, with an undefined reference point, so only differences between calls are valid; system-wide since Python 3.5, and on macOS since 3.10. Also [time.time()](https://docs.python.org/3.12/library/time.html#time.time), which can go back when the system clock is set back.
- [CPython v3.12.5 source of the `time` documentation](https://raw.githubusercontent.com/python/cpython/v3.12.5/Doc/library/time.rst) — the same `monotonic()` text as tagged for the 3.12.5 release.
- [Python 3.12.5 release](https://www.python.org/downloads/release/python-3125/) — released 6 August 2024, before this post's editorial date.
