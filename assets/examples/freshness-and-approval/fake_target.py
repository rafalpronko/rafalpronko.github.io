"""In-memory fake target, approval ledger and dispatcher.

Synthetic teaching example. Everything is serialized with in-process locks;
this is NOT a production design for a real device, service or database.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from freshness_gate import (
    Approval,
    Command,
    Decision,
    Policy,
    Snapshot,
    Status,
    command_fingerprint,
    evaluate,
    issue_approval,
)


class ManualClock:
    """Test clock: time moves only when the test says so."""

    def __init__(self, clock_id: str, start: float = 0.0) -> None:
        self.clock_id = clock_id
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class MonotonicClock:
    """time.monotonic() with a fresh clock_id per instance (per process start).

    Since Python 3.5 the clock is system-wide, so another process on the same
    machine, or this one after a restart, reads the same clock. The reference
    point is undefined, so a value is not portable across machines and nothing
    makes it comparable across an OS boot. A new id per instance is a stricter
    policy of ours, not a property of the clock: the gate rejects snapshots and
    approvals stamped by any previous instance, so a restart forces a fresh read.
    """

    def __init__(self) -> None:
        self.clock_id = f"monotonic-{uuid.uuid4()}"

    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class WriteResult:
    status: str  # applied | precondition_failed | duplicate | key_conflict | wrong_target
    version: int


class FakeTarget:
    """Stands in for the system that owns the state. Its lock models the
    target's own atomic compare-and-set, which callers cannot bypass.

    read() is a synchronous, authoritative read from the owner: the version it
    returns is current at the moment of the call. A cache, a queue or an
    upstream event feed would not give that guarantee.
    """

    def __init__(self, name: str, level: int = 0, version: int = 1) -> None:
        self.name = name
        self._state: dict[str, object] = {"level": level}
        self._version = version
        self._lock = threading.Lock()
        # idempotency_key -> (command fingerprint, recorded result)
        self._seen: dict[str, tuple[str, WriteResult]] = {}

    def read(self) -> tuple[int, dict[str, object]]:
        with self._lock:
            return self._version, dict(self._state)

    def _apply(self, command: Command) -> None:
        self._state.update(dict(command.params))
        self._version += 1

    def write_if_match(self, *, expected_version: int, command: Command,
                       idempotency_key: str) -> WriteResult:
        # The fingerprint ignores state version and policy: a retry of the same
        # command with the same key is a duplicate even though the version moved.
        fingerprint = command_fingerprint(command)
        with self._lock:
            if command.target != self.name:
                return WriteResult("wrong_target", self._version)
            if idempotency_key in self._seen:
                seen_fingerprint, result = self._seen[idempotency_key]
                if seen_fingerprint != fingerprint:
                    return WriteResult("key_conflict", self._version)
                return WriteResult("duplicate", result.version)
            if self._version != expected_version:
                return WriteResult("precondition_failed", self._version)
            self._apply(command)
            result = WriteResult("applied", self._version)
            self._seen[idempotency_key] = (fingerprint, result)
            return result

    def write_unconditional(self, command: Command) -> WriteResult:
        """The hazard: last writer wins, whatever happened since the check.

        It still refuses a command addressed to another target: this path
        demonstrates the state race, not a routing error.
        """
        with self._lock:
            if command.target != self.name:
                return WriteResult("wrong_target", self._version)
            self._apply(command)
            return WriteResult("applied", self._version)

    def external_change(self, level: int) -> None:
        """Someone else (operator, other service, the world) changes the target."""
        with self._lock:
            self._state["level"] = level
            self._version += 1

    def lookup(self, idempotency_key: str) -> WriteResult | None:
        with self._lock:
            entry = self._seen.get(idempotency_key)
            return entry[1] if entry else None


class ApprovalLedger:
    """Shared record of approvals. authorize_and_consume() enforces single use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Approval, Status]] = {}

    def add(self, approval: Approval) -> None:
        with self._lock:
            if approval.approval_id in self._entries:
                raise ValueError(f"approval id already exists: {approval.approval_id}")
            self._entries[approval.approval_id] = (approval, Status.ACTIVE)

    def get(self, approval_id: str) -> tuple[Approval | None, Status | None]:
        with self._lock:
            return self._entries.get(approval_id, (None, None))

    def revoke(self, approval_id: str) -> bool:
        """ACTIVE -> REVOKED. False if it was already USED: too late to recall."""
        with self._lock:
            approval, status = self._entries[approval_id]
            if status is not Status.ACTIVE:
                return False
            self._entries[approval_id] = (approval, Status.REVOKED)
            return True

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


class Dispatcher:
    """One process's view: its own clock and its own *local* lock."""

    def __init__(self, target: FakeTarget, ledger: ApprovalLedger, policy: Policy,
                 clock: ManualClock | MonotonicClock) -> None:
        self.target = target
        self.ledger = ledger
        self.policy = policy
        self.clock = clock
        self._local_lock = threading.Lock()
        # Test seam: runs before the ledger's atomic step, standing in for a
        # scheduling pause or a wait for the ledger lock.
        self.before_consume: Callable[[], None] | None = None
        # Test seam: runs after the gate passed and before the write is sent.
        self.before_write: Callable[[], None] | None = None

    def observe(self) -> Snapshot:
        # Stamped when the synchronous read returns. With a real network read,
        # stamping before the request would be the conservative choice.
        version, _ = self.target.read()
        return Snapshot(self.target.name, version, self.clock.clock_id,
                        self.clock.now())

    def approve(self, approval_id: str, command: Command,
                snapshot: Snapshot) -> Approval:
        approval = issue_approval(approval_id, command, snapshot, self.policy,
                                  self.clock.clock_id, self.clock.now())
        self.ledger.add(approval)
        return approval

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
