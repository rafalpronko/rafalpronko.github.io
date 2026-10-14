"""Pure freshness and approval gate. Synthetic teaching example, not production code.

The gate never reads a clock, a ledger or a target itself: the caller passes
everything in, so every decision is a plain function of its inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum


class Status(Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    USED = "used"


@dataclass(frozen=True)
class Policy:
    """Limits chosen per risk class by the owner of the action; no defaults on purpose.

    Approvals bind every field, not just policy_id: changing a limit under the
    same id invalidates approvals issued before the change.
    """

    policy_id: str
    max_state_age_s: float
    approval_ttl_s: float

    def __post_init__(self) -> None:
        for name in ("max_state_age_s", "approval_ttl_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
            # 5 and 5.0 are the same limit; JSON would spell them differently.
            object.__setattr__(self, name, float(value))


def _scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float parameter")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"parameter must be a JSON scalar, got {type(value).__name__}")


@dataclass(frozen=True)
class Command:
    target: str
    action: str
    params: tuple[tuple[str, object], ...]

    @classmethod
    def create(cls, target: str, action: str, params: Mapping[str, object]) -> Command:
        # Copy and sort: later changes to the caller's dict cannot change the command.
        items = tuple(sorted((str(k), _scalar(v)) for k, v in params.items()))
        return cls(target, action, items)


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


@dataclass(frozen=True)
class Approval:
    approval_id: str
    digest: str
    clock_id: str
    issued_at: float
    expires_at: float


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _sha256_json(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def command_fingerprint(command: Command) -> str:
    """What the command does, without state or policy: used for idempotency keys."""
    return _sha256_json({
        "target": command.target,
        "action": command.action,
        "params": [list(p) for p in command.params],
    })


def binding_digest(command: Command, state_version: int, policy: Policy) -> str:
    return _sha256_json({
        "target": command.target,
        "action": command.action,
        "params": [list(p) for p in command.params],
        "state_version": state_version,
        "policy": asdict(policy),  # every field, not only policy_id
    })


def issue_approval(approval_id: str, command: Command, snapshot: Snapshot,
                   policy: Policy, clock_id: str, now: float) -> Approval:
    # The digest holds the version number only, so the snapshot must be of the
    # command's own target: two targets can both be at version 7.
    if snapshot.target != command.target:
        raise ValueError("snapshot is of another target than the command")
    if snapshot.version is None:
        raise ValueError("cannot approve a command against unknown state")
    if not math.isfinite(now):
        raise ValueError("non-finite issue time")
    digest = binding_digest(command, snapshot.version, policy)
    return Approval(approval_id, digest, clock_id, now, now + policy.approval_ttl_s)


def _deny(reason: str) -> Decision:
    return Decision(False, reason)


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
