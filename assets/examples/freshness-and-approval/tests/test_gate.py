"""Tests for the pure gate. All values are synthetic."""

import math

import pytest

from freshness_gate import (
    Command,
    Policy,
    Snapshot,
    Status,
    evaluate,
    issue_approval,
)

CLOCK = "boot-A"
# Illustrative numbers for a test, not recommended values.
POLICY = Policy("demo-low-risk", max_state_age_s=5.0, approval_ttl_s=60.0)


def cmd(level: int = 3) -> Command:
    return Command.create("demo-actuator-1", "set_level", {"level": level})


def snap(version: int | None = 7, received_at: float = 100.0,
         clock_id: str = CLOCK, target: str = "demo-actuator-1") -> Snapshot:
    return Snapshot(target, version, clock_id, received_at)


def check(*, command=None, snapshot="default", approval="default",
          status=Status.ACTIVE, policy=POLICY, clock_id=CLOCK, now=101.0):
    command = command or cmd()
    if snapshot == "default":
        snapshot = snap()
    if approval == "default":
        approval = issue_approval("ap-1", cmd(), snap(), POLICY, CLOCK, now=100.0)
    return evaluate(command=command, snapshot=snapshot, approval=approval,
                    approval_status=status, policy=policy, clock_id=clock_id, now=now)


def test_fresh_state_and_matching_approval_is_allowed():
    assert check() == check().__class__(True, "ok")


# --- state: missing, unknown, foreign, stale ----------------------------------

def test_missing_state_fails_closed():
    assert check(snapshot=None).reason == "missing_state"


def test_unknown_state_version_fails_closed():
    assert check(snapshot=snap(version=None)).reason == "unknown_state"


def test_snapshot_of_another_target_is_rejected():
    assert check(snapshot=snap(target="demo-actuator-2")).reason == "state_target_mismatch"


def test_state_older_than_policy_limit_is_stale():
    assert check(now=105.5).reason == "stale_state"


def test_state_exactly_at_limit_is_still_fresh():
    assert check(now=105.0).allowed


def test_state_received_in_the_future_is_rejected():
    assert check(snapshot=snap(received_at=200.0), now=101.0).reason == "state_from_future"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_now_is_rejected(bad):
    assert check(now=bad).reason == "nonfinite_time"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_received_at_is_rejected(bad):
    # NaN compares False with everything, so a naive "age > limit" would pass it.
    assert check(snapshot=snap(received_at=bad)).reason == "nonfinite_time"


def test_snapshot_from_another_clock_domain_is_rejected():
    assert check(snapshot=snap(clock_id="boot-B")).reason == "clock_domain_mismatch"


# --- approval: binding --------------------------------------------------------

def test_approval_for_different_parameters_is_rejected():
    assert check(command=cmd(level=9)).reason == "approval_binding_mismatch"


def test_approval_for_older_state_version_is_rejected():
    assert check(snapshot=snap(version=8)).reason == "approval_binding_mismatch"


def test_approval_under_different_policy_is_rejected():
    strict = Policy("demo-high-risk", max_state_age_s=5.0, approval_ttl_s=60.0)
    assert check(policy=strict).reason == "approval_binding_mismatch"


@pytest.mark.parametrize("age,ttl", [(5.0, 1.0), (4.0, 60.0), (5.0, 600.0)])
def test_changed_limits_under_the_same_policy_id_are_rejected(age, ttl):
    # The whole policy is bound, not only its name.
    changed = Policy("demo-low-risk", max_state_age_s=age, approval_ttl_s=ttl)
    assert check(policy=changed).reason == "approval_binding_mismatch"


def test_int_and_float_limits_are_the_same_policy():
    same = Policy("demo-low-risk", max_state_age_s=5, approval_ttl_s=60)
    assert check(policy=same).allowed


def test_command_is_a_copy_of_caller_params():
    params = {"level": 3}
    command = Command.create("demo-actuator-1", "set_level", params)
    params["level"] = 9
    assert command == cmd(level=3)


def test_command_rejects_nonscalar_and_nonfinite_params():
    with pytest.raises(ValueError):
        Command.create("t", "a", {"level": [1, 2]})
    with pytest.raises(ValueError):
        Command.create("t", "a", {"level": math.nan})


def test_param_order_does_not_change_command():
    a = Command.create("t", "a", {"x": 1, "y": 2})
    b = Command.create("t", "a", {"y": 2, "x": 1})
    assert a == b


# --- approval: lifecycle ------------------------------------------------------

def test_missing_approval_fails_closed():
    assert check(approval=None).reason == "missing_approval"


def test_unknown_approval_status_fails_closed():
    assert check(status=None).reason == "unknown_approval"


def test_revoked_approval_is_rejected():
    assert check(status=Status.REVOKED).reason == "approval_revoked"


def test_used_approval_is_rejected():
    assert check(status=Status.USED).reason == "approval_used"


def test_approval_expires_at_ttl():
    # TTL 60 s from issue at 100.0; state refreshed so only expiry can fail.
    late = check(snapshot=snap(received_at=159.0), now=160.0)
    assert late.reason == "approval_expired"


def test_approval_from_another_clock_domain_is_rejected():
    foreign = issue_approval("ap-2", cmd(), snap(clock_id="boot-B"), POLICY, "boot-B", 100.0)
    assert check(approval=foreign).reason == "clock_domain_mismatch"


def test_approval_issued_in_the_future_is_rejected():
    future = issue_approval("ap-3", cmd(), snap(), POLICY, CLOCK, now=150.0)
    assert check(approval=future, now=101.0).reason == "approval_from_future"


def test_cannot_issue_approval_for_unknown_state():
    with pytest.raises(ValueError):
        issue_approval("ap-4", cmd(), snap(version=None), POLICY, CLOCK, 100.0)


def test_cannot_issue_approval_against_another_targets_snapshot():
    # The digest holds only the version number; both targets can be at version 7.
    with pytest.raises(ValueError):
        issue_approval("ap-5", cmd(), snap(target="demo-actuator-2"), POLICY, CLOCK,
                       100.0)


# --- policy -------------------------------------------------------------------

@pytest.mark.parametrize("age,ttl", [(0, 60), (-1, 60), (5, 0), (math.nan, 60), (5, math.inf)])
def test_policy_rejects_nonpositive_or_nonfinite_limits(age, ttl):
    with pytest.raises(ValueError):
        Policy("bad", max_state_age_s=age, approval_ttl_s=ttl)
