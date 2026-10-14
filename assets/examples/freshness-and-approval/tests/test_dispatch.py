"""Dispatch through the gate to a fake in-memory target. All values are synthetic."""

import pytest

from fake_target import (
    ApprovalLedger,
    Dispatcher,
    FakeTarget,
    ManualClock,
    MonotonicClock,
)
from freshness_gate import Command, Policy, Status

# Illustrative numbers for a test, not recommended values.
POLICY = Policy("demo-low-risk", max_state_age_s=5.0, approval_ttl_s=60.0)


def setup(clock_id: str = "boot-A"):
    target = FakeTarget("demo-actuator-1", level=1, version=7)
    ledger = ApprovalLedger()
    clock = ManualClock(clock_id, start=100.0)
    return target, ledger, clock, Dispatcher(target, ledger, POLICY, clock)


def set_level(level: int) -> Command:
    return Command.create("demo-actuator-1", "set_level", {"level": level})


def test_happy_path_applies_once_and_bumps_version():
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "applied"
    assert target.read() == (8, {"level": 3})


def test_approval_is_single_use():
    _, ledger, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.dispatch(set_level(3), snap, "ap-1", "key-1")
    assert ledger.get("ap-1")[1] is Status.USED
    assert d.dispatch(set_level(3), snap, "ap-1", "key-2") == "approval_used"


def test_revoked_approval_does_not_reach_target():
    target, ledger, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    assert ledger.revoke("ap-1") is True
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_revoked"
    assert target.read() == (7, {"level": 1})


def test_unknown_approval_id_fails_closed():
    target, _, _, d = setup()
    snap = d.observe()
    assert d.dispatch(set_level(3), snap, "never-issued", "key-1") == "missing_approval"
    assert target.read()[0] == 7


def test_human_delay_makes_state_stale_even_with_valid_approval():
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    clock.advance(30)  # approval still inside its TTL, state is not
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "stale_state"
    assert target.read()[0] == 7


def test_fresh_reread_after_change_needs_new_approval():
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    target.external_change(level=5)  # someone else acted: version 8
    clock.advance(1)
    fresh = d.observe()
    assert d.dispatch(set_level(3), fresh, "ap-1", "key-1") == "approval_binding_mismatch"


def test_toctou_conditional_write_rejects_change_after_check():
    """Gate passes, then the target changes before the write lands."""
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.before_write = lambda: target.external_change(level=5)
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "precondition_failed"
    assert target.read() == (8, {"level": 5})


def test_toctou_unconditional_write_silently_overwrites():
    """The same race without a target-side condition: the hazard the test above blocks."""
    target, _, _, _ = setup()
    version_seen = target.read()[0]
    target.external_change(level=5)
    target.write_unconditional(set_level(3))
    assert version_seen == 7 and target.read() == (9, {"level": 3})


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


def test_restart_invalidates_old_approvals_and_snapshots():
    """Our policy, not a property of the clock: a new instance means a new clock id."""
    target, ledger, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    restarted = Dispatcher(target, ledger, POLICY, ManualClock("boot-B", 100.0))
    assert restarted.dispatch(set_level(3), snap, "ap-1", "key-1") == "clock_domain_mismatch"
    assert target.read()[0] == 7


def test_monotonic_clock_instances_are_separate_domains():
    a, b = MonotonicClock(), MonotonicClock()
    assert a.clock_id != b.clock_id
    assert a.now() <= a.now()


def test_missing_state_fails_closed_at_dispatch():
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    assert d.dispatch(set_level(3), None, "ap-1", "key-1") == "missing_state"
    assert target.read()[0] == 7


# --- target routing: issuance, dispatcher and target all check it -------------

def two_targets():
    a = FakeTarget("demo-actuator-1", level=1, version=7)
    b = FakeTarget("demo-actuator-2", level=1, version=7)  # same version on purpose
    ledger = ApprovalLedger()
    clock = ManualClock("boot-A", start=100.0)
    da = Dispatcher(a, ledger, POLICY, clock)
    db = Dispatcher(b, ledger, POLICY, clock)
    return a, b, ledger, da, db


def test_approval_for_one_target_is_not_dispatched_to_another():
    a, b, ledger, da, db = two_targets()
    snap_a = da.observe()
    da.approve("ap-1", set_level(3), snap_a)
    assert db.dispatch(set_level(3), snap_a, "ap-1", "key-1") == "wrong_target"
    assert a.read() == (7, {"level": 1}) and b.read() == (7, {"level": 1})
    assert ledger.get("ap-1")[1] is Status.ACTIVE  # refused before it was spent


def test_issuance_refuses_a_snapshot_of_another_target():
    a, b, ledger, da, db = two_targets()
    snap_b = db.observe()
    with pytest.raises(ValueError):
        da.approve("ap-1", set_level(3), snap_b)
    assert ledger.get("ap-1") == (None, None)


def test_target_refuses_conditional_write_addressed_elsewhere():
    b = FakeTarget("demo-actuator-2", level=1, version=7)
    result = b.write_if_match(expected_version=7, command=set_level(3),
                              idempotency_key="k")
    assert result.status == "wrong_target"
    assert b.read() == (7, {"level": 1}) and b.lookup("k") is None


def test_unconditional_write_still_refuses_another_target():
    """The unconditional path demonstrates the state race, never a routing error."""
    b = FakeTarget("demo-actuator-2", level=1, version=7)
    assert b.write_unconditional(set_level(3)).status == "wrong_target"
    assert b.read() == (7, {"level": 1})


# --- policy: the whole policy is bound, not only its id -----------------------

def test_changed_limits_under_same_policy_id_invalidate_approval():
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.policy = Policy("demo-low-risk", max_state_age_s=5.0, approval_ttl_s=1.0)
    clock.advance(2)
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_binding_mismatch"
    assert target.read()[0] == 7


# --- authorization cutoff: the successful consume -----------------------------

def test_revocation_after_consume_cannot_recall_the_write():
    target, ledger, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    revoked = []
    d.before_write = lambda: revoked.append(ledger.revoke("ap-1"))
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "applied"
    assert revoked == [False]  # too late: the approval was already USED
    assert target.read() == (8, {"level": 3})


def test_expiry_is_not_rechecked_after_consume():
    """Known limit: the version condition checks state, not the approval deadline."""
    target, _, clock, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)  # expires at 160.0
    d.before_write = lambda: clock.advance(61)  # the write lands at 161.0
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "applied"
    assert target.read() == (8, {"level": 3})


# --- the cutoff's clock sample is taken inside the ledger's atomic step -------

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


def test_state_going_stale_before_consume_is_denied_and_not_spent():
    target, ledger, clock, d = setup()
    snap = d.observe()  # received at 100.0, stale after 105.0
    d.approve("ap-1", set_level(3), snap)
    clock.advance(4.9)
    d.before_consume = lambda: clock.advance(0.2)  # the consume runs at 105.1
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "stale_state"
    assert target.read() == (7, {"level": 1}) and target.lookup("key-1") is None
    assert ledger.get("ap-1")[1] is Status.ACTIVE


def test_revocation_before_consume_is_reported_as_revocation():
    target, ledger, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.before_consume = lambda: ledger.revoke("ap-1")
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_revoked"
    assert target.read() == (7, {"level": 1}) and target.lookup("key-1") is None
    assert ledger.get("ap-1")[1] is Status.REVOKED


class LockProbeClock(ManualClock):
    """Records, for every reading, whether the ledger lock was held at that moment."""

    def __init__(self, ledger: ApprovalLedger) -> None:
        super().__init__("boot-A", start=100.0)
        self.ledger = ledger
        self.lock_held: list[bool] = []

    def now(self) -> float:
        self.lock_held.append(self.ledger._lock.locked())
        return super().now()


def test_dispatch_reads_the_clock_only_while_holding_the_ledger_lock():
    target = FakeTarget("demo-actuator-1", level=1, version=7)
    ledger = ApprovalLedger()
    clock = LockProbeClock(ledger)
    d = Dispatcher(target, ledger, POLICY, clock)
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    clock.lock_held.clear()  # only the readings taken by dispatch() matter
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "applied"
    assert clock.lock_held and all(clock.lock_held)


# --- idempotency is not authorization ----------------------------------------

def test_lost_response_is_resolved_by_lookup_not_by_reusing_approval():
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.dispatch(set_level(3), snap, "ap-1", "key-1")  # pretend the reply was lost
    assert d.dispatch(set_level(3), snap, "ap-1", "key-1") == "approval_used"
    assert target.lookup("key-1").status == "applied"
    assert target.read()[0] == 8  # applied exactly once


def test_target_deduplicates_same_key_and_same_command():
    target = FakeTarget("demo-actuator-1", level=1, version=7)
    first = target.write_if_match(expected_version=7, command=set_level(3), idempotency_key="k")
    again = target.write_if_match(expected_version=7, command=set_level(3), idempotency_key="k")
    assert (first.status, again.status) == ("applied", "duplicate")
    assert target.read()[0] == 8


def test_reused_key_with_new_approved_command_is_a_conflict():
    target, _, _, d = setup()
    snap = d.observe()
    d.approve("ap-1", set_level(3), snap)
    d.dispatch(set_level(3), snap, "ap-1", "key-1")
    snap2 = d.observe()
    d.approve("ap-2", set_level(9), snap2)
    assert d.dispatch(set_level(9), snap2, "ap-2", "key-1") == "key_conflict"
    assert target.read() == (8, {"level": 3})


def test_idempotency_key_alone_does_not_authorize():
    target, _, _, d = setup()
    snap = d.observe()
    assert d.dispatch(set_level(3), snap, "no-approval", "key-1") == "missing_approval"
    assert target.lookup("key-1") is None
