import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from device_journal import (
    NEEDS_RECONCILIATION,
    OBSERVED_MATCH,
    PENDING,
    REJECTED,
    CommandJournal,
    InvalidCommand,
    PayloadConflict,
    reconcile_pending,
    send,
    submit,
)
from fake_device import FakeDevice, SimulatedCrash

EXAMPLES = Path(__file__).resolve().parent.parent
OFF = {"power": "off", "brightness": 0}
ON = {"power": "on", "brightness": 100}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "journal.db"


def kinds(journal, correlation_id):
    return [e["kind"] for e in journal.events(correlation_id)]


def run(journal, device, desired=ON, op="op-1", corr="corr-1"):
    return submit(journal, device, operation_id=op, correlation_id=corr, desired=desired)


def test_success_is_read_back_and_recorded_as_observed_match(db):
    journal, device = CommandJournal(db), FakeDevice("sim-lamp-1", OFF)

    assert run(journal, device) == OBSERVED_MATCH
    assert len(device.applied) == 1
    assert device.read_state() == ON
    assert kinds(journal, "corr-1") == [
        "INTENT_RECORDED", "ATTEMPT_STARTED", "ATTEMPT_ACKED", "READBACK"]


def test_same_operation_id_and_payload_is_not_sent_again(db):
    journal, device = CommandJournal(db), FakeDevice("sim-lamp-1", OFF)
    run(journal, device)

    assert run(journal, device) == OBSERVED_MATCH
    assert len(device.applied) == 1
    assert kinds(journal, "corr-1")[-1] == "DUPLICATE_SUBMIT"


def test_same_operation_id_with_different_payload_is_rejected(db):
    journal, device = CommandJournal(db), FakeDevice("sim-lamp-1", OFF)
    run(journal, device)

    with pytest.raises(PayloadConflict):
        run(journal, device, desired={"power": "on", "brightness": 40})
    with pytest.raises(PayloadConflict):
        submit(journal, FakeDevice("sim-lamp-2", OFF),
               operation_id="op-1", correlation_id="corr-1", desired=ON)
    assert len(device.applied) == 1
    assert journal.get("op-1").desired == ON


@pytest.mark.parametrize("desired", [
    {}, ["power", "on"], None, {"power": "ON"}, {"power": None}, {"power": True},
    {"brightness": True}, {"brightness": 101}, {"brightness": -1}, {"brightness": 50.0},
    {"brightness": "100"}, {"brightness": None}, {"power": "on", "color": "red"},
])
def test_invalid_command_is_refused_before_journal_and_device(db, desired):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", {}, write_faults=["timeout_before_apply"])

    with pytest.raises(InvalidCommand):
        run(journal, device, desired=desired)
    assert device.applied == [] and device.reads == 0
    assert journal.con.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert journal.con.execute("SELECT count(*) FROM operations").fetchone()[0] == 0
    assert device.write_faults == ["timeout_before_apply"]  # no call consumed a fault


@pytest.mark.parametrize("name", ["op", "corr", "device_id"])
@pytest.mark.parametrize("value", ["", " \t", None, 7, False, []])
def test_invalid_ids_are_refused_before_journal_and_device(db, name, value):
    ids = {"op": "op-1", "corr": "corr-1", "device_id": "sim-lamp-1"}
    ids[name] = value
    journal, device = CommandJournal(db), FakeDevice(ids["device_id"], OFF)

    with pytest.raises(InvalidCommand):
        run(journal, device, op=ids["op"], corr=ids["corr"])
    assert device.applied == [] and device.reads == 0
    assert journal.con.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert journal.con.execute("SELECT count(*) FROM operations").fetchone()[0] == 0


def test_boolean_brightness_does_not_count_as_the_same_integer_command(db):
    journal, device = CommandJournal(db), FakeDevice("sim-lamp-1", OFF)
    assert run(journal, device, desired={"brightness": 1}) == OBSERVED_MATCH
    events_before = journal.events("corr-1")

    with pytest.raises(InvalidCommand):
        run(journal, device, desired={"brightness": True})
    assert journal.events("corr-1") == events_before
    assert journal.get("op-1").desired == {"brightness": 1}
    assert type(journal.get("op-1").desired["brightness"]) is int
    assert len(device.applied) == 1


@pytest.mark.parametrize("reported, brightness", [
    ({}, 1), ({"brightness": None}, 1), ({"brightness": True}, 1),
    ({"brightness": False}, 0), ({"brightness": 1.0}, 1),
])
def test_readback_needs_the_field_present_with_the_same_type(db, reported, brightness):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", reported, write_faults=["timeout_before_apply"])

    assert run(journal, device, desired={"brightness": brightness}) == NEEDS_RECONCILIATION
    assert device.applied == []
    assert device.reads == 1
    assert journal.events("corr-1")[-1]["detail"]["observed"] == reported


@pytest.mark.parametrize("desired", [
    {"power": "on"}, {"power": "off"}, {"brightness": 0}, {"brightness": 1},
    {"brightness": 100}, {"power": "off", "brightness": 100},
])
def test_valid_lamp_fields_and_boundaries_match_with_unrequested_fields(db, desired):
    journal, device = CommandJournal(db), FakeDevice("sim-lamp-1", {**OFF, "extra": 42})
    assert run(journal, device, desired=desired) == OBSERVED_MATCH
    assert device.applied[0][1] == desired


def test_applied_then_timeout_is_verified_by_reading_state(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["timeout_after_apply"])

    assert run(journal, device) == OBSERVED_MATCH
    assert len(device.applied) == 1
    assert "ATTEMPT_ACKED" not in kinds(journal, "corr-1")
    assert kinds(journal, "corr-1")[-2:] == ["ATTEMPT_TIMEOUT", "READBACK"]


def test_timeout_without_apply_needs_reconciliation_not_rejection(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["timeout_before_apply"])

    assert run(journal, device) == NEEDS_RECONCILIATION
    assert device.applied == []
    assert device.read_state() == OFF


def test_rejection_is_recorded_and_device_state_is_unchanged(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["reject"])

    assert run(journal, device) == REJECTED
    assert device.applied == []
    assert device.reads == 0
    assert device.read_state() == OFF


def test_read_timeout_after_ack_needs_reconciliation(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, read_faults=["timeout"])

    assert run(journal, device) == NEEDS_RECONCILIATION
    assert len(device.applied) == 1
    assert kinds(journal, "corr-1")[-1] == "READ_TIMEOUT"


def test_acknowledged_write_with_different_state_needs_reconciliation(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["clamp"])

    assert run(journal, device) == NEEDS_RECONCILIATION
    assert device.applied[0][1] == {"power": "on", "brightness": 80}
    readback = journal.events("corr-1")[-1]
    assert readback["detail"]["observed"]["brightness"] == 80


def test_intent_and_attempt_are_committed_before_device_call(db):
    seen = {}

    class Inspecting(FakeDevice):
        def set_state(self, desired, *, attempt_id):
            other = sqlite3.connect(db)  # a separate connection sees only committed rows
            seen["status"] = other.execute(
                "SELECT status FROM operations WHERE operation_id = 'op-1'").fetchone()
            seen["attempt"] = other.execute(
                "SELECT attempt_id FROM events WHERE kind = 'ATTEMPT_STARTED'").fetchone()
            other.close()
            super().set_state(desired, attempt_id=attempt_id)

    journal, device = CommandJournal(db), Inspecting("sim-lamp-1", OFF)
    run(journal, device)

    assert seen["status"] == (PENDING,)
    assert seen["attempt"] == (device.applied[0][0],)


def test_crash_after_apply_leaves_intent_and_restart_does_not_resend(db):
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["crash_after_apply"])
    with pytest.raises(SimulatedCrash):
        run(CommandJournal(db), device)

    restarted = CommandJournal(db)
    assert [op.operation_id for op in restarted.pending()] == ["op-1"]
    with pytest.raises(RuntimeError):
        send(restarted, device, "op-1")  # outcome unknown: reconcile first
    assert reconcile_pending(restarted, device) == {"op-1": OBSERVED_MATCH}
    assert run(restarted, device) == OBSERVED_MATCH
    assert len(device.applied) == 1


def test_crash_before_send_restart_reports_mismatch_without_sending(db):
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["crash_before_send"])
    with pytest.raises(SimulatedCrash):
        run(CommandJournal(db), device)

    restarted = CommandJournal(db)
    assert reconcile_pending(restarted, device) == {"op-1": NEEDS_RECONCILIATION}
    assert device.applied == []
    assert restarted.pending() == []


def test_hard_process_exit_after_device_write_leaves_pending_intent(db):
    child = f"""
import os
from device_journal import CommandJournal, submit
from fake_device import FakeDevice

class DiesAfterWrite(FakeDevice):
    def set_state(self, desired, *, attempt_id):
        super().set_state(desired, attempt_id=attempt_id)
        os._exit(70)

submit(CommandJournal({str(db)!r}), DiesAfterWrite("sim-lamp-1", {OFF!r}),
       operation_id="op-1", correlation_id="corr-1", desired={ON!r})
"""
    result = subprocess.run([sys.executable, "-c", child], cwd=EXAMPLES)

    assert result.returncode == 70
    journal = CommandJournal(db)
    assert [op.status for op in journal.pending()] == [PENDING]
    assert kinds(journal, "corr-1") == ["INTENT_RECORDED", "ATTEMPT_STARTED"]


def test_explicit_resend_after_reconciliation_uses_new_attempt_id(db):
    journal = CommandJournal(db)
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["timeout_before_apply"])
    assert run(journal, device) == NEEDS_RECONCILIATION

    assert send(journal, device, "op-1") == OBSERVED_MATCH
    attempts = [e["attempt_id"] for e in journal.events("corr-1")
                if e["kind"] == "ATTEMPT_STARTED"]
    assert len(attempts) == len(set(attempts)) == 2
    assert device.applied == [(attempts[1], ON)]
    with pytest.raises(RuntimeError):
        send(journal, device, "op-1")  # already observed: nothing to resend


def test_crash_during_explicit_resend_blocks_another_blind_send(db):
    device = FakeDevice("sim-lamp-1", OFF,
                        write_faults=["timeout_before_apply", "crash_after_apply"])
    journal = CommandJournal(db)
    assert run(journal, device) == NEEDS_RECONCILIATION
    with pytest.raises(SimulatedCrash):
        send(journal, device, "op-1")

    restarted = CommandJournal(db)
    assert [op.operation_id for op in restarted.pending()] == ["op-1"]
    with pytest.raises(RuntimeError):
        send(restarted, device, "op-1")
    assert len(device.applied) == 1


def test_correlation_id_groups_events_of_several_operations(db):
    journal = CommandJournal(db)
    lamp1, lamp2 = FakeDevice("sim-lamp-1", OFF), FakeDevice("sim-lamp-2", OFF)
    submit(journal, lamp1, operation_id="op-a", correlation_id="req-7", desired=ON)
    submit(journal, lamp2, operation_id="op-b", correlation_id="req-7", desired=ON)
    submit(journal, lamp1, operation_id="op-c", correlation_id="req-8", desired=OFF)

    ops = {e["operation_id"] for e in journal.events("req-7")}
    assert ops == {"op-a", "op-b"}


def test_journal_rollback_does_not_undo_device_write(db):
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE note (text TEXT)")
    device = FakeDevice("sim-lamp-1", OFF)

    with pytest.raises(RuntimeError):
        with con:
            con.execute("INSERT INTO note VALUES ('sent')")
            device.set_state(ON, attempt_id="a-1")
            raise RuntimeError("failure before COMMIT")

    assert con.execute("SELECT count(*) FROM note").fetchone() == (0,)
    assert device.read_state() == ON
    con.close()


def test_repeated_toggle_differs_from_repeated_set():
    toggled = FakeDevice("sim-lamp-1", OFF)
    toggled.toggle("power")
    toggled.toggle("power")  # a "retry" of a toggle undoes the first one
    assert toggled.read_state()["power"] == "off"

    setter = FakeDevice("sim-lamp-1", OFF)
    setter.set_state({"power": "on"}, attempt_id="a-1")
    setter.set_state({"power": "on"}, attempt_id="a-2")
    assert setter.read_state()["power"] == "on"
