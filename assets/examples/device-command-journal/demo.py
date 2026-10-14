"""Walk through the fault scenarios on a synthetic device. Run: python demo.py"""
import sqlite3
from pathlib import Path

from device_journal import CommandJournal, PayloadConflict, reconcile_pending, send, submit
from fake_device import FakeDevice, SimulatedCrash

DB = Path("demo-journal.db")
OFF = {"power": "off", "brightness": 0}
ON = {"power": "on", "brightness": 100}


def scenario(journal, title, op, write_faults=(), read_faults=()):
    device = FakeDevice("sim-lamp-1", OFF, write_faults=write_faults, read_faults=read_faults)
    status = submit(journal, device, operation_id=op, correlation_id=f"req-{op}", desired=ON)
    print(f"{title:<28} status={status:<21} device writes={len(device.applied)}")
    return device


def main():
    DB.unlink(missing_ok=True)
    journal = CommandJournal(DB)

    print("== single attempts ==")
    device = scenario(journal, "success", "op-1")
    again = submit(journal, device, operation_id="op-1", correlation_id="req-op-1", desired=ON)
    print(f"{'same op, same payload':<28} status={again:<21} device writes={len(device.applied)}")
    try:
        submit(journal, device, operation_id="op-1", correlation_id="req-op-1",
               desired={"power": "on", "brightness": 40})
    except PayloadConflict as exc:
        print(f"{'same op, other payload':<28} PayloadConflict: {exc}")
    scenario(journal, "applied, then timeout", "op-2", write_faults=["timeout_after_apply"])
    lost = scenario(journal, "timeout before apply", "op-3", write_faults=["timeout_before_apply"])
    scenario(journal, "rejected", "op-4", write_faults=["reject"])
    scenario(journal, "acked, read timeout", "op-5", read_faults=["timeout"])
    scenario(journal, "acked, different state", "op-6", write_faults=["clamp"])

    print("\n== explicit resend of op-3 after reconciliation ==")
    print("status:", send(journal, lost, "op-3"), "| device writes:", len(lost.applied))
    for event in journal.events("req-op-3"):
        attempt = (event["attempt_id"] or "-")[:8]
        print(f"  {event['seq']:>3} {event['kind']:<17} attempt={attempt:<8} {event['detail'] or ''}")

    print("\n== process dies after the device applied the write ==")
    device = FakeDevice("sim-lamp-1", OFF, write_faults=["crash_after_apply"])
    try:
        submit(journal, device, operation_id="op-7", correlation_id="req-op-7", desired=ON)
    except SimulatedCrash as exc:
        print("crash:", exc)
    journal.close()
    restarted = CommandJournal(DB)
    print("pending after restart:", [op.operation_id for op in restarted.pending()])
    try:
        send(restarted, device, "op-7")
    except RuntimeError as exc:
        print("send refused:", exc)
    print("reconcile:", reconcile_pending(restarted, device), "| device writes:", len(device.applied))

    print("\n== a local rollback does not undo the device write ==")
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE note (text TEXT)")
    device = FakeDevice("sim-lamp-1", OFF)
    try:
        with con:
            con.execute("INSERT INTO note VALUES ('sent')")
            device.set_state(ON, attempt_id="a-1")
            raise RuntimeError("failure before COMMIT")
    except RuntimeError:
        pass
    print("journal rows:", con.execute("SELECT count(*) FROM note").fetchone()[0],
          "| device state:", device.read_state())
    con.close()

    print("\n== retrying a toggle versus retrying a set ==")
    toggled, setter = FakeDevice("sim-lamp-1", OFF), FakeDevice("sim-lamp-1", OFF)
    for attempt in ("a-1", "a-2"):
        toggled.toggle("power")
        setter.set_state({"power": "on"}, attempt_id=attempt)
    print("toggle twice:", toggled.read_state()["power"], "| set twice:", setter.read_state()["power"])
    restarted.close()


if __name__ == "__main__":
    main()
