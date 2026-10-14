"""Walk through the recovery scenarios on synthetic data.

Run from this directory:  python3 hitl_demo.py
Needs only hitl_db.py, hitl_device.py and hitl_coordinator.py next to it.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from hitl_coordinator import ActionSpec, Coordinator, Refused, SimulatedCrash
from hitl_device import SyntheticDevice

OPERATOR = "operator-1 (simulated, not authenticated)"
NOW = 1_000


def check(ok: bool, name: str) -> None:
    if not ok:
        raise AssertionError(f"FAIL: {name}")
    print(f"PASS: {name}")


def open_world(folder: Path):
    """(Re)start: new objects, state only from the two files on disk."""
    device = SyntheticDevice(folder / "device.db")
    return Coordinator(folder / "coordinator.db", device, lambda: NOW), device


def child(folder: str, spec: list, approval: str) -> None:
    coordinator, _ = open_world(Path(folder))
    coordinator.execute(ActionSpec(*spec), approval, fault="exit_after_device_commit")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        coordinator, device = open_world(folder)
        device.init_setting("setpoint", 205)  # 20.5 °C, in tenths

        # 1. Approved change; the worker process really exits after the device commit.
        spec = coordinator.propose_change("setpoint", 190)
        approval = coordinator.approve(spec, OPERATOR, ttl_seconds=300)
        code = subprocess.run([sys.executable, __file__, "--child", tmp,
                               repr(list(spec)), approval]).returncode
        check(code == 70, "worker exited after device commit, before acknowledgement")
        coordinator.close(); device.close()
        coordinator, device = open_world(folder)
        [change] = coordinator.unresolved()
        check(coordinator.state(change) == "INTENT", "after restart intent is durable, outcome unknown")
        check(device.read("setpoint") == (190, 2), "device already holds the new value")
        check(coordinator.reconcile() == {change: "APPLIED"}, "reconcile confirms from device log")

        # 2. The old approval cannot authorise anything else.
        try:
            coordinator.execute(spec, approval)
            check(False, "approval reuse refused")
        except Refused as error:
            check("used" in str(error), f"approval reuse refused ({error})")

        # 3. Compensation is a new action and needs a new approval.
        comp = coordinator.propose_compensation(change)
        print(f"      proposed compensation: {comp.from_value} -> {comp.to_value} "
              f"if device is still at v{comp.expected_version}")
        comp_approval = coordinator.approve(comp, OPERATOR, ttl_seconds=300)
        try:
            coordinator.execute(comp, comp_approval, fault="crash_after_device_commit")
        except SimulatedCrash:
            print("      worker crashed during compensation, after device commit")
        coordinator.close(); device.close()
        coordinator, device = open_world(folder)
        [comp_action] = coordinator.unresolved()
        check(coordinator.reconcile() == {comp_action: "APPLIED"}, "compensation reconciled")
        check(coordinator.state(change) == "COMPENSATED", "original change marked compensated")
        check(device.read("setpoint") == (205, 3), "device back to 20.5 °C, at v3")

        # 4. Someone at the device interferes; we do not overwrite them.
        spec = coordinator.propose_change("setpoint", 180)
        approval = coordinator.approve(spec, OPERATOR, ttl_seconds=300)
        action = coordinator.execute(spec, approval)
        device.external_write("setpoint", 215)
        try:
            coordinator.propose_compensation(action)
            check(False, "compensation refused after interference")
        except Refused as error:
            check("diverged" in str(error), f"compensation refused ({error})")
        check(device.read("setpoint") == (215, 5), "external value 21.5 °C preserved")

        print("\nevents for the first change:", ", ".join(coordinator.events(change)))
        coordinator.close(); device.close()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        import ast
        child(sys.argv[2], ast.literal_eval(sys.argv[3]), sys.argv[4])
    else:
        main()
