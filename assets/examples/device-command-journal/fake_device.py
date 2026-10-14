"""A purely synthetic device simulator. It models no real product or protocol."""


class WriteTimeout(Exception):
    """No answer to a write. The write may or may not have been applied."""


class ReadTimeout(Exception):
    """No answer to a read. Nothing is known about the current state."""


class DeviceRejected(Exception):
    """This simulator's contract: a rejected write is not applied."""


class SimulatedCrash(BaseException):
    """Stands in for the process dying. Journal code does not catch it."""


class FakeDevice:
    def __init__(self, device_id, state, *, write_faults=(), read_faults=()):
        self.device_id = device_id
        self._state = dict(state)
        self.write_faults = list(write_faults)  # consumed one per set_state()
        self.read_faults = list(read_faults)    # consumed one per read_state()
        self.applied = []  # (attempt_id, fields) the device actually applied
        self.reads = 0

    def set_state(self, desired, *, attempt_id):
        fault = self.write_faults.pop(0) if self.write_faults else None
        if fault == "reject":
            raise DeviceRejected("value outside the simulated allowed range")
        if fault == "crash_before_send":
            raise SimulatedCrash("process died before the write left")
        if fault == "timeout_before_apply":
            raise WriteTimeout("write lost on the way to the device")
        applied = dict(desired)
        if fault == "clamp" and "brightness" in applied:
            applied["brightness"] = min(applied["brightness"], 80)
        self._state.update(applied)
        self.applied.append((attempt_id, applied))
        if fault == "timeout_after_apply":
            raise WriteTimeout("write applied, acknowledgement lost")
        if fault == "crash_after_apply":
            raise SimulatedCrash("process died after the device applied the write")

    def toggle(self, field):
        self._state[field] = "off" if self._state[field] == "on" else "on"
        self.applied.append((None, {field: self._state[field]}))

    def read_state(self):
        self.reads += 1
        fault = self.read_faults.pop(0) if self.read_faults else None
        if fault == "timeout":
            raise ReadTimeout("no answer to the state read")
        return dict(self._state)
