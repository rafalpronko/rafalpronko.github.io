"""Synthetic device with its own persistent store.

The device is deliberately a separate SQLite file: the coordinator cannot
include it in its own transactions, just as it could not with real hardware.
"""
from hitl_db import connect, transaction

DEVICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS setting (
    name TEXT PRIMARY KEY, value INTEGER NOT NULL, version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS applied (
    command_id TEXT PRIMARY KEY, name TEXT NOT NULL,
    value INTEGER NOT NULL, version INTEGER NOT NULL);
"""


class VersionMismatch(Exception):
    """Definite rejection: the device changed nothing."""


class SyntheticDevice:
    def __init__(self, path) -> None:
        self.con = connect(path)
        self.con.executescript(DEVICE_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def init_setting(self, name: str, value: int) -> None:
        with transaction(self.con):
            self.con.execute("INSERT INTO setting VALUES (?, ?, 1)", (name, value))

    def read(self, name: str) -> tuple[int, int]:
        return self.con.execute(
            "SELECT value, version FROM setting WHERE name = ?", (name,)).fetchone()

    def applied(self, command_id: str) -> tuple[int, int] | None:
        """Did this exact command take effect? Answers from the device's own log."""
        return self.con.execute(
            "SELECT value, version FROM applied WHERE command_id = ?",
            (command_id,)).fetchone()

    def apply(self, command_id: str, name: str, expected_version: int,
              value: int) -> tuple[int, int]:
        """Conditional, deduplicated write. Returns the observed (value, version)."""
        with transaction(self.con):
            done = self.applied(command_id)
            if done is not None:
                return done
            _, version = self.read(name)
            if version != expected_version:
                raise VersionMismatch(f"{name}: expected v{expected_version}, is v{version}")
            self.con.execute("UPDATE setting SET value = ?, version = ? WHERE name = ?",
                             (value, version + 1, name))
            self.con.execute("INSERT INTO applied VALUES (?, ?, ?, ?)",
                             (command_id, name, value, version + 1))
            return value, version + 1

    def external_write(self, name: str, value: int) -> tuple[int, int]:
        """Someone else (a wall panel, another tool) changes the setting."""
        with transaction(self.con):
            _, version = self.read(name)
            self.con.execute("UPDATE setting SET value = ?, version = ? WHERE name = ?",
                             (value, version + 1, name))
            return value, version + 1
