"""Describe the machine and interpreter that produced a result file."""

from __future__ import annotations

import gc
import os
import platform
import shutil
import subprocess
import sys
import time


def _command_output(args: list[str]) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def cpu_model() -> str | None:
    system = platform.system()
    if system == "Darwin":
        return _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
    return platform.processor() or None


def describe_environment() -> dict:
    clock = time.get_clock_info("perf_counter")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": " ".join(platform.python_build()),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "uv_version": _command_output(["uv", "--version"]),
        "gc_enabled_during_timing": gc.isenabled(),
        "perf_counter": {
            "implementation": clock.implementation,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
            "reported_resolution_s": clock.resolution,
        },
        "argv": [os.path.basename(sys.argv[0]), *sys.argv[1:]],
    }
