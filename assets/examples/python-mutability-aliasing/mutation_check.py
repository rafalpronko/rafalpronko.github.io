"""Check that the tests for the fixed version can see aliasing.

Runs the "fixed_version" tests with tag_orders replaced by the buggy
tag_orders_shallow. Exit 0 means the outcome is exactly the expected
one: the four ownership tests fail, the value-only test passes and
pytest reports test failures (exit code 1).
Run: python mutation_check.py   (needs pytest)
"""
import sys

import pytest

import aliasing

EXPECTED_FAILED = {
    "test_aliasing.py::test_fixed_version_leaves_input_unchanged",
    "test_aliasing.py::test_fixed_version_keeps_sibling_results_independent",
    "test_aliasing.py::test_fixed_version_owns_every_mutable_component",
    "test_aliasing.py::"
    "test_fixed_version_separates_aliases_present_in_the_input",
}
# A value-only test cannot see aliasing: on a fresh input the buggy
# function returns equal values.
EXPECTED_PASSED = {
    "test_aliasing.py::test_fixed_version_returns_expected_values",
}


class SwapFixedForShallow:
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            assert isinstance(item, pytest.Function)
            setattr(item.module, "tag_orders", aliasing.tag_orders_shallow)


class Outcomes:
    def __init__(self) -> None:
        self.failed: set[str] = set()
        self.passed: set[str] = set()
        self.unexpected: list[str] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            if not report.passed:
                self.unexpected.append(
                    f"{report.nodeid}: {report.when} {report.outcome}"
                )
        elif report.failed:
            self.failed.add(report.nodeid)
        elif report.passed:
            self.passed.add(report.nodeid)
        else:
            self.unexpected.append(f"{report.nodeid}: {report.outcome}")


def main() -> int:
    outcomes = Outcomes()
    exit_code = pytest.main(
        ["-q", "-k", "fixed_version", "test_aliasing.py"],
        plugins=[SwapFixedForShallow(), outcomes],
    )
    print(
        "with tag_orders replaced by tag_orders_shallow:",
        f"{len(outcomes.failed)} failed, {len(outcomes.passed)} passed,",
        f"pytest exit code {int(exit_code)}",
    )
    for nodeid in sorted(outcomes.failed):
        print("DETECTED (failed):", nodeid)
    for nodeid in sorted(outcomes.passed):
        label = "EXPECTED PASS (value-only):"
        if nodeid not in EXPECTED_PASSED:
            label = "NOT DETECTED:"
        print(label, nodeid)
    ok = (
        outcomes.failed == EXPECTED_FAILED
        and outcomes.passed == EXPECTED_PASSED
        and exit_code == pytest.ExitCode.TESTS_FAILED
        and not outcomes.unexpected
    )
    for outcome in outcomes.unexpected:
        print("UNEXPECTED REPORT:", outcome)
    print("mutation check:", "as expected" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
