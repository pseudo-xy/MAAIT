"""Pytest configuration that saves each test's outcome to a CSV.

After every `pytest` run, a file `results/pytest_results.csv` is written
(overwritten each run) with one row per test:

    test_name, outcome, duration_sec

This uses pytest's built-in reporting hook — no plugin needed. We record the
"call" phase (the actual test body) so durations reflect the test itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("results") / "pytest_results.csv"

# accumulates (name, outcome, duration) as tests run
_records: list[tuple[str, str, float]] = []


def pytest_runtest_logreport(report) -> None:
    """Called for each phase (setup/call/teardown) of each test. We keep the
    'call' phase, which is the test body itself."""
    if report.when == "call":
        _records.append((report.nodeid, report.outcome, report.duration))
    # a test that errors during setup never reaches 'call' — capture that too
    elif report.when == "setup" and report.outcome == "failed":
        _records.append((report.nodeid, "error", report.duration))


def pytest_sessionfinish(session, exitstatus) -> None:
    """After the whole run, write all records to the CSV (overwrite)."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "outcome", "duration_sec"])
        for name, outcome, duration in _records:
            writer.writerow([name, outcome, round(duration, 4)])


import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level caches that persist across tests (per-USV wave-sway
    phase keyed by id) so tests don't leak state into each other."""
    from reschu import usv_movement as _um
    _um._WAVE_PHASE.clear()
    _um._WAVE_PREV.clear()
    yield
    _um._WAVE_PHASE.clear()
    _um._WAVE_PREV.clear()
