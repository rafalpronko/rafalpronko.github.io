"""Synthetic invoice-line fixtures. No real invoices, customers or tax data."""

from __future__ import annotations

from typing import Any

from invoice_contract import EXPECTED_ENVELOPE

# Synthetic reference median of net_minor for the advisory monitor.
# It is an invented number for the demo, not a measured benchmark.
SYNTHETIC_BASELINE_MEDIAN_NET = 4_550


def line(line_id: str, net: int, tax_bp: int, tax: int, gross: int,
         invoice_id: str = "INV-1") -> dict[str, Any]:
    return {"line_id": line_id, "invoice_id": invoice_id, "net_minor": net,
            "tax_bp": tax_bp, "tax_minor": tax, "gross_minor": gross}


def canonical_batch() -> dict[str, Any]:
    """Valid batch. L-3 is the half-up case: 4550 * 5% = 227.5 -> 228."""
    return {**EXPECTED_ENVELOPE, "rows": [
        line("L-1", 10_000, 2000, 2_000, 12_000),
        line("L-2", 1_999, 1000, 200, 2_199),
        line("L-3", 4_550, 500, 228, 4_778, invoice_id="INV-2"),
    ]}


def consistent_unit_lie_batch() -> dict[str, Any]:
    """Whole euros sent while the envelope declares cents.

    Every relation holds, so the hard contract cannot see the problem.
    """
    return {**EXPECTED_ENVELOPE, "rows": [
        line("M-1", 100, 2000, 20, 120),
        line("M-2", 20, 1000, 2, 22),
        line("M-3", 46, 500, 2, 48, invoice_id="INV-2"),
    ]}
