"""Minimal bill allocation helpers for the REST API (POC)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class PaymentAllocation:
    matched_bill_ids: tuple[str, ...]
    allocated_sum_minor: int
    payment_amount_minor: int

    @property
    def remaining_minor(self) -> int:
        return max(0, self.payment_amount_minor - self.allocated_sum_minor)

    @property
    def is_fully_allocated(self) -> bool:
        return self.remaining_minor == 0 and len(self.matched_bill_ids) > 0


def get_payment_allocation(
    conn: sqlite3.Connection,
    tenant_id: str,
    payment_id: str,
) -> PaymentAllocation:
    pay = conn.execute(
        "SELECT amount_minor FROM payments WHERE payment_id = ? AND tenant_id = ?",
        (payment_id, tenant_id),
    ).fetchone()
    amount = int(pay["amount_minor"]) if pay else 0
    rows = conn.execute(
        """
        SELECT bill_id, open_amount_minor FROM bills
        WHERE tenant_id = ? AND matched_payment_id = ?
        """,
        (tenant_id, payment_id),
    ).fetchall()
    bill_ids = tuple(r["bill_id"] for r in rows)
    total = sum(int(r["open_amount_minor"]) for r in rows)
    return PaymentAllocation(
        matched_bill_ids=bill_ids,
        allocated_sum_minor=total,
        payment_amount_minor=amount,
    )


def apply_accepted_bills(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    bill_ids: list[str],
) -> None:
    """Mark bills as allocated to this payment."""
    if not bill_ids:
        raise ValueError("bill_ids must be non-empty")
    for bid in bill_ids:
        row = conn.execute(
            "SELECT matched_payment_id FROM bills WHERE bill_id = ? AND tenant_id = ?",
            (bid, tenant_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown bill_id {bid}")
        if row["matched_payment_id"] and row["matched_payment_id"] != payment_id:
            raise ValueError(f"bill {bid} already allocated to {row['matched_payment_id']}")
    for bid in bill_ids:
        conn.execute(
            "UPDATE bills SET matched_payment_id = ? WHERE bill_id = ? AND tenant_id = ?",
            (payment_id, bid, tenant_id),
        )
    conn.commit()


def unmatch_payment(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
) -> list[str]:
    """Clear all bill allocations for a payment. Returns freed bill IDs."""
    rows = conn.execute(
        "SELECT bill_id FROM bills WHERE tenant_id = ? AND matched_payment_id = ?",
        (tenant_id, payment_id),
    ).fetchall()
    freed = [r["bill_id"] for r in rows]
    if freed:
        conn.execute(
            "UPDATE bills SET matched_payment_id = NULL WHERE tenant_id = ? AND matched_payment_id = ?",
            (tenant_id, payment_id),
        )
        conn.commit()
    return freed
