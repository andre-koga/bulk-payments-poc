from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from bulk_payments.fx import convert_to_ledger
from bulk_payments.synthetic import load_payment, load_tenant_config


@dataclass(frozen=True)
class PaymentAllocation:
    payment_id: str
    payment_amount_minor: int
    matched_bill_ids: tuple[str, ...]
    allocated_sum_minor: int
    remaining_minor: int
    is_fully_allocated: bool


def get_payment_allocation(conn: sqlite3.Connection, tenant_id: str, payment_id: str) -> PaymentAllocation:
    """Sum ledger amounts for bills already linked to this payment."""
    cfg = load_tenant_config(conn, tenant_id)
    payment = load_payment(conn, payment_id)
    rows = conn.execute(
        """
        SELECT bill_id, open_amount_minor, currency, open_date
        FROM bills WHERE matched_payment_id = ? AND tenant_id = ?
        ORDER BY bill_id
        """,
        (payment_id, tenant_id),
    ).fetchall()

    allocated = 0
    bill_ids: list[str] = []
    for r in rows:
        bill_ids.append(r["bill_id"])
        if r["currency"] == cfg.ledger_currency:
            allocated += int(r["open_amount_minor"])
        else:
            open_date = date.fromisoformat(r["open_date"])
            ledger_amt, _ = convert_to_ledger(
                conn,
                tenant_id=tenant_id,
                amount_minor=int(r["open_amount_minor"]),
                from_currency=r["currency"],
                to_currency=cfg.ledger_currency,
                as_of=payment.payment_date,
            )
            allocated += ledger_amt

    remaining = payment.amount_minor - allocated
    tol = cfg.amount_tolerance_minor
    fully = remaining <= tol

    return PaymentAllocation(
        payment_id=payment_id,
        payment_amount_minor=payment.amount_minor,
        matched_bill_ids=tuple(bill_ids),
        allocated_sum_minor=allocated,
        remaining_minor=max(0, remaining),
        is_fully_allocated=fully,
    )


def apply_accepted_bills(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    bill_ids: list[str],
) -> None:
    """Link bills to payment after accountant (or batch auto) acceptance."""
    cfg = load_tenant_config(conn, tenant_id)
    alloc = get_payment_allocation(conn, tenant_id, payment_id)
    new_bills = [b for b in bill_ids if b not in alloc.matched_bill_ids]
    add_sum = 0
    for bid in new_bills:
        brow = conn.execute(
            "SELECT open_amount_minor, currency, open_date FROM bills WHERE bill_id = ?",
            (bid,),
        ).fetchone()
        if brow is None:
            continue
        if brow["currency"] == cfg.ledger_currency:
            add_sum += int(brow["open_amount_minor"])
        else:
            pay = load_payment(conn, payment_id)
            add_sum += convert_to_ledger(
                conn,
                tenant_id=tenant_id,
                amount_minor=int(brow["open_amount_minor"]),
                from_currency=brow["currency"],
                to_currency=cfg.ledger_currency,
                as_of=pay.payment_date,
            )[0]
    total_after = alloc.allocated_sum_minor + add_sum
    if total_after > alloc.payment_amount_minor + cfg.amount_tolerance_minor:
        raise ValueError(
            f"Cannot allocate ${total_after / 100:.2f} to a ${alloc.payment_amount_minor / 100:.2f} payment"
        )
    for bid in bill_ids:
        conn.execute(
            "UPDATE bills SET matched_payment_id = ? WHERE bill_id = ? AND matched_payment_id IS NULL",
            (payment_id, bid),
        )
    conn.commit()


def unmatch_payment(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
) -> tuple[str, ...]:
    """Clear matched_payment_id on all bills for this payment. Returns freed bill ids."""
    payment = load_payment(conn, payment_id)
    if payment.tenant_id != tenant_id:
        raise ValueError("payment tenant mismatch")

    alloc = get_payment_allocation(conn, tenant_id, payment_id)
    if not alloc.matched_bill_ids:
        return ()

    conn.execute(
        """
        UPDATE bills SET matched_payment_id = NULL
        WHERE matched_payment_id = ? AND tenant_id = ?
        """,
        (payment_id, tenant_id),
    )
    conn.commit()
    return alloc.matched_bill_ids
