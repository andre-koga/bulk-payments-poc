"""Batch auto-match over open payments for a tenant."""
from __future__ import annotations

import sqlite3
from typing import Any

from bulk_payments.allocation import apply_accepted_bills, get_payment_allocation
from bulk_payments.db import insert_match_event
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind
from bulk_payments.synthetic import load_tenant_config


def run_batch_auto_match(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    ranker_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_tenant_config(conn, tenant_id)
    payments = conn.execute(
        "SELECT payment_id FROM payments WHERE tenant_id = ? ORDER BY payment_date",
        (tenant_id,),
    ).fetchall()

    auto_matched: list[dict[str, Any]] = []
    needs_review_results: list[tuple[Any, str | None]] = []
    no_match: list[dict[str, Any]] = []
    skipped = 0

    for row in payments:
        pid = row["payment_id"]
        alloc = get_payment_allocation(conn, tenant_id, pid)
        if alloc.is_fully_allocated:
            skipped += 1
            continue

        r = match_payment(
            conn,
            tenant_id=tenant_id,
            payment_id=pid,
            ranker_path=ranker_path,
            log_event=True,
        )
        event_row = conn.execute(
            "SELECT event_id FROM match_events WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1",
            (pid,),
        ).fetchone()
        event_id = event_row["event_id"] if event_row else None

        if r.decision == DecisionKind.AUTO_APPLIED and r.subsets:
            bill_ids = list(r.subsets[0].bill_ids)
            try:
                apply_accepted_bills(conn, tenant_id=tenant_id, payment_id=pid, bill_ids=bill_ids)
                insert_match_event(
                    conn,
                    tenant_id=tenant_id,
                    payment_id=pid,
                    bill_ids=bill_ids,
                    rules_version=cfg.rules_version,
                    features={**r.features, "batch_auto": True},
                    decision=r.decision.value,
                    outcome="accepted_as_is",
                    ranker_score=r.ranker_score,
                    calibrated_prob=r.calibrated_accept_prob,
                    reason_codes=list(r.reason_codes) + ["batch_auto_applied"],
                )
            except ValueError:
                needs_review_results.append((r, event_id))
                continue
            auto_matched.append(
                {"payment_id": pid, "status": "auto", "decision": r.decision.value, "bill_ids": bill_ids}
            )
        elif r.decision == DecisionKind.SUGGESTED:
            needs_review_results.append((r, event_id))
        else:
            no_match.append({"payment_id": pid, "status": "no_match", "decision": r.decision.value})

    return {
        "summary": {
            "processed": len(payments) - skipped,
            "auto_matched": len(auto_matched),
            "needs_review": len(needs_review_results),
            "no_match": len(no_match),
            "skipped_fully_allocated": skipped,
        },
        "auto_matched": auto_matched,
        "needs_review_results": needs_review_results,
        "no_match": no_match,
    }
