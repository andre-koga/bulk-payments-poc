from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from bulk_payments.allocation import apply_accepted_bills, get_payment_allocation
from bulk_payments.db import insert_match_event
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind
from bulk_payments.synthetic import load_tenant_config


@dataclass(frozen=True)
class BatchPaymentResult:
    payment_id: str
    status: str  # auto_matched | needs_review | no_match | skipped
    decision: str
    event_id: str | None
    bill_ids: tuple[str, ...]
    competing_subset_count: int
    reason_codes: tuple[str, ...]


def _latest_event_id(conn: sqlite3.Connection, payment_id: str) -> str | None:
    row = conn.execute(
        "SELECT event_id FROM match_events WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1",
        (payment_id,),
    ).fetchone()
    return row["event_id"] if row else None


def run_batch_auto_match(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    ranker_path: str | None = None,
) -> dict[str, Any]:
    """
    Run the matcher on every non-fully-allocated payment for a tenant.
    - auto_applied: accept and allocate bills immediately
    - suggested: leave pending for expert accountant review
    - no_candidates: record only
    """
    cfg = load_tenant_config(conn, tenant_id)
    rows = conn.execute(
        "SELECT payment_id FROM payments WHERE tenant_id = ? ORDER BY payment_date",
        (tenant_id,),
    ).fetchall()

    auto_matched: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []
    needs_review_results: list[Any] = []
    no_match: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        payment_id = row["payment_id"]
        alloc = get_payment_allocation(conn, tenant_id, payment_id)
        if alloc.is_fully_allocated:
            skipped += 1
            continue

        result = match_payment(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            ranker_path=ranker_path,
            log_event=True,
        )
        event_id = _latest_event_id(conn, payment_id)
        best_bills = tuple(result.subsets[0].bill_ids) if result.subsets else ()
        entry = {
            "payment_id": payment_id,
            "decision": result.decision.value,
            "event_id": event_id,
            "bill_ids": list(best_bills),
            "competing_subset_count": result.competing_subset_count,
            "reason_codes": list(result.reason_codes),
        }

        if result.decision == DecisionKind.AUTO_APPLIED and best_bills:
            apply_accepted_bills(
                conn,
                tenant_id=tenant_id,
                payment_id=payment_id,
                bill_ids=list(best_bills),
            )
            features = json.loads(
                conn.execute(
                    "SELECT features_json FROM match_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()["features_json"]
            )
            insert_match_event(
                conn,
                tenant_id=tenant_id,
                payment_id=payment_id,
                bill_ids=list(best_bills),
                rules_version=cfg.rules_version,
                features={**features, "batch_auto_accept": True, "correction_of": event_id},
                decision=result.decision.value,
                outcome="accepted_as_is",
                ranker_score=result.ranker_score,
                calibrated_prob=result.calibrated_accept_prob,
                reason_codes=["batch_auto_matched", *list(result.reason_codes)],
            )
            entry["status"] = "auto_matched"
            auto_matched.append(entry)
        elif result.decision == DecisionKind.SUGGESTED and result.subsets:
            entry["status"] = "needs_review"
            needs_review.append(entry)
            needs_review_results.append((result, event_id))
        else:
            entry["status"] = "no_match"
            no_match.append(entry)

    return {
        "tenant_id": tenant_id,
        "summary": {
            "processed": len(auto_matched) + len(needs_review) + len(no_match),
            "auto_matched": len(auto_matched),
            "needs_review": len(needs_review),
            "no_match": len(no_match),
            "skipped_fully_allocated": skipped,
        },
        "auto_matched": auto_matched,
        "needs_review": needs_review,
        "needs_review_results": needs_review_results,
        "no_match": no_match,
    }
