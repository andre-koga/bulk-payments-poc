from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from bulk_payments.db import update_match_event_outcome
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind


OracleKind = Literal["none", "suggest", "exact"]


@dataclass(frozen=True)
class OracleCase:
    kind: OracleKind
    bills: frozenset[str] | None = None


# Ground truth for seeded demo DB (see synthetic.seed_demo_dataset).
ORACLE: dict[str, OracleCase] = {
    "pay_bulk_1": OracleCase("suggest", None),  # multiple amount-closing subsets
    "pay_no_match": OracleCase("none", None),
    "pay_t2_ok": OracleCase("exact", frozenset({"t2_b1", "t2_b2"})),
}


def _best_bill_set(result) -> frozenset[str] | None:
    if not result.subsets:
        return None
    return frozenset(result.subsets[0].bill_ids)


def label_events_for_training(conn: sqlite3.Connection) -> int:
    """Attach synthetic accountant labels to latest match_events per payment_id."""
    updated = 0
    for payment_id, oracle in ORACLE.items():
        row = conn.execute(
            """
            SELECT event_id, decision, bill_ids_json FROM match_events
            WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (payment_id,),
        ).fetchone()
        if row is None:
            continue
        event_id = row["event_id"]
        decision = row["decision"]
        bills = frozenset(json.loads(row["bill_ids_json"]))
        outcome: str
        if oracle.kind == "none":
            outcome = "accepted_as_is" if decision == DecisionKind.NO_CANDIDATES.value else "rejected"
        elif oracle.kind == "suggest":
            outcome = "accepted_as_is" if decision == DecisionKind.SUGGESTED.value else "rejected"
        elif oracle.kind == "exact":
            if decision == DecisionKind.AUTO_APPLIED.value and bills == oracle.bills:
                outcome = "accepted_as_is"
            elif decision == DecisionKind.SUGGESTED.value and bills == oracle.bills:
                outcome = "accepted_as_is"
            else:
                outcome = "rejected"
        else:
            outcome = "rejected"
        update_match_event_outcome(conn, event_id, outcome)
        updated += 1
    return updated


def run_eval(conn: sqlite3.Connection, policy=None, ranker_path: str | None = None) -> dict[str, Any]:
    """Run matcher on oracle payments and compute simple precision/recall style checks."""
    conn.execute(
        f"DELETE FROM match_events WHERE payment_id IN ({','.join('?' for _ in ORACLE)})",
        tuple(ORACLE.keys()),
    )
    conn.commit()
    results: dict[str, Any] = {}
    auto_tp = auto_fp = auto_fn = 0
    suggest_ok = suggest_bad = 0
    none_ok = none_bad = 0

    for payment_id, oracle in ORACLE.items():
        tenant_row = conn.execute(
            "SELECT tenant_id FROM payments WHERE payment_id = ?", (payment_id,)
        ).fetchone()
        assert tenant_row is not None
        tenant_id = tenant_row["tenant_id"]
        r = match_payment(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            policy=policy,
            ranker_path=ranker_path,
        )
        best = _best_bill_set(r)

        passed = False
        if oracle.kind == "none":
            passed = r.decision == DecisionKind.NO_CANDIDATES
            if passed:
                none_ok += 1
            else:
                none_bad += 1
        elif oracle.kind == "suggest":
            passed = r.decision == DecisionKind.SUGGESTED and bool(r.subsets)
            if passed:
                suggest_ok += 1
            else:
                suggest_bad += 1
        elif oracle.kind == "exact":
            passed = r.decision == DecisionKind.AUTO_APPLIED and best == oracle.bills
            if r.decision == DecisionKind.AUTO_APPLIED:
                if passed:
                    auto_tp += 1
                else:
                    auto_fp += 1
            elif best == oracle.bills and r.decision == DecisionKind.SUGGESTED:
                auto_fn += 1

        results[payment_id] = {
            "decision": r.decision.value,
            "best_subset": sorted(best) if best else [],
            "competing_subsets": r.competing_subset_count,
            "oracle_pass": passed,
            "reasons": list(r.reason_codes),
        }

    precision_auto = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) else None
    summary = {
        "per_payment": results,
        "counts": {
            "auto_tp": auto_tp,
            "auto_fp": auto_fp,
            "auto_fn": auto_fn,
            "suggest_ok": suggest_ok,
            "suggest_bad": suggest_bad,
            "none_ok": none_ok,
            "none_bad": none_bad,
        },
        "precision_auto_apply": precision_auto,
        "oracle_all_pass": all(results[pid]["oracle_pass"] for pid in ORACLE),
    }
    return summary
