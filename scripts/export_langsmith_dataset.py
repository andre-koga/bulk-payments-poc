#!/usr/bin/env python3
"""Export demo oracle cases and labeled match_events to a LangSmith dataset.

This script creates (or updates) a LangSmith dataset called
`bulk-payments-<env>` with two kinds of examples:

1. Seed examples — the three demo oracle cases from `evaluation.ORACLE`
   (pay_bulk_1, pay_no_match, pay_t2_ok) with ground-truth expected actions.

2. Labeled examples — rows from `match_events` where `outcome != 'pending'`
   (i.e., an accountant has accepted, rejected, or corrected the suggestion).

Usage:
    pip install -e ".[agent]"
    export LANGSMITH_API_KEY=...
    python scripts/export_langsmith_dataset.py \\
        --db /tmp/bulk.db \\
        [--dataset-name bulk-payments-poc] \\
        [--seed-only]

Requirements:
    LANGSMITH_API_KEY must be set. LANGSMITH_ENDPOINT defaults to the public
    LangSmith cloud.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bulk_payments.agent.context import build_context, format_context_for_prompt
from bulk_payments.db import connect
from bulk_payments.evaluation import ORACLE
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind
from bulk_payments.synthetic import create_db_with_seed, load_payment


def _expected_action(oracle_kind: str, oracle_bills: frozenset | None) -> dict:
    if oracle_kind == "none":
        return {"action": "no_match", "bill_ids": []}
    if oracle_kind == "suggest":
        return {"action": "propose_match", "bill_ids": None}  # any valid subset is OK
    if oracle_kind == "exact":
        return {"action": "propose_match", "bill_ids": sorted(oracle_bills or [])}
    return {"action": "no_match", "bill_ids": []}


def _oracle_tenant(payment_id: str) -> str:
    return "t2" if payment_id == "pay_t2_ok" else "t1"


def export_seed_examples(conn: sqlite3.Connection, dataset) -> int:
    """Upload the three oracle demo cases as LangSmith examples."""
    created = 0
    for payment_id, oracle in ORACLE.items():
        tenant_id = _oracle_tenant(payment_id)
        try:
            result = match_payment(conn, tenant_id=tenant_id, payment_id=payment_id, log_event=False)
        except Exception as exc:
            print(f"  skip {payment_id}: {exc}", file=sys.stderr)
            continue

        ctx = build_context(conn, tenant_id=tenant_id, payment_id=payment_id, match_result=result)
        context_text = format_context_for_prompt(ctx)

        inputs = {
            "payment_id": payment_id,
            "tenant_id": tenant_id,
            "context": context_text,
            "rules_decision": result.decision.value,
            "reason_codes": list(result.reason_codes),
        }
        outputs = _expected_action(oracle.kind, oracle.bills)

        dataset.create_example(inputs=inputs, outputs=outputs, metadata={"source": "oracle_seed"})
        created += 1
        print(f"  seeded: {payment_id} → expected_action={outputs['action']}")
    return created


def export_labeled_events(conn: sqlite3.Connection, dataset) -> int:
    """Upload labeled match_events (outcome != pending) as LangSmith examples."""
    rows = conn.execute(
        """
        SELECT me.event_id, me.tenant_id, me.payment_id, me.decision,
               me.bill_ids_json, me.reason_codes_json, me.outcome
        FROM match_events me
        WHERE me.outcome != 'pending'
        ORDER BY me.created_at DESC
        """
    ).fetchall()

    created = 0
    for row in rows:
        tenant_id = row["tenant_id"]
        payment_id = row["payment_id"]
        try:
            result = match_payment(conn, tenant_id=tenant_id, payment_id=payment_id, log_event=False)
            ctx = build_context(conn, tenant_id=tenant_id, payment_id=payment_id, match_result=result)
            context_text = format_context_for_prompt(ctx)
        except Exception as exc:
            print(f"  skip {payment_id}: {exc}", file=sys.stderr)
            continue

        accepted_bills = json.loads(row["bill_ids_json"])
        outcome = row["outcome"]
        if outcome in ("accepted_as_is", "edited_subset"):
            expected_action = "propose_match"
        elif outcome in ("rejected", "rejected_by_user"):
            expected_action = "no_match"
        else:
            expected_action = "need_more_info"

        inputs = {
            "payment_id": payment_id,
            "tenant_id": tenant_id,
            "context": context_text,
            "rules_decision": row["decision"],
            "reason_codes": json.loads(row["reason_codes_json"]),
        }
        outputs = {
            "action": expected_action,
            "bill_ids": accepted_bills if expected_action == "propose_match" else [],
        }

        dataset.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata={"source": "labeled_event", "event_id": row["event_id"], "outcome": outcome},
        )
        created += 1

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--dataset-name", default="bulk-payments-poc", help="LangSmith dataset name")
    parser.add_argument("--seed-only", action="store_true", help="Only export oracle seed cases")
    args = parser.parse_args()

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: LANGSMITH_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    try:
        from langsmith import Client
    except ImportError:
        print("Error: langsmith package not installed. Run: pip install 'bulk-payments[agent]'", file=sys.stderr)
        return 1

    client = Client(api_key=api_key)

    # Get or create the dataset
    existing = [d for d in client.list_datasets() if d.name == args.dataset_name]
    if existing:
        dataset = existing[0]
        print(f"Using existing dataset '{args.dataset_name}' (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=args.dataset_name,
            description="Bulk payment to bill matching evaluation dataset",
        )
        print(f"Created dataset '{args.dataset_name}' (id={dataset.id})")

    conn = connect(args.db)

    print("Exporting seed oracle examples...")
    n_seed = export_seed_examples(conn, dataset)
    print(f"  → {n_seed} seed examples uploaded")

    if not args.seed_only:
        print("Exporting labeled match_events...")
        n_labeled = export_labeled_events(conn, dataset)
        print(f"  → {n_labeled} labeled examples uploaded")

    conn.close()
    print(f"\nDone. View dataset at: https://smith.langchain.com/datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
