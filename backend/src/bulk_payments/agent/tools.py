"""LangChain tool definitions for the resolver agent.

Each tool is a closure over a sqlite3.Connection, tenant_id, payment_id, and
the current MatchResult. Tools are read-only except `submit_resolution`, which
validates the proposed allocation but does NOT persist anything — that step
belongs to the orchestrator after the agent finishes.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Annotated, Any

from langchain_core.tools import tool

from bulk_payments.models import MatchResult, TenantConfig
from bulk_payments.synthetic import load_bills, load_payment, load_tenant_config


# ---------------------------------------------------------------------------
# Tool factory — returns a list of tools pre-bound to the current request
# ---------------------------------------------------------------------------

def make_tools(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    match_result: MatchResult,
    context_text: str,
) -> list:
    """Return LangChain tools bound to the current matching context."""

    # Cache data for this request so tools don't hit the DB repeatedly
    cfg: TenantConfig = load_tenant_config(conn, tenant_id)
    payment = load_payment(conn, payment_id)
    all_bills = {b.bill_id: b for b in load_bills(conn, tenant_id)}

    @tool
    def get_match_context() -> str:
        """Return the full matching context: payment details, open bills, ranked subsets,
        and an explanation of why the rules engine did not auto-apply.
        Read this first before calling other tools."""
        return context_text

    @tool
    def list_feasible_subsets() -> str:
        """List all feasible bill subsets that the rules engine found (up to 10), ranked
        by residual then by subset size. Returns JSON array."""
        subsets = [
            {
                "bill_ids": list(s.bill_ids),
                "sum_minor": s.sum_minor,
                "abs_residual_minor": s.abs_residual_minor,
            }
            for s in match_result.subsets
        ]
        return json.dumps(subsets, indent=2)

    @tool
    def get_bill_details(bill_id: Annotated[str, "The bill_id to look up"]) -> str:
        """Return detailed information for a single bill by bill_id. Returns JSON."""
        bill = all_bills.get(bill_id)
        if bill is None:
            return json.dumps({"error": f"bill_id '{bill_id}' not found for this tenant"})
        vendor_row = conn.execute(
            "SELECT display_name, bank_aliases_json FROM vendors WHERE vendor_id = ?",
            (bill.vendor_id,),
        ).fetchone()
        vendor_name = vendor_row["display_name"] if vendor_row else bill.vendor_id
        return json.dumps(
            {
                "bill_id": bill.bill_id,
                "vendor_id": bill.vendor_id,
                "vendor_name": vendor_name,
                "open_amount_minor": bill.open_amount_minor,
                "open_amount_decimal": f"{bill.open_amount_minor / 100:.2f}",
                "currency": bill.currency,
                "open_date": bill.open_date.isoformat(),
                "due_date": bill.due_date.isoformat() if bill.due_date else None,
                "already_allocated": bill.matched_payment_id is not None,
            }
        )

    @tool
    def get_vendor_aliases(
        vendor_id: Annotated[str, "The vendor_id to look up"]
    ) -> str:
        """Return the display name and bank name aliases for a vendor. Useful for
        reasoning about counterparty name matching. Returns JSON."""
        row = conn.execute(
            "SELECT display_name, bank_aliases_json FROM vendors WHERE vendor_id = ?",
            (vendor_id,),
        ).fetchone()
        if row is None:
            return json.dumps({"error": f"vendor_id '{vendor_id}' not found"})
        return json.dumps(
            {
                "vendor_id": vendor_id,
                "display_name": row["display_name"],
                "bank_aliases": json.loads(row["bank_aliases_json"]),
            }
        )

    @tool
    def submit_resolution(
        action: Annotated[
            str,
            "One of: propose_match, no_match, need_more_info",
        ],
        bill_ids: Annotated[
            list[str],
            "Bill IDs to propose for allocation. Must be empty for no_match / need_more_info.",
        ],
        confidence: Annotated[float, "Confidence score between 0.0 and 1.0"],
        reasoning: Annotated[
            str,
            "Accountant-style explanation justifying the proposed decision.",
        ],
    ) -> str:
        """Validate and submit the final resolution.

        Call this when you have reached a conclusion. The tool validates that:
        - For propose_match: bill_ids are non-empty, all belong to this tenant,
          none are already allocated, currencies match the ledger, and the sum
          is within the tenant's amount tolerance.
        - For no_match / need_more_info: bill_ids must be empty.

        Returns a JSON result with 'status' = 'ok' or 'error' with details.
        If the status is 'error', fix the issue and call submit_resolution again.
        """
        errors: list[str] = []

        valid_actions = {"propose_match", "no_match", "need_more_info"}
        if action not in valid_actions:
            errors.append(f"action must be one of {sorted(valid_actions)}, got '{action}'")

        if not 0.0 <= confidence <= 1.0:
            errors.append(f"confidence must be between 0.0 and 1.0, got {confidence}")

        if not reasoning or not reasoning.strip():
            errors.append("reasoning must be a non-empty string")

        if action in ("no_match", "need_more_info") and bill_ids:
            errors.append(f"bill_ids must be empty when action is '{action}'")

        if action == "propose_match":
            if not bill_ids:
                errors.append("bill_ids must be non-empty when action is 'propose_match'")
            else:
                total = 0
                for bid in bill_ids:
                    b = all_bills.get(bid)
                    if b is None:
                        errors.append(f"bill_id '{bid}' not found for tenant '{tenant_id}'")
                        continue
                    if b.matched_payment_id is not None:
                        errors.append(f"bill_id '{bid}' is already allocated to payment '{b.matched_payment_id}'")
                    if b.currency != cfg.ledger_currency:
                        errors.append(
                            f"bill_id '{bid}' currency '{b.currency}' does not match"
                            f" ledger currency '{cfg.ledger_currency}'"
                        )
                    total += b.open_amount_minor

                residual = abs(total - payment.amount_minor)
                if not errors and residual > cfg.amount_tolerance_minor:
                    errors.append(
                        f"proposed sum {total / 100:.2f} differs from payment amount"
                        f" {payment.amount_minor / 100:.2f} by {residual / 100:.2f}"
                        f" which exceeds tolerance {cfg.amount_tolerance_minor / 100:.2f}"
                    )

        if errors:
            return json.dumps({"status": "error", "errors": errors})

        return json.dumps(
            {
                "status": "ok",
                "action": action,
                "bill_ids": bill_ids,
                "confidence": confidence,
            }
        )

    return [get_match_context, list_feasible_subsets, get_bill_details, get_vendor_aliases, submit_resolution]
