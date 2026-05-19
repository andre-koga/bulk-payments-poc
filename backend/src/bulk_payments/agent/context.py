"""Build a compact, LLM-friendly context blob from the DB + MatchResult.

All functions are pure DB reads — no LLM calls, no side effects.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from bulk_payments.models import DecisionKind, MatchResult, TenantConfig
from bulk_payments.synthetic import (
    load_bills,
    load_payment,
    load_tenant_config,
    load_vendors,
)


def _minor_to_decimal(minor: int) -> str:
    """Format minor currency units as a human-readable decimal string."""
    return f"{minor / 100:.2f}"


def _bill_summary(bill_row: Any, vendor_map: dict[str, str]) -> dict[str, Any]:
    return {
        "bill_id": bill_row.bill_id,
        "vendor": vendor_map.get(bill_row.vendor_id, bill_row.vendor_id),
        "amount": _minor_to_decimal(bill_row.open_amount_minor),
        "currency": bill_row.currency,
        "open_date": bill_row.open_date.isoformat(),
        "due_date": bill_row.due_date.isoformat() if bill_row.due_date else None,
        "already_allocated": bill_row.matched_payment_id is not None,
    }


def build_context(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    match_result: MatchResult,
) -> dict[str, Any]:
    """Assemble a structured context dict for the resolver agent.

    Includes payment details, tenant config, all open candidate bills, ranked
    subsets from the rules engine, and a clear explanation of why the rules
    engine did not auto-apply. For NO_CANDIDATES, we include the retrieved-but-
    gated near-miss bills so the agent can reason about why nothing matched.
    """
    cfg: TenantConfig = load_tenant_config(conn, tenant_id)
    payment = load_payment(conn, payment_id)
    all_bills = load_bills(conn, tenant_id)
    vendors_raw = load_vendors(conn, tenant_id)
    vendor_name: dict[str, str] = {
        vid: data[2] for vid, data in vendors_raw.items()
    }
    vendor_aliases: dict[str, list[str]] = {
        vid: list(data[3]) for vid, data in vendors_raw.items()
    }

    # Include all bills (including already-allocated ones, flagged), so the agent
    # has full context and can reason about near-misses. The submit_resolution tool
    # will reject already-allocated bills if the agent tries to propose them.
    bill_summaries = [_bill_summary(b, vendor_name) for b in all_bills]

    subsets = [
        {
            "bill_ids": list(s.bill_ids),
            "sum": _minor_to_decimal(s.sum_minor),
            "residual": _minor_to_decimal(s.abs_residual_minor),
        }
        for s in match_result.subsets
    ]

    return {
        "payment": {
            "payment_id": payment.payment_id,
            "amount": _minor_to_decimal(payment.amount_minor),
            "currency": cfg.ledger_currency,
            "date": payment.payment_date.isoformat(),
            "counterparty_bank_name": payment.counterparty_bank_name,
            "description": payment.description,
        },
        "tenant": {
            "tenant_id": tenant_id,
            "ledger_currency": cfg.ledger_currency,
            "amount_tolerance": _minor_to_decimal(cfg.amount_tolerance_minor),
        },
        "rules_engine": {
            "decision": match_result.decision.value,
            "reason_codes": list(match_result.reason_codes),
            "competing_subset_count": match_result.competing_subset_count,
            "unique_best": match_result.unique_best,
        },
        "ranked_subsets": subsets,
        "open_bills": bill_summaries,
        "vendor_aliases": vendor_aliases,
    }


def format_context_for_prompt(ctx: dict[str, Any]) -> str:
    """Render the context dict as a structured text block for the system prompt."""
    p = ctx["payment"]
    t = ctx["tenant"]
    r = ctx["rules_engine"]

    lines: list[str] = [
        "=== PAYMENT ===",
        f"ID: {p['payment_id']}",
        f"Amount: {p['amount']} {p['currency']}",
        f"Date: {p['date']}",
        f"Counterparty bank: {p['counterparty_bank_name']}",
        f"Description: {p['description'] or '(none)'}",
        "",
        "=== TENANT CONFIG ===",
        f"Ledger currency: {t['ledger_currency']}",
        f"Amount tolerance: ±{t['amount_tolerance']} {t['ledger_currency']}",
        "",
        "=== RULES ENGINE RESULT ===",
        f"Decision: {r['decision']}",
        f"Reason codes: {', '.join(r['reason_codes']) or 'none'}",
        f"Competing subsets: {r['competing_subset_count']}",
        "",
    ]

    if ctx["ranked_subsets"]:
        lines.append("=== RANKED SUBSETS FROM RULES ENGINE ===")
        for i, s in enumerate(ctx["ranked_subsets"], 1):
            lines.append(
                f"  Subset {i}: bills={s['bill_ids']}  sum={s['sum']}  residual={s['residual']}"
            )
        lines.append("")

    lines.append("=== OPEN BILLS ===")
    for b in ctx["open_bills"]:
        alloc = " [ALREADY ALLOCATED]" if b["already_allocated"] else ""
        lines.append(
            f"  {b['bill_id']}: vendor={b['vendor']}  amount={b['amount']} {b['currency']}"
            f"  open={b['open_date']}{alloc}"
        )
    lines.append("")

    if ctx["vendor_aliases"]:
        lines.append("=== VENDOR BANK ALIASES ===")
        for vid, aliases in ctx["vendor_aliases"].items():
            if aliases:
                lines.append(f"  {vid}: {', '.join(aliases)}")
        lines.append("")

    return "\n".join(lines)
