"""Tests for agent/context.py — pure DB reads, no LLM calls."""
from __future__ import annotations

import pytest

from bulk_payments.agent.context import build_context, format_context_for_prompt
from bulk_payments.db import connect, init_db
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind
from bulk_payments.synthetic import seed_demo_dataset


@pytest.fixture()
def demo_conn(tmp_path):
    db = tmp_path / "test.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    return conn


def test_build_context_pay_bulk_1(demo_conn):
    """Ambiguous payment: context should show SUGGESTED with multiple subsets."""
    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    assert result.decision == DecisionKind.SUGGESTED

    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", match_result=result)

    assert ctx["payment"]["payment_id"] == "pay_bulk_1"
    assert ctx["payment"]["amount"] == "500.00"
    assert ctx["rules_engine"]["decision"] == "suggested"
    assert ctx["rules_engine"]["competing_subset_count"] > 1
    assert len(ctx["ranked_subsets"]) > 1
    assert len(ctx["open_bills"]) > 0
    # Vendor aliases should be included
    assert ctx["vendor_aliases"]


def test_build_context_pay_no_match(demo_conn):
    """No-match payment: context should explain why rules failed."""
    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_no_match", log_event=False)
    assert result.decision == DecisionKind.NO_CANDIDATES

    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_no_match", match_result=result)

    assert ctx["rules_engine"]["decision"] == "no_candidates"
    assert len(ctx["rules_engine"]["reason_codes"]) > 0
    # Even with no match, we pass open bills so agent can reason about near-misses
    assert isinstance(ctx["open_bills"], list)


def test_format_context_for_prompt_has_sections(demo_conn):
    """Formatted prompt should contain all required section headers."""
    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", match_result=result)
    text = format_context_for_prompt(ctx)

    assert "=== PAYMENT ===" in text
    assert "=== RULES ENGINE RESULT ===" in text
    assert "=== OPEN BILLS ===" in text
    assert "pay_bulk_1" in text
    assert "500.00" in text


def test_open_bills_excludes_allocated(demo_conn):
    """Bills with matched_payment_id should be flagged as already_allocated."""
    # Manually mark a bill as allocated
    demo_conn.execute("UPDATE bills SET matched_payment_id = 'p_fake' WHERE bill_id = 'b1'")
    demo_conn.commit()

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", match_result=result)

    allocated = [b for b in ctx["open_bills"] if b["bill_id"] == "b1"]
    assert allocated, "b1 should still appear in context (for transparency)"
    assert allocated[0]["already_allocated"] is True
