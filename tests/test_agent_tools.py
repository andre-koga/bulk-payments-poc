"""Tests for agent/tools.py — submit_resolution guardrails and tool responses."""
from __future__ import annotations

import json

import pytest

from bulk_payments.agent.context import build_context, format_context_for_prompt
from bulk_payments.agent.tools import make_tools
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


@pytest.fixture()
def tools_for_bulk1(demo_conn):
    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", match_result=result)
    context_text = format_context_for_prompt(ctx)
    tools = make_tools(
        demo_conn,
        tenant_id="t1",
        payment_id="pay_bulk_1",
        match_result=result,
        context_text=context_text,
    )
    return {t.name: t for t in tools}


def test_get_match_context_returns_text(tools_for_bulk1):
    result = tools_for_bulk1["get_match_context"].invoke({})
    assert "=== PAYMENT ===" in result
    assert "pay_bulk_1" in result


def test_list_feasible_subsets_returns_json(tools_for_bulk1):
    raw = tools_for_bulk1["list_feasible_subsets"].invoke({})
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) > 1  # pay_bulk_1 is ambiguous
    assert "bill_ids" in data[0]
    assert "abs_residual_minor" in data[0]


def test_get_bill_details_valid(tools_for_bulk1):
    raw = tools_for_bulk1["get_bill_details"].invoke({"bill_id": "b1"})
    data = json.loads(raw)
    assert data["bill_id"] == "b1"
    assert "open_amount_decimal" in data
    assert data["already_allocated"] is False


def test_get_bill_details_unknown(tools_for_bulk1):
    raw = tools_for_bulk1["get_bill_details"].invoke({"bill_id": "no_such_bill"})
    data = json.loads(raw)
    assert "error" in data


def test_get_vendor_aliases_valid(tools_for_bulk1):
    raw = tools_for_bulk1["get_vendor_aliases"].invoke({"vendor_id": "v_acme"})
    data = json.loads(raw)
    assert data["vendor_id"] == "v_acme"
    assert isinstance(data["bank_aliases"], list)
    assert "ACME BANK" in data["bank_aliases"]


def test_get_vendor_aliases_unknown(tools_for_bulk1):
    raw = tools_for_bulk1["get_vendor_aliases"].invoke({"vendor_id": "v_fake"})
    data = json.loads(raw)
    assert "error" in data


# ---------------------------------------------------------------------------
# submit_resolution guardrails
# ---------------------------------------------------------------------------

def test_submit_resolution_ok(tools_for_bulk1):
    """Valid propose_match: b2 + b4 = 250 + 200 = 450 — not 500, so should fail tolerance."""
    # First find a valid subset: b1+b2+b3 = 100+250+150 = 500
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "propose_match",
            "bill_ids": ["b1", "b2", "b3"],
            "confidence": 0.9,
            "reasoning": "Unique sum of 500 matching the payment amount exactly.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert data["action"] == "propose_match"
    assert set(data["bill_ids"]) == {"b1", "b2", "b3"}


def test_submit_resolution_no_match_ok(tools_for_bulk1):
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "no_match",
            "bill_ids": [],
            "confidence": 0.8,
            "reasoning": "Could not find a suitable match.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert data["action"] == "no_match"


def test_submit_resolution_rejects_unknown_bill(tools_for_bulk1):
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "propose_match",
            "bill_ids": ["DOES_NOT_EXIST"],
            "confidence": 0.5,
            "reasoning": "Testing unknown bill.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert any("not found" in e for e in data["errors"])


def test_submit_resolution_rejects_over_tolerance(tools_for_bulk1, demo_conn):
    """b1 alone = 100, payment = 500. Residual = 400, well over tolerance=2."""
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "propose_match",
            "bill_ids": ["b1"],
            "confidence": 0.3,
            "reasoning": "Testing over-tolerance.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert any("tolerance" in e for e in data["errors"])


def test_submit_resolution_rejects_already_allocated(tools_for_bulk1, demo_conn):
    """Bills with matched_payment_id should be rejected."""
    demo_conn.execute("UPDATE bills SET matched_payment_id = 'p_other' WHERE bill_id = 'b1'")
    demo_conn.commit()

    # Rebuild tools with fresh state
    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    ctx = build_context(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", match_result=result)
    tools = make_tools(
        demo_conn,
        tenant_id="t1",
        payment_id="pay_bulk_1",
        match_result=result,
        context_text=format_context_for_prompt(ctx),
    )
    tool_map = {t.name: t for t in tools}

    raw = tool_map["submit_resolution"].invoke(
        {
            "action": "propose_match",
            "bill_ids": ["b1", "b2", "b3"],
            "confidence": 0.9,
            "reasoning": "Testing allocated bill.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert any("already allocated" in e for e in data["errors"])


def test_submit_resolution_rejects_bills_in_no_match(tools_for_bulk1):
    """no_match action should not include bill_ids."""
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "no_match",
            "bill_ids": ["b1"],
            "confidence": 0.5,
            "reasoning": "Testing invalid no_match with bills.",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert any("must be empty" in e for e in data["errors"])


def test_submit_resolution_rejects_empty_reasoning(tools_for_bulk1):
    raw = tools_for_bulk1["submit_resolution"].invoke(
        {
            "action": "no_match",
            "bill_ids": [],
            "confidence": 0.5,
            "reasoning": "",
        }
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert any("non-empty" in e for e in data["errors"])
