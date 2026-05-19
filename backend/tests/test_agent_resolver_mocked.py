"""Integration tests for the resolver agent using a mock LLM.

Tests run without real API keys by patching get_chat_model to return a
FakeListChatModel that emits pre-scripted tool call / text sequences.
These cover:
  - AUTO_APPLIED short-circuit (pay_t2_ok)
  - Successful propose_match via tool loop (pay_bulk_1)
  - Fallback to no_match (pay_no_match)
  - Exhausted iterations fallback
  - match_payment_with_agent orchestrator persistence

Integration tests against real LLMs are tagged @pytest.mark.integration
and skipped unless OPENAI_API_KEY / ANTHROPIC_API_KEY are present.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from bulk_payments.agent.schemas import AgentAction
from bulk_payments.db import connect, init_db
from bulk_payments.models import DecisionKind
from bulk_payments.synthetic import seed_demo_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def demo_conn(tmp_path):
    db = tmp_path / "test.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    return conn


def _make_tool_call_message(name: str, args: dict, call_id: str = "c1"):
    """Build a minimal AIMessage with one tool call."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _make_text_message(text: str):
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)


# ---------------------------------------------------------------------------
# AUTO_APPLIED short-circuit — no LLM calls expected
# ---------------------------------------------------------------------------

def test_auto_applied_does_not_invoke_llm(demo_conn):
    """pay_t2_ok auto-applies; resolver must short-circuit without LLM."""
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(demo_conn, tenant_id="t2", payment_id="pay_t2_ok", log_event=False)
    assert result.decision == DecisionKind.AUTO_APPLIED

    with patch("bulk_payments.agent.resolver.get_chat_model") as mock_factory:
        resolution = resolve_with_agent(
            demo_conn,
            tenant_id="t2",
            payment_id="pay_t2_ok",
            match_result=result,
        )
        mock_factory.assert_not_called()

    assert resolution.action == AgentAction.PROPOSE_MATCH
    assert resolution.model_id == "rules_engine"
    assert resolution.confidence == 1.0
    assert set(resolution.bill_ids) == {"t2_b2", "t2_b4"}


# ---------------------------------------------------------------------------
# Mocked propose_match for pay_bulk_1 (SUGGESTED)
# ---------------------------------------------------------------------------

def test_mocked_propose_match_pay_bulk_1(demo_conn):
    """Agent calls submit_resolution with a valid subset on the first attempt."""
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    assert result.decision == DecisionKind.SUGGESTED

    # Sequence: LLM calls submit_resolution directly with b1+b2+b3=500
    submit_call = _make_tool_call_message(
        "submit_resolution",
        {
            "action": "propose_match",
            "bill_ids": ["b1", "b2", "b3"],
            "confidence": 0.92,
            "reasoning": "Bills b1+b2+b3 sum to exactly 500.00, matching payment amount.",
        },
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = submit_call

    with patch("bulk_payments.agent.resolver.get_chat_model", return_value=mock_llm):
        resolution = resolve_with_agent(
            demo_conn,
            tenant_id="t1",
            payment_id="pay_bulk_1",
            match_result=result,
        )

    assert resolution.action == AgentAction.PROPOSE_MATCH
    assert set(resolution.bill_ids) == {"b1", "b2", "b3"}
    assert resolution.confidence == pytest.approx(0.92)
    assert "500" in resolution.reasoning or "b1" in resolution.reasoning


# ---------------------------------------------------------------------------
# Mocked no_match for pay_no_match (NO_CANDIDATES)
# ---------------------------------------------------------------------------

def test_mocked_no_match_pay_no_match(demo_conn):
    """Agent agrees with rules engine: no feasible bills for pay_no_match."""
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(
        demo_conn, tenant_id="t2", payment_id="pay_t2_no_match", log_event=False
    )
    assert result.decision == DecisionKind.NO_CANDIDATES

    no_match_call = _make_tool_call_message(
        "submit_resolution",
        {
            "action": "no_match",
            "bill_ids": [],
            "confidence": 0.95,
            "reasoning": "No open bills close the payment amount in the omega week.",
        },
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = no_match_call

    with patch("bulk_payments.agent.resolver.get_chat_model", return_value=mock_llm):
        resolution = resolve_with_agent(
            demo_conn,
            tenant_id="t2",
            payment_id="pay_t2_no_match",
            match_result=result,
        )

    assert resolution.action == AgentAction.NO_MATCH
    assert resolution.bill_ids == []
    assert resolution.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Exhausted iterations fallback
# ---------------------------------------------------------------------------

def test_exhausted_iterations_fallback(demo_conn):
    """When the LLM never calls submit_resolution, return NEED_MORE_INFO."""
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)

    # LLM always returns plain text, no tool calls
    no_tool_msg = _make_text_message("I need more information about the vendor.")
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = no_tool_msg

    with patch("bulk_payments.agent.resolver.get_chat_model", return_value=mock_llm):
        resolution = resolve_with_agent(
            demo_conn,
            tenant_id="t1",
            payment_id="pay_bulk_1",
            match_result=result,
            max_iterations=3,
        )

    assert resolution.action == AgentAction.NEED_MORE_INFO
    assert resolution.confidence == 0.0
    assert "3 iterations" in resolution.reasoning


# ---------------------------------------------------------------------------
# Tool validation loop: agent recovers from submit_resolution error
# ---------------------------------------------------------------------------

def test_agent_recovers_from_submit_error(demo_conn):
    """Agent calls submit_resolution with a bad bill_id, gets error, retries with valid bills."""
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment
    from langchain_core.messages import AIMessage

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)

    bad_call = _make_tool_call_message(
        "submit_resolution",
        {
            "action": "propose_match",
            "bill_ids": ["FAKE_BILL"],
            "confidence": 0.8,
            "reasoning": "Trying a fake bill.",
        },
        call_id="c1",
    )
    good_call = _make_tool_call_message(
        "submit_resolution",
        {
            "action": "propose_match",
            "bill_ids": ["b1", "b2", "b3"],
            "confidence": 0.9,
            "reasoning": "Corrected to valid subset summing to 500.",
        },
        call_id="c2",
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    # First invoke returns bad call, second returns good call
    mock_llm.invoke.side_effect = [bad_call, good_call]

    with patch("bulk_payments.agent.resolver.get_chat_model", return_value=mock_llm):
        resolution = resolve_with_agent(
            demo_conn,
            tenant_id="t1",
            payment_id="pay_bulk_1",
            match_result=result,
            max_iterations=5,
        )

    assert resolution.action == AgentAction.PROPOSE_MATCH
    assert set(resolution.bill_ids) == {"b1", "b2", "b3"}


# ---------------------------------------------------------------------------
# Orchestrator persistence test
# ---------------------------------------------------------------------------

def test_match_payment_with_agent_persists_resolution(demo_conn):
    """match_payment_with_agent should write both match_event and agent_resolution rows."""
    from bulk_payments.match_with_agent import match_payment_with_agent

    submit_call = _make_tool_call_message(
        "submit_resolution",
        {
            "action": "propose_match",
            "bill_ids": ["b1", "b2", "b3"],
            "confidence": 0.88,
            "reasoning": "Unique subset matching payment amount.",
        },
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = submit_call

    with patch("bulk_payments.agent.resolver.get_chat_model", return_value=mock_llm):
        combined = match_payment_with_agent(
            demo_conn,
            tenant_id="t1",
            payment_id="pay_bulk_1",
        )

    assert combined.event_id is not None
    assert combined.resolution_id is not None
    assert combined.agent is not None
    assert combined.agent.action == AgentAction.PROPOSE_MATCH

    # Verify DB rows
    event_row = demo_conn.execute(
        "SELECT * FROM match_events WHERE event_id = ?", (combined.event_id,)
    ).fetchone()
    assert event_row is not None
    assert event_row["payment_id"] == "pay_bulk_1"

    resolution_row = demo_conn.execute(
        "SELECT * FROM agent_resolutions WHERE resolution_id = ?", (combined.resolution_id,)
    ).fetchone()
    assert resolution_row is not None
    assert resolution_row["event_id"] == combined.event_id
    assert resolution_row["action"] == "propose_match"
    assert json.loads(resolution_row["bill_ids_json"]) == ["b1", "b2", "b3"]
    assert resolution_row["reasoning"] != ""


def test_match_payment_with_agent_auto_skips_agent(demo_conn):
    """AUTO_APPLIED path: no agent_resolution row should be created."""
    from bulk_payments.match_with_agent import match_payment_with_agent

    with patch("bulk_payments.agent.resolver.get_chat_model") as mock_factory:
        combined = match_payment_with_agent(
            demo_conn,
            tenant_id="t2",
            payment_id="pay_t2_ok",
        )
        mock_factory.assert_not_called()

    assert combined.rules.decision == DecisionKind.AUTO_APPLIED
    assert combined.agent is None
    assert combined.resolution_id is None

    # No agent_resolution row should exist
    row = demo_conn.execute(
        "SELECT * FROM agent_resolutions WHERE event_id = ?", (combined.event_id,)
    ).fetchone()
    assert row is None


def test_agent_disabled_via_env(demo_conn, monkeypatch):
    """BULK_AGENT_ENABLED=false should skip the agent even for SUGGESTED."""
    from bulk_payments.match_with_agent import match_payment_with_agent

    monkeypatch.setenv("BULK_AGENT_ENABLED", "false")

    with patch("bulk_payments.agent.resolver.get_chat_model") as mock_factory:
        combined = match_payment_with_agent(
            demo_conn,
            tenant_id="t1",
            payment_id="pay_bulk_1",
        )
        mock_factory.assert_not_called()

    assert combined.agent is None
    assert combined.resolution_id is None


# ---------------------------------------------------------------------------
# Live LLM integration tests (skipped in CI without API keys)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live LLM test",
)
def test_live_gpt_resolves_pay_bulk_1(demo_conn):
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_bulk_1", log_event=False)
    resolution = resolve_with_agent(
        demo_conn,
        tenant_id="t1",
        payment_id="pay_bulk_1",
        match_result=result,
        model="gpt-4o-mini",
    )
    assert resolution.action in (AgentAction.PROPOSE_MATCH, AgentAction.NEED_MORE_INFO)
    assert resolution.reasoning.strip()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live LLM test",
)
def test_live_claude_resolves_pay_no_match(demo_conn):
    from bulk_payments.agent.resolver import resolve_with_agent
    from bulk_payments.matcher import match_payment

    result = match_payment(demo_conn, tenant_id="t1", payment_id="pay_no_match", log_event=False)
    resolution = resolve_with_agent(
        demo_conn,
        tenant_id="t1",
        payment_id="pay_no_match",
        match_result=result,
        model="claude-3-5-haiku-20241022",
    )
    assert resolution.action == AgentAction.NO_MATCH
    assert resolution.reasoning.strip()
