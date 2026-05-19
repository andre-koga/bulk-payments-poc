"""Orchestrator: rules engine + AI resolver as a single call.

`match_payment_with_agent` is the primary public API for AI-augmented matching.
It runs the rules engine first; if the decision is SUGGESTED or NO_CANDIDATES it
invokes the AI resolver and persists both the match_event and agent_resolution
atomically before returning.

The rules engine result is always persisted to match_events. The agent result is
persisted to agent_resolutions, linked by event_id. This keeps both layers auditable
and allows offline evaluation by comparing agent decisions against accountant labels.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

from bulk_payments.agent.resolver import resolve_with_agent
from bulk_payments.agent.schemas import AgentResolution
from bulk_payments.db import insert_agent_resolution, insert_match_event, migrate_db
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind, MatchResult
from bulk_payments.policy import PolicyConfig
from bulk_payments.synthetic import load_tenant_config


@dataclass
class MatchWithAgentResult:
    """Combined result from rules engine + optional AI resolver."""

    rules: MatchResult
    agent: AgentResolution | None
    event_id: str
    resolution_id: str | None


def match_payment_with_agent(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    policy: PolicyConfig | None = None,
    ranker_path: str | None = None,
    model: str = "gpt-4o-mini",
    max_iterations: int = 5,
    agent_enabled: bool | None = None,
) -> MatchWithAgentResult:
    """Run the rules engine and, when needed, the AI resolver.

    The agent is invoked when the rules engine returns SUGGESTED or NO_CANDIDATES.
    For AUTO_APPLIED decisions, the agent is skipped; the rules engine result is
    persisted and returned immediately.

    Args:
        conn: Active SQLite connection.
        tenant_id: Tenant scope.
        payment_id: Payment to match.
        policy: Optional PolicyConfig override (defaults to PolicyConfig()).
        ranker_path: Optional path to a trained ranker joblib file.
        model: LLM model string for the AI resolver.
        max_iterations: Maximum ReAct loop iterations for the agent.
        agent_enabled: Override for the BULK_AGENT_ENABLED env flag. Pass False
            to skip the agent even for non-auto decisions (rules-only fallback).

    Returns:
        MatchWithAgentResult with both the rules and agent outcomes.
    """
    migrate_db(conn)

    if agent_enabled is None:
        agent_enabled = os.environ.get("BULK_AGENT_ENABLED", "true").lower() not in ("false", "0", "no")

    cfg = load_tenant_config(conn, tenant_id)

    rules_result = match_payment(
        conn,
        tenant_id=tenant_id,
        payment_id=payment_id,
        policy=policy,
        ranker_path=ranker_path,
        log_event=False,
    )

    needs_agent = (
        agent_enabled
        and rules_result.decision in (DecisionKind.SUGGESTED, DecisionKind.NO_CANDIDATES)
    )

    best_bill_ids = list(rules_result.subsets[0].bill_ids) if rules_result.subsets else []

    event_id = insert_match_event(
        conn,
        tenant_id=tenant_id,
        payment_id=payment_id,
        bill_ids=best_bill_ids,
        rules_version=cfg.rules_version,
        features=rules_result.features,
        decision=rules_result.decision.value,
        ranker_score=rules_result.ranker_score,
        calibrated_prob=rules_result.calibrated_accept_prob,
        reason_codes=list(dict.fromkeys(rules_result.reason_codes)),
    )

    if not needs_agent:
        return MatchWithAgentResult(
            rules=rules_result,
            agent=None,
            event_id=event_id,
            resolution_id=None,
        )

    agent_result = resolve_with_agent(
        conn,
        tenant_id=tenant_id,
        payment_id=payment_id,
        match_result=rules_result,
        model=model,
        max_iterations=max_iterations,
    )

    resolution_id = insert_agent_resolution(
        conn,
        event_id=event_id,
        model_id=agent_result.model_id,
        action=agent_result.action.value,
        bill_ids=agent_result.bill_ids,
        confidence=agent_result.confidence,
        reasoning=agent_result.reasoning,
        langsmith_run_id=agent_result.langsmith_run_id,
    )

    return MatchWithAgentResult(
        rules=rules_result,
        agent=agent_result,
        event_id=event_id,
        resolution_id=resolution_id,
    )
