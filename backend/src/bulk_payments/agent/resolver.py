"""ReAct resolver agent that handles SUGGESTED and NO_CANDIDATES decisions.

The agent is given read-only tools to inspect the payment, bills, and the
rules-engine output, plus a `submit_resolution` tool that validates and
records the final answer. Structured output is extracted from the validated
`submit_resolution` call so we never rely on free-form LLM text parsing.

LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true; each call
to resolve_with_agent() becomes one parent run with child LLM and tool spans.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from bulk_payments.agent.context import build_context, format_context_for_prompt
from bulk_payments.agent.models import get_chat_model
from bulk_payments.agent.schemas import AgentAction, AgentResolution
from bulk_payments.agent.tools import make_tools
from bulk_payments.agent.tracing import run_metadata
from bulk_payments.models import DecisionKind, MatchResult

_SYSTEM_PROMPT = """\
You are an expert accounts-receivable accountant assistant helping allocate \
incoming bank payments to open vendor bills.

The rules engine has already tried to match this payment automatically but \
could not do so with high confidence. Your job is to review the evidence and \
reach a decision.

RULES:
1. Only propose bill IDs that appear in the open-bills list. Never invent IDs.
2. The proposed allocation must sum to within the tenant's amount tolerance of the \
payment amount.
3. Prefer fewer bills when multiple subsets work equally well.
4. Consider the counterparty bank name and vendor aliases when assessing vendor match.
5. If you cannot find a satisfactory match, call submit_resolution with action=no_match.
6. Always call submit_resolution as your final action — even if the answer is no_match.
7. If information is genuinely insufficient, use action=need_more_info, but prefer \
a definitive answer when possible.

PROCESS:
- Call get_match_context first to review all available information.
- Use list_feasible_subsets and get_bill_details as needed.
- Call submit_resolution once you are confident. If it returns an error, fix the \
issue and call it again.
- Do not call submit_resolution more than once successfully.
"""


def _extract_resolution_from_messages(
    messages: list,
    model_id: str,
    run_id: str | None,
) -> AgentResolution | None:
    """Parse the last successful submit_resolution tool call from message history.

    Reasoning is taken directly from the tool call arguments (the LLM-supplied
    `reasoning` field in the submit_resolution call), falling back to the last
    non-empty AIMessage content if the args don't include it.
    """
    # Build an index of tool_call_id → tool call args from AIMessages
    tool_call_args: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in (msg.tool_calls or []):
                tool_call_args[tc["id"]] = tc.get("args", {})

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "submit_resolution":
            try:
                data = json.loads(msg.content)
                if data.get("status") == "ok":
                    # Prefer reasoning from the tool call args (always present in submit_resolution)
                    args = tool_call_args.get(msg.tool_call_id, {})
                    reasoning = (
                        args.get("reasoning")
                        or _extract_reasoning_from_messages(messages)
                    )
                    return AgentResolution(
                        action=AgentAction(data["action"]),
                        bill_ids=data.get("bill_ids", []),
                        confidence=data.get("confidence", 0.5),
                        reasoning=reasoning,
                        model_id=model_id,
                        langsmith_run_id=run_id,
                    )
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    return None


def _extract_reasoning_from_messages(messages: list) -> str:
    """Extract the last non-empty AI text content as the accountant reasoning."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                combined = " ".join(filter(None, text_parts)).strip()
                if combined:
                    return combined
    return "No reasoning provided."


@traceable(name="bulk_payments.resolve_with_agent")
def resolve_with_agent(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    match_result: MatchResult | None = None,
    model: str = "gpt-4o-mini",
    max_iterations: int = 5,
    langsmith_run_id: str | None = None,
) -> AgentResolution:
    """Run the ReAct resolver agent for one payment.

    If match_result is None, the rules engine is called first (without logging)
    to produce a MatchResult. If the decision is AUTO_APPLIED, the agent is
    skipped and a high-confidence AgentResolution reflecting the auto decision
    is returned immediately.

    Args:
        conn: Active SQLite connection.
        tenant_id: Tenant scope.
        payment_id: Payment to resolve.
        match_result: Pre-computed MatchResult (optional; avoids duplicate run).
        model: LLM model string — see agent.models.get_chat_model for supported values.
        max_iterations: Maximum ReAct loop iterations before giving up.
        langsmith_run_id: Caller-supplied run ID (usually set by @traceable automatically).

    Returns:
        AgentResolution with the final decision, bill_ids, confidence, and reasoning.
    """
    from bulk_payments.matcher import match_payment

    if match_result is None:
        match_result = match_payment(conn, tenant_id=tenant_id, payment_id=payment_id, log_event=False)

    # Short-circuit: agent is not needed for auto-applied decisions
    if match_result.decision == DecisionKind.AUTO_APPLIED:
        best_bill_ids = list(match_result.subsets[0].bill_ids) if match_result.subsets else []
        return AgentResolution(
            action=AgentAction.PROPOSE_MATCH,
            bill_ids=best_bill_ids,
            confidence=1.0,
            reasoning="Rules engine auto-applied a unique, unambiguous match. No AI review needed.",
            model_id="rules_engine",
            langsmith_run_id=None,
        )

    ctx = build_context(conn, tenant_id=tenant_id, payment_id=payment_id, match_result=match_result)
    context_text = format_context_for_prompt(ctx)
    tools = make_tools(conn, tenant_id=tenant_id, payment_id=payment_id, match_result=match_result, context_text=context_text)

    llm = get_chat_model(model)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: list[Any] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Please resolve the following payment.\n\n{context_text}"),
    ]

    resolved = False
    for _iteration in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No tool call — model gave a final text answer without calling submit_resolution.
            # Treat this as need_more_info to avoid silent failures.
            break

        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                tool_result = json.dumps({"error": f"unknown tool '{tc['name']}'"})
            else:
                try:
                    tool_result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    tool_result = json.dumps({"error": str(exc)})

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"], name=tc["name"])
            )

            if tc["name"] == "submit_resolution":
                try:
                    data = json.loads(str(tool_result))
                    if data.get("status") == "ok":
                        resolved = True
                except (json.JSONDecodeError, AttributeError):
                    pass

        if resolved:
            break

    resolution = _extract_resolution_from_messages(messages, model_id=model, run_id=langsmith_run_id)

    if resolution is None:
        # Fallback: agent exhausted iterations or never called submit_resolution
        reasoning = _extract_reasoning_from_messages(messages)
        resolution = AgentResolution(
            action=AgentAction.NEED_MORE_INFO,
            bill_ids=[],
            confidence=0.0,
            reasoning=f"Agent did not reach a resolution within {max_iterations} iterations. "
                      f"Last reasoning: {reasoning}",
            model_id=model,
            langsmith_run_id=langsmith_run_id,
        )

    return resolution
