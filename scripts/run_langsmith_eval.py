#!/usr/bin/env python3
"""Run LangSmith evaluation across one or more models on the bulk-payments dataset.

For each model, this script:
1. Loads examples from the named LangSmith dataset.
2. Runs `resolve_with_agent` for each example.
3. Evaluates four metrics: subset_exact_match, decision_agreement,
   amount_residual_ok, reasoning_present.
4. Writes results back to LangSmith so you can compare models in the UI.

Usage:
    export LANGSMITH_API_KEY=...
    export OPENAI_API_KEY=...          # for gpt-* models
    export ANTHROPIC_API_KEY=...       # for claude-* models
    python scripts/run_langsmith_eval.py \\
        --db /tmp/bulk.db \\
        --models gpt-4o-mini,claude-3-5-haiku-20241022 \\
        [--dataset-name bulk-payments-poc] \\
        [--project bulk-payments-poc]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bulk_payments.db import connect
from bulk_payments.synthetic import load_tenant_config


# ---------------------------------------------------------------------------
# Evaluator functions (passed to langsmith.evaluate)
# ---------------------------------------------------------------------------

def subset_exact_match(run, example) -> dict:
    """1.0 if the proposed bill_ids exactly match the expected set."""
    expected = set(example.outputs.get("bill_ids") or [])
    predicted = set((run.outputs or {}).get("bill_ids") or [])
    # If expected is None/empty (oracle_kind=suggest), any non-empty prediction counts
    if not expected:
        score = 1.0 if predicted else 0.0
    else:
        score = 1.0 if predicted == expected else 0.0
    return {"key": "subset_exact_match", "score": score}


def decision_agreement(run, example) -> dict:
    """1.0 if agent action agrees with expected action."""
    expected = example.outputs.get("action", "")
    predicted = (run.outputs or {}).get("action", "")
    # For oracle_kind=suggest, both propose_match and need_more_info are acceptable
    if expected == "propose_match" and example.outputs.get("bill_ids") is None:
        score = 1.0 if predicted in ("propose_match", "need_more_info") else 0.0
    else:
        score = 1.0 if predicted == expected else 0.0
    return {"key": "decision_agreement", "score": score}


def amount_residual_ok(run, example) -> dict:
    """1.0 if sum of proposed bills is within tolerance (checked via context metadata)."""
    # We don't have the DB connection here so we check the residual from the run metadata
    residual = (run.outputs or {}).get("_residual_minor", None)
    tolerance = (run.outputs or {}).get("_tolerance_minor", 200)  # default 200 cents
    if residual is None:
        return {"key": "amount_residual_ok", "score": None}
    score = 1.0 if residual <= tolerance else 0.0
    return {"key": "amount_residual_ok", "score": score}


def reasoning_present(run, example) -> dict:
    """1.0 if the agent produced a non-empty reasoning string."""
    reasoning = (run.outputs or {}).get("reasoning", "")
    score = 1.0 if isinstance(reasoning, str) and len(reasoning.strip()) > 10 else 0.0
    return {"key": "reasoning_present", "score": score}


# ---------------------------------------------------------------------------
# Target function: called once per example per model
# ---------------------------------------------------------------------------

def make_target(db_path: str, model: str):
    """Return a target function for langsmith.evaluate bound to a DB + model."""

    def target(inputs: dict) -> dict:
        conn = connect(db_path)
        try:
            from bulk_payments.agent.resolver import resolve_with_agent
            from bulk_payments.matcher import match_payment
            from bulk_payments.models import DecisionKind

            tenant_id = inputs["tenant_id"]
            payment_id = inputs["payment_id"]

            result = match_payment(conn, tenant_id=tenant_id, payment_id=payment_id, log_event=False)
            cfg = load_tenant_config(conn, tenant_id)

            resolution = resolve_with_agent(
                conn,
                tenant_id=tenant_id,
                payment_id=payment_id,
                match_result=result,
                model=model,
            )

            # Compute residual for amount_residual_ok evaluator
            if resolution.bill_ids:
                bill_rows = conn.execute(
                    f"SELECT open_amount_minor FROM bills WHERE bill_id IN ({','.join('?' * len(resolution.bill_ids))})",
                    tuple(resolution.bill_ids),
                ).fetchall()
                total = sum(r["open_amount_minor"] for r in bill_rows)
                payment_row = conn.execute(
                    "SELECT amount_minor FROM payments WHERE payment_id = ?", (payment_id,)
                ).fetchone()
                payment_amount = payment_row["amount_minor"] if payment_row else 0
                residual = abs(total - payment_amount)
            else:
                residual = None

            return {
                "action": resolution.action.value,
                "bill_ids": resolution.bill_ids,
                "confidence": resolution.confidence,
                "reasoning": resolution.reasoning,
                "model_id": resolution.model_id,
                "langsmith_run_id": resolution.langsmith_run_id,
                "_residual_minor": residual,
                "_tolerance_minor": cfg.amount_tolerance_minor,
            }
        finally:
            conn.close()

    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--models", default="gpt-4o-mini", help="Comma-separated model IDs")
    parser.add_argument("--dataset-name", default="bulk-payments-poc")
    parser.add_argument("--project", default="bulk-payments-poc")
    args = parser.parse_args()

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: LANGSMITH_API_KEY is not set.", file=sys.stderr)
        return 1

    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("Error: langsmith not installed. Run: pip install 'bulk-payments[agent]'", file=sys.stderr)
        return 1

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", args.project)

    client = Client(api_key=api_key)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in models:
        print(f"\nEvaluating model: {model}")
        experiment_prefix = f"{model.replace('/', '-')}"

        evaluate(
            make_target(args.db, model),
            data=args.dataset_name,
            evaluators=[subset_exact_match, decision_agreement, amount_residual_ok, reasoning_present],
            experiment_prefix=experiment_prefix,
            metadata={"model": model, "db": args.db},
            client=client,
        )
        print(f"  Done. Results visible in LangSmith under project '{args.project}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
