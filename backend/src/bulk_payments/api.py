from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bulk_payments.allocation import apply_accepted_bills, get_payment_allocation, unmatch_payment
from bulk_payments.batch import run_batch_auto_match
from bulk_payments.db import connect, insert_match_event, init_db, migrate_db
from bulk_payments.evaluation import extended_eval_report
from bulk_payments.matcher import match_payment
from bulk_payments.synthetic import (
    load_bills,
    load_payment,
    load_tenant_config,
    load_vendors,
)
from bulk_payments.models import DecisionKind, OutcomeLabel

DB_PATH = os.environ.get("BULK_DB", "/tmp/bulk.db")
RANKER_PATH = os.environ.get("BULK_RANKER", None)
DEFAULT_AGENT_MODEL = os.environ.get("BULK_AGENT_MODEL", "gpt-4o-mini")

app = FastAPI(title="Bulk Payment Matcher API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_conn() -> sqlite3.Connection:
    conn = connect(DB_PATH)
    init_db(conn)
    migrate_db(conn)
    return conn


# ---------- response shapes ----------

class SubsetOut(BaseModel):
    bill_ids: list[str]
    sum_minor: int
    abs_residual_minor: int
    sum_display: str


class MatchResultOut(BaseModel):
    payment_id: str
    tenant_id: str
    decision: str
    unique_best: bool
    competing_subset_count: int
    reason_codes: list[str]
    subsets: list[SubsetOut]
    ranker_score: float | None
    calibrated_accept_prob: float | None
    event_id: str | None
    allocated_sum_minor: int = 0
    remaining_minor: int | None = None
    is_fully_allocated: bool = False


class PaymentSettlementOut(BaseModel):
    bill_ids: list[str]
    allocated_sum_minor: int
    remaining_minor: int
    is_fully_allocated: bool


class PaymentOut(BaseModel):
    payment_id: str
    tenant_id: str
    amount_minor: int
    amount_display: str
    payment_date: str
    counterparty_bank_name: str
    description: str
    source_currency: str | None
    settlement: PaymentSettlementOut | None = None


class BillOut(BaseModel):
    bill_id: str
    vendor_id: str
    vendor_name: str
    open_amount_minor: int
    amount_display: str
    currency: str
    open_date: str
    due_date: str | None
    matched_payment_id: str | None


class AgentResolutionOut(BaseModel):
    action: str
    bill_ids: list[str]
    confidence: float
    reasoning: str
    model_id: str
    langsmith_run_id: str | None = None
    langsmith_trace_url: str | None = None


class AgentResolveIn(BaseModel):
    model: str = Field(default_factory=lambda: DEFAULT_AGENT_MODEL)


class MatchWithAgentOut(BaseModel):
    payment_id: str
    tenant_id: str
    event_id: str
    resolution_id: str | None
    rules: MatchResultOut
    agent: AgentResolutionOut | None


class OutcomeIn(BaseModel):
    outcome: str  # accepted_as_is | rejected | edited_subset | manual_alternative
    corrected_bill_ids: list[str] | None = None
    user_id: str | None = None
    accountant_reasoning: str | None = None


class OutcomeOut(BaseModel):
    event_id: str
    outcome: str


class UnmatchOut(BaseModel):
    payment_id: str
    freed_bill_ids: list[str]
    event_id: str | None = None


class BatchAutoMatchOut(BaseModel):
    tenant_id: str
    summary: dict[str, int]
    auto_matched: list[dict[str, Any]]
    needs_review: list[MatchResultOut]
    no_match: list[dict[str, Any]]


# ---------- helpers ----------

def _fmt_minor(minor: int, currency: str = "USD") -> str:
    return f"{currency} {minor / 100:.2f}"


def _payment_settlement(
    conn: sqlite3.Connection, tenant_id: str, payment_id: str
) -> PaymentSettlementOut | None:
    """Bills already allocated to this payment (survives page reload)."""
    alloc = get_payment_allocation(conn, tenant_id, payment_id)
    if not alloc.matched_bill_ids:
        return None
    return PaymentSettlementOut(
        bill_ids=list(alloc.matched_bill_ids),
        allocated_sum_minor=alloc.allocated_sum_minor,
        remaining_minor=alloc.remaining_minor,
        is_fully_allocated=alloc.is_fully_allocated,
    )


def _match_result_out(
    conn: sqlite3.Connection,
    tenant_id: str,
    r,
    event_id: str | None,
) -> MatchResultOut:
    cfg = load_tenant_config(conn, tenant_id)
    return MatchResultOut(
        payment_id=r.payment_id,
        tenant_id=r.tenant_id,
        decision=r.decision.value,
        unique_best=r.unique_best,
        competing_subset_count=r.competing_subset_count,
        reason_codes=list(r.reason_codes),
        subsets=[
            SubsetOut(
                bill_ids=list(s.bill_ids),
                sum_minor=s.sum_minor,
                abs_residual_minor=s.abs_residual_minor,
                sum_display=_fmt_minor(s.sum_minor, cfg.ledger_currency),
            )
            for s in r.subsets
        ],
        ranker_score=r.ranker_score,
        calibrated_accept_prob=r.calibrated_accept_prob,
        event_id=event_id,
        allocated_sum_minor=r.allocated_sum_minor,
        remaining_minor=r.remaining_minor,
        is_fully_allocated=r.is_fully_allocated,
    )


def _latest_event_for_payment(conn: sqlite3.Connection, payment_id: str) -> str | None:
    row = conn.execute(
        "SELECT event_id FROM match_events WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1",
        (payment_id,),
    ).fetchone()
    return row["event_id"] if row else None


# ---------- endpoints ----------



def _langsmith_trace_url(run_id: str | None) -> str | None:
    if not run_id:
        return None
    project = os.environ.get("LANGSMITH_PROJECT", "bulk-payments-poc")
    return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"


def _agent_resolution_out(agent) -> "AgentResolutionOut":
    run_id = agent.langsmith_run_id
    return AgentResolutionOut(
        action=agent.action.value,
        bill_ids=list(agent.bill_ids),
        confidence=agent.confidence,
        reasoning=agent.reasoning,
        model_id=agent.model_id,
        langsmith_run_id=run_id,
        langsmith_trace_url=_langsmith_trace_url(run_id),
    )

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tenants")
def list_tenants() -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM tenants").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/tenants/{tenant_id}/payments", response_model=list[PaymentOut])
def list_payments(tenant_id: str) -> list[PaymentOut]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM payments WHERE tenant_id = ? ORDER BY payment_date DESC",
        (tenant_id,),
    ).fetchall()
    out: list[PaymentOut] = []
    for r in rows:
        pid = r["payment_id"]
        out.append(
            PaymentOut(
                payment_id=pid,
                tenant_id=r["tenant_id"],
                amount_minor=r["amount_minor"],
                amount_display=_fmt_minor(r["amount_minor"], r["source_currency"] or "USD"),
                payment_date=r["payment_date"],
                counterparty_bank_name=r["counterparty_bank_name"],
                description=r["description"] or "",
                source_currency=r["source_currency"],
                settlement=_payment_settlement(conn, tenant_id, pid),
            )
        )
    conn.close()
    return out


@app.get("/tenants/{tenant_id}/bills", response_model=list[BillOut])
def list_bills(tenant_id: str, unmatched_only: bool = False) -> list[BillOut]:
    conn = _get_conn()
    vendors_raw = load_vendors(conn, tenant_id)
    vendor_names = {vid: data[2] for vid, data in vendors_raw.items()}
    q = "SELECT * FROM bills WHERE tenant_id = ?"
    params: list[Any] = [tenant_id]
    if unmatched_only:
        q += " AND matched_payment_id IS NULL"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [
        BillOut(
            bill_id=r["bill_id"],
            vendor_id=r["vendor_id"],
            vendor_name=vendor_names.get(r["vendor_id"], r["vendor_id"]),
            open_amount_minor=r["open_amount_minor"],
            amount_display=_fmt_minor(r["open_amount_minor"], r["currency"]),
            currency=r["currency"],
            open_date=r["open_date"],
            due_date=r["due_date"],
            matched_payment_id=r["matched_payment_id"],
        )
        for r in rows
    ]


@app.post("/tenants/{tenant_id}/payments/{payment_id}/match", response_model=MatchResultOut)
def run_match(tenant_id: str, payment_id: str) -> MatchResultOut:
    conn = _get_conn()
    try:
        r = match_payment(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            ranker_path=RANKER_PATH,
            log_event=True,
        )
    except KeyError as e:
        conn.close()
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))

    event_id = _latest_event_for_payment(conn, payment_id)
    out = _match_result_out(conn, tenant_id, r, event_id)
    conn.close()
    return out


@app.post(
    "/tenants/{tenant_id}/payments/{payment_id}/agent-resolve",
    response_model=MatchWithAgentOut,
)
def agent_resolve(
    tenant_id: str,
    payment_id: str,
    body: AgentResolveIn | None = None,
) -> MatchWithAgentOut:
    """Run rules engine + AI resolver; persist match_event and agent_resolution."""
    try:
        from bulk_payments.match_with_agent import match_payment_with_agent
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Agent dependencies not installed. Run: pip install -e '.[agent,api]'",
        ) from e

    model = (body.model if body else None) or DEFAULT_AGENT_MODEL
    conn = _get_conn()
    try:
        combined = match_payment_with_agent(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            ranker_path=RANKER_PATH,
            model=model,
        )
    except KeyError as e:
        conn.close()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e)) from e

    rules_out = _match_result_out(conn, tenant_id, combined.rules, combined.event_id)
    agent_out = _agent_resolution_out(combined.agent) if combined.agent else None
    conn.close()
    return MatchWithAgentOut(
        payment_id=payment_id,
        tenant_id=tenant_id,
        event_id=combined.event_id,
        resolution_id=combined.resolution_id,
        rules=rules_out,
        agent=agent_out,
    )


@app.get(
    "/tenants/{tenant_id}/payments/{payment_id}/agent-resolution/latest",
    response_model=AgentResolutionOut | None,
)
def latest_agent_resolution(tenant_id: str, payment_id: str) -> AgentResolutionOut | None:
    """Return the most recent agent resolution for this payment, if any."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT ar.* FROM agent_resolutions ar
        JOIN match_events me ON me.event_id = ar.event_id
        WHERE me.tenant_id = ? AND me.payment_id = ?
        ORDER BY ar.created_at DESC LIMIT 1
        """,
        (tenant_id, payment_id),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    run_id = row["langsmith_run_id"]
    return AgentResolutionOut(
        action=row["action"],
        bill_ids=json.loads(row["bill_ids_json"]),
        confidence=row["confidence"],
        reasoning=row["reasoning"],
        model_id=row["model_id"],
        langsmith_run_id=run_id,
        langsmith_trace_url=_langsmith_trace_url(run_id),
    )


@app.post("/tenants/{tenant_id}/payments/{payment_id}/unmatch", response_model=UnmatchOut)
def unmatch_payment_endpoint(tenant_id: str, payment_id: str) -> UnmatchOut:
    """Release all bills allocated to this payment so it can be matched again from scratch."""
    conn = _get_conn()
    try:
        load_payment(conn, payment_id)
    except KeyError:
        conn.close()
        raise HTTPException(status_code=404, detail="payment not found")

    try:
        freed = unmatch_payment(conn, tenant_id=tenant_id, payment_id=payment_id)
    except ValueError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))

    event_id: str | None = None
    if freed:
        cfg = load_tenant_config(conn, tenant_id)
        event_id = insert_match_event(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            bill_ids=list(freed),
            rules_version=cfg.rules_version,
            features={"action": "unmatch", "freed_bill_count": len(freed)},
            decision="manual",
            outcome="manual_alternative",
            reason_codes=["user_unmatch"],
        )
    conn.close()
    return UnmatchOut(
        payment_id=payment_id,
        freed_bill_ids=list(freed),
        event_id=event_id,
    )


@app.post("/match-events/{event_id}/outcome", response_model=OutcomeOut)
def record_outcome(event_id: str, body: OutcomeIn) -> OutcomeOut:
    """Record accountant feedback by inserting a correction event (append-only)."""
    valid_outcomes = {o.value for o in OutcomeLabel}
    if body.outcome not in valid_outcomes:
        raise HTTPException(
            status_code=422,
            detail=f"outcome must be one of {sorted(valid_outcomes)}",
        )
    conn = _get_conn()
    orig = conn.execute(
        "SELECT * FROM match_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    if orig is None:
        conn.close()
        raise HTTPException(status_code=404, detail="event_id not found")

    corrected_bills = body.corrected_bill_ids or json.loads(orig["bill_ids_json"])
    features = json.loads(orig["features_json"])
    if body.accountant_reasoning:
        features["accountant_reasoning"] = body.accountant_reasoning

    if body.outcome in ("accepted_as_is", "edited_subset"):
        try:
            apply_accepted_bills(
                conn,
                tenant_id=orig["tenant_id"],
                payment_id=orig["payment_id"],
                bill_ids=corrected_bills,
            )
        except ValueError as e:
            conn.close()
            raise HTTPException(status_code=422, detail=str(e))

    new_event_id = insert_match_event(
        conn,
        tenant_id=orig["tenant_id"],
        payment_id=orig["payment_id"],
        bill_ids=corrected_bills,
        rules_version=orig["rules_version"],
        features={**features, "correction_of": event_id},
        decision=orig["decision"],
        outcome=body.outcome,
        ranker_score=orig["ranker_score"],
        calibrated_prob=orig["calibrated_prob"],
        reason_codes=json.loads(orig["reason_codes_json"]),
        user_id=body.user_id,
        accountant_reasoning=body.accountant_reasoning,
    )

    conn.close()
    return OutcomeOut(event_id=new_event_id, outcome=body.outcome)


@app.post("/tenants/{tenant_id}/auto-match", response_model=BatchAutoMatchOut)
def batch_auto_match(tenant_id: str) -> BatchAutoMatchOut:
    """
    Run matcher on all open payments: auto-apply confident matches,
    queue ambiguous suggestions for expert review.
    """
    conn = _get_conn()
    row = conn.execute("SELECT tenant_id FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="tenant not found")

    raw = run_batch_auto_match(conn, tenant_id=tenant_id, ranker_path=RANKER_PATH)

    needs_review_out: list[MatchResultOut] = []
    for r, event_id in raw.get("needs_review_results", []):
        needs_review_out.append(_match_result_out(conn, tenant_id, r, event_id))

    conn.close()
    return BatchAutoMatchOut(
        tenant_id=tenant_id,
        summary=raw["summary"],
        auto_matched=raw["auto_matched"],
        needs_review=needs_review_out,
        no_match=raw["no_match"],
    )


@app.post("/tenants/{tenant_id}/ranker-gate")
def set_ranker_gate(tenant_id: str, enable: bool = True) -> dict[str, Any]:
    """Enable or disable ranker-probability gating for auto-apply on this tenant."""
    conn = _get_conn()
    row = conn.execute("SELECT tenant_id FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="tenant not found")
    conn.execute(
        "UPDATE tenants SET use_ranker_threshold = ? WHERE tenant_id = ?",
        (1 if enable else 0, tenant_id),
    )
    conn.commit()
    conn.close()
    return {"tenant_id": tenant_id, "use_ranker_threshold": enable}


@app.get("/tenants/{tenant_id}/eval")
def eval_report(tenant_id: str) -> dict[str, Any]:
    """Extended metrics: suggestion acceptance, calibration bins, slice breakdown, stop-loss."""
    conn = _get_conn()
    report = extended_eval_report(conn, tenant_id=tenant_id)
    conn.close()
    return report


@app.get("/match-events")
def list_events(tenant_id: str | None = None, payment_id: str | None = None) -> list[dict[str, Any]]:
    conn = _get_conn()
    q = "SELECT * FROM match_events WHERE 1=1"
    params: list[Any] = []
    if tenant_id:
        q += " AND tenant_id = ?"
        params.append(tenant_id)
    if payment_id:
        q += " AND payment_id = ?"
        params.append(payment_id)
    q += " ORDER BY created_at DESC LIMIT 200"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
