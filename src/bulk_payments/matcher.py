from __future__ import annotations

import json
from typing import Any

import sqlite3

from bulk_payments.db import insert_match_event
from bulk_payments.models import Bill, DecisionKind, MatchResult, Payment, SubsetCandidate, TenantConfig, Vendor
from bulk_payments.policy import PolicyConfig, choose_decision
from bulk_payments.ranker import load_ranker, predict_accept_prob
from bulk_payments.retrieval import apply_hard_gates, retrieve_candidates
from bulk_payments.subset_solver import find_feasible_subsets, rank_subsets
from bulk_payments.synthetic import load_bills, load_payment, load_tenant_config, load_vendors


def _vendors_as_objects(raw: dict[str, tuple]) -> dict[str, Vendor]:
    out: dict[str, Vendor] = {}
    for vid, (_vid, tenant_id, display_name, aliases) in raw.items():
        out[vid] = Vendor(
            vendor_id=vid,
            tenant_id=tenant_id,
            display_name=display_name,
            bank_name_aliases=aliases,
        )
    return out


def _dedupe_subsets(subs: tuple[SubsetCandidate, ...]) -> tuple[SubsetCandidate, ...]:
    seen: set[frozenset[str]] = set()
    out: list[SubsetCandidate] = []
    for s in subs:
        key = frozenset(s.bill_ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)


def _vendor_link_for_subset(
    bill_ids: frozenset[str],
    retrieved_by_id: dict[str, Any],
) -> float:
    scores = []
    for bid in bill_ids:
        rb = retrieved_by_id.get(bid)
        if rb is not None:
            scores.append(rb.vendor_link_score)
    return max(scores) if scores else 0.0


def _feature_vector(
    *,
    payment: Payment,
    subset: SubsetCandidate,
    retrieved_by_id: dict[str, Any],
    competing_subset_count: int,
    unique_closure: bool,
) -> dict[str, Any]:
    max_link = _vendor_link_for_subset(frozenset(subset.bill_ids), retrieved_by_id)
    min_date_delta = min(
        (retrieved_by_id[b].date_delta_days for b in subset.bill_ids if b in retrieved_by_id),
        default=0,
    )
    return {
        "payment_amount_minor": payment.amount_minor,
        "subset_sum_minor": subset.sum_minor,
        "abs_residual_minor": subset.abs_residual_minor,
        "subset_size": len(subset.bill_ids),
        "competing_subset_count": competing_subset_count,
        "unique_amount_closure": unique_closure,
        "max_vendor_link_score": max_link,
        "min_date_delta_days": min_date_delta,
        "desc_len": len(payment.description or ""),
    }


def match_payment(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    policy: PolicyConfig | None = None,
    ranker_path: str | None = None,
    log_event: bool = True,
) -> MatchResult:
    policy = policy or PolicyConfig()
    cfg = load_tenant_config(conn, tenant_id)
    payment = load_payment(conn, payment_id)
    if payment.tenant_id != tenant_id:
        raise ValueError("payment tenant mismatch")
    bills = load_bills(conn, tenant_id)
    vendors = _vendors_as_objects(load_vendors(conn, tenant_id))

    retrieved = retrieve_candidates(payment=payment, cfg=cfg, bills=bills, vendors=vendors)
    gates = apply_hard_gates(payment=payment, cfg=cfg, candidates=retrieved)
    reason_codes: list[str] = list(gates.reason_codes)

    if not gates.ok or not gates.bills:
        res = MatchResult(
            payment_id=payment_id,
            tenant_id=tenant_id,
            decision=DecisionKind.NO_CANDIDATES,
            subsets=(),
            unique_best=False,
            competing_subset_count=0,
            reason_codes=tuple(reason_codes) or ("retrieval_or_gates_failed",),
            features={},
        )
        if log_event:
            insert_match_event(
                conn,
                tenant_id=tenant_id,
                payment_id=payment_id,
                bill_ids=[],
                rules_version=cfg.rules_version,
                features=res.features,
                decision=res.decision.value,
                reason_codes=list(res.reason_codes),
            )
        return res

    feasible = find_feasible_subsets(
        gates.bills,
        payment.amount_minor,
        cfg.amount_tolerance_minor,
        max_solutions=50,
    )
    feasible = _dedupe_subsets(feasible)
    feasible_ranked = rank_subsets(feasible)
    competing = len(feasible_ranked)
    unique_closure = competing == 1

    best = feasible_ranked[0] if feasible_ranked else None
    second = feasible_ranked[1] if len(feasible_ranked) > 1 else None

    retrieved_by_id = {rb.bill.bill_id: rb for rb in gates.bills}
    vendor_link_best = _vendor_link_for_subset(
        frozenset(best.bill_ids) if best else frozenset(),
        retrieved_by_id,
    )

    feats = (
        _feature_vector(
            payment=payment,
            subset=best,
            retrieved_by_id=retrieved_by_id,
            competing_subset_count=competing,
            unique_closure=unique_closure,
        )
        if best
        else {}
    )

    ranker = load_ranker(ranker_path) if ranker_path else None
    cal_prob, raw_score = predict_accept_prob(ranker, feats)

    decision, dreasons = choose_decision(
        payment=payment,
        cfg=cfg,
        policy=policy,
        feasible=feasible_ranked,
        unique_amount_closure=unique_closure,
        competing_subset_count=competing,
        best=best,
        second=second,
        vendor_link_best=vendor_link_best,
        calibrated_prob=cal_prob,
        ranker_score=raw_score,
    )
    reason_codes.extend(dreasons)

    res = MatchResult(
        payment_id=payment_id,
        tenant_id=tenant_id,
        decision=decision,
        subsets=tuple(feasible_ranked[:10]),
        unique_best=unique_closure,
        competing_subset_count=competing,
        reason_codes=tuple(reason_codes),
        features=feats,
        ranker_score=raw_score,
        calibrated_accept_prob=cal_prob,
    )

    if log_event:
        insert_match_event(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            bill_ids=list(best.bill_ids) if best else [],
            rules_version=cfg.rules_version,
            features=feats,
            decision=decision.value,
            ranker_score=raw_score,
            calibrated_prob=cal_prob,
            reason_codes=list(dict.fromkeys(reason_codes)),
        )
    return res
