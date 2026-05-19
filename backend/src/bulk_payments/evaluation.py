from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from bulk_payments.db import update_match_event_outcome
from bulk_payments.matcher import match_payment
from bulk_payments.models import DecisionKind


OracleKind = Literal["none", "suggest", "exact"]


@dataclass(frozen=True)
class OracleCase:
    kind: OracleKind
    bills: frozenset[str] | None = None
    note: str = ""


# Expected outcomes after seed_demo_dataset (see synthetic.py).
# Each case is asserted by run_eval(); covers retrieval, FX, allocation,
# ambiguity, tolerance, vendor gating, and exposure caps.
ORACLE: dict[str, OracleCase] = {
    # --- t1: dense AR ledger (stress — overlapping vendor pool, many subsets) -----
    # Ambiguity-by-design: should NOT auto-apply even though a "natural" subset exists.
    "pay_bulk_1": OracleCase("suggest", note="$500: many ambiguous subsets in shared acme pool"),
    "pay_ambig_750": OracleCase("suggest", note="$750: b1+b2+b6 vs b3+b4+b6 plus near-misses"),
    "pay_tolerance": OracleCase("suggest", note="$500.01 within 2¢ tolerance, but pool is ambiguous"),
    "pay_fuzzy_acme": OracleCase("suggest", note="alias counterparty (ACME CORP PAYMENTS) → many $350 subsets"),
    "pay_single_400": OracleCase("suggest", note="b6=$400 exists but pool offers other $400 subsets"),
    "pay_delta_300": OracleCase("suggest", note="delta vendor lives in same date window as acme noise"),
    "pay_seek_250": OracleCase("suggest", note="$250 closes via b2 alone OR b_rem_250 OR multi-bill"),
    "pay_wrong_vendor": OracleCase("suggest", note="$999 closure possible from acme pool ignoring beta"),
    "pay_no_match": OracleCase("suggest", note="$777 from UNKNOWN BANK: weak link survives; closures exist"),
    # Hard-no-match: gates / allocation block any decision.
    "pay_fx_eur": OracleCase("exact", frozenset({"b_eur"}), note="auto: EUR €90 → $99 via FX"),
    "pay_gbp_gap": OracleCase("none", note="GBP bill has no FX rate → retrieval drops it"),
    "pay_prealloc_500": OracleCase("none", note="payment fully allocated to b_lock_a+b_lock_b"),
    "pay_all_matched": OracleCase("none", note="all b_cent_* pre-matched elsewhere"),
    "pay_exact_10": OracleCase(
        "none",
        note="10×$1 subset exists but max_candidates=28 + retrieval ranking truncates the dime ladder",
    ),
    "pay_cap_stress": OracleCase("none", note="$36 target lost behind higher-ranked acme bills in top-28 cap"),
    # --- t2: medium tenant, gamma and omega weeks isolated by 7-day window --------
    "pay_t2_ok": OracleCase(
        "exact",
        frozenset({"t2_b2", "t2_b4"}),
        note="auto: $125 closes uniquely via t2_b2+t2_b4 (t2_b1=60 prevents alt)",
    ),
    "pay_t2_ambig": OracleCase("suggest", note="$135: t2_b1+t2_b2 vs t2_b1+t2_b3+t2_b4"),
    "pay_t2_no_match": OracleCase("none", note="$333 has no closure in omega week"),
    "pay_t2_omega": OracleCase(
        "exact",
        frozenset({"o1", "o2"}),
        note="auto: $100 closes uniquely via o1+o2",
    ),
    # --- t3: regression — one scenario per week, tight 3-day window ----------------
    "pay_iso_single": OracleCase("exact", frozenset({"iso_b6"}), note="auto: lone $400 bill in week"),
    "pay_iso_ambig": OracleCase("suggest", note="$500 = iso_x+iso_y vs iso_z+iso_w"),
    "pay_iso_fx": OracleCase("exact", frozenset({"iso_eur"}), note="auto: EUR via FX"),
    "pay_iso_tolerance": OracleCase(
        "exact",
        frozenset({"iso_t1", "iso_t2", "iso_t3"}),
        note="auto: $500.01 closes 100+250+150 within 1¢ tolerance",
    ),
    "pay_iso_nomatch": OracleCase("none", note="$99,999.99 has no closure"),
    "pay_iso_prealloc": OracleCase("none", note="payment fully allocated"),
    "pay_iso_seek": OracleCase(
        "exact",
        frozenset({"iso_rem_250"}),
        note="auto: only iso_rem_250 open in week (locks ignored)",
    ),
    "pay_iso_exact10": OracleCase(
        "exact",
        frozenset({f"iso_dime_{i}" for i in range(1, 11)}),
        note="auto: 10×$1 closes uniquely under max_auto_bill_count=50",
    ),
    "pay_iso_wrong_vendor": OracleCase("none", note="ZULU EXPORT vs Beta → weak_vendor_link gates out"),
    "pay_iso_gbp": OracleCase("none", note="GBP no FX rate → retrieval drops the only bill"),
    "pay_iso_delta": OracleCase("exact", frozenset({"iso_d3"}), note="auto: DELTA SAVINGS → iso_d3"),
}


def _best_bill_set(result) -> frozenset[str] | None:
    if not result.subsets:
        return None
    return frozenset(result.subsets[0].bill_ids)


def label_events_for_training(conn: sqlite3.Connection) -> int:
    """Attach synthetic accountant labels to latest match_events per payment_id."""
    updated = 0
    for payment_id, oracle in ORACLE.items():
        row = conn.execute(
            """
            SELECT event_id, decision, bill_ids_json FROM match_events
            WHERE payment_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (payment_id,),
        ).fetchone()
        if row is None:
            continue
        event_id = row["event_id"]
        decision = row["decision"]
        bills = frozenset(json.loads(row["bill_ids_json"]))
        outcome: str
        if oracle.kind == "none":
            outcome = "accepted_as_is" if decision == DecisionKind.NO_CANDIDATES.value else "rejected"
        elif oracle.kind == "suggest":
            outcome = "accepted_as_is" if decision == DecisionKind.SUGGESTED.value else "rejected"
        elif oracle.kind == "exact":
            if decision == DecisionKind.AUTO_APPLIED.value and bills == oracle.bills:
                outcome = "accepted_as_is"
            elif decision == DecisionKind.SUGGESTED.value and bills == oracle.bills:
                outcome = "accepted_as_is"
            else:
                outcome = "rejected"
        else:
            outcome = "rejected"
        update_match_event_outcome(conn, event_id, outcome)
        updated += 1
    return updated


def run_eval(conn: sqlite3.Connection, policy=None, ranker_path: str | None = None) -> dict[str, Any]:
    """Run matcher on oracle payments and compute precision/recall style metrics."""
    conn.execute(
        f"DELETE FROM match_events WHERE payment_id IN ({','.join('?' for _ in ORACLE)})",
        tuple(ORACLE.keys()),
    )
    conn.commit()
    results: dict[str, Any] = {}
    auto_tp = auto_fp = auto_fn = 0
    suggest_ok = suggest_bad = 0
    none_ok = none_bad = 0

    for payment_id, oracle in ORACLE.items():
        tenant_row = conn.execute(
            "SELECT tenant_id FROM payments WHERE payment_id = ?", (payment_id,)
        ).fetchone()
        assert tenant_row is not None
        tenant_id = tenant_row["tenant_id"]
        r = match_payment(
            conn,
            tenant_id=tenant_id,
            payment_id=payment_id,
            policy=policy,
            ranker_path=ranker_path,
        )
        best = _best_bill_set(r)

        passed = False
        if oracle.kind == "none":
            passed = r.decision == DecisionKind.NO_CANDIDATES
            if passed:
                none_ok += 1
            else:
                none_bad += 1
        elif oracle.kind == "suggest":
            passed = r.decision == DecisionKind.SUGGESTED and bool(r.subsets)
            if passed:
                suggest_ok += 1
            else:
                suggest_bad += 1
        elif oracle.kind == "exact":
            passed = r.decision == DecisionKind.AUTO_APPLIED and best == oracle.bills
            if r.decision == DecisionKind.AUTO_APPLIED:
                if passed:
                    auto_tp += 1
                else:
                    auto_fp += 1
            elif best == oracle.bills and r.decision == DecisionKind.SUGGESTED:
                auto_fn += 1

        results[payment_id] = {
            "decision": r.decision.value,
            "best_subset": sorted(best) if best else [],
            "competing_subsets": r.competing_subset_count,
            "oracle_pass": passed,
            "reasons": list(r.reason_codes),
            "note": oracle.note,
        }

    precision_auto = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) else None
    summary = {
        "per_payment": results,
        "counts": {
            "auto_tp": auto_tp,
            "auto_fp": auto_fp,
            "auto_fn": auto_fn,
            "suggest_ok": suggest_ok,
            "suggest_bad": suggest_bad,
            "none_ok": none_ok,
            "none_bad": none_bad,
        },
        "precision_auto_apply": precision_auto,
        "oracle_all_pass": all(results[pid]["oracle_pass"] for pid in ORACLE),
    }
    return summary


def suggestion_acceptance_rate(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
    exclude_pending: bool = True,
) -> dict[str, Any]:
    base = """
        SELECT decision, outcome FROM match_events
        WHERE decision IN ('suggested', 'auto_applied')
    """
    params: list[Any] = []
    if tenant_id:
        base += " AND tenant_id = ?"
        params.append(tenant_id)
    if exclude_pending:
        base += " AND outcome != 'pending'"
    rows = conn.execute(base, params).fetchall()

    suggest_accepted = suggest_total = 0
    auto_accepted = auto_total = 0
    for r in rows:
        decision, outcome = r["decision"], r["outcome"]
        accepted = outcome == "accepted_as_is"
        if decision == "suggested":
            suggest_total += 1
            if accepted:
                suggest_accepted += 1
        elif decision == "auto_applied":
            auto_total += 1
            if accepted:
                auto_accepted += 1

    return {
        "suggestion_acceptance_rate": suggest_accepted / suggest_total if suggest_total else None,
        "suggestion_accepted": suggest_accepted,
        "suggestion_total": suggest_total,
        "auto_precision": auto_accepted / auto_total if auto_total else None,
        "auto_accepted": auto_accepted,
        "auto_total": auto_total,
    }


def calibration_bins(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    base = """
        SELECT calibrated_prob, outcome FROM match_events
        WHERE calibrated_prob IS NOT NULL
          AND outcome IN ('accepted_as_is', 'rejected', 'edited_subset', 'manual_alternative')
    """
    params: list[Any] = []
    if tenant_id:
        base += " AND tenant_id = ?"
        params.append(tenant_id)
    rows = conn.execute(base, params).fetchall()
    if not rows:
        return []

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for r in rows:
        prob = float(r["calibrated_prob"])
        y = 1 if r["outcome"] == "accepted_as_is" else 0
        idx = min(int(prob * n_bins), n_bins - 1)
        buckets[idx].append((prob, y))

    out: list[dict[str, Any]] = []
    for i, bucket in enumerate(buckets):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not bucket:
            out.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "mean_prob": None, "frac_accepted": None})
        else:
            probs, ys = zip(*bucket)
            out.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": len(bucket),
                    "mean_prob": sum(probs) / len(probs),
                    "frac_accepted": sum(ys) / len(ys),
                }
            )
    return out


def slice_metrics(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    base = """
        SELECT features_json, decision, outcome, calibrated_prob
        FROM match_events
        WHERE outcome IN ('accepted_as_is', 'rejected', 'edited_subset', 'manual_alternative')
    """
    params: list[Any] = []
    if tenant_id:
        base += " AND tenant_id = ?"
        params.append(tenant_id)
    rows = conn.execute(base, params).fetchall()

    slices: dict[str, dict[str, int]] = {
        "weak_vendor_link": {"total": 0, "accepted": 0},
        "ambiguous_subsets": {"total": 0, "accepted": 0},
        "large_subset": {"total": 0, "accepted": 0},
        "fx_payment": {"total": 0, "accepted": 0},
    }

    for r in rows:
        feats: dict[str, Any] = json.loads(r["features_json"])
        outcome = r["outcome"]
        accepted = 1 if outcome == "accepted_as_is" else 0

        if float(feats.get("max_vendor_link_score", 1.0)) < 0.5:
            slices["weak_vendor_link"]["total"] += 1
            slices["weak_vendor_link"]["accepted"] += accepted
        if int(feats.get("competing_subset_count", 0)) > 1:
            slices["ambiguous_subsets"]["total"] += 1
            slices["ambiguous_subsets"]["accepted"] += accepted
        if int(feats.get("subset_size", 0)) >= 5:
            slices["large_subset"]["total"] += 1
            slices["large_subset"]["accepted"] += accepted
        if feats.get("fx_rate_used", 1.0) != 1.0:
            slices["fx_payment"]["total"] += 1
            slices["fx_payment"]["accepted"] += accepted

    out: dict[str, Any] = {}
    for name, s in slices.items():
        rate = s["accepted"] / s["total"] if s["total"] else None
        out[name] = {"total": s["total"], "accepted": s["accepted"], "acceptance_rate": rate}
    return out


def precision_stop_loss(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
    window: int = 50,
    min_precision: float = 0.95,
) -> dict[str, Any]:
    base = """
        SELECT outcome FROM match_events
        WHERE decision = 'auto_applied'
          AND outcome IN ('accepted_as_is', 'rejected', 'edited_subset', 'manual_alternative')
    """
    params: list[Any] = []
    if tenant_id:
        base += " AND tenant_id = ?"
        params.append(tenant_id)
    base += " ORDER BY created_at DESC LIMIT ?"
    params.append(window)
    rows = conn.execute(base, params).fetchall()
    total = len(rows)
    accepted = sum(1 for r in rows if r["outcome"] == "accepted_as_is")
    precision = accepted / total if total else None
    return {
        "window": window,
        "total_auto": total,
        "accepted": accepted,
        "precision": precision,
        "above_threshold": precision >= min_precision if precision is not None else None,
        "min_precision": min_precision,
    }


def extended_eval_report(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
    ranker_path: str | None = None,
) -> dict[str, Any]:
    return {
        "suggestion_acceptance": suggestion_acceptance_rate(conn, tenant_id=tenant_id),
        "calibration_bins": calibration_bins(conn, tenant_id=tenant_id),
        "slice_metrics": slice_metrics(conn, tenant_id=tenant_id),
        "stop_loss": precision_stop_loss(conn, tenant_id=tenant_id),
    }
