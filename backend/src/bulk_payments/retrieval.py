from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher

from bulk_payments.fx import FXError, convert_to_ledger
from bulk_payments.models import Bill, Payment, TenantConfig, Vendor


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def counterparty_vendor_score(counterparty: str, vendor: Vendor) -> float:
    """0..1 fuzzy match between bank counterparty string and vendor display + aliases."""
    cp = _norm(counterparty)
    needles = [_norm(vendor.display_name), *[_norm(a) for a in vendor.bank_name_aliases]]
    best = 0.0
    for n in needles:
        if not n:
            continue
        ratio = SequenceMatcher(None, cp, n).ratio()
        if n in cp or cp in n:
            ratio = max(ratio, 0.85)
        best = max(best, ratio)
    return best


@dataclass(frozen=True)
class RetrievedBill:
    bill: Bill
    vendor_link_score: float
    date_delta_days: int
    ledger_amount_minor: int  # bill amount converted to tenant ledger currency
    fx_rate_used: float  # 1.0 if same currency


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason_codes: tuple[str, ...]
    bills: tuple[RetrievedBill, ...]


def apply_hard_gates(
    *,
    payment: Payment,
    cfg: TenantConfig,
    candidates: tuple[RetrievedBill, ...],
) -> GateResult:
    """Double-spend check. Currency normalization already happened in retrieve_candidates."""
    reasons: list[str] = []
    kept: list[RetrievedBill] = []
    for rb in candidates:
        b = rb.bill
        if b.matched_payment_id is not None:
            reasons.append(f"skip_already_matched:{b.bill_id}")
            continue
        kept.append(rb)
    if not kept:
        return GateResult(False, tuple(reasons) or ("no_candidates_after_gates",), ())
    return GateResult(True, tuple(reasons), tuple(kept))


def retrieve_candidates(
    *,
    payment: Payment,
    cfg: TenantConfig,
    bills: list[Bill],
    vendors: dict[str, Vendor],
    conn: sqlite3.Connection | None = None,
) -> tuple[RetrievedBill, ...]:
    """
    Filter by date window, fuzzy-rank by vendor linkage, normalize bill amounts to
    ledger currency via FX (fail closed when rate missing), cap by max_candidates.
    """
    win_start: date = payment.payment_date - timedelta(days=cfg.date_window_days)
    win_end: date = payment.payment_date + timedelta(days=cfg.date_window_days)

    scored: list[RetrievedBill] = []
    for b in bills:
        if b.tenant_id != payment.tenant_id:
            continue
        if b.open_date < win_start or b.open_date > win_end:
            continue
        vendor = vendors.get(b.vendor_id)
        if vendor is None:
            continue

        # FX normalization: convert bill to ledger currency
        if b.currency == cfg.ledger_currency:
            ledger_amt = b.open_amount_minor
            rate = 1.0
        elif conn is not None:
            try:
                ledger_amt, rate = convert_to_ledger(
                    conn,
                    tenant_id=cfg.tenant_id,
                    amount_minor=b.open_amount_minor,
                    from_currency=b.currency,
                    to_currency=cfg.ledger_currency,
                    as_of=payment.payment_date,
                )
            except FXError:
                # Fail closed: skip bill if FX rate unavailable
                continue
        else:
            # No DB connection provided and bill is foreign currency — skip
            continue

        link = counterparty_vendor_score(payment.counterparty_bank_name, vendor)
        delta = abs((b.open_date - payment.payment_date).days)
        scored.append(
            RetrievedBill(
                bill=b,
                vendor_link_score=link,
                date_delta_days=delta,
                ledger_amount_minor=ledger_amt,
                fx_rate_used=rate,
            )
        )

    scored.sort(key=lambda rb: (-rb.vendor_link_score, rb.date_delta_days))
    return tuple(scored[: cfg.max_candidates])
