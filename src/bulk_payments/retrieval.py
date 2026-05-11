from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher

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
    """Currency match to ledger, double-spend (already matched bills)."""
    reasons: list[str] = []
    kept: list[RetrievedBill] = []
    for rb in candidates:
        b = rb.bill
        if b.matched_payment_id is not None:
            reasons.append(f"skip_already_matched:{b.bill_id}")
            continue
        if b.currency != cfg.ledger_currency:
            reasons.append(f"currency_mismatch:{b.bill_id}")
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
) -> tuple[RetrievedBill, ...]:
    """Filter by date window around payment_date, rank by vendor linkage, cap by max_candidates."""
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
        link = counterparty_vendor_score(payment.counterparty_bank_name, vendor)
        delta = abs((b.open_date - payment.payment_date).days)
        scored.append(RetrievedBill(bill=b, vendor_link_score=link, date_delta_days=delta))

    scored.sort(key=lambda rb: (-rb.vendor_link_score, rb.date_delta_days))
    return tuple(scored[: cfg.max_candidates])
