from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class DecisionKind(str, Enum):
    AUTO_APPLIED = "auto_applied"
    SUGGESTED = "suggested"
    REJECTED_BY_USER = "rejected_by_user"
    NO_CANDIDATES = "no_candidates"
    MANUAL = "manual"


class OutcomeLabel(str, Enum):
    PENDING = "pending"
    ACCEPTED_AS_IS = "accepted_as_is"
    EDITED_SUBSET = "edited_subset"
    REJECTED = "rejected"
    MANUAL_ALTERNATIVE = "manual_alternative"


@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    ledger_currency: str = "USD"
    amount_tolerance_minor: int = 2  # ± cents in ledger minor units
    date_window_days: int = 90
    max_candidates: int = 28
    max_auto_amount_minor: int | None = 10_000_000_000  # cap in minor units; None = no cap
    max_auto_bill_count: int = 50
    rules_version: str = "v1"


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    tenant_id: str
    display_name: str
    bank_name_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bill:
    bill_id: str
    tenant_id: str
    vendor_id: str
    open_amount_minor: int  # ledger currency, minor units (e.g. cents)
    currency: str  # original bill currency (for audit)
    open_date: date
    due_date: date | None = None
    matched_payment_id: str | None = None  # ledger: already allocated


@dataclass(frozen=True)
class Payment:
    payment_id: str
    tenant_id: str
    amount_minor: int  # ledger currency
    payment_date: date
    counterparty_bank_name: str
    description: str = ""
    fx_rate_used: float = 1.0
    source_currency: str | None = None


@dataclass(frozen=True)
class SubsetCandidate:
    bill_ids: tuple[str, ...]
    sum_minor: int
    abs_residual_minor: int


@dataclass(frozen=True)
class MatchResult:
    payment_id: str
    tenant_id: str
    decision: DecisionKind
    subsets: tuple[SubsetCandidate, ...]
    unique_best: bool
    competing_subset_count: int
    reason_codes: tuple[str, ...]
    features: dict[str, Any] = field(default_factory=dict)
    ranker_score: float | None = None
    calibrated_accept_prob: float | None = None


@dataclass(frozen=True)
class MatchEventRow:
    """Append-only audit row (mirrors DB)."""

    event_id: str | None
    tenant_id: str
    payment_id: str
    bill_ids_json: str
    rules_version: str
    features_json: str
    decision: str
    outcome: str
    ranker_score: float | None
    calibrated_prob: float | None
    reason_codes_json: str
    created_at_iso: str
