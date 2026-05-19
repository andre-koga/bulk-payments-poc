from __future__ import annotations

import sqlite3
from datetime import date


class FXError(Exception):
    """Raised when FX rate is unavailable and fail-closed is required."""


def get_fx_rate(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    from_currency: str,
    to_currency: str,
    as_of: date,
) -> float:
    """Return the rate (from → to) for the given tenant on or before as_of.

    Raises FXError when no rate is found so callers can fail closed.
    Returns 1.0 immediately for same-currency conversions.
    """
    if from_currency == to_currency:
        return 1.0
    row = conn.execute(
        """
        SELECT rate FROM fx_rates
        WHERE tenant_id = ?
          AND from_currency = ?
          AND to_currency = ?
          AND effective_date <= ?
        ORDER BY effective_date DESC
        LIMIT 1
        """,
        (tenant_id, from_currency, to_currency, as_of.isoformat()),
    ).fetchone()
    if row is None:
        raise FXError(
            f"No FX rate for {from_currency}→{to_currency} on {as_of} (tenant={tenant_id})"
        )
    return float(row["rate"])


def convert_to_ledger(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    amount_minor: int,
    from_currency: str,
    to_currency: str,
    as_of: date,
) -> tuple[int, float]:
    """Convert amount_minor from from_currency to to_currency.

    Returns (converted_minor, rate_used).
    Raises FXError if rate is missing.
    """
    rate = get_fx_rate(
        conn,
        tenant_id=tenant_id,
        from_currency=from_currency,
        to_currency=to_currency,
        as_of=as_of,
    )
    converted = int(round(amount_minor * rate))
    return converted, rate


def upsert_fx_rate(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    from_currency: str,
    to_currency: str,
    rate: float,
    effective_date: date,
) -> None:
    from datetime import datetime, timezone

    conn.execute(
        """
        INSERT INTO fx_rates(tenant_id, from_currency, to_currency, rate, effective_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            tenant_id,
            from_currency,
            to_currency,
            rate,
            effective_date.isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
