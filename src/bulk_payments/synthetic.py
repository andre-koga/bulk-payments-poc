from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from typing import Sequence

from bulk_payments.db import connect, init_db


def _usd_minor(dollars: float) -> int:
    return int(round(dollars * 100))


def seed_demo_dataset(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """
    Insert tenants, vendors, bills, and payments including:
    - clear bulk match (unique subset)
    - ambiguous two subsets (same sum) -> suggest only
    - noise bills outside window
    - optional FX payment (rate != 1) if we add EUR bill normalized upstream — here all USD ledger for POC
    """
    init_db(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM match_events")
    cur.execute("DELETE FROM payments")
    cur.execute("DELETE FROM bills")
    cur.execute("DELETE FROM vendors")
    cur.execute("DELETE FROM tenants")

    tenants = [
        (
            "t1",
            "USD",
            2,
            120,
            28,
            10_000_000_000,
            50,
            "v1",
        ),
        (
            "t2",
            "USD",
            2,
            60,
            20,
            None,
            20,
            "v1",
        ),
    ]
    cur.executemany(
        """
        INSERT INTO tenants(
          tenant_id, ledger_currency, amount_tolerance_minor, date_window_days,
          max_candidates, max_auto_amount_minor, max_auto_bill_count, rules_version
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        tenants,
    )

    vendors_t1 = [
        ("v_acme", "t1", "Acme Corp", json.dumps(["ACME BANK", "Acme National"])),
        ("v_beta", "t1", "Beta LLC", json.dumps(["BETA SAVINGS"])),
    ]
    cur.executemany(
        """
        INSERT INTO vendors(vendor_id, tenant_id, display_name, bank_aliases_json)
        VALUES (?,?,?,?)
        """,
        vendors_t1,
    )

    base = date(2025, 1, 15)
    bills_t1: list[tuple] = []
    # Open bills for Acme — scenario A: 100+250+150 = 500 (unique)
    bills_t1.extend(
        [
            ("b1", "t1", "v_acme", _usd_minor(100), "USD", base.isoformat(), None, None),
            ("b2", "t1", "v_acme", _usd_minor(250), "USD", base.isoformat(), None, None),
            ("b3", "t1", "v_acme", _usd_minor(150), "USD", base.isoformat(), None, None),
        ]
    )
    # Scenario B: add 200+300 and 400+100 both = 500 -> ambiguous if all Acme candidates
    bills_t1.append(("b4", "t1", "v_acme", _usd_minor(200), "USD", base.isoformat(), None, None))
    bills_t1.append(("b5", "t1", "v_acme", _usd_minor(300), "USD", base.isoformat(), None, None))
    bills_t1.append(("b6", "t1", "v_acme", _usd_minor(400), "USD", base.isoformat(), None, None))
    # Old bill outside date window (retrieval should drop for pay_bulk_1 date)
    old = base - timedelta(days=200)
    bills_t1.append(("b_old", "t1", "v_acme", _usd_minor(500), "USD", old.isoformat(), None, None))
    # Beta vendor noise
    bills_t1.append(("b_beta", "t1", "v_beta", _usd_minor(999), "USD", base.isoformat(), None, None))

    cur.executemany(
        """
        INSERT INTO bills(
          bill_id, tenant_id, vendor_id, open_amount_minor, currency,
          open_date, due_date, matched_payment_id
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        bills_t1,
    )

    # Tenant t2: small set for eval / ranker smoke
    cur.execute(
        """
        INSERT INTO vendors(vendor_id, tenant_id, display_name, bank_aliases_json)
        VALUES ('v_gamma','t2','Gamma Inc','["GAMMA BANK"]')
        """
    )
    b7 = base + timedelta(days=1)
    cur.executemany(
        """
        INSERT INTO bills(
          bill_id, tenant_id, vendor_id, open_amount_minor, currency,
          open_date, due_date, matched_payment_id
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        [
            ("t2_b1", "t2", "v_gamma", _usd_minor(50), "USD", b7.isoformat(), None, None),
            ("t2_b2", "t2", "v_gamma", _usd_minor(75), "USD", b7.isoformat(), None, None),
        ],
    )

    pay_day = base + timedelta(days=5)
    payments = [
        (
            "pay_bulk_1",
            "t1",
            _usd_minor(500),
            pay_day.isoformat(),
            "ACME BANK",
            "ACH batch",
            1.0,
            None,
        ),
        (
            "pay_no_match",
            "t1",
            _usd_minor(777),
            pay_day.isoformat(),
            "UNKNOWN BANK",
            "",
            1.0,
            None,
        ),
        (
            "pay_t2_ok",
            "t2",
            _usd_minor(125),
            (b7 + timedelta(days=2)).isoformat(),
            "GAMMA BANK",
            "wire",
            1.0,
            None,
        ),
    ]
    cur.executemany(
        """
        INSERT INTO payments(
          payment_id, tenant_id, amount_minor, payment_date,
          counterparty_bank_name, description, fx_rate_used, source_currency
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        payments,
    )
    conn.commit()

    return {
        "tenants": [r[0] for r in tenants],
        "payments": [p[0] for p in payments],
        "bills_t1": [b[0] for b in bills_t1],
    }


def load_tenant_config(conn: sqlite3.Connection, tenant_id: str):
    from bulk_payments.models import TenantConfig

    row = conn.execute("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown tenant {tenant_id}")
    return TenantConfig(
        tenant_id=row["tenant_id"],
        ledger_currency=row["ledger_currency"],
        amount_tolerance_minor=row["amount_tolerance_minor"],
        date_window_days=row["date_window_days"],
        max_candidates=row["max_candidates"],
        max_auto_amount_minor=row["max_auto_amount_minor"],
        max_auto_bill_count=row["max_auto_bill_count"],
        rules_version=row["rules_version"],
    )


def load_vendors(conn: sqlite3.Connection, tenant_id: str) -> dict[str, tuple]:
    out = {}
    for r in conn.execute(
        "SELECT * FROM vendors WHERE tenant_id = ?", (tenant_id,)
    ).fetchall():
        out[r["vendor_id"]] = (
            r["vendor_id"],
            r["tenant_id"],
            r["display_name"],
            tuple(json.loads(r["bank_aliases_json"])),
        )
    return out


def load_bills(conn: sqlite3.Connection, tenant_id: str) -> list:
    from bulk_payments.models import Bill

    bills = []
    for r in conn.execute("SELECT * FROM bills WHERE tenant_id = ?", (tenant_id,)).fetchall():
        bills.append(
            Bill(
                bill_id=r["bill_id"],
                tenant_id=r["tenant_id"],
                vendor_id=r["vendor_id"],
                open_amount_minor=r["open_amount_minor"],
                currency=r["currency"],
                open_date=date.fromisoformat(r["open_date"]),
                due_date=date.fromisoformat(r["due_date"]) if r["due_date"] else None,
                matched_payment_id=r["matched_payment_id"],
            )
        )
    return bills


def load_payment(conn: sqlite3.Connection, payment_id: str):
    from bulk_payments.models import Payment

    r = conn.execute("SELECT * FROM payments WHERE payment_id = ?", (payment_id,)).fetchone()
    if r is None:
        raise KeyError(f"unknown payment {payment_id}")
    return Payment(
        payment_id=r["payment_id"],
        tenant_id=r["tenant_id"],
        amount_minor=r["amount_minor"],
        payment_date=date.fromisoformat(r["payment_date"]),
        counterparty_bank_name=r["counterparty_bank_name"],
        description=r["description"] or "",
        fx_rate_used=float(r["fx_rate_used"]),
        source_currency=r["source_currency"],
    )


def create_db_with_seed(db_path: str) -> dict[str, Sequence[str]]:
    conn = connect(db_path)
    info = seed_demo_dataset(conn)
    conn.close()
    return info
