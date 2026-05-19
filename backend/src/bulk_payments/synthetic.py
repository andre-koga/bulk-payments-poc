from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from typing import Sequence

from bulk_payments.db import connect, init_db
from bulk_payments.fx import upsert_fx_rate


def _usd_minor(dollars: float) -> int:
    return int(round(dollars * 100))


def _eur_minor(euros: float) -> int:
    return int(round(euros * 100))


def _gbp_minor(pounds: float) -> int:
    return int(round(pounds * 100))


# ---------------------------------------------------------------------------
# Demo dataset — realistic mix of edge cases for matcher / policy / allocation
# See evaluation.ORACLE for expected outcomes per payment_id.
# ---------------------------------------------------------------------------

def seed_demo_dataset(conn: sqlite3.Connection) -> dict[str, list[str]]:
    init_db(conn)
    cur = conn.cursor()
    for table in ("match_events", "payments", "bills", "vendors", "fx_rates", "tenants"):
        try:
            cur.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass

    tenants = [
        # t1: dense AR ledger — wide window, many overlapping bills (stress / ambiguity)
        ("t1", "USD", 2, 120, 28, 10_000_000_000, 50, "v1", 0),
        # t2: smaller cap, 7-day window isolates gamma vs omega weeks
        ("t2", "USD", 2, 7, 20, None, 20, "v1", 0),
        # t3: regression tenant — one scenario per week, tight 3-day window
        ("t3", "USD", 2, 3, 40, 10_000_000_000, 50, "v1", 0),
    ]
    cur.executemany(
        """
        INSERT INTO tenants(
          tenant_id, ledger_currency, amount_tolerance_minor, date_window_days,
          max_candidates, max_auto_amount_minor, max_auto_bill_count, rules_version,
          use_ranker_threshold
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        tenants,
    )

    vendors = [
        ("v_acme", "t1", "Acme Corp", json.dumps(["ACME BANK", "Acme National", "ACME CORP PAYMENTS"])),
        ("v_beta", "t1", "Beta LLC", json.dumps(["BETA SAVINGS", "BETA LLC"])),
        ("v_delta", "t1", "Delta Industries", json.dumps(["DELTA SAVINGS", "DELTA CORP", "DELTA INDUSTRIES"])),
        ("v_gamma", "t2", "Gamma Inc", json.dumps(["GAMMA BANK", "GAMMA INC"])),
        ("v_omega", "t2", "Omega Partners", json.dumps(["OMEGA FINANCIAL"])),
        ("v3_acme", "t3", "Acme Corp", json.dumps(["ACME BANK", "Acme National", "ACME CORP PAYMENTS"])),
        ("v3_beta", "t3", "Beta LLC", json.dumps(["BETA SAVINGS", "BETA LLC"])),
        ("v3_delta", "t3", "Delta Industries", json.dumps(["DELTA SAVINGS", "DELTA CORP"])),
    ]
    cur.executemany(
        """
        INSERT INTO vendors(vendor_id, tenant_id, display_name, bank_aliases_json)
        VALUES (?,?,?,?)
        """,
        vendors,
    )

    base = date(2025, 1, 15)
    pay_day = base + timedelta(days=5)
    old = base - timedelta(days=200)
    future = base + timedelta(days=200)

    bills: list[tuple] = []

    def bill(
        bid: str,
        tenant: str,
        vendor: str,
        usd: float | None = None,
        *,
        eur: float | None = None,
        gbp: float | None = None,
        open_d: date | None = None,
        matched: str | None = None,
    ) -> None:
        if eur is not None:
            amt, ccy = _eur_minor(eur), "EUR"
        elif gbp is not None:
            amt, ccy = _gbp_minor(gbp), "GBP"
        else:
            amt, ccy = _usd_minor(usd or 0), "USD"
        bills.append(
            (
                bid,
                tenant,
                vendor,
                amt,
                ccy,
                (open_d or base).isoformat(),
                None,
                matched,
            )
        )

    # --- Acme open bills (core amount ladder) --------------------------------
    bill("b1", "t1", "v_acme", 100)
    bill("b2", "t1", "v_acme", 250)
    bill("b3", "t1", "v_acme", 150)
    bill("b4", "t1", "v_acme", 200)
    bill("b5", "t1", "v_acme", 300)
    bill("b6", "t1", "v_acme", 400)
    # Ambiguous pair for $750: (b1,b2,b6) and (b3,b4,b6)
    bill("b7", "t1", "v_acme", 50)
    bill("b8", "t1", "v_acme", 75)

    # Outside date window (retrieval drop)
    bill("b_old", "t1", "v_acme", 500, open_d=old)
    bill("b_future", "t1", "v_acme", 500, open_d=future)

    # Beta noise / wrong-vendor trap
    bill("b_beta", "t1", "v_beta", 999)

    # FX: EUR with rate; EUR without rate; GBP without rate
    bill("b_eur", "t1", "v_acme", eur=90)
    bill("b_eur_nofx", "t1", "v_acme", eur=200)
    bill("b_gbp_nofx", "t1", "v_acme", gbp=80)

    # Delta vendor only (clean single-vendor match)
    bill("d1", "t1", "v_delta", 120)
    bill("d2", "t1", "v_delta", 180)
    bill("d3", "t1", "v_delta", 300)

    # Pre-allocated to other payments (double-spend / remaining-amount tests)
    bill("b_lock_a", "t1", "v_acme", 200, matched="pay_prealloc_500")
    bill("b_lock_b", "t1", "v_acme", 300, matched="pay_prealloc_500")
    bill("b_rem_250", "t1", "v_acme", 250)  # open; for pay_seek_250 after $500 locked

    # Penny ladder: pay_exact_10 uses b_dime_*; pay_all_matched pre-allocates b_cent_*
    for i in range(1, 11):
        bill(f"b_cent_{i}", "t1", "v_acme", 1.00)
    for i in range(1, 11):
        bill(f"b_dime_{i}", "t1", "v_acme", 1.00)

    # Many similar amounts to stress candidate cap (max_candidates=28 on t1)
    for i in range(1, 16):
        bill(f"b_cap_{i}", "t1", "v_acme", 10.00 + i)  # 11.00 .. 25.00

    # --- Tenant t2 (gamma week vs omega week, 7-day retrieval window) ---------------
    t2_gamma_day = base + timedelta(days=1)
    t2_omega_day = base + timedelta(days=30)
    bill("t2_b1", "t2", "v_gamma", 60, open_d=t2_gamma_day)  # 60+75≠125 → unique t2_b2+t2_b4
    bill("t2_b2", "t2", "v_gamma", 75, open_d=t2_gamma_day)
    bill("t2_b3", "t2", "v_gamma", 25, open_d=t2_gamma_day)
    bill("t2_b4", "t2", "v_gamma", 50, open_d=t2_gamma_day)
    bill("o1", "t2", "v_omega", 40, open_d=t2_omega_day)
    bill("o2", "t2", "v_omega", 60, open_d=t2_omega_day)

    # --- Tenant t3: isolated scenarios (one open-date cluster per payment) ----------
    iso_base = date(2025, 6, 1)

    def iso_week(week: int) -> date:
        return iso_base + timedelta(days=7 * week)

    w0, w1, w2, w3, w4, w5, w6, w7, w8 = (iso_week(i) for i in range(9))

    bill("iso_b6", "t3", "v3_acme", 400, open_d=w0)
    bill("iso_x", "t3", "v3_acme", 200, open_d=w1)
    bill("iso_y", "t3", "v3_acme", 300, open_d=w1)
    bill("iso_z", "t3", "v3_acme", 100, open_d=w1)
    bill("iso_w", "t3", "v3_acme", 400, open_d=w1)
    bill("iso_eur", "t3", "v3_acme", eur=90, open_d=w2)
    bill("iso_t1", "t3", "v3_acme", 100, open_d=w3)
    bill("iso_t2", "t3", "v3_acme", 250, open_d=w3)
    bill("iso_t3", "t3", "v3_acme", 150, open_d=w3)
    bill("iso_lock_a", "t3", "v3_acme", 200, open_d=w5, matched="pay_iso_prealloc")
    bill("iso_lock_b", "t3", "v3_acme", 300, open_d=w5, matched="pay_iso_prealloc")
    bill("iso_rem_250", "t3", "v3_acme", 250, open_d=w5)
    for i in range(1, 11):
        bill(f"iso_dime_{i}", "t3", "v3_acme", 1.00, open_d=w6)
    bill("iso_beta_trap", "t3", "v3_beta", 999, open_d=w7)
    bill("iso_gbp", "t3", "v3_acme", gbp=80, open_d=w8)
    bill("iso_d3", "t3", "v3_delta", 300, open_d=w4)

    cur.executemany(
        """
        INSERT INTO bills(
          bill_id, tenant_id, vendor_id, open_amount_minor, currency,
          open_date, due_date, matched_payment_id
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        bills,
    )

    def pay(
        pid: str,
        tenant: str,
        amount: int,
        pday: date,
        bank: str,
        desc: str,
        *,
        fx: float = 1.0,
        src_ccy: str | None = None,
    ) -> tuple:
        return (pid, tenant, amount, pday.isoformat(), bank, desc, fx, src_ccy)

    payments = [
        # --- Classic POC scenarios (t1) -----------------------------------------
        pay("pay_bulk_1", "t1", _usd_minor(500), pay_day, "ACME BANK", "ACH batch — ambiguous subsets"),
        pay("pay_no_match", "t1", _usd_minor(777), pay_day, "UNKNOWN BANK", "no amount closure"),
        pay("pay_fx_eur", "t1", _usd_minor(99), pay_day, "Acme National", "EUR €90 @ 1.10"),
        pay("pay_single_400", "t1", _usd_minor(400), pay_day, "ACME BANK", "one bill b6"),
        pay("pay_ambig_750", "t1", _usd_minor(750), pay_day, "ACME BANK", "two valid bulk subsets"),
        pay("pay_tolerance", "t1", _usd_minor(500.01), pay_day, "ACME BANK", "1¢ residual"),
        pay("pay_fuzzy_acme", "t1", _usd_minor(350), pay_day, "ACME CORP PAYMENTS", "b3+b4"),
        pay("pay_delta_300", "t1", _usd_minor(300), pay_day, "DELTA SAVINGS", "d3 only"),
        pay("pay_wrong_vendor", "t1", _usd_minor(999), pay_day, "ACME BANK", "only b_beta fits amount"),
        pay("pay_gbp_gap", "t1", _usd_minor(88), pay_day, "ACME BANK", "b_gbp_nofx skipped"),
        pay(
            "pay_prealloc_500",
            "t1",
            _usd_minor(500),
            pay_day - timedelta(days=1),
            "ACME BANK",
            "already matched bills",
        ),
        pay("pay_seek_250", "t1", _usd_minor(250), pay_day, "ACME BANK", "open b_rem_250 only"),
        pay("pay_exact_10", "t1", _usd_minor(10), pay_day, "ACME BANK", "b_dime_1..10"),
        pay("pay_all_matched", "t1", _usd_minor(100), pay_day, "ACME BANK", "b_cent_* pre-matched"),
        pay("pay_cap_stress", "t1", _usd_minor(36), pay_day, "ACME BANK", "b_cap_1+b_cap_2+b_cap_3"),
        # --- Tenant t2 ------------------------------------------------------------
        pay("pay_t2_ok", "t2", _usd_minor(125), t2_gamma_day, "GAMMA BANK", "unique: t2_b2+t2_b4"),
        pay(
            "pay_t2_ambig",
            "t2",
            _usd_minor(135),
            t2_gamma_day,
            "GAMMA BANK",
            "ambiguous: t2_b1+t2_b2 vs t2_b1+t2_b3+t2_b4",
        ),
        pay("pay_t2_no_match", "t2", _usd_minor(333), t2_omega_day, "OMEGA FINANCIAL", "no closure"),
        pay("pay_t2_omega", "t2", _usd_minor(100), t2_omega_day, "OMEGA FINANCIAL", "o1+o2=100"),
        # --- Tenant t3 (isolated regression) --------------------------------------
        pay("pay_iso_single", "t3", _usd_minor(400), w0, "ACME BANK", "single bill"),
        pay("pay_iso_ambig", "t3", _usd_minor(500), w1, "ACME BANK", "iso_x+iso_y vs iso_z+iso_w"),
        pay("pay_iso_fx", "t3", _usd_minor(99), w2, "Acme National", "EUR bill"),
        pay("pay_iso_tolerance", "t3", _usd_minor(500.01), w3, "ACME BANK", "1¢ residual"),
        pay("pay_iso_nomatch", "t3", 9_999_999, w4, "UNKNOWN BANK", "no closure"),
        pay("pay_iso_prealloc", "t3", _usd_minor(500), w5, "ACME BANK", "fully allocated"),
        pay("pay_iso_seek", "t3", _usd_minor(250), w5, "ACME BANK", "iso_rem_250"),
        pay("pay_iso_exact10", "t3", _usd_minor(10), w6, "ACME BANK", "10×$1"),
        pay("pay_iso_wrong_vendor", "t3", _usd_minor(999), w7, "ZULU EXPORT", "weak vendor link gates out"),
        pay("pay_iso_gbp", "t3", _usd_minor(88), w8, "ACME BANK", "GBP no FX"),
        pay("pay_iso_delta", "t3", _usd_minor(300), w4, "DELTA SAVINGS", "iso_d3"),
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

    # Pre-match all b_cent_* to pay_all_matched for "no open bills" scenario
    for i in range(1, 11):
        cur.execute(
            "UPDATE bills SET matched_payment_id = ? WHERE bill_id = ?",
            ("pay_all_matched", f"b_cent_{i}"),
        )

    conn.commit()

    upsert_fx_rate(conn, tenant_id="t1", from_currency="EUR", to_currency="USD", rate=1.10, effective_date=base)
    upsert_fx_rate(conn, tenant_id="t3", from_currency="EUR", to_currency="USD", rate=1.10, effective_date=iso_base)
    # GBP deliberately has no rate on t1/t3

    return {
        "tenants": [t[0] for t in tenants],
        "payments": [p[0] for p in payments],
        "bills": [b[0] for b in bills],
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
        use_ranker_threshold=bool(row["use_ranker_threshold"]),
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
