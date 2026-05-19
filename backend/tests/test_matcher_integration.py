from bulk_payments.allocation import get_payment_allocation, unmatch_payment
from bulk_payments.db import connect, init_db
from bulk_payments.evaluation import run_eval
from bulk_payments.matcher import match_payment
from bulk_payments.synthetic import seed_demo_dataset


def test_demo_oracle_passes(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    conn.close()
    conn = connect(db)
    summary = run_eval(conn)
    conn.close()
    assert summary["oracle_all_pass"] is True
    assert summary["counts"]["auto_tp"] >= 1


def test_unmatch_clears_bill_allocations(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    match_payment(conn, tenant_id="t2", payment_id="pay_t2_ok", log_event=False)
    conn.execute(
        "UPDATE bills SET matched_payment_id = 'pay_t2_ok' WHERE bill_id IN ('t2_b1', 't2_b2')"
    )
    conn.commit()
    alloc = get_payment_allocation(conn, "t2", "pay_t2_ok")
    assert len(alloc.matched_bill_ids) == 2

    freed = unmatch_payment(conn, tenant_id="t2", payment_id="pay_t2_ok")
    assert set(freed) == {"t2_b1", "t2_b2"}
    alloc_after = get_payment_allocation(conn, "t2", "pay_t2_ok")
    assert alloc_after.matched_bill_ids == ()
    conn.close()
