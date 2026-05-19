from bulk_payments.allocation import get_payment_allocation, unmatch_payment
from bulk_payments.batch import run_batch_auto_match
from bulk_payments.db import connect, init_db, insert_match_event
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
        "UPDATE bills SET matched_payment_id = 'pay_t2_ok' WHERE bill_id IN ('t2_b2', 't2_b4')"
    )
    conn.commit()
    alloc = get_payment_allocation(conn, "t2", "pay_t2_ok")
    assert len(alloc.matched_bill_ids) == 2

    freed = unmatch_payment(conn, tenant_id="t2", payment_id="pay_t2_ok")
    assert set(freed) == {"t2_b2", "t2_b4"}
    alloc_after = get_payment_allocation(conn, "t2", "pay_t2_ok")
    assert alloc_after.matched_bill_ids == ()
    conn.close()


def test_batch_auto_match_allocates_confident_matches(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    result = run_batch_auto_match(conn, tenant_id="t3")
    assert result["summary"]["auto_matched"] >= 5
    assert result["summary"]["needs_review"] >= 1
    alloc = get_payment_allocation(conn, "t3", "pay_iso_single")
    assert alloc.is_fully_allocated
    assert "iso_b6" in alloc.matched_bill_ids
    conn.close()


def test_accountant_reasoning_stored_on_outcome(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    init_db(conn)
    seed_demo_dataset(conn)
    event_id = insert_match_event(
        conn,
        tenant_id="t3",
        payment_id="pay_iso_ambig",
        bill_ids=["iso_x", "iso_y"],
        rules_version="v1",
        features={},
        decision="suggested",
        outcome="pending",
    )
    from bulk_payments.allocation import apply_accepted_bills

    apply_accepted_bills(
        conn,
        tenant_id="t3",
        payment_id="pay_iso_ambig",
        bill_ids=["iso_x", "iso_y"],
    )
    new_id = insert_match_event(
        conn,
        tenant_id="t3",
        payment_id="pay_iso_ambig",
        bill_ids=["iso_x", "iso_y"],
        rules_version="v1",
        features={"correction_of": event_id},
        decision="suggested",
        outcome="accepted_as_is",
        accountant_reasoning="Vendor memo matches PO 445",
    )
    row = conn.execute(
        "SELECT accountant_reasoning FROM match_events WHERE event_id = ?",
        (new_id,),
    ).fetchone()
    assert row["accountant_reasoning"] == "Vendor memo matches PO 445"
    conn.close()
