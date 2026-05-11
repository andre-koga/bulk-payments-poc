from bulk_payments.db import connect, init_db
from bulk_payments.evaluation import run_eval
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
