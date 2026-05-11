from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bulk_payments.db import connect, init_db
from bulk_payments.evaluation import label_events_for_training, run_eval
from bulk_payments.matcher import match_payment
from bulk_payments.synthetic import create_db_with_seed
from bulk_payments.train import train_and_save, train_per_tenant_models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bulk-match")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-db", help="Create SQLite schema")
    p_init.add_argument("--db", required=True)

    p_seed = sub.add_parser("seed", help="Reset and load demo data")
    p_seed.add_argument("--db", required=True)

    p_match = sub.add_parser("match", help="Run matcher for one payment")
    p_match.add_argument("--db", required=True)
    p_match.add_argument("--tenant", required=True)
    p_match.add_argument("--payment", required=True)
    p_match.add_argument("--ranker", default=None)

    p_eval = sub.add_parser("eval", help="Run evaluation harness on demo oracle")
    p_eval.add_argument("--db", required=True)
    p_eval.add_argument("--ranker", default=None)

    p_train = sub.add_parser("train-ranker", help="Train calibrated ranker from labeled match_events")
    p_train.add_argument("--db", required=True)
    p_train.add_argument("--out", required=True)

    p_train_t = sub.add_parser("train-ranker-tenants", help="Train one ranker per tenant into a directory")
    p_train_t.add_argument("--db", required=True)
    p_train_t.add_argument("--out-dir", required=True)

    p_label = sub.add_parser("label-demo", help="Apply synthetic labels to latest events for training")
    p_label.add_argument("--db", required=True)

    args = p.parse_args(argv)

    if args.cmd == "init-db":
        conn = connect(args.db)
        init_db(conn)
        conn.close()
        return 0

    if args.cmd == "seed":
        create_db_with_seed(args.db)
        return 0

    if args.cmd == "match":
        conn = connect(args.db)
        r = match_payment(conn, tenant_id=args.tenant, payment_id=args.payment, ranker_path=args.ranker)
        conn.close()
        print(json.dumps(_result_to_json(r), indent=2))
        return 0

    if args.cmd == "eval":
        conn = connect(args.db)
        summary = run_eval(conn, ranker_path=args.ranker)
        conn.close()
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("oracle_all_pass") else 1

    if args.cmd == "label-demo":
        conn = connect(args.db)
        n = label_events_for_training(conn)
        conn.close()
        print(f"labeled_events={n}")
        return 0

    if args.cmd == "train-ranker":
        conn = connect(args.db)
        art = train_and_save(conn, args.out)
        conn.close()
        if art is None:
            print("not_enough_labeled_rows", file=sys.stderr)
            return 1
        print(f"wrote_model={args.out}")
        return 0

    if args.cmd == "train-ranker-tenants":
        conn = connect(args.db)
        paths = train_per_tenant_models(conn, args.out_dir)
        conn.close()
        print(json.dumps(paths, indent=2))
        return 0

    return 1


def _result_to_json(r):
    return {
        "payment_id": r.payment_id,
        "tenant_id": r.tenant_id,
        "decision": r.decision.value,
        "unique_best": r.unique_best,
        "competing_subset_count": r.competing_subset_count,
        "reason_codes": list(r.reason_codes),
        "ranker_score": r.ranker_score,
        "calibrated_accept_prob": r.calibrated_accept_prob,
        "subsets": [
            {"bill_ids": list(s.bill_ids), "sum_minor": s.sum_minor, "abs_residual_minor": s.abs_residual_minor}
            for s in r.subsets
        ],
        "features": r.features,
    }


if __name__ == "__main__":
    raise SystemExit(main())
