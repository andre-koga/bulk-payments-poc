from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bulk_payments.db import connect, init_db
from bulk_payments.evaluation import (
    extended_eval_report,
    label_events_for_training,
    precision_stop_loss,
    run_eval,
)
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

    p_eval_ext = sub.add_parser("eval-extended", help="Suggestion acceptance, calibration, slices, stop-loss")
    p_eval_ext.add_argument("--db", required=True)
    p_eval_ext.add_argument("--tenant", default=None)
    p_eval_ext.add_argument("--ranker", default=None)

    p_train = sub.add_parser("train-ranker", help="Train calibrated ranker from labeled match_events")
    p_train.add_argument("--db", required=True)
    p_train.add_argument("--out", required=True)

    p_train_t = sub.add_parser("train-ranker-tenants", help="Train one ranker per tenant into a directory")
    p_train_t.add_argument("--db", required=True)
    p_train_t.add_argument("--out-dir", required=True)

    p_serve = sub.add_parser("serve", help="Start the FastAPI HTTP server")
    p_serve.add_argument("--db", required=True)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--ranker", default=None)

    p_set_ranker = sub.add_parser("set-ranker-gate", help="Enable/disable ranker threshold for auto-apply on a tenant")
    p_set_ranker.add_argument("--db", required=True)
    p_set_ranker.add_argument("--tenant", required=True)
    p_set_ranker.add_argument("--enable", action="store_true", default=False)
    p_set_ranker.add_argument("--disable", dest="enable", action="store_false")

    p_validate = sub.add_parser(
        "validate-ranker-gate",
        help="Check rolling precision; print whether ranker gating is safe to enable",
    )
    p_validate.add_argument("--db", required=True)
    p_validate.add_argument("--tenant", required=True)
    p_validate.add_argument("--window", type=int, default=50)
    p_validate.add_argument("--min-precision", type=float, default=0.95)

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

    if args.cmd == "set-ranker-gate":
        conn = connect(args.db)
        conn.execute(
            "UPDATE tenants SET use_ranker_threshold = ? WHERE tenant_id = ?",
            (1 if args.enable else 0, args.tenant),
        )
        conn.commit()
        state = "enabled" if args.enable else "disabled"
        print(f"ranker_gate={state} for tenant={args.tenant}")
        conn.close()
        return 0

    if args.cmd == "validate-ranker-gate":
        conn = connect(args.db)
        result = precision_stop_loss(
            conn,
            tenant_id=args.tenant,
            window=args.window,
            min_precision=args.min_precision,
        )
        conn.close()
        print(json.dumps(result, indent=2))
        safe = result.get("above_threshold")
        if safe is True:
            print(f"\nSafe to enable: bulk-match set-ranker-gate --db {args.db} --tenant {args.tenant} --enable")
            return 0
        if safe is False:
            print("\nNot safe: precision below threshold. Accumulate more corrections first.")
            return 1
        print("\nNot enough labeled auto-apply events yet.")
        return 1

    if args.cmd == "eval-extended":
        conn = connect(args.db)
        report = extended_eval_report(conn, tenant_id=args.tenant, ranker_path=args.ranker)
        conn.close()
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "serve":
        import os
        import uvicorn

        os.environ["BULK_DB"] = args.db
        if args.ranker:
            os.environ["BULK_RANKER"] = args.ranker
        uvicorn.run(
            "bulk_payments.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

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
