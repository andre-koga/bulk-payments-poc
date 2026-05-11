# Bulk payment matching (POC)

Precision-first matching of one bank payment to multiple open bills: retrieval, hard gates, subset-sum search, auto vs suggest policy, audit logging, and optional calibrated ranker training.

## Setup

```bash
cd /path/to/bulk-payments
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

```bash
bulk-match init-db --db /tmp/bulk.db
bulk-match seed --db /tmp/bulk.db
bulk-match match --db /tmp/bulk.db --tenant t1 --payment pay_bulk_1
bulk-match eval --db /tmp/bulk.db
bulk-match train-ranker --db /tmp/bulk.db --out /tmp/ranker.joblib
bulk-match train-ranker-tenants --db /tmp/bulk.db --out-dir /tmp/rankers/
```

## Library usage

```python
from bulk_payments.matcher import match_payment
from bulk_payments.db import connect

conn = connect("/tmp/bulk.db")
result = match_payment(conn, tenant_id="t1", payment_id="pay_bulk_1")
```
