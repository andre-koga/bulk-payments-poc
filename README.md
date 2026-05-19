# Bulk payment matching (POC)

Precision-first matching of one bank payment to multiple open bills: retrieval, hard gates, subset-sum search, auto vs suggest policy, audit logging, optional calibrated ranker training, and an AI agent layer for ambiguous cases.

## Setup

```bash
cd /path/to/bulk-payments
python -m venv .venv && source .venv/bin/activate

# Core only (rules engine + ranker)
pip install -e ".[dev]"

# Core + AI agent (LangChain / LangSmith)
pip install -e ".[dev,agent]"
```

## CLI — rules engine

```bash
bulk-match init-db --db /tmp/bulk.db
bulk-match seed --db /tmp/bulk.db
bulk-match match --db /tmp/bulk.db --tenant t1 --payment pay_bulk_1
bulk-match eval --db /tmp/bulk.db
bulk-match train-ranker --db /tmp/bulk.db --out /tmp/ranker.joblib
bulk-match train-ranker-tenants --db /tmp/bulk.db --out-dir /tmp/rankers/
bulk-match label-demo --db /tmp/bulk.db
```

## CLI — AI agent

```bash
# Single payment — runs rules engine + AI resolver, prints JSON
bulk-match agent-resolve --db /tmp/bulk.db --tenant t1 --payment pay_bulk_1 --model gpt-4o-mini

# Evaluate agent against oracle across multiple models
bulk-match agent-eval --db /tmp/bulk.db --models gpt-4o-mini,claude-3-5-haiku-20241022

# Batch-resolve all pending SUGGESTED / NO_CANDIDATES for a tenant
bulk-match agent-batch --db /tmp/bulk.db --tenant t1 --model gpt-4o-mini
```

## Environment variables

| Variable | Required for | Purpose |
|----------|-------------|---------|
| `OPENAI_API_KEY` | gpt-* models | OpenAI provider auth |
| `ANTHROPIC_API_KEY` | claude-* models | Anthropic provider auth |
| `LANGCHAIN_TRACING_V2=true` | LangSmith | Enable run tracing |
| `LANGSMITH_API_KEY` | LangSmith | LangSmith auth |
| `LANGSMITH_PROJECT` | LangSmith | Project name (default: `bulk-payments-poc`) |
| `BULK_AGENT_ENABLED=false` | agent commands | Disable AI agent (rules-only fallback) |

## Library usage — rules engine

```python
from bulk_payments.matcher import match_payment
from bulk_payments.db import connect

conn = connect("/tmp/bulk.db")
result = match_payment(conn, tenant_id="t1", payment_id="pay_bulk_1")
print(result.decision, result.reason_codes)
```

## Library usage — AI agent

```python
from bulk_payments.match_with_agent import match_payment_with_agent
from bulk_payments.db import connect

conn = connect("/tmp/bulk.db")
combined = match_payment_with_agent(
    conn,
    tenant_id="t1",
    payment_id="pay_bulk_1",
    model="gpt-4o-mini",
)

print(combined.rules.decision)        # rules engine decision
if combined.agent:
    print(combined.agent.action)      # propose_match / no_match / need_more_info
    print(combined.agent.bill_ids)    # proposed allocation
    print(combined.agent.reasoning)   # accountant-style explanation
    print(combined.agent.langsmith_run_id)  # link to LangSmith trace
```

## LangSmith datasets and evaluation

```bash
# Export oracle cases and labeled events to a LangSmith dataset
python scripts/export_langsmith_dataset.py --db /tmp/bulk.db --dataset-name bulk-payments-poc

# Run LangSmith evaluate() across multiple models and compare results in the UI
python scripts/run_langsmith_eval.py \
    --db /tmp/bulk.db \
    --models gpt-4o-mini,claude-3-5-haiku-20241022 \
    --dataset-name bulk-payments-poc
```

## Architecture overview

```
Payment
  └─► rules engine (retrieval → gates → subset-sum → policy)
        ├─ AUTO_APPLIED  → log match_event, done
        ├─ SUGGESTED     → AI resolver agent (LangChain ReAct)
        └─ NO_CANDIDATES → AI resolver agent (LangChain ReAct)
                                └─ LangChain tools (get_match_context, list_feasible_subsets,
                                                    get_bill_details, get_vendor_aliases,
                                                    submit_resolution)
                                └─ AgentResolution (action, bill_ids, confidence, reasoning)
                                └─ SQLite agent_resolutions table + LangSmith trace
```

The AI agent is tool-grounded: it can only propose bill IDs that exist in the DB,
and `submit_resolution` validates the sum is within the tenant's tolerance before
accepting the answer. Full LLM message traces are stored in LangSmith; the
`reasoning` field (accountant-style narrative) is also persisted in SQLite for
offline evaluation and future UI display.
