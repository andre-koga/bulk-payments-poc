from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tenants (
  tenant_id TEXT PRIMARY KEY,
  ledger_currency TEXT NOT NULL DEFAULT 'USD',
  amount_tolerance_minor INTEGER NOT NULL DEFAULT 2,
  date_window_days INTEGER NOT NULL DEFAULT 90,
  max_candidates INTEGER NOT NULL DEFAULT 28,
  max_auto_amount_minor INTEGER,
  max_auto_bill_count INTEGER NOT NULL DEFAULT 50,
  rules_version TEXT NOT NULL DEFAULT 'v1'
);

CREATE TABLE IF NOT EXISTS vendors (
  vendor_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
  display_name TEXT NOT NULL,
  bank_aliases_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS bills (
  bill_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
  vendor_id TEXT NOT NULL REFERENCES vendors(vendor_id),
  open_amount_minor INTEGER NOT NULL,
  currency TEXT NOT NULL,
  open_date TEXT NOT NULL,
  due_date TEXT,
  matched_payment_id TEXT
);

CREATE TABLE IF NOT EXISTS payments (
  payment_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
  amount_minor INTEGER NOT NULL,
  payment_date TEXT NOT NULL,
  counterparty_bank_name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  fx_rate_used REAL NOT NULL DEFAULT 1.0,
  source_currency TEXT
);

-- Append-only audit log for learning / evaluation
CREATE TABLE IF NOT EXISTS match_events (
  event_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  payment_id TEXT NOT NULL,
  bill_ids_json TEXT NOT NULL,
  rules_version TEXT NOT NULL,
  features_json TEXT NOT NULL,
  decision TEXT NOT NULL,
  outcome TEXT NOT NULL DEFAULT 'pending',
  ranker_score REAL,
  calibrated_prob REAL,
  reason_codes_json TEXT NOT NULL,
  user_id TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_match_events_tenant_time
  ON match_events(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_match_events_payment
  ON match_events(payment_id);

-- AI agent resolution log (schema v2)
-- One row per agent invocation; linked to match_events via event_id FK.
CREATE TABLE IF NOT EXISTS agent_resolutions (
  resolution_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES match_events(event_id),
  model_id TEXT NOT NULL,
  action TEXT NOT NULL,
  bill_ids_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  reasoning TEXT NOT NULL,
  langsmith_run_id TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_resolutions_event
  ON agent_resolutions(event_id);
CREATE INDEX IF NOT EXISTS idx_agent_resolutions_model
  ON agent_resolutions(model_id);
"""

_MIGRATION_V2 = """
CREATE TABLE IF NOT EXISTS agent_resolutions (
  resolution_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES match_events(event_id),
  model_id TEXT NOT NULL,
  action TEXT NOT NULL,
  bill_ids_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  reasoning TEXT NOT NULL,
  langsmith_run_id TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_resolutions_event
  ON agent_resolutions(event_id);
CREATE INDEX IF NOT EXISTS idx_agent_resolutions_model
  ON agent_resolutions(model_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def migrate_db(conn: sqlite3.Connection) -> None:
    """Apply incremental migrations to an existing database.

    Safe to call on a fresh DB (all migrations are idempotent via IF NOT EXISTS).
    Checks the stored schema_version and applies only the needed migrations.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row["value"]) if row else 0

    if current < 2:
        conn.executescript(_MIGRATION_V2)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '2')"
        )
        conn.commit()


def insert_match_event(
    conn: sqlite3.Connection,
    *,
    tenant_id: str,
    payment_id: str,
    bill_ids: list[str],
    rules_version: str,
    features: dict[str, Any],
    decision: str,
    outcome: str = "pending",
    ranker_score: float | None = None,
    calibrated_prob: float | None = None,
    reason_codes: list[str] | None = None,
    user_id: str | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO match_events(
          event_id, tenant_id, payment_id, bill_ids_json, rules_version,
          features_json, decision, outcome, ranker_score, calibrated_prob,
          reason_codes_json, user_id, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            event_id,
            tenant_id,
            payment_id,
            json.dumps(bill_ids),
            rules_version,
            json.dumps(features),
            decision,
            outcome,
            ranker_score,
            calibrated_prob,
            json.dumps(reason_codes or []),
            user_id,
            now,
        ),
    )
    conn.commit()
    return event_id


def update_match_event_outcome(
    conn: sqlite3.Connection,
    event_id: str,
    outcome: str,
    user_id: str | None = None,
) -> None:
    # Append-only policy: we do not mutate rows; insert correction event in production.
    # POC convenience: allow updating outcome for evaluation labelling.
    conn.execute(
        """
        UPDATE match_events SET outcome = ?, user_id = COALESCE(?, user_id)
        WHERE event_id = ?
        """,
        (outcome, user_id, event_id),
    )
    conn.commit()


def insert_agent_resolution(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    model_id: str,
    action: str,
    bill_ids: list[str],
    confidence: float,
    reasoning: str,
    langsmith_run_id: str | None = None,
) -> str:
    """Persist an AI agent resolution linked to a match_event row."""
    resolution_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO agent_resolutions(
          resolution_id, event_id, model_id, action,
          bill_ids_json, confidence, reasoning, langsmith_run_id, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            resolution_id,
            event_id,
            model_id,
            action,
            json.dumps(bill_ids),
            confidence,
            reasoning,
            langsmith_run_id,
            now,
        ),
    )
    conn.commit()
    return resolution_id
