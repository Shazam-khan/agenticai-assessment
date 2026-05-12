"""SQLite store for traces, evals, memory, HITL queue, circuit-breaker state.

One file per environment. Tasks each contribute their own tables but they all live here.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "./data/agentic.db")

# Task 1 + Task 4C own the trace + eval_runs tables.
# Task 2 will append memory tables; Task 3 will append HITL + circuit-breaker tables.
SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  span_id          TEXT PRIMARY KEY,
  trace_id         TEXT NOT NULL,
  parent_span_id   TEXT,
  ts               TEXT NOT NULL,
  actor            TEXT NOT NULL,
  actor_kind       TEXT NOT NULL,          -- 'agent' | 'tool' | 'llm' | 'eval'
  model            TEXT,
  prompt_version   TEXT,
  input_json       TEXT,
  output_json      TEXT,
  status           TEXT NOT NULL,
  error_class      TEXT,
  input_tokens     INTEGER DEFAULT 0,
  output_tokens    INTEGER DEFAULT 0,
  latency_ms       INTEGER DEFAULT 0,
  cost_usd         REAL    DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_traces_trace ON traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_actor_ts ON traces(actor, ts);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id           TEXT NOT NULL,
  scenario_id      TEXT NOT NULL,
  suite            TEXT NOT NULL,
  ts               TEXT NOT NULL,
  passed           INTEGER NOT NULL,
  score            REAL,
  details_json     TEXT,
  prompt_version   TEXT,
  PRIMARY KEY (run_id, scenario_id)
);

-- Task 2 memory tables --

CREATE TABLE IF NOT EXISTS episodic_turns (
  turn_id      TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  customer_id  TEXT NOT NULL,
  channel      TEXT,
  role         TEXT NOT NULL,                 -- 'user' | 'agent'
  content      TEXT NOT NULL,
  ts           TEXT NOT NULL,
  summarised   INTEGER DEFAULT 0              -- 1 once folded into working_memory
);
CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_customer ON episodic_turns(customer_id);

CREATE TABLE IF NOT EXISTS semantic_facts (
  fact_id           TEXT PRIMARY KEY,
  customer_id       TEXT NOT NULL,
  category          TEXT NOT NULL,
  entity            TEXT,
  value             TEXT NOT NULL,
  confidence        REAL DEFAULT 0.8,
  source_turn_id    TEXT,
  created_at        TEXT NOT NULL,
  last_accessed_at  TEXT NOT NULL,
  access_count      INTEGER DEFAULT 0,
  dedupe_key        TEXT NOT NULL UNIQUE      -- sha256(customer + category + entity + value_norm)
);
CREATE INDEX IF NOT EXISTS idx_facts_customer ON semantic_facts(customer_id);

CREATE TABLE IF NOT EXISTS working_memory (
  session_id          TEXT PRIMARY KEY,
  customer_id         TEXT,
  summary             TEXT,
  summary_token_count INTEGER DEFAULT 0,
  last_updated_at     TEXT
);

-- Task 3 tables --

CREATE TABLE IF NOT EXISTS pending_confirmations (
  confirmation_id   TEXT PRIMARY KEY,
  tool_name         TEXT NOT NULL,
  inputs_json       TEXT NOT NULL,
  reason            TEXT,
  trace_id          TEXT,
  created_at        TEXT NOT NULL,
  expires_at        TEXT NOT NULL,
  approved_at       TEXT,
  approved_by       TEXT,
  status            TEXT NOT NULL          -- pending | approved | rejected | expired | executed
);
CREATE INDEX IF NOT EXISTS idx_confirm_status ON pending_confirmations(status);

CREATE TABLE IF NOT EXISTS purchase_orders (
  po_id            TEXT PRIMARY KEY,
  material         TEXT NOT NULL,
  quantity         REAL NOT NULL,
  supplier_id      TEXT NOT NULL,
  urgency          TEXT NOT NULL,
  status           TEXT NOT NULL,           -- 'created' | 'cancelled'
  created_at       TEXT NOT NULL,
  trace_id         TEXT,
  confirmation_id  TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id    TEXT PRIMARY KEY,
  order_id    TEXT NOT NULL,
  reason      TEXT NOT NULL,
  severity    TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  trace_id    TEXT
);

CREATE TABLE IF NOT EXISTS circuit_state (
  tool_name             TEXT PRIMARY KEY,
  state                 TEXT NOT NULL,       -- 'closed' | 'open' | 'half_open'
  consecutive_failures  INTEGER DEFAULT 0,
  opened_at             TEXT,
  cooldown_seconds      INTEGER DEFAULT 60
);

CREATE TABLE IF NOT EXISTS paused_runs (
  trace_id     TEXT PRIMARY KEY,
  objective    TEXT NOT NULL,
  paused_at    TEXT NOT NULL,
  resumed_at   TEXT,
  status       TEXT NOT NULL                 -- 'paused' | 'resumed' | 'cancelled'
);

-- Task 4 tables --

CREATE TABLE IF NOT EXISTS leads (
  customer_id     TEXT NOT NULL,
  field_name      TEXT NOT NULL,
  value           TEXT NOT NULL,
  language        TEXT,
  source_turn_id  TEXT,
  trace_id        TEXT,
  updated_at      TEXT NOT NULL,
  PRIMARY KEY (customer_id, field_name)
);
CREATE INDEX IF NOT EXISTS idx_leads_customer ON leads(customer_id);

CREATE TABLE IF NOT EXISTS session_language (
  session_id  TEXT PRIMARY KEY,
  language    TEXT NOT NULL,                 -- locked after first detection
  detected_at TEXT NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def reset_db() -> None:
    """Wipe the file. Tests only."""
    Path(DB_PATH).unlink(missing_ok=True)
