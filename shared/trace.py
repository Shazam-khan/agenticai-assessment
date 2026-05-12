"""Trace context manager. Every agent / tool / LLM call wraps its work in one of these.

The resulting rows in `traces` are the single source of truth for the Task 4C dashboard.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from .db import get_conn
from .ids import new_id


@dataclass
class TraceRecord:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    actor: str
    actor_kind: str
    model: str | None = None
    prompt_version: str | None = None
    input_payload: Any = None
    output_payload: Any = None
    status: str = "success"
    error_class: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


@contextmanager
def trace(
    actor: str,
    actor_kind: str,
    *,
    trace_id: str,
    parent_span_id: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    input_payload: Any = None,
) -> Iterator[TraceRecord]:
    rec = TraceRecord(
        span_id=new_id("span"),
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        actor=actor,
        actor_kind=actor_kind,
        model=model,
        prompt_version=prompt_version,
        input_payload=input_payload,
    )
    start = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        rec.status = "error"
        rec.error_class = type(e).__name__
        rec.output_payload = {"exception": str(e)[:500]}
        raise
    finally:
        rec.latency_ms = int((time.perf_counter() - start) * 1000)
        _persist(rec)


def _persist(rec: TraceRecord) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO traces
               (span_id, trace_id, parent_span_id, ts, actor, actor_kind, model, prompt_version,
                input_json, output_json, status, error_class,
                input_tokens, output_tokens, latency_ms, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.span_id,
                rec.trace_id,
                rec.parent_span_id,
                datetime.now(timezone.utc).isoformat(),
                rec.actor,
                rec.actor_kind,
                rec.model,
                rec.prompt_version,
                _safe_json(rec.input_payload),
                _safe_json(rec.output_payload),
                rec.status,
                rec.error_class,
                rec.input_tokens,
                rec.output_tokens,
                rec.latency_ms,
                rec.cost_usd,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _safe_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_repr": str(obj)[:500]})
