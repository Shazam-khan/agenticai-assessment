"""Production tools: get_production_schedule, flag_bottleneck.

flag_bottleneck has a real side effect: writes to `alerts` AND appends to
data/notifications.log. The log file is the documented integration seam —
swap out the writer to push to Slack/PagerDuty/etc.

Idempotency: alert_id = sha256(order_id + reason_norm + utc_day). Same flag
on the same day = same alert. Different day or different reason = new alert.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ..db import get_conn
from .base import ToolResult, utc_day, validated_tool

from task1.mock_data import PRODUCTION_ORDERS, get_production_schedule as _mock_schedule

NOTIFICATIONS_LOG = os.getenv("NOTIFICATIONS_LOG", "./data/notifications.log")


# ---------- schemas ----------

class GetProductionScheduleInput(BaseModel):
    date_range: tuple[str, str]


class FlagBottleneckInput(BaseModel):
    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    severity: str  # validated against enum below


VALID_SEVERITIES = {"low", "medium", "high"}


# ---------- get_production_schedule ----------

@validated_tool(name="get_production_schedule", input_schema=GetProductionScheduleInput)
def get_production_schedule(args: GetProductionScheduleInput, *, trace_id: str,
                            parent_span_id: str | None = None) -> ToolResult:
    # Cross-field validation that can't fit on a single field.
    try:
        start = datetime.fromisoformat(args.date_range[0])
        end = datetime.fromisoformat(args.date_range[1])
    except (ValueError, TypeError) as e:
        return ToolResult.err(
            "schema_violation",
            f"date_range entries must be ISO dates: {e}",
            retryable=False,
        )
    if start > end:
        return ToolResult.err(
            "schema_violation",
            f"start {start.date()} is after end {end.date()}",
            retryable=False,
        )

    orders = _mock_schedule(args.date_range)
    return ToolResult.ok({"orders": orders})


# ---------- flag_bottleneck ----------

def _alert_id(order_id: str, reason: str) -> str:
    raw = f"{order_id}|{reason.lower().strip()}|{utc_day()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _existing_alert(alert_id: str) -> dict | None:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT alert_id, order_id, reason, severity FROM alerts WHERE alert_id = ?",
            (alert_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(r) if r else None


def _notify(line: str) -> None:
    """Adapter point — production replaces this with Slack/PagerDuty/email."""
    path = Path(NOTIFICATIONS_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _known_order_ids() -> set[str]:
    return {o["order_id"] for o in PRODUCTION_ORDERS}


@validated_tool(name="flag_bottleneck", input_schema=FlagBottleneckInput)
def flag_bottleneck(args: FlagBottleneckInput, *, trace_id: str,
                    parent_span_id: str | None = None) -> ToolResult:
    if args.severity not in VALID_SEVERITIES:
        return ToolResult.err(
            "invalid_severity",
            f"severity must be one of {VALID_SEVERITIES}, got {args.severity!r}",
            retryable=False,
        )
    known = _known_order_ids()
    if args.order_id not in known:
        return ToolResult.err(
            "unknown_order",
            f"order_id '{args.order_id}' not in known orders: {sorted(known)}",
            retryable=False,
        )

    aid = _alert_id(args.order_id, args.reason)
    existing = _existing_alert(aid)
    if existing:
        return ToolResult.ok({
            "alert_id": aid, "status": "existing",
            "order_id": existing["order_id"], "severity": existing["severity"],
        })

    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO alerts (alert_id, order_id, reason, severity, created_at, trace_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (aid, args.order_id, args.reason, args.severity,
             datetime.now(timezone.utc).isoformat(), trace_id),
        )
        conn.commit()
    finally:
        conn.close()

    _notify(
        f"[{datetime.now(timezone.utc).isoformat()}] "
        f"[ALERT severity={args.severity}] order={args.order_id} reason={args.reason}"
    )

    return ToolResult.ok({
        "alert_id": aid, "status": "created",
        "order_id": args.order_id, "severity": args.severity,
    })
