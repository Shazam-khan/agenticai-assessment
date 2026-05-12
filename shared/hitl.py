"""Human-in-the-loop confirmation primitive.

The brief's hard question: "how does an autonomous agent pause, request
confirmation, and resume?". Our answer: persist-and-exit.

When a tool needs confirmation, it (via `create_pending`) writes a
`pending_confirmations` row keyed by an inputs-derived `confirmation_id` and
returns `ToolResult.needs_confirmation(...)`. Approval is out-of-band — a CLI
command, an API endpoint, or whatever else flips the row to 'approved'. On
resume, the tool re-checks the row; if approved, it executes and writes the
side effect.

Idempotency falls out for free because the confirmation_id is derived from
the inputs (see `tools.base.inputs_idempotency_key`).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from .db import get_conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def make_confirmation_id(tool_name: str, inputs: dict) -> str:
    return hashlib.sha256(f"{tool_name}|{_canon(inputs)}".encode()).hexdigest()


def create_pending(
    *,
    tool_name: str,
    inputs: dict,
    reason: str,
    trace_id: str | None = None,
    ttl_minutes: int = 60,
) -> str:
    """Upsert a pending confirmation. Returns confirmation_id.

    If a row already exists with status 'pending' or 'approved', it is
    preserved (the second submission of the same call does NOT create a second
    pending row, and does NOT downgrade an approval back to pending).
    """
    cid = make_confirmation_id(tool_name, inputs)
    now = _now()
    expires = (now + timedelta(minutes=ttl_minutes)).isoformat()
    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT status FROM pending_confirmations WHERE confirmation_id = ?",
            (cid,),
        ).fetchone()
        if existing and existing["status"] in ("pending", "approved", "executed"):
            return cid
        if existing:
            # Re-create after a 'rejected' or 'expired' lifecycle.
            conn.execute(
                """UPDATE pending_confirmations
                   SET status = 'pending', inputs_json = ?, reason = ?, trace_id = ?,
                       created_at = ?, expires_at = ?, approved_at = NULL, approved_by = NULL
                   WHERE confirmation_id = ?""",
                (_canon(inputs), reason, trace_id, now.isoformat(), expires, cid),
            )
        else:
            conn.execute(
                """INSERT INTO pending_confirmations
                   (confirmation_id, tool_name, inputs_json, reason, trace_id,
                    created_at, expires_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (cid, tool_name, _canon(inputs), reason, trace_id,
                 now.isoformat(), expires),
            )
        conn.commit()
    finally:
        conn.close()
    return cid


def get_status(confirmation_id: str) -> dict | None:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT * FROM pending_confirmations WHERE confirmation_id = ?",
            (confirmation_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(r) if r else None


def is_approved(confirmation_id: str) -> bool:
    row = get_status(confirmation_id)
    return bool(row) and row["status"] == "approved"


def approve(confirmation_id: str, approver: str) -> bool:
    conn = get_conn()
    try:
        cur = conn.execute(
            """UPDATE pending_confirmations
               SET status = 'approved', approved_at = ?, approved_by = ?
               WHERE confirmation_id = ? AND status = 'pending'""",
            (_now().isoformat(), approver, confirmation_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def reject(confirmation_id: str, approver: str) -> bool:
    conn = get_conn()
    try:
        cur = conn.execute(
            """UPDATE pending_confirmations
               SET status = 'rejected', approved_at = ?, approved_by = ?
               WHERE confirmation_id = ? AND status = 'pending'""",
            (_now().isoformat(), approver, confirmation_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def mark_executed(confirmation_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE pending_confirmations SET status = 'executed' WHERE confirmation_id = ?",
            (confirmation_id,),
        )
        conn.commit()
    finally:
        conn.close()


def list_pending() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT confirmation_id, tool_name, inputs_json, reason, trace_id,
                      created_at, expires_at, status
               FROM pending_confirmations
               WHERE status IN ('pending', 'approved')
               ORDER BY created_at DESC"""
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
