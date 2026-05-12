"""Intake tools: record_lead_fields, get_lead_fields, lock_session_language.

These are the tools the rewritten Task 4A intake agent uses. `record_lead_fields`
is the one the brief calls out by name — must be called immediately when a
value is learned, never batched across turns.

Storage:
  - `leads` table keyed on (customer_id, field_name). Upsert semantics.
  - `session_language` table — first write wins. Enforces the brief's
    "no mid-conversation switching" rule at the storage layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ..db import get_conn
from .base import ToolResult, validated_tool

LEAD_FIELDS = (
    "company_name",
    "contact_name",
    "product_need",
    "quantity",
    "timeline",
    "budget",
)
LeadField = Literal[
    "company_name", "contact_name", "product_need", "quantity", "timeline", "budget"
]
LanguageCode = Literal["en", "ur", "mixed"]


# ---------- schemas ----------

class RecordLeadFieldInput(BaseModel):
    customer_id: str = Field(min_length=1)
    field: LeadField
    value: str = Field(min_length=1)
    language: LanguageCode | None = None
    source_turn_id: str | None = None


class GetLeadFieldsInput(BaseModel):
    customer_id: str = Field(min_length=1)


class LockSessionLanguageInput(BaseModel):
    session_id: str = Field(min_length=1)
    language: LanguageCode


class GetSessionLanguageInput(BaseModel):
    session_id: str = Field(min_length=1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- record_lead_fields ----------

@validated_tool(name="record_lead_fields", input_schema=RecordLeadFieldInput)
def record_lead_fields(args: RecordLeadFieldInput, *, trace_id: str,
                       parent_span_id: str | None = None) -> ToolResult:
    """Upsert one lead field. Last-write-wins on (customer_id, field)."""
    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT value FROM leads WHERE customer_id = ? AND field_name = ?",
            (args.customer_id, args.field),
        ).fetchone()

        if existing and existing["value"].strip().lower() == args.value.strip().lower():
            return ToolResult.ok({
                "status": "existing",
                "field": args.field,
                "value": existing["value"],
            })

        conn.execute(
            """INSERT INTO leads
               (customer_id, field_name, value, language, source_turn_id, trace_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(customer_id, field_name) DO UPDATE SET
                 value = excluded.value,
                 language = excluded.language,
                 source_turn_id = excluded.source_turn_id,
                 trace_id = excluded.trace_id,
                 updated_at = excluded.updated_at""",
            (args.customer_id, args.field, args.value, args.language,
             args.source_turn_id, trace_id, _now()),
        )
        conn.commit()
    finally:
        conn.close()

    return ToolResult.ok({
        "status": "updated" if existing else "created",
        "field": args.field,
        "value": args.value,
        "previous_value": existing["value"] if existing else None,
    })


# ---------- get_lead_fields ----------

@validated_tool(name="get_lead_fields", input_schema=GetLeadFieldsInput)
def get_lead_fields(args: GetLeadFieldsInput, *, trace_id: str,
                    parent_span_id: str | None = None) -> ToolResult:
    """Returns the current snapshot for one customer + which fields are still missing."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT field_name, value FROM leads WHERE customer_id = ?",
            (args.customer_id,),
        ).fetchall()
    finally:
        conn.close()
    collected = {r["field_name"]: r["value"] for r in rows}
    missing = [f for f in LEAD_FIELDS if f not in collected]
    return ToolResult.ok({"collected": collected, "missing": missing})


# ---------- lock_session_language ----------

@validated_tool(name="lock_session_language", input_schema=LockSessionLanguageInput)
def lock_session_language(args: LockSessionLanguageInput, *, trace_id: str,
                          parent_span_id: str | None = None) -> ToolResult:
    """First write wins. Subsequent calls with a DIFFERENT language are no-ops
    that return the originally-locked language."""
    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT language FROM session_language WHERE session_id = ?",
            (args.session_id,),
        ).fetchone()
        if existing:
            return ToolResult.ok({
                "status": "existing",
                "language": existing["language"],
                "requested": args.language,
            })
        conn.execute(
            """INSERT INTO session_language (session_id, language, detected_at)
               VALUES (?, ?, ?)""",
            (args.session_id, args.language, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return ToolResult.ok({
        "status": "locked",
        "language": args.language,
    })


@validated_tool(name="get_session_language", input_schema=GetSessionLanguageInput)
def get_session_language(args: GetSessionLanguageInput, *, trace_id: str,
                         parent_span_id: str | None = None) -> ToolResult:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT language FROM session_language WHERE session_id = ?",
            (args.session_id,),
        ).fetchone()
    finally:
        conn.close()
    return ToolResult.ok({"language": row["language"] if row else None})
