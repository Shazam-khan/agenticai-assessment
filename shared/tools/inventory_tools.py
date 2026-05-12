"""Inventory tools: get_stock_levels, create_purchase_order.

create_purchase_order is the one with REAL SIDE EFFECTS:
  - writes a row to `purchase_orders`
  - idempotent per (material, quantity, supplier_id, ordering_day_utc)
  - if quantity > threshold (default 500), persists a pending_confirmation
    and returns needs_confirmation instead of executing
  - if an approved confirmation exists for the same inputs, executes and
    marks that confirmation 'executed'
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .. import hitl
from ..db import get_conn
from ..ids import new_id
from .base import PendingConfirmation, ToolResult, utc_day, validated_tool

from task1.mock_data import KNOWN_MATERIALS, KNOWN_SUPPLIERS, STOCK

PO_THRESHOLD = int(os.getenv("HITL_PO_THRESHOLD_UNITS", "500"))


# ---------- schemas ----------

class GetStockLevelsInput(BaseModel):
    materials: list[str] = Field(min_length=1)


class CreatePurchaseOrderInput(BaseModel):
    material: str
    quantity: float = Field(gt=0)
    supplier_id: str
    urgency: str = "standard"


# ---------- get_stock_levels ----------

@validated_tool(name="get_stock_levels", input_schema=GetStockLevelsInput)
def get_stock_levels(args: GetStockLevelsInput, *, trace_id: str,
                     parent_span_id: str | None = None) -> ToolResult:
    levels = []
    unknown = []
    for m in args.materials:
        row = STOCK.get(m)
        if not row:
            unknown.append(m)
            continue
        days = float("inf") if row["daily_usage"] == 0 else row["quantity"] / row["daily_usage"]
        levels.append({
            "material": m,
            "quantity": row["quantity"],
            "unit": row["unit"],
            "reorder_level": row["reorder_level"],
            "days_until_stockout": round(days, 1),
        })
    return ToolResult.ok({"levels": levels, "unknown": unknown})


# ---------- create_purchase_order ----------

VALID_URGENCIES = {"low", "standard", "high", "critical"}


def _po_id(material: str, quantity: float, supplier_id: str) -> str:
    raw = f"{material}|{quantity}|{supplier_id}|{utc_day()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _existing_po(po_id: str) -> dict | None:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT po_id, material, quantity, supplier_id, urgency, status, created_at "
            "FROM purchase_orders WHERE po_id = ?",
            (po_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(r) if r else None


def _insert_po(*, po_id: str, args: CreatePurchaseOrderInput, trace_id: str,
               confirmation_id: str | None) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO purchase_orders
               (po_id, material, quantity, supplier_id, urgency, status, created_at, trace_id, confirmation_id)
               VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)""",
            (po_id, args.material, args.quantity, args.supplier_id, args.urgency,
             datetime.now(timezone.utc).isoformat(), trace_id, confirmation_id),
        )
        conn.commit()
    finally:
        conn.close()


@validated_tool(name="create_purchase_order", input_schema=CreatePurchaseOrderInput)
def create_purchase_order(
    args: CreatePurchaseOrderInput,
    *,
    trace_id: str,
    parent_span_id: str | None = None,
    confirmed: bool = False,
) -> ToolResult:
    # Cross-field validations that can't be expressed in the schema alone.
    if args.material not in KNOWN_MATERIALS:
        return ToolResult.err(
            "unknown_material",
            f"material '{args.material}' is not in the known list: {KNOWN_MATERIALS}",
            retryable=False,
        )
    if args.supplier_id not in KNOWN_SUPPLIERS:
        return ToolResult.err(
            "unknown_supplier",
            f"supplier_id '{args.supplier_id}' is not in the known list: {KNOWN_SUPPLIERS}",
            retryable=False,
        )
    if args.urgency not in VALID_URGENCIES:
        return ToolResult.err(
            "invalid_urgency",
            f"urgency must be one of {VALID_URGENCIES}",
            retryable=False,
        )

    po_id = _po_id(args.material, args.quantity, args.supplier_id)

    # Idempotency #1: a PO already exists for these inputs today.
    existing = _existing_po(po_id)
    if existing:
        return ToolResult.ok({
            "po_id": po_id, "status": "existing",
            "material": existing["material"], "quantity": existing["quantity"],
            "supplier_id": existing["supplier_id"], "urgency": existing["urgency"],
        })

    inputs_dict = args.model_dump()

    # HITL gate: any quantity strictly greater than the threshold needs approval.
    if args.quantity > PO_THRESHOLD and not confirmed:
        # If the approval already came in out-of-band, execute now.
        cid_existing = hitl.make_confirmation_id("create_purchase_order", inputs_dict)
        if hitl.is_approved(cid_existing):
            _insert_po(po_id=po_id, args=args, trace_id=trace_id,
                       confirmation_id=cid_existing)
            hitl.mark_executed(cid_existing)
            return ToolResult.ok({
                "po_id": po_id, "status": "created",
                "confirmation_id": cid_existing,
                "note": "executed after prior approval",
            })

        # Otherwise persist a pending confirmation.
        reason = (
            f"Quantity {args.quantity} exceeds threshold {PO_THRESHOLD} "
            f"for material '{args.material}'. Human approval required."
        )
        cid = hitl.create_pending(
            tool_name="create_purchase_order",
            inputs=inputs_dict,
            reason=reason,
            trace_id=trace_id,
        )
        # expires_at is set inside hitl.create_pending; load it back.
        row = hitl.get_status(cid)
        return ToolResult.needs_confirmation(PendingConfirmation(
            confirmation_id=cid,
            tool_name="create_purchase_order",
            inputs=inputs_dict,
            reason=reason,
            expires_at=row["expires_at"] if row else "",
        ))

    # Sub-threshold OR explicitly confirmed: execute.
    cid_to_mark: str | None = None
    if confirmed:
        cid_to_mark = hitl.make_confirmation_id("create_purchase_order", inputs_dict)
    _insert_po(po_id=po_id, args=args, trace_id=trace_id,
               confirmation_id=cid_to_mark)
    if cid_to_mark and hitl.get_status(cid_to_mark):
        hitl.mark_executed(cid_to_mark)
    return ToolResult.ok({
        "po_id": po_id, "status": "created",
        "confirmation_id": cid_to_mark,
    })
