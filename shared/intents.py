"""Typed per-intent payloads. Agents validate inputs/outputs against these.

A wrong-shape payload is a `schema_violation` error — the supervisor catches it
and retries with a corrective instruction (Task 1 hard requirement #2).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- inventory ----------

class CheckStockInput(BaseModel):
    materials: list[str] = Field(min_length=1)


class StockLevel(BaseModel):
    material: str
    quantity: float
    unit: str
    reorder_level: float
    days_until_stockout: float


class StockAlert(BaseModel):
    material: str
    severity: Literal["info", "warning", "critical"]
    suggested_action: str


class CheckStockOutput(BaseModel):
    levels: list[StockLevel]
    alerts: list[StockAlert]


# ---------- production ----------

class CheckScheduleInput(BaseModel):
    date_range: tuple[str, str]  # ISO dates


class ProductionOrder(BaseModel):
    order_id: str
    product: str
    line: str
    completion_pct: float
    eta: str  # ISO datetime


class Bottleneck(BaseModel):
    order_id: str
    reason: str
    severity: Literal["low", "medium", "high"]


class CheckScheduleOutput(BaseModel):
    orders: list[ProductionOrder]
    bottlenecks: list[Bottleneck]


# ---------- report ----------

class ReportSectionInput(BaseModel):
    heading: str
    data: dict  # arbitrary structured payload from another agent


class DraftReportInput(BaseModel):
    title: str
    sections: list[ReportSectionInput]


class ReportSectionOutput(BaseModel):
    heading: str
    body: str  # human-readable markdown-ish text


class DraftReportOutput(BaseModel):
    title: str
    summary: str
    sections: list[ReportSectionOutput]


# ---------- place_order (Task 3) ----------

class PlaceOrderInput(BaseModel):
    material: str
    quantity: float = Field(gt=0)
    supplier_id: str
    urgency: Literal["low", "standard", "high", "critical"] = "standard"


class PlaceOrderOutput(BaseModel):
    po_id: str | None = None
    status: Literal["created", "existing", "needs_confirmation"]
    confirmation_id: str | None = None
    reason: str | None = None


# ---------- supervisor planning ----------

class PlannedSubtask(BaseModel):
    agent: Literal["inventory", "production", "report"]
    intent: Literal["check_stock", "check_schedule", "draft_report", "place_order"]
    payload: dict


class SupervisorPlan(BaseModel):
    subtasks: list[PlannedSubtask]
    needs_report: bool = False
    reasoning: str = ""


INTENT_INPUT_SCHEMA: dict[str, type[BaseModel]] = {
    "check_stock": CheckStockInput,
    "check_schedule": CheckScheduleInput,
    "draft_report": DraftReportInput,
    "place_order": PlaceOrderInput,
}

INTENT_OUTPUT_SCHEMA: dict[str, type[BaseModel]] = {
    "check_stock": CheckStockOutput,
    "check_schedule": CheckScheduleOutput,
    "draft_report": DraftReportOutput,
    "place_order": PlaceOrderOutput,
}
