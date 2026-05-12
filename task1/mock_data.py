"""Deterministic stand-in for the manufacturing client's ERP / MES.

Kept dead simple so tests are reproducible. Replace with real adapters in production.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_NOW = datetime.now(timezone.utc)


STOCK: dict[str, dict] = {
    "rose_oil":       {"quantity": 12.0,  "unit": "L",     "reorder_level": 20.0, "daily_usage": 1.5},
    "glycerin":       {"quantity": 8.0,   "unit": "L",     "reorder_level": 25.0, "daily_usage": 2.0},
    "beeswax":        {"quantity": 60.0,  "unit": "kg",    "reorder_level": 15.0, "daily_usage": 1.0},
    "vitamin_e":      {"quantity": 3.5,   "unit": "L",     "reorder_level": 5.0,  "daily_usage": 0.5},
    "packaging_30ml": {"quantity": 1200,  "unit": "units", "reorder_level": 800,  "daily_usage": 120},
    "packaging_50ml": {"quantity": 400,   "unit": "units", "reorder_level": 800,  "daily_usage": 80},
}

KNOWN_MATERIALS: list[str] = list(STOCK.keys())


SUPPLIERS: dict[str, dict] = {
    "SUP-001": {"name": "Aroma Source Co.",     "specialties": ["rose_oil", "vitamin_e"]},
    "SUP-002": {"name": "Glycerin Direct",      "specialties": ["glycerin", "beeswax"]},
    "SUP-003": {"name": "PackRight Industries", "specialties": ["packaging_30ml", "packaging_50ml"]},
}
KNOWN_SUPPLIERS: list[str] = list(SUPPLIERS.keys())


PRODUCTION_ORDERS: list[dict] = [
    {
        "order_id": "PO-7741",
        "product": "Rose Hydrating Serum 30ml",
        "line": "line_1",
        "completion_pct": 62.0,
        "eta": (_NOW + timedelta(hours=8)).isoformat(),
        "blocked": False,
        "block_reason": None,
    },
    {
        "order_id": "PO-7742",
        "product": "Lavender Body Oil 50ml",
        "line": "line_2",
        "completion_pct": 18.0,
        "eta": (_NOW + timedelta(days=2)).isoformat(),
        "blocked": True,
        "block_reason": "Awaiting packaging_50ml restock (below reorder level).",
    },
    {
        "order_id": "PO-7743",
        "product": "Beeswax Lip Balm batch",
        "line": "line_3",
        "completion_pct": 92.0,
        "eta": (_NOW + timedelta(hours=2)).isoformat(),
        "blocked": False,
        "block_reason": None,
    },
]


def get_stock_levels(materials: list[str]) -> list[dict]:
    """Return raw stock rows. Unknown materials are silently skipped — the agent
    decides how to surface that as an alert."""
    out: list[dict] = []
    for m in materials:
        row = STOCK.get(m)
        if not row:
            continue
        days = float("inf") if row["daily_usage"] == 0 else row["quantity"] / row["daily_usage"]
        out.append(
            {
                "material": m,
                "quantity": row["quantity"],
                "unit": row["unit"],
                "reorder_level": row["reorder_level"],
                "days_until_stockout": round(days, 1),
            }
        )
    return out


def get_production_schedule(date_range: tuple[str, str]) -> list[dict]:
    """For this mock, the date range is ignored — we return today's active orders."""
    _ = date_range
    return [dict(o) for o in PRODUCTION_ORDERS]
