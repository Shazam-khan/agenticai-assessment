"""Adversarial input coverage for all four tools.

Brief hard requirement #5: 3+ adversarial cases per tool — malformed, boundary,
and plausible-but-wrong LLM inputs. Each test below also names which of those
three flavours it represents in the docstring.

Brief hard requirement #1 (in passing): every malformed input must come back as
a structured `ToolResult.error`, never raise an exception. The tests would
notice an unhandled exception even without an explicit assertion because
pytest would mark them errored, not failed.
"""
from __future__ import annotations

import pytest

from shared.tools.inventory_tools import create_purchase_order, get_stock_levels
from shared.tools.production_tools import flag_bottleneck, get_production_schedule


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    import shared.db
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    yield


# ===================== get_stock_levels =====================

class TestGetStockLevels:
    def test_malformed_none_materials(self):
        """Malformed: materials=None must produce schema_violation, not raise."""
        r = get_stock_levels(materials=None, trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_malformed_wrong_inner_types(self):
        """Malformed: integers inside the list."""
        r = get_stock_levels(materials=[1, 2, 3], trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_boundary_empty_list(self):
        """Boundary: empty list (min_length=1 on the schema)."""
        r = get_stock_levels(materials=[], trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_plausible_but_wrong_typo(self):
        """Plausible-but-wrong: 'glycerine' (typo of 'glycerin'). Tool succeeds
        but surfaces the unknown material rather than silently dropping it."""
        r = get_stock_levels(materials=["glycerine"], trace_id="t")
        assert r.status == "ok"
        assert r.output["levels"] == []
        assert r.output["unknown"] == ["glycerine"]


# ===================== create_purchase_order =====================

class TestCreatePurchaseOrder:
    def test_malformed_quantity_not_a_number(self):
        """Malformed: quantity='ten'."""
        r = create_purchase_order(material="glycerin", quantity="ten",
                                   supplier_id="SUP-002", urgency="standard",
                                   trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_boundary_quantity_zero(self):
        """Boundary: quantity=0 violates gt=0."""
        r = create_purchase_order(material="glycerin", quantity=0,
                                   supplier_id="SUP-002", urgency="standard",
                                   trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_boundary_at_threshold_auto_executes(self):
        """Boundary: quantity=500 is AT the threshold — threshold is strictly
        'above', so this auto-executes."""
        r = create_purchase_order(material="beeswax", quantity=500,
                                   supplier_id="SUP-002", urgency="standard",
                                   trace_id="t")
        assert r.status == "ok"
        assert r.output["status"] == "created"

    def test_boundary_just_above_threshold_needs_confirmation(self):
        """Boundary: quantity=501 -> needs_confirmation."""
        r = create_purchase_order(material="rose_oil", quantity=501,
                                   supplier_id="SUP-001", urgency="high",
                                   trace_id="t")
        assert r.status == "needs_confirmation"
        assert r.confirmation is not None
        assert r.confirmation.confirmation_id  # non-empty hex string

    def test_plausible_but_wrong_supplier(self):
        """Plausible-but-wrong: SUP-9999 looks like a real supplier id but isn't."""
        r = create_purchase_order(material="glycerin", quantity=100,
                                   supplier_id="SUP-9999", urgency="standard",
                                   trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "unknown_supplier"
        assert r.error.retryable is False

    def test_idempotency_same_inputs_returns_existing(self):
        """Same valid sub-threshold call twice = ONE PO; second returns 'existing'."""
        r1 = create_purchase_order(material="glycerin", quantity=200,
                                    supplier_id="SUP-002", urgency="standard",
                                    trace_id="t")
        assert r1.status == "ok"
        assert r1.output["status"] == "created"
        po_id = r1.output["po_id"]

        r2 = create_purchase_order(material="glycerin", quantity=200,
                                    supplier_id="SUP-002", urgency="standard",
                                    trace_id="t-2")
        assert r2.status == "ok"
        assert r2.output["status"] == "existing"
        assert r2.output["po_id"] == po_id


# ===================== get_production_schedule =====================

class TestGetProductionSchedule:
    def test_malformed_none(self):
        r = get_production_schedule(date_range=None, trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_malformed_not_iso(self):
        r = get_production_schedule(date_range=("not-a-date", "also-not"), trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_boundary_start_after_end(self):
        r = get_production_schedule(
            date_range=("2026-12-31", "2026-01-01"), trace_id="t",
        )
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"


# ===================== flag_bottleneck =====================

class TestFlagBottleneck:
    def test_malformed_severity_not_in_enum(self):
        r = flag_bottleneck(order_id="PO-7742", reason="x",
                            severity="catastrophic", trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "invalid_severity"

    def test_malformed_missing_order_id(self):
        r = flag_bottleneck(order_id=None, reason="x", severity="high", trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_boundary_empty_reason(self):
        r = flag_bottleneck(order_id="PO-7742", reason="", severity="high", trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "schema_violation"

    def test_plausible_but_wrong_unknown_order(self):
        r = flag_bottleneck(order_id="PO-0000", reason="x",
                            severity="medium", trace_id="t")
        assert r.status == "error"
        assert r.error.error_class == "unknown_order"

    def test_idempotency_same_flag_same_day(self):
        r1 = flag_bottleneck(order_id="PO-7742", reason="packaging shortage",
                              severity="high", trace_id="t")
        assert r1.status == "ok"
        assert r1.output["status"] == "created"

        r2 = flag_bottleneck(order_id="PO-7742", reason="packaging shortage",
                              severity="high", trace_id="t-2")
        assert r2.status == "ok"
        assert r2.output["status"] == "existing"
        assert r2.output["alert_id"] == r1.output["alert_id"]
