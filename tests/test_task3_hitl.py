"""HITL pause/resume lifecycle tests.

These exercise the persist-and-exit mechanism the brief explicitly calls out
as the architectural answer to "how does an agent pause, request confirmation,
and resume."
"""
from __future__ import annotations

import pytest

from shared import hitl
from shared.tools.inventory_tools import create_purchase_order


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    import shared.db
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    yield


def test_pending_confirmation_persists_and_is_idempotent():
    """Submitting the same over-threshold PO twice without approving = ONE pending row."""
    r1 = create_purchase_order(material="glycerin", quantity=1500,
                                supplier_id="SUP-002", urgency="high",
                                trace_id="t")
    assert r1.status == "needs_confirmation"
    cid = r1.confirmation.confirmation_id

    r2 = create_purchase_order(material="glycerin", quantity=1500,
                                supplier_id="SUP-002", urgency="high",
                                trace_id="t2")
    assert r2.status == "needs_confirmation"
    assert r2.confirmation.confirmation_id == cid

    pending = hitl.list_pending()
    assert len([p for p in pending if p["confirmation_id"] == cid]) == 1


def test_approve_then_execute_creates_po():
    """Submit -> pending -> approve -> resubmit -> tool sees approved -> executes."""
    r1 = create_purchase_order(material="glycerin", quantity=1500,
                                supplier_id="SUP-002", urgency="high",
                                trace_id="t")
    assert r1.status == "needs_confirmation"
    cid = r1.confirmation.confirmation_id

    assert hitl.approve(cid, approver="test_user") is True

    # Re-submit with confirmed=False; tool detects the approved row.
    r2 = create_purchase_order(material="glycerin", quantity=1500,
                                supplier_id="SUP-002", urgency="high",
                                trace_id="t-resume")
    assert r2.status == "ok", f"expected ok after approval, got {r2}"
    assert r2.output["status"] == "created"

    # And the confirmation is marked executed.
    row = hitl.get_status(cid)
    assert row["status"] == "executed"


def test_rejected_confirmation_does_not_execute():
    r1 = create_purchase_order(material="rose_oil", quantity=2000,
                                supplier_id="SUP-001", urgency="critical",
                                trace_id="t")
    cid = r1.confirmation.confirmation_id

    assert hitl.reject(cid, approver="test_user") is True

    # Resubmit; tool still sees a row but its status is 'rejected', not 'approved'.
    # `create_pending` will reset it back to 'pending'.
    r2 = create_purchase_order(material="rose_oil", quantity=2000,
                                supplier_id="SUP-001", urgency="critical",
                                trace_id="t2")
    assert r2.status == "needs_confirmation"
    # No PO created yet.
    from shared.db import get_conn
    conn = get_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_approve_idempotent():
    r1 = create_purchase_order(material="beeswax", quantity=600,
                                supplier_id="SUP-002", urgency="standard",
                                trace_id="t")
    cid = r1.confirmation.confirmation_id

    assert hitl.approve(cid, approver="user_a") is True
    # Second approval is a no-op (already approved).
    assert hitl.approve(cid, approver="user_b") is False
    row = hitl.get_status(cid)
    assert row["approved_by"] == "user_a"  # first approval sticks


def test_unconfirmed_resubmissions_do_not_create_second_pending():
    cids = []
    for i in range(3):
        r = create_purchase_order(material="glycerin", quantity=1500,
                                   supplier_id="SUP-002", urgency="high",
                                   trace_id=f"t{i}")
        cids.append(r.confirmation.confirmation_id)
    assert len(set(cids)) == 1  # all three got the same id

    from shared.db import get_conn
    conn = get_conn()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM pending_confirmations WHERE confirmation_id = ?",
            (cids[0],),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1
