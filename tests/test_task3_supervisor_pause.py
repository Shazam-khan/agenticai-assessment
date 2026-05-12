"""Supervisor pause/resume end-to-end with a stub LLM.

These tests prove the full HITL flow through the orchestration layer without
hitting Groq:

1. supervisor.run() with an over-threshold place_order plan -> status='needs_human'.
2. paused_runs row is written.
3. After hitl.approve(), supervisor.resume(trace_id) executes the PO and marks
   the run resumed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shared import hitl
from shared.ids import new_id
from shared.llm import LLMResponse
from task1.agents.production import ProductionAgent
from task1.agents.supervisor import SupervisorAgent


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    import shared.db
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    yield


def _stub_llm(responses: list[dict]) -> MagicMock:
    """LLM stub returning canned JSON for `complete_json` and a fixed text for `complete`."""
    llm = MagicMock()
    it = iter(responses)

    def complete_json(*, system, user, schema, trace_id, parent_span_id=None,
                       prompt_version="v1", model=None):
        payload = next(it)
        parsed = schema.model_validate(payload)
        return parsed, LLMResponse(content=json.dumps(payload), tokens_in=5, tokens_out=10,
                                    latency_ms=2, model="stub", cost_usd=0.0)

    def complete(*, system, user, trace_id, parent_span_id=None, prompt_version="v1",
                  json_mode=False, temperature=0.0, model=None, max_tokens=2048):
        return LLMResponse(content="ok", tokens_in=5, tokens_out=10,
                            latency_ms=2, model="stub", cost_usd=0.0)

    llm.complete_json = complete_json
    llm.complete = complete
    return llm


def _high_qty_plan() -> dict:
    return {
        "subtasks": [{
            "agent": "production",
            "intent": "place_order",
            "payload": {
                "material": "glycerin",
                "quantity": 1500,
                "supplier_id": "SUP-002",
                "urgency": "high",
            },
        }],
        "needs_report": False,
        "reasoning": "",
    }


def test_run_returns_needs_human_and_writes_paused_row():
    llm = _stub_llm([_high_qty_plan()])
    registry = {"production": ProductionAgent(llm=llm)}
    supervisor = SupervisorAgent(llm=llm, registry=registry, max_retries=0)

    trace_id = new_id("trace")
    out = supervisor.run("Order 1500 units of glycerin from SUP-002, high urgency",
                          trace_id=trace_id)

    assert out["status"] == "needs_human"
    assert len(out["pending"]) == 1
    cid = out["pending"][0]["confirmation_id"]
    assert cid

    from shared.db import get_conn
    conn = get_conn()
    paused = conn.execute("SELECT * FROM paused_runs WHERE trace_id = ?",
                          (trace_id,)).fetchone()
    conn.close()
    assert paused is not None
    assert paused["status"] == "paused"


def test_resume_after_approval_executes_po():
    # We re-plan on resume too, so the stub needs two plans + a synthesis text.
    llm = _stub_llm([_high_qty_plan(), _high_qty_plan()])
    registry = {"production": ProductionAgent(llm=llm)}
    supervisor = SupervisorAgent(llm=llm, registry=registry, max_retries=0)

    trace_id = new_id("trace")
    out1 = supervisor.run("Order 1500 units of glycerin from SUP-002, high urgency",
                           trace_id=trace_id)
    assert out1["status"] == "needs_human"
    cid = out1["pending"][0]["confirmation_id"]

    assert hitl.approve(cid, approver="test_user") is True

    out2 = supervisor.resume(trace_id)
    assert out2["status"] == "ok", f"expected ok, got: {out2}"

    # PO was created.
    from shared.db import get_conn
    conn = get_conn()
    pos = conn.execute("SELECT * FROM purchase_orders").fetchall()
    conn.close()
    assert len(pos) == 1
    assert pos[0]["material"] == "glycerin"
    assert pos[0]["quantity"] == 1500

    # paused_runs marked resumed.
    conn = get_conn()
    paused = conn.execute("SELECT status FROM paused_runs WHERE trace_id = ?",
                          (trace_id,)).fetchone()
    conn.close()
    assert paused["status"] == "resumed"


def test_resume_without_approval_re_pauses():
    """If the user calls resume() before approving, the run pauses again — no
    silent execution. Proves the safety property."""
    llm = _stub_llm([_high_qty_plan(), _high_qty_plan()])
    registry = {"production": ProductionAgent(llm=llm)}
    supervisor = SupervisorAgent(llm=llm, registry=registry, max_retries=0)

    trace_id = new_id("trace")
    out1 = supervisor.run("Order 1500 units of glycerin from SUP-002, high urgency",
                           trace_id=trace_id)
    assert out1["status"] == "needs_human"
    # Skip approval; call resume directly.
    out2 = supervisor.resume(trace_id)
    assert out2["status"] == "needs_human"

    from shared.db import get_conn
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
    conn.close()
    assert n == 0
