"""Stub-LLM unit tests for the supervisor's control flow.

These don't hit Groq — they swap a deterministic LLM stub into the agents so we
can verify the orchestration mechanics (planning, retry-on-error, escalation,
graceful degradation) without burning tokens or depending on network.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shared.db import get_conn, reset_db
from shared.ids import new_id
from shared.intents import (
    CheckStockOutput,
    StockAlert,
    StockLevel,
    SupervisorPlan,
    PlannedSubtask,
)
from shared.llm import LLMResponse
from shared.messages import AgentError, AgentMessage, MessageMetadata

from task1.agents.inventory import InventoryAgent
from task1.agents.supervisor import SupervisorAgent
from task1.eval.doubles import AlwaysFailingProduction, HallucinatingInventoryOnce


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db))
    import shared.db
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    yield


def _stub_llm(responses: list) -> MagicMock:
    """Returns a stub LLMClient whose complete/complete_json return canned values."""
    llm = MagicMock()
    json_iter = iter(responses)

    def complete_json(*, system, user, schema, trace_id, parent_span_id=None,
                      prompt_version="v1", model=None):
        payload = next(json_iter)
        parsed = schema.model_validate(payload)
        resp = LLMResponse(content=json.dumps(payload), tokens_in=10, tokens_out=20,
                            latency_ms=5, model="stub", cost_usd=0.0)
        return parsed, resp

    def complete(*, system, user, trace_id, parent_span_id=None, prompt_version="v1",
                 json_mode=False, temperature=0.0, model=None, max_tokens=2048):
        return LLMResponse(content="stub answer text", tokens_in=5, tokens_out=10,
                            latency_ms=3, model="stub", cost_usd=0.0)

    llm.complete_json = complete_json
    llm.complete = complete
    return llm


def test_supervisor_escalates_after_failure_and_continues():
    """Production hard-fails; supervisor should escalate AND still produce an answer."""
    plan_payload = {
        "subtasks": [
            {"agent": "inventory", "intent": "check_stock",
             "payload": {"materials": ["glycerin"]}},
            {"agent": "production", "intent": "check_schedule",
             "payload": {"date_range": ["2026-05-12", "2026-05-12"]}},
        ],
        "needs_report": False,
        "reasoning": "",
    }
    # inventory makes one LLM call internally; supervisor makes 1 plan + 1 retry-fixup + 1 synth.
    # Production agent is the failing double — it does NOT call the LLM at all.
    inventory_alerts = {"alerts": [
        {"material": "glycerin", "severity": "critical", "suggested_action": "Reorder now."}
    ]}
    retry_payload = {"payload": {"date_range": ["2026-05-12", "2026-05-12"]}}

    llm = _stub_llm([plan_payload, inventory_alerts, retry_payload])
    registry = {
        "inventory": InventoryAgent(llm=llm),
        "production": AlwaysFailingProduction(llm=llm),
    }
    supervisor = SupervisorAgent(llm=llm, registry=registry, max_retries=1)

    trace_id = new_id("t")
    out = supervisor.run("status of glycerin and production", trace_id=trace_id)

    statuses = {r["agent"]: r["status"] for r in out["results"]}
    assert statuses["inventory"] == "success"
    assert statuses["production"] == "escalated"
    assert out["answer"]  # non-empty synthesis even with one specialist down

    # And the trace table has rows for both specialists.
    conn = get_conn()
    rows = conn.execute(
        "SELECT actor, status FROM traces WHERE trace_id = ? AND actor_kind='agent'",
        (trace_id,),
    ).fetchall()
    conn.close()
    actors = [r["actor"] for r in rows]
    assert "inventory" in actors
    assert "production" in actors
    assert "supervisor" in actors


def test_supervisor_retries_hallucinated_output_and_recovers():
    """Inventory returns malformed payload on call 1, valid on call 2."""
    plan_payload = {
        "subtasks": [
            {"agent": "inventory", "intent": "check_stock",
             "payload": {"materials": ["glycerin"]}},
        ],
        "needs_report": False,
        "reasoning": "",
    }
    retry_payload = {"payload": {"materials": ["glycerin"]}}

    llm = _stub_llm([plan_payload, retry_payload])
    registry = {"inventory": HallucinatingInventoryOnce(llm=llm)}
    supervisor = SupervisorAgent(llm=llm, registry=registry, max_retries=1)

    out = supervisor.run("glycerin?", trace_id=new_id("t"))

    statuses = {r["agent"]: r["status"] for r in out["results"]}
    assert statuses["inventory"] == "success", f"expected success after retry, got {statuses}"
    inv_result = next(r for r in out["results"] if r["agent"] == "inventory")
    assert inv_result["attempts"] == 2, "should have taken 2 attempts to recover"


def test_unknown_agent_returns_structured_error_not_crash():
    plan_payload = {
        "subtasks": [
            {"agent": "inventory", "intent": "check_stock",
             "payload": {"materials": ["x"]}},
        ],
        "needs_report": False,
        "reasoning": "",
    }
    llm = _stub_llm([plan_payload])
    supervisor = SupervisorAgent(llm=llm, registry={}, max_retries=0)
    out = supervisor.run("anything", trace_id=new_id("t"))
    assert out["results"][0]["status"] == "error"
    assert out["results"][0]["error"]["error_class"] == "unknown_agent"
