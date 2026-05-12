"""Stub-LLM unit tests for Task 4A's intake-v2 discipline.

These prove the agent loop's contract independent of the model:
  - record_lead_fields is idempotent.
  - session_language locks on first write.
  - For every fields_learned entry the loop calls record_lead_fields.
  - Phantom tool_calls (no matching fields_learned) are rejected, not executed.
  - A model-emitted language different from the locked one is detected.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shared.llm import LLMResponse
from shared.messages import AgentMessage
from shared.tools.intake_tools import (
    get_lead_fields,
    get_session_language,
    lock_session_language,
    record_lead_fields,
)
from task4.intake_v2 import IntakeAgentV2


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    chroma = tmp_path / "chroma"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("CHROMA_PATH", str(chroma))
    import shared.db
    import shared.memory.semantic as semantic_mod
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    monkeypatch.setattr(semantic_mod, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(semantic_mod, "_chroma_collection", None)
    yield


def _stub_llm(intake_output: dict, *, summariser: str = "summary") -> MagicMock:
    llm = MagicMock()
    json_calls = [intake_output]

    def complete_json(*, system, user, schema, trace_id, parent_span_id=None,
                       prompt_version="v1", model=None):
        payload = json_calls.pop(0)
        parsed = schema.model_validate(payload)
        return parsed, LLMResponse(content=json.dumps(payload), tokens_in=5, tokens_out=10,
                                    latency_ms=2, model="stub", cost_usd=0.0)

    def complete(*, system, user, trace_id, parent_span_id=None, prompt_version="v1",
                  json_mode=False, temperature=0.0, model=None, max_tokens=2048):
        return LLMResponse(content=summariser, tokens_in=5, tokens_out=10,
                            latency_ms=2, model="stub", cost_usd=0.0)

    llm.complete_json = complete_json
    llm.complete = complete
    return llm


# ---------- record_lead_fields ----------

def test_record_lead_fields_idempotent_for_identical_value():
    r1 = record_lead_fields(customer_id="c1", field="company_name",
                             value="Lumen Botanicals", language="en",
                             source_turn_id="t1", trace_id="tr")
    assert r1.status == "ok"
    assert r1.output["status"] == "created"

    r2 = record_lead_fields(customer_id="c1", field="company_name",
                             value="Lumen Botanicals", language="en",
                             source_turn_id="t2", trace_id="tr")
    assert r2.status == "ok"
    assert r2.output["status"] == "existing"


def test_record_lead_fields_update_changes_value():
    record_lead_fields(customer_id="c1", field="quantity",
                       value="10000 units", trace_id="tr")
    r2 = record_lead_fields(customer_id="c1", field="quantity",
                             value="15000 units", trace_id="tr")
    assert r2.status == "ok"
    assert r2.output["status"] == "updated"
    assert r2.output["previous_value"] == "10000 units"


def test_record_lead_fields_rejects_unknown_field():
    r = record_lead_fields(customer_id="c1", field="zodiac_sign",
                            value="virgo", trace_id="tr")
    assert r.status == "error"
    assert r.error.error_class == "schema_violation"


# ---------- session_language ----------

def test_session_language_locks_first_write_only():
    r1 = lock_session_language(session_id="s1", language="en", trace_id="tr")
    assert r1.output["status"] == "locked"
    assert r1.output["language"] == "en"

    r2 = lock_session_language(session_id="s1", language="ur", trace_id="tr")
    assert r2.output["status"] == "existing"
    assert r2.output["language"] == "en"  # original wins

    r3 = get_session_language(session_id="s1", trace_id="tr")
    assert r3.output["language"] == "en"


# ---------- IntakeAgentV2 ----------

def _msg(content: str, *, customer_id="c1", session_id="s1") -> AgentMessage:
    return AgentMessage(
        trace_id="tr",
        from_agent="user", to_agent="intake", intent="intake_turn",
        payload={
            "customer_id": customer_id, "session_id": session_id,
            "channel": "test", "content": content,
        },
    )


def test_intake_v2_records_each_learned_field():
    """Two learned fields, both with matching tool_calls, both persisted."""
    output = {
        "language": "en",
        "response_to_user": "Thanks Priya — what product are you looking to manufacture?",
        "fields_learned": [
            {"field": "contact_name", "value": "Priya", "confidence": 0.95},
            {"field": "company_name", "value": "Lumen Botanicals", "confidence": 0.95},
        ],
        "asked_for_field": "product_need",
        "tool_calls": [
            {"tool": "record_lead_fields", "field": "contact_name", "value": "Priya"},
            {"tool": "record_lead_fields", "field": "company_name", "value": "Lumen Botanicals"},
        ],
    }
    llm = _stub_llm(output)
    agent = IntakeAgentV2(llm=llm)
    reply = agent.handle(_msg("Hi I'm Priya from Lumen Botanicals."))
    assert reply.status == "success"
    recorded_fields = {r["field"] for r in reply.payload["fields_recorded"]}
    assert recorded_fields == {"contact_name", "company_name"}

    snapshot = get_lead_fields(customer_id="c1", trace_id="tr").output
    assert "contact_name" in snapshot["collected"]
    assert "company_name" in snapshot["collected"]
    assert "product_need" in snapshot["missing"]


def test_intake_v2_rejects_phantom_tool_call():
    """A tool_call without a matching fields_learned entry must NOT be persisted."""
    output = {
        "language": "en",
        "response_to_user": "What's your company name?",
        "fields_learned": [
            {"field": "contact_name", "value": "Priya", "confidence": 0.9},
        ],
        "asked_for_field": "company_name",
        "tool_calls": [
            {"tool": "record_lead_fields", "field": "contact_name", "value": "Priya"},
            # Phantom — the model emitted a tool_call without listing the field
            # as learned. We should NOT persist this.
            {"tool": "record_lead_fields", "field": "budget", "value": "PKR 10M"},
        ],
    }
    llm = _stub_llm(output)
    agent = IntakeAgentV2(llm=llm)
    reply = agent.handle(_msg("Hi I'm Priya."))
    assert reply.status == "success"

    snapshot = get_lead_fields(customer_id="c1", trace_id="tr").output
    assert "budget" not in snapshot["collected"], "phantom write leaked into storage"
    assert ("budget", "pkr 10m") in [tuple(x) for x in reply.payload["phantom_writes_rejected"]]


def test_intake_v2_skips_field_learned_without_tool_call():
    """A fields_learned entry without a matching tool_call must NOT be persisted
    either — Task 4A's discipline goes both directions."""
    output = {
        "language": "en",
        "response_to_user": "What product are you looking for?",
        "fields_learned": [
            {"field": "contact_name", "value": "Priya", "confidence": 0.9},
            {"field": "company_name", "value": "Lumen Botanicals", "confidence": 0.9},
        ],
        "asked_for_field": "product_need",
        "tool_calls": [
            # Only one tool_call — agent learned 2 fields but only committed to recording 1.
            {"tool": "record_lead_fields", "field": "contact_name", "value": "Priya"},
        ],
    }
    llm = _stub_llm(output)
    agent = IntakeAgentV2(llm=llm)
    reply = agent.handle(_msg("Hi I'm Priya from Lumen Botanicals."))
    assert reply.status == "success"

    snapshot = get_lead_fields(customer_id="c1", trace_id="tr").output
    assert "contact_name" in snapshot["collected"]
    assert "company_name" not in snapshot["collected"]
    assert reply.payload["unrecorded_learnings"]


def test_intake_v2_locks_language_on_first_turn():
    """The agent detects + locks the language on the first turn."""
    output_en = {
        "language": "en", "response_to_user": "Hi! What's your company name?",
        "fields_learned": [], "asked_for_field": "company_name", "tool_calls": [],
    }
    llm = _stub_llm(output_en)
    agent = IntakeAgentV2(llm=llm)
    agent.handle(_msg("hello there"))

    lang = get_session_language(session_id="s1", trace_id="tr").output["language"]
    assert lang == "en"


def test_intake_v2_flags_language_violation_against_locked_session():
    """If the model returns the wrong language for a locked session, flag it."""
    # First lock to English.
    lock_session_language(session_id="s1", language="en", trace_id="tr")

    output_ur = {
        "language": "ur",          # wrong — session is locked to 'en'
        "response_to_user": "آپ کا نام کیا ہے؟",
        "fields_learned": [], "asked_for_field": "company_name", "tool_calls": [],
    }
    llm = _stub_llm(output_ur)
    agent = IntakeAgentV2(llm=llm)
    reply = agent.handle(_msg("hello"))

    assert reply.status == "success"
    assert reply.payload["language_violation"] is True
    assert reply.payload["language"] == "en"  # effective language = locked one
