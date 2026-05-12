"""Circuit-breaker state-machine tests.

closed --(N failures)--> open --(cooldown elapsed)--> half_open
half_open --(success)--> closed
half_open --(failure)--> open (cooldown doubled)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from shared import circuit_breaker as cb
from shared.tools.base import ToolResult, validated_tool


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db))
    import shared.db
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    yield


class _NoArgs(BaseModel):
    pass


def _make_failing_tool(name: str):
    @validated_tool(name=name, input_schema=_NoArgs)
    def fail(_args, *, trace_id, parent_span_id=None):
        raise RuntimeError("upstream down")
    return fail


def _make_passing_tool(name: str):
    @validated_tool(name=name, input_schema=_NoArgs)
    def succeed(_args, *, trace_id, parent_span_id=None):
        return ToolResult.ok({"hello": "world"})
    return succeed


def _state(name: str) -> str:
    from shared.db import get_conn
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT state FROM circuit_state WHERE tool_name = ?", (name,)
        ).fetchone()
    finally:
        conn.close()
    return row["state"] if row else "missing"


def test_three_failures_open_circuit():
    fail = _make_failing_tool("test_tool_1")
    for _ in range(3):
        r = fail(trace_id="t")
        assert r.status == "error"
    assert _state("test_tool_1") == "open"


def test_open_circuit_short_circuits_call_without_invoking_tool():
    name = "test_tool_2"
    fail = _make_failing_tool(name)
    for _ in range(3):
        fail(trace_id="t")
    assert _state(name) == "open"

    # Next call: should return circuit_open without ever entering the function.
    # We assert that by also installing a side-effect counter via a fresh tool
    # under the same name pointing at a counter — actually we just check the
    # error class and that consecutive_failures didn't increment further.
    from shared.db import get_conn
    conn = get_conn()
    before = conn.execute(
        "SELECT consecutive_failures FROM circuit_state WHERE tool_name = ?", (name,)
    ).fetchone()["consecutive_failures"]
    conn.close()

    r = fail(trace_id="t")
    assert r.status == "error"
    assert r.error.error_class == "circuit_open"

    conn = get_conn()
    after = conn.execute(
        "SELECT consecutive_failures FROM circuit_state WHERE tool_name = ?", (name,)
    ).fetchone()["consecutive_failures"]
    conn.close()
    assert after == before  # short-circuited; failure NOT recorded again


def test_cooldown_transitions_to_half_open_and_success_closes(monkeypatch):
    name = "test_tool_3"
    fail = _make_failing_tool(name)
    for _ in range(3):
        fail(trace_id="t")
    assert _state(name) == "open"

    # Move opened_at far enough into the past that the cooldown is elapsed.
    from shared.db import get_conn
    conn = get_conn()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    conn.execute(
        "UPDATE circuit_state SET opened_at = ? WHERE tool_name = ?",
        (past, name),
    )
    conn.commit()
    conn.close()

    # Now redefine the tool to succeed and call it once; should flip
    # closed via half_open path.
    succeed = _make_passing_tool(name)
    r = succeed(trace_id="t")
    assert r.status == "ok"
    assert _state(name) == "closed"


def test_half_open_probe_fails_back_to_open_with_doubled_cooldown():
    name = "test_tool_4"
    fail = _make_failing_tool(name)
    for _ in range(3):
        fail(trace_id="t")

    from shared.db import get_conn
    conn = get_conn()
    initial_cooldown = conn.execute(
        "SELECT cooldown_seconds FROM circuit_state WHERE tool_name = ?", (name,)
    ).fetchone()["cooldown_seconds"]
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    conn.execute(
        "UPDATE circuit_state SET opened_at = ? WHERE tool_name = ?",
        (past, name),
    )
    conn.commit()
    conn.close()

    # Half-open probe fails -> open with doubled cooldown.
    r = fail(trace_id="t")
    assert r.status == "error"
    assert _state(name) == "open"
    conn = get_conn()
    new_cooldown = conn.execute(
        "SELECT cooldown_seconds FROM circuit_state WHERE tool_name = ?", (name,)
    ).fetchone()["cooldown_seconds"]
    conn.close()
    assert new_cooldown == min(initial_cooldown * 2, 300)
