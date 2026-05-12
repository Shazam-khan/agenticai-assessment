"""Episodic memory — raw turn-by-turn conversation log.

This is the lowest layer of the three-layer memory architecture. Every user and
agent turn lands here. Promotion to semantic memory happens via `extractor.py`.
"""
from __future__ import annotations

from ..db import get_conn
from ..trace import trace
from .schema import Turn


def store_turn(
    *,
    session_id: str,
    customer_id: str,
    role: str,
    content: str,
    channel: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> Turn:
    turn = Turn(
        session_id=session_id,
        customer_id=customer_id,
        channel=channel,
        role=role,  # type: ignore[arg-type]
        content=content,
    )
    with trace(
        "memory.episodic.store_turn",
        "tool",
        trace_id=trace_id or session_id,
        parent_span_id=parent_span_id,
        input_payload={"session_id": session_id, "role": role, "content_len": len(content)},
    ) as rec:
        conn = get_conn()
        try:
            conn.execute(
                """INSERT INTO episodic_turns
                   (turn_id, session_id, customer_id, channel, role, content, ts, summarised)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (turn.turn_id, turn.session_id, turn.customer_id, turn.channel,
                 turn.role, turn.content, turn.ts),
            )
            conn.commit()
        finally:
            conn.close()
        rec.output_payload = {"turn_id": turn.turn_id}
    return turn


def list_turns(session_id: str) -> list[Turn]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT turn_id, session_id, customer_id, channel, role, content, ts
               FROM episodic_turns WHERE session_id = ? ORDER BY ts""",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [Turn(**dict(r)) for r in rows]


def list_unsummarised_turns(session_id: str) -> list[Turn]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT turn_id, session_id, customer_id, channel, role, content, ts
               FROM episodic_turns
               WHERE session_id = ? AND summarised = 0
               ORDER BY ts""",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [Turn(**dict(r)) for r in rows]


def mark_summarised(turn_ids: list[str]) -> None:
    if not turn_ids:
        return
    placeholders = ",".join(["?"] * len(turn_ids))
    conn = get_conn()
    try:
        conn.execute(
            f"UPDATE episodic_turns SET summarised = 1 WHERE turn_id IN ({placeholders})",
            turn_ids,
        )
        conn.commit()
    finally:
        conn.close()
