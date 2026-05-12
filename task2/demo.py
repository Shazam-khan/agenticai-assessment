"""Two-session demo for customer 'lumen_botanicals'.

Session 1: customer introduces themselves and their needs.
Session 2: same customer returns; the agent must recall facts from session 1
           without being re-told them.

Run:
    python -m task2.demo

Prints, per turn: extracted facts, recalled facts (with decay+similarity),
agent response, token count. End-of-run prints a recall map.
"""
from __future__ import annotations

import sys

import shared.config  # noqa: F401 — loads .env
from shared.db import get_conn
from shared.ids import new_id
from shared.llm import LLMClient
from shared.messages import AgentMessage

from .intake_agent import IntakeAgent


CUSTOMER_ID = "lumen_botanicals"

SESSION_1 = [
    "Hi, I'm Priya from Lumen Botanicals. We're a skincare startup based in Karachi.",
    "We're looking to white-label a hydrating face serum.",
    "We need about 10,000 units, ideally in 30ml glass bottles.",
    "Our target launch is Q3 next year, so we'd need delivery by end of June.",
    "Our budget is around PKR 4 million for this batch.",
]

SESSION_2 = [
    "Hey, it's Priya again. Just checking in on next steps.",
    "Did you finalise the timeline we discussed?",
    "And does our budget constraint still work for you?",
]


def _print_turn_result(turn_idx: int, user_msg: str, result: dict) -> None:
    print(f"\n--- Turn {turn_idx} ---")
    print(f"USER: {user_msg}")
    if result.get("facts_extracted"):
        print(f"  facts extracted ({len(result['facts_extracted'])}):")
        for f in result["facts_extracted"]:
            ent = f"{f['entity']} | " if f["entity"] else ""
            print(f"    + [{f['category']}] {ent}{f['value']}")
    if result.get("facts_recalled"):
        print(f"  facts recalled ({len(result['facts_recalled'])}):")
        for f in result["facts_recalled"]:
            ent = f"{f['entity']} | " if f["entity"] else ""
            print(f"    <- [{f['category']}] {ent}{f['value']}  "
                  f"sim={f['similarity']} decay={f['decay']} score={f['score']}")
    print(f"  token_count={result['token_count']} "
          f"summary_iterations={result['summary_iterations']}")
    print(f"AGENT: {result['response'].strip()}")


def _run_session(agent: IntakeAgent, session_id: str, turns: list[str]) -> None:
    print(f"\n========== Session {session_id} ==========")
    for i, msg_text in enumerate(turns, start=1):
        message = AgentMessage(
            trace_id=new_id("trace"),
            from_agent="user",
            to_agent="intake",
            intent="intake_turn",
            payload={
                "customer_id": CUSTOMER_ID,
                "session_id": session_id,
                "channel": "demo",
                "content": msg_text,
            },
        )
        reply = agent.handle(message)
        if reply.status != "success":
            print(f"ERROR on turn {i}: {reply.error}")
            return
        _print_turn_result(i, msg_text, reply.payload)


def _print_recall_map() -> None:
    """List all stored facts for the customer — proves persistence."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT category, entity, value, access_count, last_accessed_at
               FROM semantic_facts WHERE customer_id = ?
               ORDER BY category, value""",
            (CUSTOMER_ID,),
        ).fetchall()
    finally:
        conn.close()
    print("\n--- All persisted facts for this customer ---")
    for r in rows:
        ent = f"{r['entity']} | " if r["entity"] else ""
        print(f"  [{r['category']}] {ent}{r['value']}  "
              f"(accessed {r['access_count']}x, last={r['last_accessed_at']})")


def main() -> int:
    llm = LLMClient()
    agent = IntakeAgent(llm=llm)

    session_1_id = new_id("sess1")
    session_2_id = new_id("sess2")

    _run_session(agent, session_1_id, SESSION_1)
    _run_session(agent, session_2_id, SESSION_2)
    _print_recall_map()
    return 0


if __name__ == "__main__":
    sys.exit(main())
