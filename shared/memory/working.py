"""Working memory — the per-session, token-budgeted context window.

`build_context` is the single function the intake agent calls before its
reasoning LLM call. It enforces the token budget by promoting raw turns into a
rolling summary when needed, so the prompt never exceeds the budget no matter
how long the conversation runs.

Flow:
  summary (rolling, persisted in working_memory)
  + top-3 semantic facts (decay-weighted)
  + last N unsummarised raw turns (most recent first)
  -> if over budget: summarise oldest unsummarised turns into the summary,
     mark them summarised, recompute, loop (max 3 iterations).
"""
from __future__ import annotations

import os

from pydantic import BaseModel

from ..db import get_conn
from ..llm import LLMClient, LLMError
from ..trace import trace
from .episodic import list_unsummarised_turns, mark_summarised
from .schema import RetrievedFact, Turn, _now_iso
from .semantic import retrieve_facts

DEFAULT_BUDGET = int(os.getenv("TOKEN_BUDGET", "4000"))
FAST_MODEL = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

SUMMARISER_SYSTEM = """You consolidate older customer conversation turns into a
running summary. Preserve every concrete fact (names, numbers, dates, materials,
quantities, constraints). Drop pleasantries and filler. Output 1-3 sentences."""


def count_tokens(text: str) -> int:
    """Approximate. 4 chars per token is a reasonable rule-of-thumb for English
    + Llama tokenisers. See Task 2 tradeoffs for the upgrade path to tiktoken."""
    return len(text) // 4


class WorkingContext(BaseModel):
    session_id: str
    summary: str
    facts: list[RetrievedFact]
    raw_turns: list[Turn]
    token_count: int
    iterations: int = 0          # how many summarise-passes were needed


def _load_summary(session_id: str) -> tuple[str, int]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT summary, summary_token_count FROM working_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row["summary"]:
        return "", 0
    return row["summary"], row["summary_token_count"] or 0


def _save_summary(session_id: str, customer_id: str, summary: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO working_memory (session_id, customer_id, summary, summary_token_count, last_updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 summary = excluded.summary,
                 summary_token_count = excluded.summary_token_count,
                 last_updated_at = excluded.last_updated_at""",
            (session_id, customer_id, summary, count_tokens(summary), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def _format_fact(rf: RetrievedFact) -> str:
    f = rf.fact
    bits = [f"[{f.category}]"]
    if f.entity:
        bits.append(f"({f.entity})")
    bits.append(f.value)
    return " ".join(bits) + f"  ⟨score={rf.final_score:.2f}⟩"


def _total_tokens(summary: str, facts: list[RetrievedFact], turns: list[Turn]) -> int:
    t = count_tokens(summary)
    for rf in facts:
        t += count_tokens(_format_fact(rf))
    for tu in turns:
        t += count_tokens(f"{tu.role}: {tu.content}")
    return t


def _summarise_oldest(
    *,
    session_id: str,
    customer_id: str,
    existing_summary: str,
    turns: list[Turn],
    n_to_fold: int,
    llm: LLMClient,
    trace_id: str,
    parent_span_id: str | None = None,
) -> tuple[str, list[str]]:
    """Fold the oldest `n_to_fold` turns into the running summary. Returns the
    new summary and the list of turn_ids that were folded."""
    fold = turns[:n_to_fold]
    if not fold:
        return existing_summary, []

    transcript = "\n".join(f"{t.role}: {t.content}" for t in fold)
    user = (
        (f"Existing summary:\n{existing_summary}\n\n" if existing_summary else "")
        + f"New turns to fold in:\n{transcript}\n\n"
        "Produce an updated summary."
    )
    try:
        resp = llm.complete(
            system=SUMMARISER_SYSTEM,
            user=user,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            prompt_version="summariser.v1",
            model=FAST_MODEL,
            max_tokens=400,
        )
        new_summary = resp.content.strip()
    except LLMError:
        # Fall back to a plain concatenation rather than failing the agent turn.
        new_summary = (existing_summary + " " + transcript)[:2000]

    folded_ids = [t.turn_id for t in fold]
    _save_summary(session_id, customer_id, new_summary)
    mark_summarised(folded_ids)
    return new_summary, folded_ids


def build_context(
    *,
    session_id: str,
    customer_id: str,
    query: str,
    llm: LLMClient,
    budget: int = DEFAULT_BUDGET,
    max_raw_turns: int = 10,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> WorkingContext:
    """Assemble a token-budgeted context for the next agent reasoning call."""
    with trace(
        "memory.working.build_context",
        "tool",
        trace_id=trace_id or session_id,
        parent_span_id=parent_span_id,
        input_payload={"session_id": session_id, "budget": budget, "query": query[:200]},
    ) as rec:
        summary, _ = _load_summary(session_id)
        facts = retrieve_facts(
            customer_id=customer_id, query=query,
            trace_id=trace_id, parent_span_id=parent_span_id,
        )
        turns = list_unsummarised_turns(session_id)
        # Keep at most max_raw_turns most-recent unsummarised turns *in the prompt*;
        # the rest are candidates for summarisation if we're over budget.
        recent = turns[-max_raw_turns:]
        older = turns[:-max_raw_turns] if len(turns) > max_raw_turns else []

        total = _total_tokens(summary, facts, recent)
        iterations = 0

        # If we're already over budget, fold older turns first, then recent ones.
        candidates_to_fold: list[Turn] = older + recent[: max(0, len(recent) - 2)]

        while total > budget and iterations < 3 and candidates_to_fold:
            # Fold a chunk of the oldest candidates.
            n = max(2, len(candidates_to_fold) // 2)
            summary, _folded = _summarise_oldest(
                session_id=session_id,
                customer_id=customer_id,
                existing_summary=summary,
                turns=candidates_to_fold,
                n_to_fold=n,
                llm=llm,
                trace_id=trace_id or session_id,
                parent_span_id=parent_span_id,
            )
            # Reload the unsummarised set and recompute.
            turns = list_unsummarised_turns(session_id)
            recent = turns[-max_raw_turns:]
            older = turns[:-max_raw_turns] if len(turns) > max_raw_turns else []
            candidates_to_fold = older + recent[: max(0, len(recent) - 2)]
            total = _total_tokens(summary, facts, recent)
            iterations += 1

        ctx = WorkingContext(
            session_id=session_id,
            summary=summary,
            facts=facts,
            raw_turns=recent,
            token_count=total,
            iterations=iterations,
        )
        rec.output_payload = {
            "token_count": total,
            "n_facts": len(facts),
            "n_raw_turns": len(recent),
            "summary_iterations": iterations,
        }
        return ctx


def format_for_prompt(ctx: WorkingContext) -> str:
    """Render a WorkingContext into a single string for the agent's user prompt."""
    parts: list[str] = []
    if ctx.summary:
        parts.append(f"Conversation summary so far:\n{ctx.summary}")
    if ctx.facts:
        fact_lines = "\n".join(f"- {_format_fact(rf)}" for rf in ctx.facts)
        parts.append(f"Recalled facts about this customer:\n{fact_lines}")
    if ctx.raw_turns:
        recent_lines = "\n".join(f"{t.role}: {t.content}" for t in ctx.raw_turns)
        parts.append(f"Recent turns:\n{recent_lines}")
    return "\n\n".join(parts) if parts else "(no prior context)"
