"""Customer Intake Agent — Task 2 thin version.

Single turn in → store episodic → extract facts → upsert semantic → build
token-budgeted working context → reasoning LLM → store agent reply → return.

The system prompt here is intentionally minimal. Task 4A rewrites it to add the
deterministic-JSON output, jailbreak resistance, language commitment, one-field-
per-turn discipline, and tool-call rules. Keeping it minimal here lets us
demonstrate the memory plumbing without conflating the prompt-engineering work.
"""
from __future__ import annotations

from shared.agent_base import Agent
from shared.memory import episodic, working
from shared.memory.extractor import extract_facts
from shared.memory.semantic import store_fact
from shared.memory.working import format_for_prompt
from shared.messages import AgentMessage, MessageMetadata

INTAKE_SYSTEM_PLACEHOLDER = """You are the Customer Intake Agent for a cosmetics
manufacturer's new-business desk. You help customers describe what they want to
make. Be concise and professional. If the customer asks about something not in
the recalled facts or recent turns, say you don't have that information yet —
do NOT invent details. This is a placeholder prompt; Task 4A replaces it."""


class IntakeAgent(Agent):
    name = "intake"
    prompt_version = "intake.t2.v1"

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        # We do NOT register intake_turn in INTENT_INPUT_SCHEMA — validate inline.
        p = message.payload
        for k in ("customer_id", "session_id", "content"):
            if not p.get(k):
                from shared.messages import AgentError
                return message.reply(
                    status="error",
                    error=AgentError(
                        error_class="schema_violation",
                        message=f"intake payload missing '{k}'",
                        retryable=False,
                    ),
                )

        customer_id = p["customer_id"]
        session_id = p["session_id"]
        channel = p.get("channel")
        content = p["content"]

        # 1. Store the user's turn.
        user_turn = episodic.store_turn(
            session_id=session_id, customer_id=customer_id,
            channel=channel, role="user", content=content,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 2. Extract facts from the user turn (fast model).
        new_facts = extract_facts(
            turn=user_turn, llm=self.llm,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 3. Persist each fact (upserts via dedupe_key).
        persisted: list[dict] = []
        for f in new_facts:
            stored = store_fact(f, trace_id=message.trace_id, parent_span_id=parent_span_id)
            persisted.append({
                "category": stored.category, "entity": stored.entity, "value": stored.value,
            })

        # 4. Assemble working context with token-budget enforcement.
        ctx = working.build_context(
            session_id=session_id, customer_id=customer_id,
            query=content, llm=self.llm,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 5. Reasoning LLM call with recalled context + the new turn.
        user_prompt = (
            f"{format_for_prompt(ctx)}\n\n"
            f"New customer message:\n{content}\n\n"
            "Reply briefly and only with information present above. If something "
            "wasn't discussed, say you don't have it yet."
        )
        resp = self.llm.complete(
            system=INTAKE_SYSTEM_PLACEHOLDER,
            user=user_prompt,
            trace_id=message.trace_id,
            parent_span_id=parent_span_id,
            prompt_version=self.prompt_version,
        )

        # 6. Persist the agent's reply turn.
        episodic.store_turn(
            session_id=session_id, customer_id=customer_id,
            channel=channel, role="agent", content=resp.content,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        return message.reply(
            status="success",
            payload={
                "response": resp.content,
                "facts_extracted": persisted,
                "facts_recalled": [
                    {
                        "category": rf.fact.category, "entity": rf.fact.entity,
                        "value": rf.fact.value,
                        "similarity": round(rf.similarity, 3),
                        "decay": round(rf.decay_weight, 3),
                        "score": round(rf.final_score, 3),
                    }
                    for rf in ctx.facts
                ],
                "token_count": ctx.token_count,
                "summary_iterations": ctx.iterations,
            },
            metadata=MessageMetadata(
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                latency_ms=resp.latency_ms,
                model=resp.model,
                prompt_version=self.prompt_version,
                cost_usd=resp.cost_usd,
            ),
        )
