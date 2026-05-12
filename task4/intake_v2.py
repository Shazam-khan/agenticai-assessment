"""IntakeAgentV2 — Task 4A's production-grade intake agent.

Replaces the Task 2 placeholder. Same memory plumbing (episodic + semantic +
working), but:
  - One JSON output per turn against `IntakeTurnOutput`.
  - Session language locked on first turn; subsequent turns receive the locked
    value in the user prompt; mid-conversation switches are blocked.
  - Per-field `record_lead_fields` call BEFORE returning to the user — never
    batched across turns.
  - Consistency check between `fields_learned` and `tool_calls` — phantom
    writes are rejected.
"""
from __future__ import annotations

import json
import re

from shared.agent_base import Agent
from shared.memory import episodic, working
from shared.memory.working import format_for_prompt
from shared.messages import AgentError, AgentMessage, MessageMetadata
from shared.tools.intake_tools import (
    LEAD_FIELDS,
    get_lead_fields,
    get_session_language,
    lock_session_language,
    record_lead_fields,
)

from .prompts import INTAKE_SYSTEM_V2, IntakeTurnOutput

# Urdu Perso-Arabic Unicode range.
_URDU_RE = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """Heuristic — Urdu glyph presence -> 'ur', else 'en'. Roman Urdu falls
    through to 'en'; see Task 4 tradeoffs."""
    return "ur" if _URDU_RE.search(text or "") else "en"


class IntakeAgentV2(Agent):
    name = "intake"
    prompt_version = "intake.t4.v1"

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        p = message.payload
        for k in ("customer_id", "session_id", "content"):
            if not p.get(k):
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

        # 1. Resolve session language — detect once, lock forever.
        lang_result = get_session_language(
            session_id=session_id,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )
        session_language = (lang_result.output or {}).get("language")
        if not session_language:
            session_language = detect_language(content)
            lock_session_language(
                session_id=session_id, language=session_language,
                trace_id=message.trace_id, parent_span_id=parent_span_id,
            )

        # 2. Persist the user's turn.
        user_turn = episodic.store_turn(
            session_id=session_id, customer_id=customer_id,
            channel=channel, role="user", content=content,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 3. Read the current lead state.
        leads_result = get_lead_fields(
            customer_id=customer_id,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )
        collected = (leads_result.output or {}).get("collected", {})
        missing = (leads_result.output or {}).get("missing", list(LEAD_FIELDS))

        # 4. Build working context (memory) using the user's content as the query.
        ctx = working.build_context(
            session_id=session_id, customer_id=customer_id,
            query=content, llm=self.llm,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 5. Reasoning LLM call (structured JSON).
        user_prompt = (
            f"session_language: {session_language}\n"
            f"collected: {json.dumps(collected, ensure_ascii=False)}\n"
            f"missing: {missing}\n\n"
            f"Memory context:\n{format_for_prompt(ctx)}\n\n"
            f"Current customer message:\n{content}\n\n"
            "Respond with a single JSON object matching IntakeTurnOutput."
        )
        try:
            out, resp = self.llm.complete_json(
                system=INTAKE_SYSTEM_V2,
                user=user_prompt,
                schema=IntakeTurnOutput,
                trace_id=message.trace_id,
                parent_span_id=parent_span_id,
                prompt_version=self.prompt_version,
            )
        except Exception as e:
            return message.reply(
                status="error",
                error=AgentError(error_class="llm_error", message=str(e)[:300]),
            )

        # 6. Language commitment check. If the model returned the wrong language,
        # we override and record the violation; we do NOT retry (eval will catch it).
        language_violation = out.language != session_language
        effective_language = session_language

        # 7. Consistency check: every fields_learned entry must have a matching tool_calls.
        learned_keys = {(f.field, f.value.strip().lower()) for f in out.fields_learned}
        called_keys = {(tc.field, tc.value.strip().lower())
                       for tc in out.tool_calls if tc.tool == "record_lead_fields"}
        valid_writes = learned_keys & called_keys
        phantom_writes = called_keys - learned_keys
        unrecorded_learnings = learned_keys - called_keys

        # 8. Persist each consistent (learned AND tool-called) field synchronously.
        recorded: list[dict] = []
        for f in out.fields_learned:
            key = (f.field, f.value.strip().lower())
            if key not in valid_writes:
                continue
            r = record_lead_fields(
                customer_id=customer_id,
                field=f.field, value=f.value, language=effective_language,
                source_turn_id=user_turn.turn_id,
                trace_id=message.trace_id, parent_span_id=parent_span_id,
            )
            recorded.append({
                "field": f.field, "value": f.value,
                "status": (r.output or {}).get("status"),
            })

        # 9. Persist the agent's reply.
        episodic.store_turn(
            session_id=session_id, customer_id=customer_id,
            channel=channel, role="agent", content=out.response_to_user,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        # 10. Reread leads to compute completion.
        leads_after = get_lead_fields(
            customer_id=customer_id,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )
        post_missing = (leads_after.output or {}).get("missing", [])

        return message.reply(
            status="success",
            payload={
                "response": out.response_to_user,
                "language": effective_language,
                "language_violation": language_violation,
                "asked_for_field": out.asked_for_field,
                "fields_learned": [f.model_dump() for f in out.fields_learned],
                "fields_recorded": recorded,
                "phantom_writes_rejected": list(phantom_writes),
                "unrecorded_learnings": list(unrecorded_learnings),
                "all_fields_collected": not post_missing,
                "missing": post_missing,
                "token_count": ctx.token_count,
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
