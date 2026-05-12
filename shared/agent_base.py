"""Base class for all agents.

Enforces the message-in / message-out contract, wraps every call in a trace span,
validates payloads against the per-intent schema registry, and converts any
exception into an `AgentError` reply so the supervisor sees structured failures
instead of crashes (Task 1 hard requirement #4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import ValidationError

from .intents import INTENT_INPUT_SCHEMA, INTENT_OUTPUT_SCHEMA
from .llm import LLMClient, LLMError
from .messages import AgentError, AgentMessage, AgentName
from .trace import trace


class Agent(ABC):
    name: ClassVar[AgentName]
    prompt_version: ClassVar[str] = "v1"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def handle(self, message: AgentMessage) -> AgentMessage:
        """Public entrypoint. Wraps `_handle` in a trace + structured error handling."""
        with trace(
            self.name,
            "agent",
            trace_id=message.trace_id,
            parent_span_id=message.parent_span_id,
            prompt_version=self.prompt_version,
            input_payload={"intent": message.intent, "payload": message.payload},
        ) as rec:
            # 1) validate input payload
            input_schema = INTENT_INPUT_SCHEMA.get(message.intent)
            if input_schema is not None:
                try:
                    input_schema.model_validate(message.payload)
                except ValidationError as e:
                    err = AgentError(
                        error_class="schema_violation",
                        message=f"Input payload does not match {input_schema.__name__}: {e}",
                        retryable=False,
                    )
                    rec.status = "error"
                    rec.error_class = err.error_class
                    return message.reply(status="error", error=err)

            # 2) delegate to subclass
            try:
                reply = self._handle(message, parent_span_id=rec.span_id)
            except LLMError as e:
                err = AgentError(error_class="llm_error", message=str(e), retryable=True)
                rec.status = "error"
                rec.error_class = err.error_class
                return message.reply(status="error", error=err)
            except Exception as e:  # pragma: no cover — defensive catch-all
                err = AgentError(
                    error_class=type(e).__name__, message=str(e)[:300], retryable=True
                )
                rec.status = "error"
                rec.error_class = err.error_class
                return message.reply(status="error", error=err)

            # 3) validate output payload (catches hallucinated/malformed agent output)
            output_schema = INTENT_OUTPUT_SCHEMA.get(message.intent)
            if reply.status == "success" and output_schema is not None:
                try:
                    output_schema.model_validate(reply.payload)
                except ValidationError as e:
                    err = AgentError(
                        error_class="schema_violation",
                        message=f"Output payload does not match {output_schema.__name__}: {e}",
                        retryable=True,
                    )
                    rec.status = "error"
                    rec.error_class = err.error_class
                    return message.reply(status="error", error=err)

            rec.status = reply.status
            rec.output_payload = {"status": reply.status, "payload": reply.payload}
            rec.input_tokens = reply.metadata.tokens_in
            rec.output_tokens = reply.metadata.tokens_out
            rec.cost_usd = reply.metadata.cost_usd
            if reply.error:
                rec.error_class = reply.error.error_class
            return reply

    @abstractmethod
    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        ...
