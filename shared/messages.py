"""The cross-task message envelope. Every inter-agent communication is one of these.

This is the contract Task 1 builds on, Task 3 extends (with `needs_human` status),
and Task 4 reads (for observability). Do not pass raw strings between agents.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .ids import new_id

AgentName = Literal[
    "supervisor", "inventory", "production", "report", "intake", "user", "system"
]

Intent = Literal[
    "plan",            # internal: supervisor decomposes objective
    "check_stock",     # supervisor -> inventory
    "check_schedule",  # supervisor -> production
    "flag_bottleneck", # production -> supervisor (alert)
    "draft_report",    # supervisor -> report
    "place_order",     # supervisor -> production (Task 3, HITL-gated)
    "synthesise",      # internal: supervisor builds final answer
    "clarify",         # any -> any (request more info)
    "escalate",        # any -> user (human needed)
    "intake_turn",     # used in Task 2/4 for customer intake
]

Status = Literal["pending", "success", "error", "needs_human", "timeout"]


class MessageMetadata(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model: str | None = None
    prompt_version: str | None = None
    cost_usd: float = 0.0


class AgentError(BaseModel):
    error_class: str  # e.g. 'schema_violation', 'timeout', 'tool_failure', 'circuit_open'
    message: str
    retryable: bool = True


class AgentMessage(BaseModel):
    trace_id: str
    span_id: str = Field(default_factory=lambda: new_id("span"))
    parent_span_id: str | None = None
    from_agent: AgentName
    to_agent: AgentName
    intent: Intent
    payload: dict[str, Any] = Field(default_factory=dict)
    status: Status = "pending"
    attempts: int = 0
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    error: AgentError | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def reply(
        self,
        *,
        status: Status,
        payload: dict | None = None,
        error: AgentError | None = None,
        metadata: MessageMetadata | None = None,
    ) -> "AgentMessage":
        return AgentMessage(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            intent=self.intent,
            payload=payload or {},
            status=status,
            attempts=self.attempts,
            error=error,
            metadata=metadata or MessageMetadata(),
        )
