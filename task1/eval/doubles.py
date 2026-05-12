"""Test doubles for the eval harness — injected via the registry, never patched.

- AlwaysFailingProduction: returns a non-retryable error every call. Lets us prove
  the supervisor escalates AND continues with the remaining subtasks (graceful
  degradation, Task 1 hard requirement #4).

- HallucinatingInventoryOnce: returns a malformed payload on the first call, then
  a valid payload on the second. Proves the supervisor's retry-with-modified-
  instruction works (Task 1 hard requirement #2). We do this WITHOUT making the
  agent inherit the base class's auto-validation — we want to surface the
  schema_violation from the base class, not from inside the agent.
"""
from __future__ import annotations

from shared.agent_base import Agent
from shared.intents import (
    CheckScheduleOutput,
    CheckStockInput,
    CheckStockOutput,
    StockAlert,
    StockLevel,
)
from shared.messages import AgentError, AgentMessage, MessageMetadata

from task1.mock_data import get_stock_levels


class AlwaysFailingProduction(Agent):
    name = "production"
    prompt_version = "prod.fail.v1"

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        return message.reply(
            status="error",
            error=AgentError(
                error_class="upstream_unavailable",
                message="Mock production ERP is offline.",
                retryable=True,
            ),
            metadata=MessageMetadata(prompt_version=self.prompt_version),
        )


class HallucinatingInventoryOnce(Agent):
    """First call returns a payload missing required fields; second returns valid data."""

    name = "inventory"
    prompt_version = "inv.hallu.v1"

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.calls = 0

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        self.calls += 1
        req = CheckStockInput.model_validate(message.payload)
        if self.calls == 1:
            # Intentionally malformed: 'levels' has wrong shape; base-class
            # output validator will reject it as schema_violation.
            return message.reply(
                status="success",
                payload={"levels": "not a list", "alerts": []},
                metadata=MessageMetadata(prompt_version=self.prompt_version),
            )
        # Second call: produce valid output deterministically (no LLM).
        rows = get_stock_levels(req.materials)
        levels = [StockLevel(**r) for r in rows]
        alerts = [
            StockAlert(
                material=l.material,
                severity="critical" if l.days_until_stockout <= 3 else "info",
                suggested_action="Reorder soon." if l.days_until_stockout <= 3 else "No action.",
            )
            for l in levels
        ]
        output = CheckStockOutput(levels=levels, alerts=alerts)
        return message.reply(
            status="success",
            payload=output.model_dump(),
            metadata=MessageMetadata(prompt_version=self.prompt_version),
        )
