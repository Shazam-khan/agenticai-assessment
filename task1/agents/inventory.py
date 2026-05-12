"""Inventory Agent.

Deterministic data fetch + LLM-driven reasoning over alert severity and action.
The LLM does NOT see raw inventory data only; it sees the structured rows we
fetched and is asked to classify severity and recommend an action. Its output
is validated against `CheckStockOutput` by the base class.
"""
from __future__ import annotations

from shared.agent_base import Agent
from shared.intents import CheckStockInput, CheckStockOutput, StockAlert, StockLevel
from shared.messages import AgentMessage, MessageMetadata

from ..mock_data import get_stock_levels, KNOWN_MATERIALS

SYSTEM_PROMPT = """You are the Inventory Agent for a cosmetics manufacturer.

You receive a list of materials and their current stock levels. For each material,
classify the situation and recommend an action.

Severity rules:
- 'critical' if days_until_stockout <= 3 OR quantity <= 0.5 * reorder_level
- 'warning'  if quantity < reorder_level but not critical
- 'info'     if quantity >= reorder_level

The 'suggested_action' must be one short imperative sentence."""


class InventoryAgent(Agent):
    name = "inventory"
    prompt_version = "inv.v1"

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        req = CheckStockInput.model_validate(message.payload)
        rows = get_stock_levels(req.materials)
        unknown = [m for m in req.materials if m not in KNOWN_MATERIALS]

        # Deterministic part: assemble the StockLevel list.
        levels = [StockLevel(**r) for r in rows]

        # LLM part: classify each row + craft alerts. We always include unknowns as
        # an explicit alert so the supervisor sees them.
        levels_summary = "\n".join(
            f"- {l.material}: {l.quantity}{l.unit}, "
            f"reorder_level={l.reorder_level}, days_until_stockout={l.days_until_stockout}"
            for l in levels
        ) or "(no known materials in request)"

        user_prompt = (
            f"Materials with stock data:\n{levels_summary}\n\n"
            f"Unknown materials (not in our inventory): {unknown or 'none'}\n\n"
            "For each material with stock data, produce a StockAlert. "
            "For unknown materials, produce a StockAlert with severity='warning' "
            "and suggested_action explaining they aren't tracked."
        )

        alerts_envelope, resp = self.llm.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_AlertEnvelope,
            trace_id=message.trace_id,
            parent_span_id=parent_span_id,
            prompt_version=self.prompt_version,
        )

        output = CheckStockOutput(levels=levels, alerts=alerts_envelope.alerts)
        return message.reply(
            status="success",
            payload=output.model_dump(),
            metadata=MessageMetadata(
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                latency_ms=resp.latency_ms,
                model=resp.model,
                prompt_version=self.prompt_version,
                cost_usd=resp.cost_usd,
            ),
        )


# Wrapper so we can ask the LLM for a single JSON object {"alerts": [...]} —
# Groq's JSON mode requires a top-level object, not a bare list.
from pydantic import BaseModel  # noqa: E402


class _AlertEnvelope(BaseModel):
    alerts: list[StockAlert]
