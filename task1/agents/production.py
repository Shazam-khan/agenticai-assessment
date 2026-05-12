"""Production Agent.

Two intents:

1. `check_schedule` — reads the production schedule via `tools.get_production_schedule`,
   asks the LLM to identify and classify bottlenecks, and persists each bottleneck
   via `tools.flag_bottleneck` (idempotent on (order_id, reason, day)).

2. `place_order` (Task 3) — calls `tools.create_purchase_order`. If quantity is
   over threshold and not yet confirmed, the tool returns `needs_confirmation`;
   the agent surfaces that to the supervisor as `status='needs_human'` so the
   supervisor can pause the run.
"""
from __future__ import annotations

from pydantic import BaseModel

from shared.agent_base import Agent
from shared.intents import (
    Bottleneck,
    CheckScheduleInput,
    CheckScheduleOutput,
    PlaceOrderInput,
    PlaceOrderOutput,
    ProductionOrder,
)
from shared.messages import AgentError, AgentMessage, MessageMetadata
from shared.tools.inventory_tools import create_purchase_order
from shared.tools.production_tools import flag_bottleneck, get_production_schedule

SYSTEM_PROMPT = """You are the Production Agent for a cosmetics manufacturer.

You receive a list of active production orders, each with completion_pct, ETA,
line assignment, and a 'blocked' flag with optional reason.

Identify the bottlenecks. For each blocked order, produce one Bottleneck record.
Severity rules:
- 'high'   if blocked AND completion_pct < 30
- 'medium' if blocked AND completion_pct between 30 and 70
- 'low'    if blocked AND completion_pct >= 70 (i.e. nearly done despite block)

Do NOT invent bottlenecks for orders that are not blocked."""


class _BottleneckEnvelope(BaseModel):
    bottlenecks: list[Bottleneck]


class ProductionAgent(Agent):
    name = "production"
    prompt_version = "prod.v2"          # bumped: now uses tools

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        if message.intent == "place_order":
            return self._handle_place_order(message, parent_span_id=parent_span_id)
        return self._handle_check_schedule(message, parent_span_id=parent_span_id)

    # ---- check_schedule ---------------------------------------------------

    def _handle_check_schedule(
        self, message: AgentMessage, *, parent_span_id: str
    ) -> AgentMessage:
        req = CheckScheduleInput.model_validate(message.payload)

        # Read via tool so the call is traced + circuit-broken like everything else.
        sched_result = get_production_schedule(
            date_range=req.date_range,
            trace_id=message.trace_id,
            parent_span_id=parent_span_id,
        )
        if sched_result.status != "ok":
            err = sched_result.error
            return message.reply(
                status="error",
                error=AgentError(
                    error_class=err.error_class if err else "tool_failure",
                    message=err.message if err else "get_production_schedule failed",
                    retryable=bool(err.retryable) if err else True,
                ),
            )
        raw = sched_result.output["orders"]

        orders = [
            ProductionOrder(
                order_id=o["order_id"], product=o["product"], line=o["line"],
                completion_pct=o["completion_pct"], eta=o["eta"],
            )
            for o in raw
        ]
        orders_summary = "\n".join(
            f"- {o['order_id']}: {o['product']} on {o['line']}, "
            f"{o['completion_pct']}% complete, eta={o['eta']}, "
            f"blocked={o['blocked']}, reason={o['block_reason']}"
            for o in raw
        )

        envelope, resp = self.llm.complete_json(
            system=SYSTEM_PROMPT,
            user=(
                f"Active production orders:\n{orders_summary}\n\n"
                "Return the bottlenecks list. If nothing is blocked, return an empty list."
            ),
            schema=_BottleneckEnvelope,
            trace_id=message.trace_id,
            parent_span_id=parent_span_id,
            prompt_version=self.prompt_version,
        )

        # Persist each bottleneck via flag_bottleneck. Idempotent: re-runs on the
        # same day with the same reason don't create duplicate alerts.
        for b in envelope.bottlenecks:
            flag_bottleneck(
                order_id=b.order_id, reason=b.reason, severity=b.severity,
                trace_id=message.trace_id, parent_span_id=parent_span_id,
            )

        output = CheckScheduleOutput(orders=orders, bottlenecks=envelope.bottlenecks)
        return message.reply(
            status="success",
            payload=output.model_dump(),
            metadata=MessageMetadata(
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                latency_ms=resp.latency_ms, model=resp.model,
                prompt_version=self.prompt_version, cost_usd=resp.cost_usd,
            ),
        )

    # ---- place_order (Task 3) --------------------------------------------

    def _handle_place_order(
        self, message: AgentMessage, *, parent_span_id: str
    ) -> AgentMessage:
        req = PlaceOrderInput.model_validate(message.payload)

        # `confirmed=True` is set by the resume-after-approval path. The tool
        # also auto-detects an approved pending row even when confirmed=False,
        # so resuming via a simple replay also works.
        confirmed = bool(message.payload.get("__confirmed__", False))
        result = create_purchase_order(
            material=req.material, quantity=req.quantity,
            supplier_id=req.supplier_id, urgency=req.urgency,
            confirmed=confirmed,
            trace_id=message.trace_id, parent_span_id=parent_span_id,
        )

        if result.status == "error":
            err = result.error
            return message.reply(
                status="error",
                error=AgentError(
                    error_class=err.error_class if err else "tool_failure",
                    message=err.message if err else "create_purchase_order failed",
                    retryable=bool(err.retryable) if err else False,
                ),
            )

        if result.status == "needs_confirmation":
            c = result.confirmation
            return message.reply(
                status="needs_human",
                payload=PlaceOrderOutput(
                    status="needs_confirmation",
                    confirmation_id=c.confirmation_id if c else None,
                    reason=c.reason if c else None,
                ).model_dump(),
            )

        out = result.output or {}
        return message.reply(
            status="success",
            payload=PlaceOrderOutput(
                po_id=out.get("po_id"),
                status="existing" if out.get("status") == "existing" else "created",
                confirmation_id=out.get("confirmation_id"),
            ).model_dump(),
        )
