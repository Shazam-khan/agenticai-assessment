"""Report Agent.

Takes structured outputs from other agents and produces a management briefing.
Output validated against `DraftReportOutput`.
"""
from __future__ import annotations

from shared.agent_base import Agent
from shared.intents import DraftReportInput, DraftReportOutput
from shared.messages import AgentMessage, MessageMetadata

SYSTEM_PROMPT = """You are the Report Agent. You compile a concise management
briefing from structured data supplied by other agents.

Rules:
- Be terse. One paragraph for the summary, one short paragraph per section body.
- Do NOT invent facts. Only restate what the input data contains.
- If a section's data is empty or trivial, say so explicitly rather than padding.
- Output JSON matching the requested schema. No markdown fences."""


class ReportAgent(Agent):
    name = "report"
    prompt_version = "rep.v1"

    def _handle(self, message: AgentMessage, *, parent_span_id: str) -> AgentMessage:
        req = DraftReportInput.model_validate(message.payload)

        sections_text = "\n\n".join(
            f"### {s.heading}\n```json\n{s.data}\n```"
            for s in req.sections
        )

        user_prompt = (
            f"Briefing title: {req.title}\n\n"
            f"Source data from other agents:\n\n{sections_text}\n\n"
            "Produce a DraftReportOutput. The summary should call out the most "
            "important point first."
        )

        output, resp = self.llm.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=DraftReportOutput,
            trace_id=message.trace_id,
            parent_span_id=parent_span_id,
            prompt_version=self.prompt_version,
        )

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
