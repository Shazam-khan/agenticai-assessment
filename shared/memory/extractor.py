"""LLM-based fact extraction.

The promotion logic from episodic -> semantic. Called once per user turn with
the fast model (Llama 3.1 8B Instant). Returns zero or more Facts that the
intake agent then persists via `semantic.store_fact` (which upserts on dedupe).

The prompt is intentionally conservative: extract only what is *explicitly*
said. We rely on the agent's *response* to ask follow-up questions for things
that need clarification; the extractor never infers.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from ..llm import LLMClient, LLMError
from .schema import Fact, FactCategory, Turn

FAST_MODEL = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

EXTRACTOR_SYSTEM = """You extract explicit, concrete facts from a single customer message.

Rules:
- Extract ONLY what is explicitly stated. Do not infer, guess, or paraphrase aggressively.
- One fact = one atomic, self-contained piece of information.
- If the message is small talk, a question, or contains no concrete information, return an empty list.

For each fact, choose ONE category:
  company_info   — the customer's company name, industry, location, size
  contact        — names, roles, email, phone
  product_need   — what they want made (e.g. "hydrating face serum")
  quantity       — numeric amounts (e.g. "10,000 units", "5 batches")
  timeline       — dates, deadlines, target quarters (e.g. "Q3 2026", "by end of June")
  budget         — money amounts (e.g. "PKR 4M", "$50k")
  constraint     — explicit constraints / requirements (packaging type, certifications, etc.)
  preference     — soft preferences (e.g. "prefers glass over plastic")

Fields:
  category   — one of the above
  entity     — the subject the fact attaches to (company name, person, etc.) — may be null
  value      — the fact itself, terse, lower-case unless proper noun
  confidence — 0..1, how clearly the message stated this (0.9 = explicit, 0.6 = implied-but-clear)
"""


class _ExtractedFact(BaseModel):
    category: FactCategory
    entity: str | None = None
    value: str
    confidence: float = 0.8


class _ExtractionEnvelope(BaseModel):
    facts: list[_ExtractedFact]


def extract_facts(
    *,
    turn: Turn,
    llm: LLMClient,
    trace_id: str,
    parent_span_id: str | None = None,
) -> list[Fact]:
    """Returns a list of (un-persisted) Facts derived from the user's turn."""
    if turn.role != "user":
        return []

    user_prompt = f"Customer message:\n\"\"\"\n{turn.content}\n\"\"\""

    try:
        envelope, _resp = llm.complete_json(
            system=EXTRACTOR_SYSTEM,
            user=user_prompt,
            schema=_ExtractionEnvelope,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            prompt_version="extractor.v1",
            model=FAST_MODEL,
        )
    except LLMError:
        # Extraction is best-effort; a parsing failure shouldn't block the agent's reply.
        return []

    facts: list[Fact] = []
    for ef in envelope.facts:
        f = Fact(
            customer_id=turn.customer_id,
            category=ef.category,
            entity=ef.entity,
            value=ef.value.strip(),
            confidence=ef.confidence,
            source_turn_id=turn.turn_id,
        ).with_dedupe_key()
        facts.append(f)
    return facts
