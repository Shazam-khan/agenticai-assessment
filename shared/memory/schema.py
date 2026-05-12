"""Pydantic models + dedupe-key helper for the memory layer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ..ids import new_id

FactCategory = Literal[
    "company_info",
    "contact",
    "product_need",
    "quantity",
    "timeline",
    "budget",
    "constraint",
    "preference",
    "other",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_dedupe_key(customer_id: str, category: str, entity: str | None, value: str) -> str:
    """Exact-text dedupe (see Task 2 tradeoffs). Two paraphrases get two rows."""
    norm = value.lower().strip()
    raw = f"{customer_id}|{category}|{entity or ''}|{norm}"
    return hashlib.sha256(raw.encode()).hexdigest()


class Turn(BaseModel):
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    session_id: str
    customer_id: str
    channel: str | None = None
    role: Literal["user", "agent"]
    content: str
    ts: str = Field(default_factory=_now_iso)


class Fact(BaseModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    customer_id: str
    category: FactCategory
    entity: str | None = None
    value: str
    confidence: float = 0.8
    source_turn_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    last_accessed_at: str = Field(default_factory=_now_iso)
    access_count: int = 0
    dedupe_key: str = ""

    def with_dedupe_key(self) -> "Fact":
        if not self.dedupe_key:
            self.dedupe_key = make_dedupe_key(
                self.customer_id, self.category, self.entity, self.value
            )
        return self


class RetrievedFact(BaseModel):
    fact: Fact
    similarity: float       # cosine similarity 0..1
    decay_weight: float     # 0..1
    final_score: float      # similarity * decay_weight
