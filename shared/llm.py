"""Groq client wrapper.

Captures tokens, latency, and cost on every call into the shared trace store.
Exposes `complete` (text) and `complete_json` (validated against a pydantic schema).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from .trace import trace

# Approximate public Groq pricing (USD per million tokens). Update if pricing changes.
PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str
    cost_usd: float


class LLMError(Exception):
    """Raised when the LLM returns unparseable or schema-violating output."""


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GROQ_MODEL_REASONING", "llama-3.3-70b-versatile")
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = None  # lazy: avoid requiring API key at import time

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("GROQ_API_KEY not set. Copy .env.example to .env.")
            from groq import Groq  # lazy import so tests can stub LLMClient.complete
            self._client = Groq(api_key=self._api_key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        trace_id: str,
        parent_span_id: str | None = None,
        prompt_version: str = "v1",
        json_mode: bool = False,
        temperature: float = 0.0,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        model = model or self.model
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        with trace(
            "llm",
            "llm",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            model=model,
            prompt_version=prompt_version,
            input_payload={"system": system[:300], "user": user[:600], "json_mode": json_mode},
        ) as rec:
            start = time.perf_counter()
            resp = self.client.chat.completions.create(**kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            in_price, out_price = PRICING.get(model, (0.0, 0.0))
            cost = (in_tok / 1_000_000) * in_price + (out_tok / 1_000_000) * out_price

            rec.output_payload = {"content": content[:500]}
            rec.input_tokens = in_tok
            rec.output_tokens = out_tok
            rec.cost_usd = cost

            return LLMResponse(
                content=content,
                tokens_in=in_tok,
                tokens_out=out_tok,
                latency_ms=elapsed_ms,
                model=model,
                cost_usd=cost,
            )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        trace_id: str,
        parent_span_id: str | None = None,
        prompt_version: str = "v1",
        model: str | None = None,
    ) -> tuple[T, LLMResponse]:
        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system}\n\n"
            "Respond with a single JSON object that matches this schema. "
            "Do not include any prose, markdown fences, or explanation outside the JSON.\n\n"
            f"{schema_hint}"
        )
        resp = self.complete(
            system=full_system,
            user=user,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            prompt_version=prompt_version,
            json_mode=True,
            model=model,
        )
        try:
            raw = json.loads(resp.content)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned invalid JSON: {e}. Content: {resp.content[:300]}")
        try:
            parsed = schema.model_validate(raw)
        except ValidationError as e:
            raise LLMError(f"LLM JSON did not match {schema.__name__}: {e}")
        return parsed, resp
