"""Tool primitives: discriminated-union result, structured error, validated_tool decorator.

Every tool function in this codebase looks like:

    @validated_tool(name="get_stock_levels", input_schema=GetStockLevelsInput)
    def get_stock_levels(args: GetStockLevelsInput, *, trace_id, parent_span_id) -> ToolResult:
        ...

The decorator handles:
  1. Pydantic input-schema validation -> ToolResult.error('schema_violation', ...) on failure.
     (Critical: this is `ToolResult.error`, NOT an exception. Brief hard requirement #1.)
  2. Circuit-breaker check -> ToolResult.error('circuit_open', ...) if open.
  3. trace() span with actor_kind='tool' so Task 4C dashboard surfaces tool spans.
  4. record_success / record_failure on the breaker.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Literal, Type

from pydantic import BaseModel, ValidationError

from ..circuit_breaker import check_circuit, record_failure, record_success
from ..trace import trace


# ---------- types ----------

class ToolError(BaseModel):
    error_class: str
    message: str
    retryable: bool = True


class PendingConfirmation(BaseModel):
    confirmation_id: str
    tool_name: str
    inputs: dict
    reason: str
    expires_at: str


class ToolResult(BaseModel):
    status: Literal["ok", "error", "needs_confirmation"]
    output: dict | None = None
    error: ToolError | None = None
    confirmation: PendingConfirmation | None = None

    @classmethod
    def ok(cls, output: dict) -> "ToolResult":
        return cls(status="ok", output=output)

    @classmethod
    def err(cls, error_class: str, message: str, retryable: bool = True) -> "ToolResult":
        return cls(status="error", error=ToolError(
            error_class=error_class, message=message, retryable=retryable
        ))

    @classmethod
    def needs_confirmation(cls, confirmation: PendingConfirmation) -> "ToolResult":
        return cls(status="needs_confirmation", confirmation=confirmation)


# ---------- idempotency key helper ----------

def canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def inputs_idempotency_key(tool_name: str, inputs: dict) -> str:
    raw = f"{tool_name}|{canonical_json(inputs)}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------- decorator ----------

def validated_tool(
    name: str,
    input_schema: Type[BaseModel],
) -> Callable[[Callable], Callable]:
    """Wraps a tool function. The wrapped function receives the validated pydantic
    model as its first positional arg. All other args must be keyword-only and
    include `trace_id` (required) and `parent_span_id` (optional)."""

    def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            trace_id = kwargs.get("trace_id") or "tool-no-trace"
            parent_span_id = kwargs.get("parent_span_id")

            # Allow callers to pass raw kwargs OR an already-built input model.
            raw_inputs: dict
            if args and isinstance(args[0], BaseModel):
                raw_inputs = args[0].model_dump()
            else:
                # Strip out plumbing kwargs from the schema-validation set.
                raw_inputs = {
                    k: v for k, v in kwargs.items()
                    if k not in ("trace_id", "parent_span_id", "_caller")
                }

            # 1. Input validation. Failure -> structured ToolError, not exception.
            try:
                parsed = input_schema.model_validate(raw_inputs)
            except ValidationError as e:
                # Log a trace span for the rejected call so Task 4C still sees it.
                with trace(
                    name, "tool",
                    trace_id=trace_id, parent_span_id=parent_span_id,
                    input_payload=raw_inputs,
                ) as rec:
                    rec.status = "error"
                    rec.error_class = "schema_violation"
                    rec.output_payload = {"validation_error": str(e)[:500]}
                return ToolResult.err(
                    "schema_violation", str(e)[:500], retryable=False
                )

            # 2. Circuit check.
            allowed, reason = check_circuit(name)
            if not allowed:
                with trace(
                    name, "tool",
                    trace_id=trace_id, parent_span_id=parent_span_id,
                    input_payload=raw_inputs,
                ) as rec:
                    rec.status = "error"
                    rec.error_class = "circuit_open"
                    rec.output_payload = {"reason": reason}
                return ToolResult.err(
                    "circuit_open",
                    f"circuit for {name} is {reason}; call blocked",
                    retryable=True,
                )

            # 3. Invoke + trace.
            with trace(
                name, "tool",
                trace_id=trace_id, parent_span_id=parent_span_id,
                input_payload=raw_inputs,
            ) as rec:
                try:
                    result: ToolResult = fn(parsed, **{
                        k: v for k, v in kwargs.items()
                        if k in ("trace_id", "parent_span_id", "confirmed", "_caller")
                    })
                except Exception as e:
                    rec.status = "error"
                    rec.error_class = type(e).__name__
                    rec.output_payload = {"exception": str(e)[:300]}
                    record_failure(name)
                    return ToolResult.err(
                        "tool_exception", f"{type(e).__name__}: {e}", retryable=True
                    )

                rec.status = result.status
                if result.status == "error" and result.error:
                    rec.error_class = result.error.error_class
                    rec.output_payload = result.error.model_dump()
                    record_failure(name)
                elif result.status == "needs_confirmation" and result.confirmation:
                    rec.output_payload = result.confirmation.model_dump()
                    # Pending isn't a failure; reset failure counter.
                    record_success(name)
                else:
                    rec.output_payload = result.output
                    record_success(name)
                return result

        return wrapper

    return decorator


# ---------- iso-day helper (used by idempotency keys with daily granularity) ----------

def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()
